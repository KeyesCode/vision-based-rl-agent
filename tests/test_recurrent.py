"""Tests for the recurrent PPO path.

Protects two correctness properties that feedforward PPO can't catch:

1. ``episode_starts == 1`` must zero the LSTM hidden state *before* that step.
2. Full end-to-end one-update smoke — rollout collection + sequence-replay update —
   must complete without NaNs or shape mismatches.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv

from osrs_rl.agents.ppo import RecurrentPPOActorCritic, RecurrentPPOTrainer
from osrs_rl.agents.rollout_buffer import RecurrentRolloutBuffer
from osrs_rl.env.action_space import ActionDecoder
from osrs_rl.env.osrs_env import make_env
from osrs_rl.utils.config import EnvConfig, PPOConfig, RewardConfig, VisionConfig


def _tiny_policy(batch: int = 2, hidden: int = 16) -> RecurrentPPOActorCritic:
    return RecurrentPPOActorCritic(
        num_actions=ActionDecoder.n_actions(),
        in_channels=2,
        input_hw=(84, 84),
        feature_dim=64,
        hidden_size=hidden,
    )


def test_initial_hidden_shape():
    policy = _tiny_policy(hidden=24)
    h, c = policy.initial_hidden(5, torch.device("cpu"))
    assert h.shape == (1, 5, 24)
    assert c.shape == (1, 5, 24)
    assert torch.all(h == 0) and torch.all(c == 0)


def test_episode_start_resets_hidden_state():
    """A step with episode_starts=1 must be identical to a step from zero hidden."""
    policy = _tiny_policy(hidden=16)
    policy.eval()
    obs = torch.randint(0, 255, (1, 2, 84, 84), dtype=torch.uint8)
    device = torch.device("cpu")

    # Path A: start from nonzero hidden, but pass episode_starts=1 -> LSTM must
    # reset before stepping, so the output matches path B.
    non_zero = (
        torch.randn(1, 1, 16) * 5.0,
        torch.randn(1, 1, 16) * 5.0,
    )
    with torch.no_grad():
        _, _, v_a, _ = policy.act(obs, non_zero, torch.ones(1), deterministic=True)

    # Path B: start from zero hidden, episode_starts=0 -> same effective starting state.
    zero = policy.initial_hidden(1, device)
    with torch.no_grad():
        _, _, v_b, _ = policy.act(obs, zero, torch.zeros(1), deterministic=True)

    assert torch.allclose(v_a, v_b, atol=1e-6), (v_a, v_b)


def test_hidden_state_persists_without_reset():
    """Without a reset signal, two consecutive calls must advance the hidden state."""
    policy = _tiny_policy(hidden=16)
    policy.eval()
    obs = torch.randint(0, 255, (1, 2, 84, 84), dtype=torch.uint8)
    hidden = policy.initial_hidden(1, torch.device("cpu"))
    _, _, _, next_hidden = policy.act(obs, hidden, torch.zeros(1), deterministic=True)
    h0, _ = hidden
    h1, _ = next_hidden
    # Non-trivial obs must have moved the hidden vector off the zero origin.
    assert not torch.allclose(h0, h1)


def test_recurrent_ppo_smoke_update():
    """One full rollout + one PPO update. Must finish with finite losses."""
    env_cfg = EnvConfig(max_episode_steps=32)
    reward_cfg = RewardConfig()
    vision_cfg = VisionConfig(frame_stack=2)
    ppo_cfg = PPOConfig(
        total_timesteps=64,
        num_envs=2,
        rollout_steps=16,
        num_epochs=1,
        num_minibatches=2,
        recurrent=True,
        lstm_hidden_size=16,
    )

    envs = SyncVectorEnv(
        [make_env(env_cfg, vision_cfg, reward_cfg, seed=0, idx=i) for i in range(ppo_cfg.num_envs)]
    )
    obs, _ = envs.reset(seed=0)

    obs_shape = envs.single_observation_space.shape
    device = torch.device("cpu")
    policy = RecurrentPPOActorCritic(
        num_actions=ActionDecoder.n_actions(),
        in_channels=obs_shape[0],
        input_hw=(obs_shape[1], obs_shape[2]),
        hidden_size=ppo_cfg.lstm_hidden_size,
    )
    trainer = RecurrentPPOTrainer(policy, ppo_cfg, device)

    buf = RecurrentRolloutBuffer(
        rollout_steps=ppo_cfg.rollout_steps,
        num_envs=ppo_cfg.num_envs,
        obs_shape=obs_shape,
        hidden_size=ppo_cfg.lstm_hidden_size,
        device=device,
    )
    hidden = policy.initial_hidden(ppo_cfg.num_envs, device)
    buf.set_initial_hidden(hidden)

    obs_t = torch.as_tensor(obs, device=device)
    done_t = torch.zeros(ppo_cfg.num_envs, device=device)

    for _ in range(ppo_cfg.rollout_steps):
        with torch.no_grad():
            action, logp, value, hidden = policy.act(obs_t, hidden, done_t)
        next_obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)
        buf.add(obs_t, action, logp, reward, done_t, value)
        obs_t = torch.as_tensor(next_obs, device=device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=device)

    with torch.no_grad():
        last_values = policy.get_value(obs_t, hidden, done_t)
    buf.compute_returns_and_advantages(last_values, done_t, gamma=0.99, gae_lambda=0.95)

    metrics = trainer.update(buf)
    assert np.isfinite(metrics.policy_loss)
    assert np.isfinite(metrics.value_loss)
    assert np.isfinite(metrics.entropy)

    envs.close()
