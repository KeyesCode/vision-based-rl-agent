"""Smoke test: run one short PPO rollout + update without error."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv

from osrs_rl.agents.ppo import PPOActorCritic, PPOTrainer
from osrs_rl.agents.rollout_buffer import RolloutBuffer
from osrs_rl.env.action_space import ActionDecoder
from osrs_rl.env.osrs_env import make_env
from osrs_rl.utils.config import EnvConfig, PPOConfig, RewardConfig, VisionConfig


def test_ppo_one_update():
    env_cfg = EnvConfig(max_episode_steps=64)
    reward_cfg = RewardConfig()
    vision_cfg = VisionConfig(frame_stack=2)
    ppo_cfg = PPOConfig(
        total_timesteps=256,
        num_envs=2,
        rollout_steps=16,
        num_epochs=1,
        num_minibatches=2,
    )

    envs = SyncVectorEnv(
        [make_env(env_cfg, vision_cfg, reward_cfg, seed=0, idx=i) for i in range(ppo_cfg.num_envs)]
    )
    obs, _ = envs.reset(seed=0)

    obs_shape = envs.single_observation_space.shape
    device = torch.device("cpu")
    policy = PPOActorCritic(
        num_actions=ActionDecoder.n_actions(),
        in_channels=obs_shape[0],
        input_hw=(obs_shape[1], obs_shape[2]),
    ).to(device)
    trainer = PPOTrainer(policy, ppo_cfg, device)

    buf = RolloutBuffer(
        rollout_steps=ppo_cfg.rollout_steps,
        num_envs=ppo_cfg.num_envs,
        obs_shape=obs_shape,
        device=device,
    )

    obs_t = torch.as_tensor(obs, device=device)
    done_t = torch.zeros(ppo_cfg.num_envs, device=device)

    for _ in range(ppo_cfg.rollout_steps):
        with torch.no_grad():
            action, logp, value = policy.act(obs_t)
        next_obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)
        buf.add(obs_t, action, logp, reward, done_t, value)
        obs_t = torch.as_tensor(next_obs, device=device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=device)

    with torch.no_grad():
        last_values = policy.get_value(obs_t)
    buf.compute_returns_and_advantages(last_values, done_t, gamma=0.99, gae_lambda=0.95)

    metrics = trainer.update(buf)

    assert np.isfinite(metrics.policy_loss)
    assert np.isfinite(metrics.value_loss)
    assert np.isfinite(metrics.entropy)
    assert 0.0 <= metrics.clip_fraction <= 1.0

    envs.close()
