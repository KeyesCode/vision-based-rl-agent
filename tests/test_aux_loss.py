"""Tests for the adjacency auxiliary-loss plumbing.

Guards the four moving pieces introduced in the representation milestone:

1. ``OSRSEnv`` info carries a 0/1 ``adjacent_to_tree`` scalar every reset and step.
2. ``RolloutBuffer.add`` stores the adjacency label.
3. ``PPOActorCritic.evaluate_actions`` returns aux logits as a 4-tuple.
4. ``RecurrentPPOActorCritic.evaluate_sequence`` returns aux logits on the
   correct shape ``(T, B)``.
"""

from __future__ import annotations

import numpy as np
import torch

from osrs_rl.agents.ppo import (
    PPOActorCritic,
    RecurrentPPOActorCritic,
)
from osrs_rl.agents.rollout_buffer import RolloutBuffer
from osrs_rl.env.action_space import ActionDecoder, ActionType
from osrs_rl.env.osrs_env import OSRSEnv
from osrs_rl.env.simulator.mock_osrs import MockOSRSClient
from osrs_rl.rewards.components import build_reward
from osrs_rl.utils.config import EnvConfig, RewardConfig


def test_reset_info_has_adjacent_to_tree_scalar():
    env = OSRSEnv(EnvConfig(), build_reward(RewardConfig()))
    _, info = env.reset(seed=0)
    assert "adjacent_to_tree" in info
    assert info["adjacent_to_tree"] in (0, 1)


def test_step_info_has_adjacent_to_tree_scalar():
    env = OSRSEnv(EnvConfig(), build_reward(RewardConfig()))
    env.reset(seed=0)
    _, _, _, _, info = env.step(int(ActionType.IDLE))
    assert "adjacent_to_tree" in info
    assert info["adjacent_to_tree"] in (0, 1)


def test_adjacent_label_is_1_when_agent_is_next_to_alive_tree():
    """White-box: position the agent adjacent to a tree and assert label=1."""
    cfg = EnvConfig(num_trees=1, grid_size=4, max_inventory=4, max_episode_steps=50)
    client = MockOSRSClient(cfg)
    client.reset(seed=0)
    tree = client._trees[0]
    client._agent.x, client._agent.y = tree.x, (tree.y + 1) % cfg.grid_size
    if client._agent.y == tree.y:
        return  # unlikely corner: tree placed at edge; skip
    env = OSRSEnv(cfg, build_reward(RewardConfig()), client=client)
    env._prev_state = client._state()
    _, _, _, _, info = env.step(int(ActionType.IDLE))
    assert info["adjacent_to_tree"] == 1


def test_rollout_buffer_stores_adjacency():
    device = torch.device("cpu")
    buf = RolloutBuffer(rollout_steps=3, num_envs=2, obs_shape=(4, 8, 8), device=device)
    obs = torch.zeros(2, 4, 8, 8, dtype=torch.uint8)
    action = torch.zeros(2, dtype=torch.long)
    logp = torch.zeros(2)
    reward = np.zeros(2, dtype=np.float32)
    done = np.zeros(2, dtype=np.float32)
    value = torch.zeros(2)
    buf.add(obs, action, logp, reward, done, value, adjacency=np.array([1.0, 0.0]))
    buf.add(obs, action, logp, reward, done, value, adjacency=np.array([0.0, 1.0]))
    assert torch.allclose(buf.adjacency[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(buf.adjacency[1], torch.tensor([0.0, 1.0]))


def test_feedforward_evaluate_actions_returns_aux_logits():
    policy = PPOActorCritic(
        num_actions=ActionDecoder.n_actions(),
        in_channels=4,
        input_hw=(84, 84),
        feature_dim=64,
    )
    obs = torch.randint(0, 255, (3, 4, 84, 84), dtype=torch.uint8)
    actions = torch.tensor([0, 1, 2])
    logp, ent, val, aux = policy.evaluate_actions(obs, actions)
    assert logp.shape == (3,)
    assert ent.shape == (3,)
    assert val.shape == (3,)
    assert aux.shape == (3,)
    # Aux logits are unconstrained reals; sigmoid would compress to (0, 1).
    assert torch.isfinite(aux).all()


def test_recurrent_evaluate_sequence_returns_aux_logits_shape_tb():
    policy = RecurrentPPOActorCritic(
        num_actions=ActionDecoder.n_actions(),
        in_channels=4,
        input_hw=(84, 84),
        feature_dim=64,
        hidden_size=16,
    )
    T, B = 5, 2
    obs = torch.randint(0, 255, (T, B, 4, 84, 84), dtype=torch.uint8)
    actions = torch.zeros(T, B, dtype=torch.long)
    h0 = torch.zeros(1, B, 16)
    starts = torch.zeros(T, B)
    logp, ent, val, aux = policy.evaluate_sequence(obs, actions, (h0, h0.clone()), starts)
    assert logp.shape == (T, B)
    assert aux.shape == (T, B)
    assert torch.isfinite(aux).all()
