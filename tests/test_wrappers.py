"""Test the EpisodeStatsWrapper emits the expected fields on termination."""

from __future__ import annotations

import numpy as np

from osrs_rl.env.action_space import ActionDecoder, ActionType
from osrs_rl.env.osrs_env import OSRSEnv
from osrs_rl.env.wrappers import EpisodeStatsWrapper
from osrs_rl.rewards.components import build_reward
from osrs_rl.utils.config import EnvConfig, RewardConfig


def test_stats_emitted_on_truncation():
    env_cfg = EnvConfig(max_episode_steps=4, num_trees=0, grid_size=4, max_inventory=1)
    env = OSRSEnv(env_cfg, build_reward(RewardConfig()))
    wrapped = EpisodeStatsWrapper(env)

    wrapped.reset(seed=0)
    info = {}
    for _ in range(env_cfg.max_episode_steps):
        _, _, terminated, truncated, info = wrapped.step(int(ActionType.IDLE))
        if terminated or truncated:
            break

    assert "episode_trees_chopped" in info
    assert "episode_success" in info
    assert "episode_invalid_ratio" in info
    assert "episode_idle_ratio" in info
    assert "episode_action_counts" in info

    # No trees placed -> no chopping possible.
    assert info["episode_trees_chopped"] == 0
    # Truncation, not termination, because inventory never filled.
    assert info["episode_success"] == 0
    # All actions were IDLE, so idle ratio == 1.
    assert info["episode_idle_ratio"] == 1.0
    # IDLE is always "valid" per the simulator's current contract.
    assert info["episode_invalid_ratio"] == 0.0
    counts = np.asarray(info["episode_action_counts"])
    assert counts.sum() == env_cfg.max_episode_steps
    assert counts[int(ActionType.IDLE)] == env_cfg.max_episode_steps
