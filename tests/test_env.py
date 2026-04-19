"""Smoke tests for the OSRS env and simulator."""

from __future__ import annotations

import numpy as np

from osrs_rl.env.action_space import ActionDecoder, ActionType
from osrs_rl.env.osrs_env import OSRSEnv, make_env
from osrs_rl.env.simulator.mock_osrs import MockOSRSClient
from osrs_rl.rewards.components import build_reward
from osrs_rl.utils.config import EnvConfig, RewardConfig, VisionConfig


def _default_configs() -> tuple[EnvConfig, RewardConfig, VisionConfig]:
    return EnvConfig(), RewardConfig(), VisionConfig()


def test_simulator_reset_and_render_shape():
    cfg = EnvConfig()
    client = MockOSRSClient(cfg)
    state = client.reset(seed=0)

    assert state.step_index == 0
    assert 0 <= state.inventory_count < cfg.max_inventory

    frame = client.render()
    expected = (cfg.grid_size * cfg.tile_size, cfg.grid_size * cfg.tile_size, 3)
    assert frame.shape == expected
    assert frame.dtype == np.uint8


def test_env_step_contract():
    env_cfg, reward_cfg, _ = _default_configs()
    env = OSRSEnv(env_cfg, build_reward(reward_cfg))
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape

    for a in range(ActionDecoder.n_actions()):
        obs, r, terminated, truncated, info = env.step(a)
        assert obs.shape == env.observation_space.shape
        assert isinstance(r, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "reward_breakdown" in info
        if terminated or truncated:
            env.reset(seed=0)


def test_wrapped_env_produces_framestacked_obs():
    env_cfg, reward_cfg, vision_cfg = _default_configs()
    env = make_env(env_cfg, vision_cfg, reward_cfg, seed=123, idx=0)()
    obs, _ = env.reset()
    # (k, H, W) with grayscale + resize
    assert obs.shape == (vision_cfg.frame_stack, vision_cfg.resize_to, vision_cfg.resize_to)
    assert obs.dtype == np.uint8


def test_interact_produces_logs_over_time():
    """Functional check: standing next to a tree and spamming INTERACT yields logs."""
    env_cfg = EnvConfig(num_trees=1, grid_size=4, chop_ticks=2, max_inventory=3, max_episode_steps=200)
    client = MockOSRSClient(env_cfg)
    client.reset(seed=0)

    # Teleport agent next to the tree (white-box test; justified because we want a
    # deterministic smoke check of chopping mechanics).
    tree = client._trees[0]
    client._agent.x = tree.x
    client._agent.y = (tree.y + 1) % env_cfg.grid_size
    # If the teleport happened to land on top of a tree tile, bail (only 1 tree so unlikely).
    if client._agent.y == tree.y:
        return

    logs_start = client._agent.inventory
    for _ in range(env_cfg.chop_ticks):
        client.step(ActionDecoder.decode(ActionType.INTERACT))
    # Tree respawns may need longer than chop_ticks; just assert at least one log gained.
    assert client._agent.inventory > logs_start
