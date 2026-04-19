"""Tests for the domain-randomization subsystem.

Two properties worth protecting:

1. **Identity when disabled.** The default :class:`RandomizationConfig` must leave
   rendered frames bit-identical to the pre-DR implementation.
2. **Diversity when enabled.** With randomization turned on, two episodes with
   different seeds must produce different pixels (same task, different appearance).
"""

from __future__ import annotations

import numpy as np

from osrs_rl.env.simulator.mock_osrs import MockOSRSClient
from osrs_rl.env.simulator.randomization import Randomizer
from osrs_rl.utils.config import EnvConfig, RandomizationConfig


def _env_cfg() -> EnvConfig:
    return EnvConfig(grid_size=8, tile_size=8, num_trees=2, max_inventory=4, max_episode_steps=16)


def test_disabled_randomization_matches_default_palette():
    """DR off => every frame uses the canonical palette and no clutter."""
    client = MockOSRSClient(_env_cfg())
    client.reset(seed=0)
    frame = client.render()

    visuals = client._visuals
    # Confirm the canonical palette survived round-tripping through the Randomizer.
    assert visuals.hud_side == "bottom"
    assert visuals.clutter == []
    assert visuals.tree_pad == 0
    # Grass color is the default dark green everywhere that isn't a tree/agent/HUD.
    assert tuple(int(v) for v in frame[0, 0]) == (48, 110, 48)


def test_enabled_randomization_changes_frames_between_seeds():
    """Same task, different seeds => visibly different pixels."""
    cfg = RandomizationConfig(
        enabled=True,
        color_jitter=0.25,
        clutter_density=0.1,
        tree_size_jitter=1,
        hud_randomize_side=True,
        brightness_jitter=0.15,
        contrast_jitter=0.1,
        pixel_noise_std=5.0,
    )
    a = MockOSRSClient(_env_cfg(), randomization_cfg=cfg)
    b = MockOSRSClient(_env_cfg(), randomization_cfg=cfg)
    a.reset(seed=0)
    b.reset(seed=1)

    fa = a.render()
    fb = b.render()
    # Must differ substantially — at least a double-digit percentage of pixels.
    diff_frac = float(np.mean(fa != fb))
    assert diff_frac > 0.1, f"frames look too similar with DR on (diff_frac={diff_frac:.3f})"


def test_apply_frame_is_noop_when_disabled():
    randomizer = Randomizer(RandomizationConfig(enabled=False))
    rng = np.random.default_rng(0)
    frame = np.full((8, 8, 3), 100, dtype=np.uint8)
    out = randomizer.apply_frame(rng, frame)
    assert np.array_equal(out, frame)


def test_apply_frame_respects_zero_knobs_even_when_enabled():
    """enabled=True but all numeric knobs zero => still identity (saves compute)."""
    cfg = RandomizationConfig(enabled=True)  # all jitter values = 0
    randomizer = Randomizer(cfg)
    rng = np.random.default_rng(0)
    frame = np.full((8, 8, 3), 100, dtype=np.uint8)
    out = randomizer.apply_frame(rng, frame)
    assert np.array_equal(out, frame)


def test_clutter_never_overlaps_occupied_tiles():
    cfg = RandomizationConfig(enabled=True, clutter_density=0.8)  # saturate request
    client = MockOSRSClient(_env_cfg(), randomization_cfg=cfg)
    client.reset(seed=42)
    occupied = {(t.x, t.y) for t in client._trees} | {(client._agent.x, client._agent.y)}
    for cx, cy, _ in client._visuals.clutter:
        assert (cx, cy) not in occupied
