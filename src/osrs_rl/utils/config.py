"""Typed configuration schema.

The dataclasses here are the single source of truth for all configurable knobs.
YAML files in ``configs/`` are loaded into these dataclasses via :func:`load_config`.
Command-line overrides are handled in :mod:`osrs_rl.training.train` via ``tyro``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml


@dataclass
class EnvConfig:
    """Woodcutting simulator parameters."""

    grid_size: int = 16
    tile_size: int = 8  # pixels per tile -> native frame is grid_size * tile_size
    num_trees: int = 6
    max_inventory: int = 28
    max_episode_steps: int = 500
    chop_ticks: int = 3  # number of INTERACT steps to yield one log
    tree_respawn_ticks: int = 20


@dataclass
class VisionConfig:
    """Frame preprocessing for the policy."""

    resize_to: int = 84
    grayscale: bool = True
    frame_stack: int = 4


@dataclass
class RewardConfig:
    """Weights for the composite reward."""

    log_collected: float = 1.0
    step_penalty: float = -0.01
    invalid_action_penalty: float = -0.1
    idle_penalty: float = -0.05  # prevents the "spam IDLE" local optimum
    distance_shaping: float = 0.1
    adjacency_bonus: float = 0.5  # fires the step agent becomes adjacent to a live tree
    full_inventory_bonus: float = 10.0


@dataclass
class RandomizationConfig:
    """Visual domain randomization for the simulator.

    Every field defaults to the no-op value, so ``RandomizationConfig(enabled=True)``
    alone changes nothing — randomization is opt-in per-family. This lets you ablate
    any single knob without touching the others.

    All jitter ranges are one-sided (``uniform(-j, j)``). Per-episode fields are
    sampled once in ``reset()``; per-frame fields are sampled every ``render()``.
    """

    enabled: bool = False
    # --- Per-episode ---
    color_jitter: float = 0.0        # 0..1; fraction of 255 added as uniform RGB noise per channel
    clutter_density: float = 0.0     # 0..1; fraction of empty tiles painted with distractor colors
    tree_size_jitter: int = 0        # pixels; trees randomly shrink by 0..N from each side
    hud_randomize_side: bool = False # if True, HUD bar randomly placed at top or bottom
    # --- Per-frame ---
    brightness_jitter: float = 0.0   # multiplicative scale in (1-j, 1+j)
    contrast_jitter: float = 0.0     # multiplicative contrast in (1-j, 1+j) around mid-gray
    pixel_noise_std: float = 0.0     # stddev of additive Gaussian noise (in 0..255 scale)


@dataclass
class PPOConfig:
    """PPO hyperparameters (defaults tuned for laptop CPU MVP)."""

    total_timesteps: int = 500_000
    num_envs: int = 8
    rollout_steps: int = 128
    num_epochs: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    max_grad_norm: float = 0.5
    target_kl: float | None = None  # None disables KL early-stop
    norm_adv: bool = True
    clip_vloss: bool = True


@dataclass
class LoggingConfig:
    run_name: str = "ppo_woodcutting"
    log_dir: str = "runs"
    log_interval_updates: int = 1
    checkpoint_interval_updates: int = 25
    eval_interval_updates: int = 25
    eval_episodes: int = 5


@dataclass
class LiveConfig:
    """Live OSRS integration — read-only by default, explicit opt-in required for input.

    Coordinates are in absolute screen pixels. ``safe_region_xywh`` is the hard bound
    that gates every cursor move or click; the kill-switch file is a dead-man's handle
    (``touch`` the file from another terminal to abort all further live actions).
    """

    # Screen capture (left, top, width, height).
    capture_region_xywh: tuple[int, int, int, int] = (100, 100, 800, 600)
    # Safe bounding box for cursor/click targets (left, top, width, height).
    safe_region_xywh: tuple[int, int, int, int] = (200, 200, 400, 400)
    # Starting cursor position (screen coordinates, must lie inside safe_region).
    initial_cursor_xy: tuple[int, int] = (400, 400)
    cursor_step_pixels: int = 25
    action_delay_seconds: float = 0.3
    # Safety — these all default to the safe, read-only posture.
    enable_live_input: bool = False
    max_actions_per_second: float = 2.0
    kill_switch_file: str = "/tmp/osrs_rl_stop"
    # Optional hotkey for DROP. Empty string => DROP becomes a no-op in live mode.
    hotkey_drop: str = ""
    # Rollout budget for live evaluation (replaces max_episode_steps from the sim).
    max_steps: int = 200


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    env: EnvConfig = field(default_factory=EnvConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


T = TypeVar("T")


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Recursively build a dataclass from a plain dict (YAML-friendly).

    Uses ``get_type_hints`` so ``from __future__ import annotations`` (which turns
    annotations into strings) does not break nested dataclass detection.
    """
    if not is_dataclass(cls):
        return data  # type: ignore[return-value]
    hints = get_type_hints(cls)
    known_fields = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known_fields:
            raise ValueError(f"Unknown config key '{key}' for {cls.__name__}")
        resolved = hints.get(key, known_fields[key].type)
        if is_dataclass(resolved) and isinstance(value, dict):
            kwargs[key] = _from_dict(resolved, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)  # type: ignore[call-arg]


def load_config(path: str | Path, cls: type[T] = TrainConfig) -> T:  # type: ignore[assignment]
    """Load a YAML file into the given dataclass schema."""
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    return _from_dict(cls, raw)


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Convert a dataclass config back into a plain dict (for logging)."""
    if is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    return dict(cfg)
