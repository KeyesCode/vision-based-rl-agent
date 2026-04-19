"""Checkpoint evaluation — the script that proves learning.

Runs ``n`` deterministic episodes with a loaded checkpoint (or a random baseline) and
reports: average reward, episode length, success rate, trees chopped, invalid/idle
ratios, and action distribution. Output is a JSON file + a console summary. Run twice
(once ``--random``, once ``--checkpoint <path>``) to produce the comparison that goes
in the README.

    osrs-eval --checkpoint runs/ppo_woodcutting/checkpoints/latest.pt --output eval_trained.json
    osrs-eval --random --output eval_random.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from osrs_rl.agents.ppo import PPOActorCritic, RecurrentPPOActorCritic
from osrs_rl.env.action_space import ActionDecoder, ActionType, build_adjacency_mask
from osrs_rl.env.game_client import GameClient
from osrs_rl.env.osrs_env import make_env
from osrs_rl.training.checkpoint import load_checkpoint
from osrs_rl.utils.config import (
    LiveConfig,
    TrainConfig,
    load_config,
)
from osrs_rl.utils.logging import get_console, setup_logger
from osrs_rl.utils.seeding import resolve_device, set_seed

_LOG = setup_logger(__name__)


@dataclass
class EvalArgs:
    checkpoint: str | None = None
    """Path to a .pt checkpoint. Omit for the random baseline."""
    random: bool = False
    """Run a random-action baseline instead of loading a checkpoint."""
    config: str = "configs/ppo_woodcutting.yaml"
    """Config file (used for env/vision/reward parameters that match training)."""
    episodes: int = 20
    """Number of evaluation episodes."""
    output: str | None = None
    """Optional JSON output path."""
    seed: int = 1000
    """Seed for the eval env (distinct from training seed by default)."""
    device: str = "auto"
    """Torch device for the policy."""
    deterministic: bool = False
    """Use argmax instead of sampling from the policy's action distribution."""
    live_config: str | None = None
    """Path to a LiveConfig YAML. When present, evaluation runs against the real OSRS
    window via LiveOSRSClient. Dry-run by default — set enable_live_input=true in the
    YAML to actually send mouse/keyboard input."""
    action_mask: bool = False
    """If true, mask the INTERACT action's logit in non-adjacent states using the
    ground-truth adjacency label from ``info['adjacent_to_tree']``. Inference-only;
    the loaded checkpoint is not modified or retrained."""


def _summary(values: list[float] | list[int]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def _build_env(cfg: TrainConfig, seed: int, client: GameClient | None = None):
    return make_env(
        cfg.env,
        cfg.vision,
        cfg.reward,
        seed=seed,
        idx=0,
        client=client,
        randomization_cfg=cfg.randomization,
    )()


def _build_live_client(cfg: TrainConfig, live_cfg: LiveConfig) -> GameClient:
    """Construct a LiveOSRSClient and adapt env_cfg so episodes truncate at max_steps."""
    # Import live-only to keep the core path usable without mss/pynput installed.
    from osrs_rl.env.live.live_client import LiveOSRSClient

    # Override the env's step budget so live evaluation stops after max_steps.
    cfg.env.max_episode_steps = live_cfg.max_steps
    return LiveOSRSClient(cfg.env, live_cfg)


def _load_policy(
    cfg: TrainConfig, checkpoint_path: str, device: torch.device
) -> PPOActorCritic | RecurrentPPOActorCritic:
    """Build a fresh sim env temporarily to determine the obs shape, then load weights.

    Probe obs shape against the simulator because the vision wrappers determine the
    policy input shape (grayscale / resize / framestack) — not the raw capture
    resolution. Recurrent or feedforward class is chosen from ``cfg.ppo.recurrent``.
    """
    env = _build_env(cfg, seed=0)
    try:
        obs_shape = env.observation_space.shape  # type: ignore[union-attr]
    finally:
        env.close()

    if cfg.ppo.recurrent:
        policy: PPOActorCritic | RecurrentPPOActorCritic = RecurrentPPOActorCritic(
            num_actions=ActionDecoder.n_actions(),
            in_channels=obs_shape[0],
            input_hw=(obs_shape[1], obs_shape[2]),
            hidden_size=cfg.ppo.lstm_hidden_size,
        ).to(device)
    else:
        policy = PPOActorCritic(
            num_actions=ActionDecoder.n_actions(),
            in_channels=obs_shape[0],
            input_hw=(obs_shape[1], obs_shape[2]),
        ).to(device)
    load_checkpoint(checkpoint_path, policy=policy, map_location=device)
    policy.eval()
    return policy


@torch.no_grad()
def evaluate(
    cfg: TrainConfig,
    policy: PPOActorCritic | None,
    episodes: int,
    device: torch.device,
    seed: int = 1000,
    deterministic: bool = False,
    live_client: GameClient | None = None,
    action_mask: bool = False,
) -> dict:
    """Run ``episodes`` rollouts and return aggregated metrics.

    If ``live_client`` is provided, rollouts run against the real OSRS window via
    :class:`LiveOSRSClient`; otherwise the fast simulator is used.
    """
    rng = np.random.default_rng(seed)
    env = _build_env(cfg, seed=seed, client=live_client)

    returns: list[float] = []
    lengths: list[int] = []
    successes: list[int] = []
    trees: list[int] = []
    invalid: list[float] = []
    idle: list[float] = []
    action_counts = np.zeros(ActionDecoder.n_actions(), dtype=np.int64)

    is_recurrent = isinstance(policy, RecurrentPPOActorCritic)

    for ep in range(episodes):
        obs, reset_info = env.reset(seed=seed + ep)
        total_r = 0.0
        steps = 0
        terminated = truncated = False
        last_info: dict = {}
        # Recurrent policies reset their hidden state at every episode boundary.
        hidden = policy.initial_hidden(1, device) if is_recurrent else None
        # Track adjacency for the current observation so we can mask INTERACT.
        last_adjacent = int(reset_info.get("adjacent_to_tree", 0))
        while not (terminated or truncated):
            if policy is None:
                action = int(rng.integers(0, ActionDecoder.n_actions()))
            else:
                obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                mask_t: torch.Tensor | None = None
                if action_mask:
                    mask_t = torch.as_tensor(
                        build_adjacency_mask(bool(last_adjacent)),
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)
                if is_recurrent:
                    start = torch.tensor([1.0 if steps == 0 else 0.0], device=device)
                    a, _, _, hidden = policy.act(  # type: ignore[misc]
                        obs_t, hidden, start, deterministic=deterministic, mask=mask_t
                    )
                else:
                    a, _, _ = policy.act(
                        obs_t, deterministic=deterministic, mask=mask_t
                    )
                action = int(a.item())
            obs, r, terminated, truncated, info = env.step(action)
            total_r += float(r)
            steps += 1
            last_info = info
            last_adjacent = int(info.get("adjacent_to_tree", last_adjacent))

        returns.append(total_r)
        lengths.append(steps)
        successes.append(int(last_info.get("episode_success", int(terminated))))
        trees.append(int(last_info.get("episode_trees_chopped", 0)))
        invalid.append(float(last_info.get("episode_invalid_ratio", 0.0)))
        idle.append(float(last_info.get("episode_idle_ratio", 0.0)))
        ac = last_info.get("episode_action_counts")
        if ac is not None:
            action_counts += np.asarray(ac, dtype=np.int64)

    env.close()

    total_actions = int(action_counts.sum()) or 1
    action_distribution = {
        ActionType(i).name: float(action_counts[i] / total_actions)
        for i in range(ActionDecoder.n_actions())
    }

    success_rate = float(np.mean(successes)) if successes else 0.0
    return {
        "num_episodes": episodes,
        "episode_return": _summary(returns),
        "episode_length": _summary(lengths),
        "trees_chopped": _summary(trees),
        "success_rate": success_rate,
        "invalid_action_ratio": float(np.mean(invalid)) if invalid else 0.0,
        "idle_ratio": float(np.mean(idle)) if idle else 0.0,
        "action_distribution": action_distribution,
    }


def _print_summary(label: str, metrics: dict) -> None:
    c = get_console()
    c.rule(f"[bold]{label}")
    ret = metrics["episode_return"]
    leng = metrics["episode_length"]
    trees = metrics["trees_chopped"]
    c.print(
        f"episodes            : [cyan]{metrics['num_episodes']}[/cyan]\n"
        f"episode_return      : [green]{ret['mean']:+.2f} ± {ret['std']:.2f}[/green]"
        f"  (min={ret['min']:+.2f}, max={ret['max']:+.2f})\n"
        f"episode_length      : {leng['mean']:.1f} ± {leng['std']:.1f}\n"
        f"success_rate        : [yellow]{metrics['success_rate']*100:.1f}%[/yellow]\n"
        f"trees_chopped / ep  : {trees['mean']:.2f} ± {trees['std']:.2f}\n"
        f"invalid_action_ratio: {metrics['invalid_action_ratio']:.3f}\n"
        f"idle_ratio          : {metrics['idle_ratio']:.3f}"
    )
    c.print("[bold]action distribution[/bold]")
    for name, frac in metrics["action_distribution"].items():
        bar = "█" * max(1, int(frac * 40)) if frac > 0 else ""
        c.print(f"  {name:<14} {frac*100:5.1f}%  {bar}")


def main(args: EvalArgs) -> None:
    if not args.random and args.checkpoint is None:
        raise SystemExit("Pass either --checkpoint <path> or --random.")
    if args.random and args.checkpoint is not None:
        raise SystemExit("--random and --checkpoint are mutually exclusive.")

    set_seed(args.seed)
    device = resolve_device(args.device)

    cfg = load_config(Path(args.config), TrainConfig)

    live_client: GameClient | None = None
    live_cfg: LiveConfig | None = None
    if args.live_config is not None:
        live_cfg = load_config(Path(args.live_config), LiveConfig)
        live_client = _build_live_client(cfg, live_cfg)
        mode = "LIVE" if live_cfg.enable_live_input else "DRY-RUN"
        _LOG.warning(
            f"Live mode ({mode}) — capture={live_cfg.capture_region_xywh} "
            f"safe_region={live_cfg.safe_region_xywh} kill_switch={live_cfg.kill_switch_file}"
        )

    policy: PPOActorCritic | None = None
    label = "random baseline"
    if not args.random:
        assert args.checkpoint is not None
        policy = _load_policy(cfg, args.checkpoint, device)
        label = f"checkpoint: {args.checkpoint}"
    if live_cfg is not None:
        label += f" [{'live' if live_cfg.enable_live_input else 'dry-run'}]"

    metrics = evaluate(
        cfg,
        policy,
        args.episodes,
        device,
        seed=args.seed,
        deterministic=args.deterministic,
        live_client=live_client,
        action_mask=args.action_mask,
    )
    metrics["label"] = label
    metrics["checkpoint"] = args.checkpoint
    metrics["config"] = args.config
    metrics["deterministic"] = args.deterministic
    metrics["action_mask"] = args.action_mask
    metrics["live_config"] = args.live_config
    if live_cfg is not None:
        metrics["live_mode"] = "live" if live_cfg.enable_live_input else "dry_run"

    _print_summary(label, metrics)

    if args.output is not None:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        _LOG.info(f"Wrote metrics to {out}")


def entrypoint() -> None:
    main(tyro.cli(EvalArgs))


if __name__ == "__main__":
    entrypoint()
