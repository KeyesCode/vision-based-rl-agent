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

from osrs_rl.agents.ppo import PPOActorCritic
from osrs_rl.env.action_space import ActionDecoder, ActionType
from osrs_rl.env.osrs_env import make_env
from osrs_rl.training.checkpoint import load_checkpoint
from osrs_rl.utils.config import EnvConfig, RewardConfig, TrainConfig, VisionConfig, load_config
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


def _build_env(cfg: TrainConfig, seed: int):
    return make_env(cfg.env, cfg.vision, cfg.reward, seed=seed, idx=0)()


def _load_policy(cfg: TrainConfig, checkpoint_path: str, device: torch.device) -> PPOActorCritic:
    """Build a fresh env temporarily to determine the obs shape, then load weights."""
    env = _build_env(cfg, seed=0)
    try:
        obs_shape = env.observation_space.shape  # type: ignore[union-attr]
    finally:
        env.close()
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
) -> dict:
    """Run ``episodes`` deterministic rollouts and return aggregated metrics."""
    rng = np.random.default_rng(seed)
    env = _build_env(cfg, seed=seed)

    returns: list[float] = []
    lengths: list[int] = []
    successes: list[int] = []
    trees: list[int] = []
    invalid: list[float] = []
    idle: list[float] = []
    action_counts = np.zeros(ActionDecoder.n_actions(), dtype=np.int64)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_r = 0.0
        steps = 0
        terminated = truncated = False
        last_info: dict = {}
        while not (terminated or truncated):
            if policy is None:
                action = int(rng.integers(0, ActionDecoder.n_actions()))
            else:
                obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                a, _, _ = policy.act(obs_t, deterministic=deterministic)
                action = int(a.item())
            obs, r, terminated, truncated, info = env.step(action)
            total_r += float(r)
            steps += 1
            last_info = info

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
    policy: PPOActorCritic | None = None
    label = "random baseline"
    if not args.random:
        assert args.checkpoint is not None
        policy = _load_policy(cfg, args.checkpoint, device)
        label = f"checkpoint: {args.checkpoint}"

    metrics = evaluate(
        cfg,
        policy,
        args.episodes,
        device,
        seed=args.seed,
        deterministic=args.deterministic,
    )
    metrics["label"] = label
    metrics["checkpoint"] = args.checkpoint
    metrics["config"] = args.config
    metrics["deterministic"] = args.deterministic

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
