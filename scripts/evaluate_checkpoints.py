"""Evaluate every saved checkpoint in a run and emit a progression JSON.

Produces a clean deterministic story of "policy quality as a function of training
steps" by running N stochastic-sample episodes per checkpoint. Much less noisy than
the 5-episode argmax eval done during training.

Usage:
    python scripts/evaluate_checkpoints.py \
        --run-dir runs/ppo_woodcutting \
        --config configs/ppo_woodcutting.yaml \
        --episodes 30
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from osrs_rl.evaluation.evaluate import _load_policy, evaluate
from osrs_rl.utils.config import TrainConfig, load_config
from osrs_rl.utils.logging import get_console, setup_logger
from osrs_rl.utils.seeding import resolve_device

_LOG = setup_logger(__name__)

_STEP_RE = re.compile(r"ckpt_upd(\d+)\.pt$")


def _list_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in sorted(ckpt_dir.glob("ckpt_upd*.pt")):
        m = _STEP_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax sampling. Default is stochastic (honest policy evaluation).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, TrainConfig)
    device = resolve_device(args.device)
    console = get_console()

    ckpt_dir = args.run_dir / "checkpoints"
    checkpoints = _list_checkpoints(ckpt_dir)
    if not checkpoints:
        raise SystemExit(f"No ckpt_upd*.pt files in {ckpt_dir}")

    # Each training update consumes num_envs * rollout_steps env-steps.
    steps_per_update = cfg.ppo.num_envs * cfg.ppo.rollout_steps
    console.print(f"[bold]Evaluating {len(checkpoints)} checkpoints from {ckpt_dir}[/bold]")

    records: list[dict] = []
    for update_idx, path in checkpoints:
        env_steps = update_idx * steps_per_update
        policy = _load_policy(cfg, str(path), device)
        metrics = evaluate(
            cfg,
            policy,
            episodes=args.episodes,
            device=device,
            seed=args.seed,
            deterministic=args.deterministic,
        )
        record = {
            "update": update_idx,
            "env_steps": env_steps,
            "checkpoint": str(path),
            "episode_return_mean": metrics["episode_return"]["mean"],
            "episode_return_std": metrics["episode_return"]["std"],
            "episode_length_mean": metrics["episode_length"]["mean"],
            "success_rate": metrics["success_rate"],
            "trees_chopped_mean": metrics["trees_chopped"]["mean"],
            "invalid_action_ratio": metrics["invalid_action_ratio"],
            "idle_ratio": metrics["idle_ratio"],
            "action_distribution": metrics["action_distribution"],
        }
        records.append(record)
        console.print(
            f"  upd={update_idx:5d} step={env_steps:>7,}  "
            f"return={record['episode_return_mean']:+7.2f}  "
            f"success={record['success_rate']*100:5.1f}%  "
            f"trees={record['trees_chopped_mean']:.2f}"
        )

    output = args.output if args.output is not None else args.run_dir / "checkpoint_progression.json"
    output.write_text(json.dumps({"episodes": args.episodes, "records": records}, indent=2))
    _LOG.info(f"Wrote progression to {output}")


if __name__ == "__main__":
    main()
