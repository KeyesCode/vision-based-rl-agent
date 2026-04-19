"""Render a 3×3 sampler of domain-randomized simulator frames.

Produces ``docs/results/dr_samples.png`` — a one-image illustration of what the
policy actually sees during DR training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from osrs_rl.env.simulator.mock_osrs import MockOSRSClient
from osrs_rl.utils.config import EnvConfig, TrainConfig, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Training config YAML (uses its env + randomization blocks).")
    parser.add_argument("--output", type=Path, default=Path("docs/results/dr_samples.png"))
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = load_config(args.config, TrainConfig)
    env_cfg: EnvConfig = cfg.env

    fig, axes = plt.subplots(args.rows, args.cols, figsize=(args.cols * 2.2, args.rows * 2.2))
    axes = axes.reshape(args.rows, args.cols)
    for r in range(args.rows):
        for c in range(args.cols):
            client = MockOSRSClient(env_cfg, randomization_cfg=cfg.randomization)
            client.reset(seed=args.seed + r * args.cols + c)
            frame = client.render()
            ax = axes[r, c]
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])

    preset = "domain-randomized" if cfg.randomization.enabled else "baseline"
    fig.suptitle(f"Simulator samples — {preset}", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
