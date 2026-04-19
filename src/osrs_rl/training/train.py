"""CLI entry point.

Two ways to invoke:

    osrs-train                                   # all defaults, tyro CLI overrides
    osrs-train --config configs/ppo_woodcutting.yaml   # load YAML config

When a ``--config`` flag is present it takes precedence; otherwise tyro parses
command-line overrides into :class:`TrainConfig`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tyro

from osrs_rl.training.trainer import Trainer
from osrs_rl.utils.config import TrainConfig, load_config


def _pop_config_arg(argv: list[str]) -> str | None:
    """Extract a ``--config <path>`` pair from ``argv`` (mutates argv)."""
    for i, tok in enumerate(argv):
        if tok == "--config" and i + 1 < len(argv):
            path = argv[i + 1]
            del argv[i : i + 2]
            return path
        if tok.startswith("--config="):
            path = tok.split("=", 1)[1]
            del argv[i]
            return path
    return None


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    config_path = _pop_config_arg(raw)

    if config_path is not None:
        if raw:
            print(
                "--config is not compositional; pass either --config or CLI overrides, not both.",
                file=sys.stderr,
            )
            sys.exit(2)
        cfg = load_config(Path(config_path), TrainConfig)
    else:
        cfg = tyro.cli(TrainConfig, args=raw)

    Trainer(cfg).train()


if __name__ == "__main__":
    main()
