"""Logging utilities: rich console + TensorBoard writer factory."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from torch.utils.tensorboard import SummaryWriter

_CONSOLE = Console()


def get_console() -> Console:
    return _CONSOLE


def setup_logger(name: str = "osrs_rl", level: int = logging.INFO) -> logging.Logger:
    """Return a rich-handled logger. Idempotent across calls."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = RichHandler(console=_CONSOLE, rich_tracebacks=True, show_path=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def create_run_dir(log_dir: str | Path, run_name: str) -> Path:
    """Create ``log_dir/run_name`` (with a numeric suffix on collision) and return it."""
    root = Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / run_name
    i = 1
    while candidate.exists():
        candidate = root / f"{run_name}_{i}"
        i += 1
    candidate.mkdir(parents=True)
    return candidate


def create_writer(run_dir: str | Path) -> SummaryWriter:
    return SummaryWriter(log_dir=str(run_dir))


def log_hparams(writer: SummaryWriter, hparams: dict[str, Any]) -> None:
    """Flatten nested dicts and write as text (TB's add_hparams is clunky for our case)."""
    flat: dict[str, Any] = {}

    def _flatten(prefix: str, d: dict[str, Any]) -> None:
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(key, v)
            else:
                flat[key] = v

    _flatten("", hparams)
    lines = [f"| `{k}` | `{v}` |" for k, v in flat.items()]
    md = "| key | value |\n| --- | --- |\n" + "\n".join(lines)
    writer.add_text("config", md, 0)
