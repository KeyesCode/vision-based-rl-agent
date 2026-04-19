"""Checkpoint save/load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from osrs_rl.agents.ppo import PPOActorCritic


def save_checkpoint(
    path: str | Path,
    policy: PPOActorCritic,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    policy: PPOActorCritic,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint into ``policy`` (and optionally ``optimizer``).

    ``strict=False`` by default so older checkpoints (pre-aux-head) still load
    into newer policies cleanly — the missing ``aux_head`` weights stay at their
    fresh orthogonal-init values, which is harmless since ``aux_adjacency_coef``
    defaults to 0.
    """
    payload = torch.load(path, map_location=map_location)
    policy.load_state_dict(payload["policy_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
