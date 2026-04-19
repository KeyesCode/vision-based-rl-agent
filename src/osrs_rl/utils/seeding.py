"""Deterministic seeding across numpy, torch, and stdlib random."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed numpy, torch, python ``random``, and env vars.

    ``deterministic_torch=True`` forces cuDNN determinism at a small throughput cost.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    """Resolve "auto"/"cpu"/"cuda"/"mps" into a concrete torch.device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)
