"""Policy interface.

Abstracting over "act" and "evaluate_actions" lets the trainer stay algorithm-agnostic:
PPO, DQN, or any future policy gradient variant implements this same surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BasePolicy(ABC):
    """Minimal interface the trainer expects."""

    @abstractmethod
    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(actions, log_probs, values)`` for the given observation batch."""

    @abstractmethod
    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(log_probs, entropy, values)`` for given (obs, action) pairs."""

    @abstractmethod
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return ``V(s)``."""
