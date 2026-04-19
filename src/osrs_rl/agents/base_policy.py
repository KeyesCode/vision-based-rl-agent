"""Policy interface.

Abstracting over "act" and "evaluate_actions" lets the trainer stay algorithm-agnostic:
PPO, DQN, or any future policy gradient variant implements this same surface.
Recurrent policies implement :class:`RecurrentPolicy` instead, which threads hidden
state through the same conceptual operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BasePolicy(ABC):
    """Stateless (feedforward) policy interface."""

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


# LSTM hidden state — a (h, c) tuple, each of shape (num_layers, batch, hidden).
RecurrentState = tuple[torch.Tensor, torch.Tensor]


class RecurrentPolicy(ABC):
    """Recurrent policy interface — threads an LSTM ``(h, c)`` through every call.

    ``episode_starts`` is a ``(batch,)`` 0/1 tensor where 1 means "obs came from a
    fresh env reset" — the LSTM resets its hidden state *before* stepping through
    such timesteps so memory from a prior episode never leaks into the next.
    """

    @abstractmethod
    def initial_hidden(self, batch_size: int, device: torch.device) -> RecurrentState:
        """Zero-valued (h, c) tuple shaped for a batch of ``batch_size`` envs."""

    @abstractmethod
    def act(
        self,
        obs: torch.Tensor,
        hidden: RecurrentState,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RecurrentState]:
        """Return ``(actions, log_probs, values, new_hidden)`` for a single step."""

    @abstractmethod
    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,            # (T, B, C, H, W)
        actions_seq: torch.Tensor,        # (T, B)
        initial_hidden: RecurrentState,   # (h, c), each (1, B, H)
        episode_starts_seq: torch.Tensor, # (T, B)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay the LSTM over ``T`` steps and return ``(log_probs, entropy, values)``."""

    @abstractmethod
    def get_value(
        self,
        obs: torch.Tensor,
        hidden: RecurrentState,
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        """One-step value estimate (used for bootstrapping at rollout end)."""
