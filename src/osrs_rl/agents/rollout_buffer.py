"""On-policy rollout buffer with GAE.

Shape convention: rollouts are stored as ``(T, N, ...)`` tensors where ``T`` is the
rollout horizon and ``N`` is the number of parallel envs. The buffer is flattened to
``(T*N, ...)`` before PPO minibatching.
"""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-size on-policy buffer (PPO-style) with GAE-λ advantage computation."""

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        device: torch.device,
        obs_dtype: torch.dtype = torch.uint8,
    ):
        self.T = rollout_steps
        self.N = num_envs
        self.device = device

        self.obs = torch.zeros((self.T, self.N, *obs_shape), dtype=obs_dtype, device=device)
        self.actions = torch.zeros((self.T, self.N), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((self.T, self.N), device=device)
        self.rewards = torch.zeros((self.T, self.N), device=device)
        self.dones = torch.zeros((self.T, self.N), device=device)
        self.values = torch.zeros((self.T, self.N), device=device)

        self.advantages = torch.zeros((self.T, self.N), device=device)
        self.returns = torch.zeros((self.T, self.N), device=device)

        self._step = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: np.ndarray | torch.Tensor,
        done: np.ndarray | torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        t = self._step
        self.obs[t] = obs
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.rewards[t] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.dones[t] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.values[t] = value
        self._step += 1

    def full(self) -> bool:
        return self._step == self.T

    def reset(self) -> None:
        self._step = 0

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE(λ) advantages + empirical returns in-place."""
        adv = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_nonterminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_nonterminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            adv = delta + gamma * gae_lambda * next_nonterminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def flatten(self) -> dict[str, torch.Tensor]:
        """Flatten ``(T, N, ...)`` -> ``(T*N, ...)`` for minibatching."""
        return {
            "obs": self.obs.reshape(self.T * self.N, *self.obs.shape[2:]),
            "actions": self.actions.reshape(-1),
            "log_probs": self.log_probs.reshape(-1),
            "advantages": self.advantages.reshape(-1),
            "returns": self.returns.reshape(-1),
            "values": self.values.reshape(-1),
        }


class RecurrentRolloutBuffer(RolloutBuffer):
    """Rollout buffer that additionally stores the LSTM hidden state at the start
    of each rollout, so sequences can be replayed during PPO updates.

    The storage shapes for everything else (obs / actions / rewards / dones /
    values) are unchanged — they're still ``(T, N, ...)``. The only reason this
    subclass exists is to hold ``initial_hidden`` and expose it to the trainer.
    """

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        hidden_size: int,
        device: torch.device,
        obs_dtype: torch.dtype = torch.uint8,
        num_layers: int = 1,
    ):
        super().__init__(rollout_steps, num_envs, obs_shape, device, obs_dtype=obs_dtype)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.initial_hidden: tuple[torch.Tensor, torch.Tensor] = (
            torch.zeros(num_layers, num_envs, hidden_size, device=device),
            torch.zeros(num_layers, num_envs, hidden_size, device=device),
        )

    def set_initial_hidden(self, hidden: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Capture the hidden state that was active at rollout-step 0."""
        h, c = hidden
        self.initial_hidden = (h.detach().clone(), c.detach().clone())
