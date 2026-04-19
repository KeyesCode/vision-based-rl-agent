"""PPO (from scratch).

Closely follows the 37-implementation-details PPO reference (Huang et al. 2022):
clipped surrogate objective, clipped value loss, advantage normalization, gradient
clipping, orthogonal init, optional target-KL early stop, and linear LR annealing.

Kept deliberately self-contained so a reviewer can see the whole algorithm in one file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from osrs_rl.agents.base_policy import BasePolicy
from osrs_rl.agents.networks import NatureCNN, layer_init
from osrs_rl.agents.rollout_buffer import RolloutBuffer
from osrs_rl.utils.config import PPOConfig


class PPOActorCritic(nn.Module, BasePolicy):
    """Shared-backbone actor-critic for discrete action spaces."""

    def __init__(
        self,
        num_actions: int,
        in_channels: int,
        input_hw: tuple[int, int] = (84, 84),
        feature_dim: int = 512,
    ):
        super().__init__()
        self.backbone = NatureCNN(in_channels, input_hw=input_hw, feature_dim=feature_dim)
        # Small-std init on the actor is a common PPO detail — prevents early-training
        # logits from saturating one action.
        self.actor = layer_init(nn.Linear(feature_dim, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(feature_dim, 1), std=1.0)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        # uint8 -> float32 in [0, 1]. Cheap and numerically safe for CNN inputs.
        x = obs.float() / 255.0
        return self.backbone(x)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self._features(obs)).squeeze(-1)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self._features(obs)
        logits = self.actor(feat)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), self.critic(feat).squeeze(-1)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self._features(obs)
        logits = self.actor(feat)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), self.critic(feat).squeeze(-1)


@dataclass
class PPOUpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    learning_rate: float


class PPOTrainer:
    """Runs the PPO update step given a filled :class:`RolloutBuffer`."""

    def __init__(
        self,
        policy: PPOActorCritic,
        cfg: PPOConfig,
        device: torch.device,
    ):
        self.policy = policy
        self.cfg = cfg
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self._batch_size = cfg.num_envs * cfg.rollout_steps
        if self._batch_size % cfg.num_minibatches != 0:
            raise ValueError("num_envs * rollout_steps must be divisible by num_minibatches")
        self._minibatch_size = self._batch_size // cfg.num_minibatches

    def set_learning_rate(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def update(self, buffer: RolloutBuffer) -> PPOUpdateMetrics:
        data = buffer.flatten()
        b_obs = data["obs"]
        b_actions = data["actions"]
        b_log_probs = data["log_probs"]
        b_advantages = data["advantages"]
        b_returns = data["returns"]
        b_values = data["values"]

        indices = np.arange(self._batch_size)
        clip_fractions: list[float] = []
        approx_kls: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []

        early_stop = False
        for _ in range(self.cfg.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, self._batch_size, self._minibatch_size):
                mb_idx = indices[start : start + self._minibatch_size]
                mb_idx_t = torch.as_tensor(mb_idx, device=self.device)

                new_logp, entropy, new_values = self.policy.evaluate_actions(
                    b_obs[mb_idx_t], b_actions[mb_idx_t]
                )
                logratio = new_logp - b_log_probs[mb_idx_t]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clip_fractions.append(
                        ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()
                    )

                mb_adv = b_advantages[mb_idx_t]
                if self.cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Clipped surrogate
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(
                    ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef
                )
                pg_loss = torch.max(pg1, pg2).mean()

                # Value loss (optionally clipped)
                if self.cfg.clip_vloss:
                    v_clipped = b_values[mb_idx_t] + torch.clamp(
                        new_values - b_values[mb_idx_t],
                        -self.cfg.clip_coef,
                        self.cfg.clip_coef,
                    )
                    v_loss_un = (new_values - b_returns[mb_idx_t]).pow(2)
                    v_loss_cl = (v_clipped - b_returns[mb_idx_t]).pow(2)
                    v_loss = 0.5 * torch.max(v_loss_un, v_loss_cl).mean()
                else:
                    v_loss = 0.5 * (new_values - b_returns[mb_idx_t]).pow(2).mean()

                ent_loss = entropy.mean()

                loss = pg_loss - self.cfg.ent_coef * ent_loss + self.cfg.vf_coef * v_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                approx_kls.append(approx_kl)
                policy_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropies.append(ent_loss.item())

                if self.cfg.target_kl is not None and approx_kl > self.cfg.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        # Explained variance is the standard value-fn diagnostic.
        y_true = b_returns.detach().cpu().numpy()
        y_pred = b_values.detach().cpu().numpy()
        var_y = float(np.var(y_true))
        explained_var = float("nan") if var_y == 0.0 else 1.0 - float(np.var(y_true - y_pred)) / var_y

        return PPOUpdateMetrics(
            policy_loss=float(np.mean(policy_losses)),
            value_loss=float(np.mean(value_losses)),
            entropy=float(np.mean(entropies)),
            approx_kl=float(np.mean(approx_kls)),
            clip_fraction=float(np.mean(clip_fractions)),
            explained_variance=explained_var,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )
