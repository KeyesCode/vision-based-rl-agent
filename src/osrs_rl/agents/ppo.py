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
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from osrs_rl.agents.base_policy import BasePolicy, RecurrentPolicy, RecurrentState
from osrs_rl.agents.networks import NatureCNN, layer_init
from osrs_rl.agents.rollout_buffer import RecurrentRolloutBuffer, RolloutBuffer
from osrs_rl.utils.config import PPOConfig


class PPOActorCritic(nn.Module, BasePolicy):
    """Shared-backbone actor-critic for discrete action spaces.

    An always-present ``aux_head`` predicts a binary label from CNN features —
    used for the adjacency auxiliary loss when ``aux_adjacency_coef > 0``. Its
    parameters are still loss-free when the coef is zero (the head just isn't
    called), so adding it costs nothing for feedforward-only runs.
    """

    def __init__(
        self,
        num_actions: int,
        in_channels: int,
        input_hw: tuple[int, int] = (84, 84),
        feature_dim: int = 512,
    ):
        super().__init__()
        self.backbone = NatureCNN(in_channels, input_hw=input_hw, feature_dim=feature_dim)
        self.actor = layer_init(nn.Linear(feature_dim, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(feature_dim, 1), std=1.0)
        # Binary classifier on CNN features. Std=1.0 mimics the critic init.
        self.aux_head = layer_init(nn.Linear(feature_dim, 1), std=1.0)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(log_probs, entropy, values, aux_logits)`` — the final element
        is the raw aux-head output; trainers apply BCE only when the aux coef > 0.
        """
        feat = self._features(obs)
        logits = self.actor(feat)
        dist = Categorical(logits=logits)
        aux = self.aux_head(feat).squeeze(-1)
        return dist.log_prob(actions), dist.entropy(), self.critic(feat).squeeze(-1), aux


@dataclass
class PPOUpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    learning_rate: float
    # Adjacency auxiliary loss — stays at 0 / nan when the aux coef is 0.
    aux_loss: float = 0.0
    aux_accuracy: float = float("nan")


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
        b_adjacency = data["adjacency"]

        indices = np.arange(self._batch_size)
        clip_fractions: list[float] = []
        approx_kls: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        aux_losses: list[float] = []
        aux_accuracies: list[float] = []
        use_aux = self.cfg.aux_adjacency_coef > 0.0

        early_stop = False
        for _ in range(self.cfg.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, self._batch_size, self._minibatch_size):
                mb_idx = indices[start : start + self._minibatch_size]
                mb_idx_t = torch.as_tensor(mb_idx, device=self.device)

                new_logp, entropy, new_values, new_aux = self.policy.evaluate_actions(
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
                if use_aux:
                    mb_adj = b_adjacency[mb_idx_t]
                    aux_loss = F.binary_cross_entropy_with_logits(new_aux, mb_adj)
                    loss = loss + self.cfg.aux_adjacency_coef * aux_loss
                    with torch.no_grad():
                        aux_pred = (torch.sigmoid(new_aux) > 0.5).float()
                        aux_acc = (aux_pred == mb_adj).float().mean().item()
                    aux_losses.append(aux_loss.item())
                    aux_accuracies.append(aux_acc)

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
            aux_loss=float(np.mean(aux_losses)) if aux_losses else 0.0,
            aux_accuracy=float(np.mean(aux_accuracies)) if aux_accuracies else float("nan"),
        )


# ============================================================================
# Recurrent PPO — adds a single LSTM layer between the CNN and the policy heads.
# Sits behind ``cfg.ppo.recurrent`` so the feedforward path is untouched.
# ============================================================================


class RecurrentPPOActorCritic(nn.Module, RecurrentPolicy):
    """CNN -> LSTM -> (actor, critic) for discrete action spaces.

    The CNN backbone is reused verbatim from :class:`PPOActorCritic`; only the
    LSTM layer and the fact that heads read from ``hidden_size`` (not
    ``feature_dim``) are new.
    """

    def __init__(
        self,
        num_actions: int,
        in_channels: int,
        input_hw: tuple[int, int] = (84, 84),
        feature_dim: int = 512,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.backbone = NatureCNN(in_channels, input_hw=input_hw, feature_dim=feature_dim)
        self.lstm = nn.LSTM(feature_dim, hidden_size, num_layers=1)
        # Orthogonal init is standard for PPO recurrent layers too.
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(hidden_size, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)
        # Aux head reads directly from CNN features (pre-LSTM) so the supervised
        # signal forces the backbone — not the LSTM — to encode adjacency.
        self.aux_head = layer_init(nn.Linear(feature_dim, 1), std=1.0)
        self.hidden_size = hidden_size
        self.num_layers = 1

    # ------------------------------------------------------------------ public API

    def initial_hidden(self, batch_size: int, device: torch.device) -> RecurrentState:
        zeros = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return zeros, zeros.clone()

    def act(
        self,
        obs: torch.Tensor,
        hidden: RecurrentState,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RecurrentState]:
        out, new_hidden = self._step(obs, hidden, episode_starts)
        logits = self.actor(out)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        value = self.critic(out).squeeze(-1)
        return action, dist.log_prob(action), value, new_hidden

    def get_value(
        self,
        obs: torch.Tensor,
        hidden: RecurrentState,
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = self._step(obs, hidden, episode_starts)
        return self.critic(out).squeeze(-1)

    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        initial_hidden: RecurrentState,
        episode_starts_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay a batch of env sequences. Returns ``(log_probs, entropy, values, aux_logits)``.

        Aux logits are computed from CNN features before the LSTM — the supervised
        signal flows directly into the backbone, bypassing the recurrent layer.
        """
        T, B = actions_seq.shape
        flat = obs_seq.reshape(T * B, *obs_seq.shape[2:])
        feat_tb = self._cnn_features(flat)            # (T*B, F)
        aux_logits = self.aux_head(feat_tb).squeeze(-1).reshape(T, B)
        feat_seq = feat_tb.reshape(T, B, -1)
        out_tbh, _ = self._lstm_with_resets(feat_seq, initial_hidden, episode_starts_seq.to(feat_seq))
        logits = self.actor(out_tbh)
        dist = Categorical(logits=logits)
        value = self.critic(out_tbh).squeeze(-1)
        return dist.log_prob(actions_seq), dist.entropy(), value, aux_logits

    # ------------------------------------------------------------------ internals

    def _cnn_features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs.float() / 255.0)

    def _step(
        self, obs: torch.Tensor, hidden: RecurrentState, episode_starts: torch.Tensor
    ) -> tuple[torch.Tensor, RecurrentState]:
        """One-step inference over a (B, ...) obs batch."""
        feat = self._cnn_features(obs).unsqueeze(0)          # (1, B, F)
        starts = episode_starts.to(feat).unsqueeze(0)        # (1, B)
        out, new_hidden = self._lstm_with_resets(feat, hidden, starts)
        return out.squeeze(0), new_hidden                    # (B, H)

    def _sequence(
        self,
        obs_seq: torch.Tensor,
        hidden: RecurrentState,
        episode_starts_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """Full (T, B, ...) replay for PPO updates."""
        T, B = obs_seq.shape[:2]
        flat = obs_seq.reshape(T * B, *obs_seq.shape[2:])
        feat = self._cnn_features(flat).reshape(T, B, -1)
        starts = episode_starts_seq.to(feat)
        return self._lstm_with_resets(feat, hidden, starts)

    def _lstm_with_resets(
        self,
        feat_tbf: torch.Tensor,
        hidden: RecurrentState,
        starts_tb: torch.Tensor,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """Run the LSTM step-by-step, zeroing ``(h, c)`` whenever an episode starts.

        Per-step loop is the clearest way to get episode-boundary resets exactly
        right. For typical ``T=128`` it is a negligible cost vs the CNN forward.
        """
        T, B, _ = feat_tbf.shape
        h, c = hidden
        outputs: list[torch.Tensor] = []
        for t in range(T):
            keep = (1.0 - starts_tb[t]).view(1, B, 1)  # (1, B, 1)
            h = h * keep
            c = c * keep
            step_out, (h, c) = self.lstm(feat_tbf[t : t + 1], (h, c))
            outputs.append(step_out)
        return torch.cat(outputs, dim=0), (h, c)


class RecurrentPPOTrainer:
    """PPO update that replays the LSTM through each minibatch sequence.

    Minibatches partition **envs**, not timesteps — each minibatch is a full
    length-``T`` sequence for a subset of envs, so temporal order is preserved.
    """

    def __init__(self, policy: RecurrentPPOActorCritic, cfg: PPOConfig, device: torch.device):
        self.policy = policy
        self.cfg = cfg
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.learning_rate, eps=1e-5)
        if cfg.num_envs % cfg.num_minibatches != 0:
            raise ValueError("num_envs must be divisible by num_minibatches for recurrent PPO")
        self._envs_per_mb = cfg.num_envs // cfg.num_minibatches

    def set_learning_rate(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def update(self, buffer: RecurrentRolloutBuffer) -> PPOUpdateMetrics:
        cfg = self.cfg
        N = buffer.N
        obs = buffer.obs              # (T, N, C, H, W)
        actions = buffer.actions      # (T, N)
        old_log_probs = buffer.log_probs
        advantages_full = buffer.advantages
        returns_full = buffer.returns
        old_values_full = buffer.values
        episode_starts = buffer.dones  # dones[t] == 1 iff obs[t] is post-reset
        h0, c0 = buffer.initial_hidden  # (1, N, H) each

        adjacency_full = buffer.adjacency  # (T, N)
        use_aux = cfg.aux_adjacency_coef > 0.0

        env_idx = np.arange(N)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []
        aux_losses: list[float] = []
        aux_accuracies: list[float] = []

        early_stop = False
        for _ in range(cfg.num_epochs):
            np.random.shuffle(env_idx)
            for start in range(0, N, self._envs_per_mb):
                mb = env_idx[start : start + self._envs_per_mb]
                mb_t = torch.as_tensor(mb, device=self.device, dtype=torch.long)

                mb_obs = obs[:, mb_t]
                mb_actions = actions[:, mb_t]
                mb_old_logp = old_log_probs[:, mb_t]
                mb_adv = advantages_full[:, mb_t]
                mb_returns = returns_full[:, mb_t]
                mb_old_values = old_values_full[:, mb_t]
                mb_starts = episode_starts[:, mb_t]
                mb_hidden = (h0[:, mb_t].contiguous(), c0[:, mb_t].contiguous())
                mb_adjacency = adjacency_full[:, mb_t]

                new_logp, entropy, new_values, new_aux = self.policy.evaluate_sequence(
                    mb_obs, mb_actions, mb_hidden, mb_starts
                )

                logratio = new_logp - mb_old_logp
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clip_fractions.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    )

                adv = mb_adv
                if cfg.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                if cfg.clip_vloss:
                    v_clipped = mb_old_values + torch.clamp(
                        new_values - mb_old_values, -cfg.clip_coef, cfg.clip_coef
                    )
                    v_un = (new_values - mb_returns).pow(2)
                    v_cl = (v_clipped - mb_returns).pow(2)
                    v_loss = 0.5 * torch.max(v_un, v_cl).mean()
                else:
                    v_loss = 0.5 * (new_values - mb_returns).pow(2).mean()

                ent_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss
                if use_aux:
                    aux_loss = F.binary_cross_entropy_with_logits(new_aux, mb_adjacency)
                    loss = loss + cfg.aux_adjacency_coef * aux_loss
                    with torch.no_grad():
                        aux_pred = (torch.sigmoid(new_aux) > 0.5).float()
                        aux_acc = (aux_pred == mb_adjacency).float().mean().item()
                    aux_losses.append(aux_loss.item())
                    aux_accuracies.append(aux_acc)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                approx_kls.append(approx_kl)
                policy_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropies.append(ent_loss.item())

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        y_true = returns_full.detach().cpu().numpy().ravel()
        y_pred = old_values_full.detach().cpu().numpy().ravel()
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
            aux_loss=float(np.mean(aux_losses)) if aux_losses else 0.0,
            aux_accuracy=float(np.mean(aux_accuracies)) if aux_accuracies else float("nan"),
        )
