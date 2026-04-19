"""PPO training loop.

Orchestrates vectorized env rollouts, PPO updates, TensorBoard logging, checkpointing,
and periodic deterministic evaluation. The loop mirrors the CleanRL reference PPO so
metrics are directly comparable to the standard literature.
"""

from __future__ import annotations

import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv

from osrs_rl.agents.ppo import PPOActorCritic, PPOTrainer
from osrs_rl.agents.rollout_buffer import RolloutBuffer
from osrs_rl.env.action_space import ActionDecoder
from osrs_rl.env.osrs_env import make_env
from osrs_rl.training.checkpoint import save_checkpoint
from osrs_rl.utils.config import TrainConfig, config_to_dict
from osrs_rl.utils.logging import create_run_dir, create_writer, log_hparams, setup_logger
from osrs_rl.utils.seeding import resolve_device, set_seed

_LOG = setup_logger(__name__)


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = resolve_device(cfg.device)
        _LOG.info(f"Using device: {self.device}")

        self.run_dir = create_run_dir(cfg.logging.log_dir, cfg.logging.run_name)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.writer = create_writer(self.run_dir)
        log_hparams(self.writer, config_to_dict(cfg))
        _LOG.info(f"Run dir: {self.run_dir}")

        self.envs = self._build_envs()
        self.eval_env = self._build_eval_env()

        obs_shape = self.envs.single_observation_space.shape  # type: ignore[union-attr]
        in_channels = obs_shape[0]
        input_hw = (obs_shape[1], obs_shape[2])
        num_actions = ActionDecoder.n_actions()

        self.policy = PPOActorCritic(
            num_actions=num_actions, in_channels=in_channels, input_hw=input_hw
        ).to(self.device)
        self.ppo = PPOTrainer(self.policy, cfg.ppo, self.device)

        self.buffer = RolloutBuffer(
            rollout_steps=cfg.ppo.rollout_steps,
            num_envs=cfg.ppo.num_envs,
            obs_shape=obs_shape,
            device=self.device,
            obs_dtype=torch.uint8,
        )

        self._episode_returns: deque[float] = deque(maxlen=100)
        self._episode_lengths: deque[int] = deque(maxlen=100)
        self._episode_success: deque[float] = deque(maxlen=100)
        self._episode_trees: deque[float] = deque(maxlen=100)
        self._episode_invalid: deque[float] = deque(maxlen=100)
        self._episode_idle: deque[float] = deque(maxlen=100)

    # ----------------------------------------------------------------- env builders

    def _build_envs(self) -> SyncVectorEnv:
        thunks = [
            make_env(self.cfg.env, self.cfg.vision, self.cfg.reward, self.cfg.seed, i)
            for i in range(self.cfg.ppo.num_envs)
        ]
        return SyncVectorEnv(thunks)

    def _build_eval_env(self) -> gym.Env:
        return make_env(
            self.cfg.env, self.cfg.vision, self.cfg.reward, self.cfg.seed + 10_000, 0
        )()

    # ----------------------------------------------------------------- main loop

    def train(self) -> None:
        cfg = self.cfg
        num_updates = cfg.ppo.total_timesteps // (cfg.ppo.num_envs * cfg.ppo.rollout_steps)
        _LOG.info(
            f"Starting training: {num_updates} updates, "
            f"{cfg.ppo.num_envs} envs × {cfg.ppo.rollout_steps} steps per rollout"
        )

        obs, _ = self.envs.reset(seed=cfg.seed)
        obs_t = torch.as_tensor(obs, device=self.device)
        done_t = torch.zeros(cfg.ppo.num_envs, device=self.device)

        global_step = 0
        start_time = time.time()

        for update in range(1, num_updates + 1):
            if cfg.ppo.anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                self.ppo.set_learning_rate(frac * cfg.ppo.learning_rate)

            self.buffer.reset()
            for _ in range(cfg.ppo.rollout_steps):
                global_step += cfg.ppo.num_envs
                with torch.no_grad():
                    action, log_prob, value = self.policy.act(obs_t)

                next_obs, reward, terminated, truncated, infos = self.envs.step(
                    action.cpu().numpy()
                )
                done = np.logical_or(terminated, truncated)

                self.buffer.add(
                    obs=obs_t,
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    done=done_t,
                    value=value,
                )

                self._record_episode_stats(infos)

                obs_t = torch.as_tensor(next_obs, device=self.device)
                done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

            # Bootstrap value from the last observation for GAE.
            with torch.no_grad():
                last_values = self.policy.get_value(obs_t)
            self.buffer.compute_returns_and_advantages(
                last_values=last_values,
                last_dones=done_t,
                gamma=cfg.ppo.gamma,
                gae_lambda=cfg.ppo.gae_lambda,
            )

            metrics = self.ppo.update(self.buffer)

            sps = int(global_step / (time.time() - start_time))
            if update % cfg.logging.log_interval_updates == 0:
                self._log_metrics(update, global_step, metrics, sps)

            if update % cfg.logging.checkpoint_interval_updates == 0 or update == num_updates:
                self._save(update, global_step)

            if update % cfg.logging.eval_interval_updates == 0 or update == num_updates:
                self._eval(global_step)

        self.writer.close()
        _LOG.info(f"Training complete. Artifacts: {self.run_dir}")

    # ----------------------------------------------------------------- helpers

    def _record_episode_stats(self, infos: dict) -> None:
        """Capture per-episode return/length/success from batched vector-env infos."""
        # Batched form (gymnasium >=0.29 default for SyncVectorEnv).
        if "episode" in infos:
            ep = infos["episode"]
            r = np.asarray(ep["r"])
            lens = np.asarray(ep["l"])
            mask = np.asarray(
                infos.get("_episode", np.ones_like(r, dtype=bool)), dtype=bool
            )
            self._capture_task_metrics(infos, mask)
            for i in range(len(mask)):
                if bool(mask[i]):
                    self._episode_returns.append(float(r[i]))
                    self._episode_lengths.append(int(lens[i]))
            return
        # Per-env list form (alternative info layouts).
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    self._episode_returns.append(float(info["episode"]["r"]))
                    self._episode_lengths.append(int(info["episode"]["l"]))
                    if "episode_success" in info:
                        self._episode_success.append(float(info["episode_success"]))
                        self._episode_trees.append(float(info["episode_trees_chopped"]))
                        self._episode_invalid.append(float(info["episode_invalid_ratio"]))
                        self._episode_idle.append(float(info["episode_idle_ratio"]))

    def _capture_task_metrics(self, infos: dict, mask: np.ndarray) -> None:
        """Pull EpisodeStatsWrapper fields out of a batched info dict."""
        keys = (
            ("episode_success", self._episode_success, float),
            ("episode_trees_chopped", self._episode_trees, float),
            ("episode_invalid_ratio", self._episode_invalid, float),
            ("episode_idle_ratio", self._episode_idle, float),
        )
        for key, buffer, cast in keys:
            if key not in infos:
                continue
            values = np.asarray(infos[key])
            # Each stats field has a companion "_<key>" mask when it's vector-batched.
            per_field_mask = infos.get(f"_{key}")
            combined = mask if per_field_mask is None else np.asarray(per_field_mask, dtype=bool)
            for i in range(len(combined)):
                if bool(combined[i]):
                    buffer.append(cast(values[i]))

    def _log_metrics(
        self, update: int, global_step: int, metrics, sps: int
    ) -> None:
        w = self.writer
        w.add_scalar("charts/learning_rate", metrics.learning_rate, global_step)
        w.add_scalar("losses/policy_loss", metrics.policy_loss, global_step)
        w.add_scalar("losses/value_loss", metrics.value_loss, global_step)
        w.add_scalar("losses/entropy", metrics.entropy, global_step)
        w.add_scalar("losses/approx_kl", metrics.approx_kl, global_step)
        w.add_scalar("losses/clip_fraction", metrics.clip_fraction, global_step)
        w.add_scalar("losses/explained_variance", metrics.explained_variance, global_step)
        w.add_scalar("charts/sps", sps, global_step)

        if self._episode_returns:
            mean_r = float(np.mean(self._episode_returns))
            mean_l = float(np.mean(self._episode_lengths))
            w.add_scalar("charts/episode_return", mean_r, global_step)
            w.add_scalar("charts/episode_length", mean_l, global_step)
            extras = ""
            if self._episode_success:
                success = float(np.mean(self._episode_success))
                trees = float(np.mean(self._episode_trees))
                invalid = float(np.mean(self._episode_invalid))
                idle = float(np.mean(self._episode_idle))
                w.add_scalar("charts/success_rate", success, global_step)
                w.add_scalar("charts/trees_chopped", trees, global_step)
                w.add_scalar("charts/invalid_action_ratio", invalid, global_step)
                w.add_scalar("charts/idle_ratio", idle, global_step)
                extras = (
                    f" success={success:.2f} trees={trees:.1f} "
                    f"invalid={invalid:.2f} idle={idle:.2f}"
                )
            _LOG.info(
                f"upd={update} step={global_step:,} "
                f"ep_ret={mean_r:+.2f} ep_len={mean_l:.0f}{extras} "
                f"sps={sps} pi_loss={metrics.policy_loss:+.3f} "
                f"v_loss={metrics.value_loss:.3f} ent={metrics.entropy:.3f} "
                f"kl={metrics.approx_kl:.4f}"
            )
        else:
            _LOG.info(
                f"upd={update} step={global_step:,} sps={sps} "
                f"pi_loss={metrics.policy_loss:+.3f} v_loss={metrics.value_loss:.3f}"
            )

    def _save(self, update: int, global_step: int) -> None:
        path = self.ckpt_dir / f"ckpt_upd{update:05d}.pt"
        save_checkpoint(
            path,
            policy=self.policy,
            optimizer=self.ppo.optimizer,
            global_step=global_step,
            extra={"update": update, "config": config_to_dict(self.cfg)},
        )
        latest = self.ckpt_dir / "latest.pt"
        save_checkpoint(
            latest,
            policy=self.policy,
            optimizer=self.ppo.optimizer,
            global_step=global_step,
            extra={"update": update, "config": config_to_dict(self.cfg)},
        )
        _LOG.info(f"Saved checkpoint to {path}")

    @torch.no_grad()
    def _eval(self, global_step: int) -> None:
        returns: list[float] = []
        lengths: list[int] = []
        for _ in range(self.cfg.logging.eval_episodes):
            obs, _ = self.eval_env.reset()
            total_r = 0.0
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                action, _, _ = self.policy.act(obs_t, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(
                    int(action.item())
                )
                total_r += float(reward)
                steps += 1
            returns.append(total_r)
            lengths.append(steps)
        self.writer.add_scalar("eval/episode_return", float(np.mean(returns)), global_step)
        self.writer.add_scalar("eval/episode_length", float(np.mean(lengths)), global_step)
        _LOG.info(
            f"[eval] step={global_step:,} ret={np.mean(returns):+.2f} len={np.mean(lengths):.0f}"
        )
