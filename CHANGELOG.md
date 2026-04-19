# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Recurrent PPO (LSTM).** `RecurrentPPOActorCritic`, `RecurrentPPOTrainer`, and
  `RecurrentRolloutBuffer` sit behind `cfg.ppo.recurrent`. Hidden state resets
  per step on `episode_starts == 1`; PPO updates minibatch by envs, not timesteps,
  preserving temporal order.
- **`configs/ppo_woodcutting_lstm.yaml`** — identical hyperparameters to the
  feedforward config aside from the recurrent block.
- **`scripts/compare_recurrent.py`** — 4-panel feedforward-vs-recurrent chart.
- **`tests/test_recurrent.py`** — hidden-state-reset correctness + full
  one-update smoke test.

### Changed

- **`Trainer` and `evaluate.py`** branch on `cfg.ppo.recurrent` for rollout
  collection, value bootstrap, and CLI/deterministic eval. Feedforward path is
  untouched.

### Findings

- The LSTM ablation produced a **clean negative result**: +1.5 stochastic return,
  identical deterministic return, identical success rate, identical argmax
  collapse vs the feedforward baseline. Documented honestly in the README as
  evidence that the remaining performance gap sits in the CNN encoder (feature
  expressivity), not in policy memory.

## [0.2.0] — 2026-04-18

Analysis, live evaluation, repo polish, and domain randomization.

### Added

- **Evaluation harness.** `osrs-eval` CLI with random-baseline, stochastic, and
  deterministic modes; JSON output plus a rich console summary
  (`src/osrs_rl/evaluation/evaluate.py`).
- **Checkpoint-progression analysis.** `scripts/evaluate_checkpoints.py` runs an
  independent stochastic eval at every saved checkpoint; `scripts/plot_training.py`
  renders the resulting progression chart alongside reward / success-rate /
  trees-chopped / action-distribution / losses PNGs.
- **`EpisodeStatsWrapper`.** Emits per-episode scalars (trees chopped, inventory
  filled, invalid/idle ratios, action counts) compatible with gymnasium vector-env
  auto-batching.
- **Live OSRS client.** `LiveOSRSClient` implements `GameClient` against real screen
  capture (`mss`) and safety-gated mouse/keyboard input (`pynput`).
- **Safety layer.** `SafetyGate` enforces `enable_live_input` opt-in, kill-switch
  file, rate limit, safe bounding-box, and audit logging on every attempted action.
  Dry-run mode rehearses the full pipeline against the live window with zero OS
  side effects (`src/osrs_rl/input_control/`).
- **Domain randomization.** Render-time palette / HUD-position / distractor-clutter /
  tree-size jitter plus per-frame brightness / contrast / Gaussian-noise
  augmentation, all behind independent config flags. Identity-tested to be bit-exact
  with the pre-DR path when disabled (`src/osrs_rl/env/simulator/randomization.py`).
- **Robustness ablation.** `configs/ppo_woodcutting_dr.yaml` + `scripts/compare_robustness.py`
  produce the 2×2 baseline-policy / DR-policy × baseline-env / DR-env chart that
  demonstrates the DR policy's success-rate advantage on perturbed visuals (0.0% →
  6.7%).
- **Continuous integration.** `.github/workflows/ci.yml` runs `ruff check` and
  `pytest -q` against Python 3.10 and 3.11 on every push and PR.
- **Architecture diagram asset.** `scripts/draw_architecture.py` regenerates
  `docs/architecture.png`; asset is committed for README rendering.
- **Reward components.** `AdjacencyBonus` (one-shot bonus on "became adjacent to a
  live tree") and `IdleActionPenalty` (discourage IDLE collapse).
- **Reward unit tests.** `tests/test_rewards.py` covers all five composable reward
  components independently.
- **`LICENSE` file.** MIT, matching `pyproject.toml`.
- **`CHANGELOG.md`.** This file.

### Changed

- **Trainer metrics.** Trainer now logs `charts/success_rate`, `charts/trees_chopped`,
  `charts/invalid_action_ratio`, and `charts/idle_ratio` in addition to reward /
  length, so TensorBoard alone tells the full learning story.
- **Reward defaults.** `log_collected` raised from 1.0 → 5.0 and
  `invalid_action_penalty` softened from −0.1 → −0.02 to make the chop signal
  dominant and keep early exploration viable; chop difficulty `chop_ticks` reduced
  from 3 → 1 so PPO discovers the reward within the first 10k steps.
- **README.** Rewritten top-to-bottom for a 2-minute recruiter read — results
  section at the top, architecture diagram embedded, copy-paste quickstart,
  explicit Limitations section, and a Next Steps sim-to-real roadmap.
- **Env factory.** `make_env` gained a `randomization_cfg` parameter; `_build_env`
  in `evaluate.py` threads it through so CLI eval honors the training DR config.

### Fixed

- **`SafetyGate` double-instantiation.** When `LiveOSRSClient` was constructed with
  an injected controller, it previously built an unused second `SafetyGate` and the
  exported `_safety.stats()` counters lagged reality. The client now reuses the
  controller's gate.
- **Typed-annotations config loader.** `load_config` now uses
  `typing.get_type_hints` so nested dataclass fields resolve correctly under
  `from __future__ import annotations`.
- **Ruff/lint pass.** Cleaned 10 lint findings across src/tests/scripts; CI
  enforces clean lint on every push.

## [0.1.0] — 2026-04-18

Initial MVP: a vision-based PPO agent trains woodcutting to a visible learning curve
in ~5 minutes on a laptop CPU.

### Added

- **Gymnasium env + abstract `GameClient` seam** so training (simulator) and
  evaluation (live) share one environment class.
- **2D woodcutting simulator** (`MockOSRSClient`) — grid world with trees, respawn
  timers, inventory, and directly-rendered RGB frames. Fast enough for
  `num_envs=8` PPO rollouts on CPU.
- **Composable reward system.** `RewardComponent` + `CompositeReward` with
  log-collection, step penalty, invalid-action penalty, distance-shaping, and
  full-inventory bonus.
- **Vision preprocessing pipeline.** Grayscale, resize to 84×84, frame-stack
  wrapper chain compatible with `gymnasium.vector.SyncVectorEnv`.
- **PPO implementation from scratch.** Nature-CNN backbone, orthogonal init,
  clipped surrogate objective, GAE(λ), advantage normalization, entropy bonus,
  linear LR annealing, optional target-KL early-stop, clipped value loss.
- **Training pipeline.** Typed-dataclass config (`tyro` + YAML), run-dir creation,
  TensorBoard scalar logging, periodic checkpointing, deterministic eval every
  N updates.
- **CLI entrypoints.** `osrs-train` and `osrs-eval` installed via
  `pip install -e .`.
- **Smoke tests.** Env shape contract, simulator chopping mechanics, wrapped
  framestack, one-update PPO smoke.
