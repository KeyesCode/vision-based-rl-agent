# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] — 2026-04-18

Inference-time action masking — the minimum-sufficient fix for the argmax
collapse diagnosed across the preceding four ablations. Caps the project's
research trajectory with a five-stage capstone chart.

### Added

- **Inference-time action masking.** Optional `mask` parameter on
  `PPOActorCritic.act` and `RecurrentPPOActorCritic.act` sets masked logits
  to `−1e8` before the Categorical distribution. `build_adjacency_mask` in
  `env.action_space` produces a length-7 mask that suppresses INTERACT when
  the agent is not adjacent to a live tree. Default `mask=None` is a pure
  no-op so every pre-M10 ablation path is unchanged and still reproducible.
- **`--action-mask` CLI flag on `osrs-eval`** threads the adjacency label
  out of `info["adjacent_to_tree"]` and builds a per-step mask for the
  policy. Inference-only; the checkpoint is not modified or retrained.
- **`scripts/compare_final.py`** — 5-stage capstone comparison chart
  (baseline → DR → LSTM → representation → + masking).
- **`docs/results/system_evolution.png`** — committed result asset.
- **`tests/test_action_masking.py`** — 5 tests: mask helper correctness,
  feedforward mask forces non-INTERACT argmax, mask is a no-op when all
  actions are allowed, recurrent path enforces the same semantics.

### Findings

- **Inference-time action masking** broke a deterministic argmax collapse
  that the preceding four training-time interventions (baseline, DR, LSTM,
  representation) had all landed at identically (−3.36 return, 100%
  INTERACT share at argmax). Masking lifted deterministic return to +4.71,
  doubled deterministic success rate (8% → 16%), and pushed INTERACT share
  from 100% to 0.8% — all without any retraining or additional parameters.
  The 5-stage capstone chart in TECHNICAL_REPORT.md is the canonical
  artifact of the project's full research trajectory.

## [0.4.0] — 2026-04-18

Representation attack — RGB 128×128 input plus an adjacency auxiliary loss on
top of the CNN features. Diagnostically completes the trilogy of ablations
(feedforward → LSTM → representation) and localizes the remaining performance
gap to the actor head's marginal action preference.

### Added

- **Representation attack (Option A + Option B, combined).**
  - *Option A* — `vision.resize_to: 128` + `vision.grayscale: false`, CNN input
    shape goes `(4, 84, 84)` → `(12, 128, 128)`.
  - *Option B* — an always-present `aux_head = Linear(feature_dim, 1)` on top of
    the CNN features of both `PPOActorCritic` and `RecurrentPPOActorCritic`.
    `evaluate_actions` / `evaluate_sequence` return aux logits alongside
    `log_probs / entropy / values`. Trainers add
    `aux_adjacency_coef * BCEWithLogitsLoss` to the PPO total when the coef is
    non-zero.
  - `info["adjacent_to_tree"]` scalar on reset and every step,
    `RolloutBuffer.adjacency` storage, `Trainer._extract_adjacency` plumbing,
    TB metrics `aux/adjacency_loss` and `aux/adjacency_accuracy`.
- **`configs/ppo_woodcutting_repr.yaml`** — 128×128 RGB + `aux_adjacency_coef: 0.1`.
- **`scripts/compare_representation.py`** — 4-panel baseline-vs-upgrade chart.
- **`tests/test_aux_loss.py`** — 6 tests covering env label emission, buffer
  storage, feedforward aux forward pass, recurrent aux forward pass (T, B shape).
- **`docs/results/representation_vs_baseline.png`** — committed result asset.

### Changed

- **`load_checkpoint` default to `strict=False`** so older checkpoints (without
  the aux head) still load cleanly into newer policy classes.

### Findings

- The aux head trained to **98.7% accuracy** on the adjacency label — the CNN
  features now provably encode adjacency as a linearly separable direction.
  **Stochastic success rate more than doubled (4% → 10%)**. Deterministic
  argmax behavior was **bit-identical** to the grayscale baseline (−3.36
  return, 100% INTERACT), establishing that the remaining failure mode is in
  the actor head's marginal preference for INTERACT — not in features and not
  in memory. README documents this with the three-experiment diagnosis
  (feedforward → LSTM → representation) and points the next milestone at
  action masking using the trained aux head at inference time.

## [0.3.0] — 2026-04-18

## [0.3.0] — 2026-04-18

Domain randomization + recurrent PPO ablation. The headline result is a
reproducible, cleanly-ablated sim-to-real robustness improvement (DR) and a
rigorously falsified hypothesis about memory (LSTM).

### Added

- **Domain randomization in the simulator** (`src/osrs_rl/env/simulator/randomization.py`).
  Per-episode palette / HUD-position / distractor-clutter / tree-size jitter plus
  per-frame brightness / contrast / Gaussian-noise augmentation — each behind an
  independent config flag, identity-tested to be bit-exact with the pre-DR path
  when disabled.
- **`configs/ppo_woodcutting_dr.yaml`** — moderate DR preset (color_jitter 0.18,
  clutter 6%, brightness 0.15, contrast 0.12, noise σ=6, HUD-side randomized,
  tree-size jitter 1).
- **`scripts/compare_robustness.py`** and **`scripts/render_dr_samples.py`** —
  2×2 robustness bar chart and 3×3 randomized-frame sampler for the README.
- **`docs/results/dr_samples.png`**, **`docs/results/robustness.png`** — committed
  result assets.
- **Recurrent PPO (LSTM).** `RecurrentPPOActorCritic`, `RecurrentPPOTrainer`, and
  `RecurrentRolloutBuffer` sit behind `cfg.ppo.recurrent`. Hidden state resets
  per step on `episode_starts == 1`; PPO updates minibatch by envs, not
  timesteps, preserving temporal order.
- **`configs/ppo_woodcutting_lstm.yaml`** — identical hyperparameters to the
  feedforward config aside from the recurrent block.
- **`scripts/compare_recurrent.py`** — 4-panel feedforward-vs-recurrent chart.
- **`tests/test_randomization.py`** — 5 tests: identity when disabled, diversity
  when enabled, no-op on zeroed knobs, frame-level no-op, clutter never overlaps
  occupied tiles.
- **`tests/test_recurrent.py`** — 4 tests: initial-hidden shape,
  episode-start resets hidden correctly, hidden persists between steps without a
  reset, full rollout + PPO update smoke.

### Changed

- **`Trainer` and `evaluate.py`** branch on `cfg.ppo.recurrent` for rollout
  collection, value bootstrap, and deterministic/stochastic eval. Feedforward
  path is untouched.
- **`MockOSRSClient`** now consumes an `EpisodeVisuals` snapshot; `make_env`
  gained a `randomization_cfg` parameter threaded through the trainer and
  evaluator.
- **README** — added a "Domain randomization" section with the 2×2 robustness
  matrix and a "Recurrent policy (LSTM) — a hypothesis, rigorously falsified"
  section with the feedforward-vs-recurrent ablation.

### Findings

- **Domain randomization acted as a regularizer** on top of improving robustness:
  the DR-trained policy was the best policy on *both* the baseline and the
  randomized evaluation envs. Success rate on randomized visuals rose from 0.0%
  (baseline policy) to 6.7% (DR policy).
- **Recurrent PPO ablation produced a clean negative result**: +1.5 stochastic
  return, identical deterministic return, identical success rate, identical
  argmax collapse vs the feedforward baseline. Documented honestly in the README
  as evidence that the remaining performance gap sits in the CNN encoder
  (feature expressivity), not in policy memory. Redirects the next milestone to
  higher input resolution + adjacency-aux-loss.

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
