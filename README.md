# OSRS-RL — Vision-Based Reinforcement Learning Agent

An end-to-end reinforcement learning project that learns to play Old School RuneScape
from raw pixel input. A convolutional policy network is trained with **PPO (implemented
from scratch)** against a fast in-house simulator, then evaluated against the live game
through the same `gymnasium.Env` interface — the standard sim-to-real pattern used in
autonomy research.

> Status: MVP — trains a woodcutting agent in a 2D OSRS-like simulator. Live-game client
> is scaffolded behind an interface and integrated in a later milestone.

## Why this project

This project intentionally mirrors the canonical autonomy stack:

```
pixels ─► preprocessing ─► CNN backbone ─► actor/critic heads ─► action
                                                  ▲                   │
                                                  │                   ▼
                                         PPO update  ◄── rewards ◄── env
```

- **Perception** — raw RGB frames from the game (or simulator), preprocessed to 84×84
  grayscale with frame-stacking.
- **Policy** — Nature-CNN backbone with separate actor and critic heads; orthogonal init.
- **Algorithm** — PPO with GAE, clipped objective, value loss, entropy bonus, advantage
  normalization, and LR annealing. Implemented in `src/osrs_rl/agents/ppo.py`.
- **Environment** — custom Gymnasium `Env` that wraps an abstract `GameClient`. The same
  env class runs against both the simulator and (later) a live screen-capture client.
- **Rewards** — composable `RewardComponent` objects summed into a `CompositeReward`, so
  new tasks can redefine the learning signal without touching the env.

## Quickstart

```bash
# 1. Create an environment and install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Smoke-test the env + PPO loop
pytest -q

# 3. Train the woodcutting agent
osrs-train --config configs/ppo_woodcutting.yaml

# 4. Watch training live
tensorboard --logdir runs

# 5. Random baseline (for comparison)
osrs-eval --random --episodes 30 --output runs/baseline_random.json

# 6. Evaluate a checkpoint (stochastic — the honest policy metric)
osrs-eval --checkpoint runs/ppo_woodcutting_v2/checkpoints/latest.pt \
          --episodes 50 --output runs/ppo_woodcutting_v2/eval_trained.json

# 7. Evaluate every checkpoint to produce a clean progression curve
python scripts/evaluate_checkpoints.py \
    --run-dir runs/ppo_woodcutting_v2 \
    --config configs/ppo_woodcutting.yaml --episodes 20

# 8. Produce README-ready plots (reward curve, progression, action-distribution)
python scripts/plot_training.py \
    --run-dir runs/ppo_woodcutting_v2 \
    --baseline-json runs/baseline_random.json \
    --trained-json runs/ppo_woodcutting_v2/eval_trained.json \
    --progression-json runs/ppo_woodcutting_v2/checkpoint_progression.json
```

A laptop CPU run reaches a visibly-learning reward curve in roughly 5–10 minutes on
the default `ppo_woodcutting.yaml` config. A single modern CPU handles `num_envs=8`
comfortably.

## Repository layout

```
src/osrs_rl/
├── env/              # Gymnasium env, GameClient abstraction
│   ├── simulator/    #   2D grid simulator (training)
│   └── live/         #   live OSRS client (evaluation)
├── vision/           # frame preprocessing + screen capture
├── input_control/    # mouse/keyboard controller + safety gate
├── rewards/          # composable reward components
├── agents/           # PPO (from scratch) + networks + rollout buffer
├── training/         # CLI entry point, trainer, checkpointing
├── evaluation/       # evaluation harness and metrics
└── utils/            # config, logging, seeding
configs/              # ppo_woodcutting.yaml (training), live.yaml (live mode)
tests/                # env / PPO / rewards / wrappers / live smoke tests
scripts/              # analysis + baselines + checkpoint-progression eval
```

## Results — PPO vs random baseline

Trained for **300,000 environment steps** (~5 minutes on a single CPU, `num_envs=8`).
Both agents were evaluated over 50 fresh episodes with the same config.

| metric | random baseline | **trained PPO** | change |
| --- | --- | --- | --- |
| episode return | −2.90 ± 3.03 | **+18.04 ± 17.93** | +20.9 pts |
| trees chopped / ep | 2.24 ± 1.74 | **3.92 ± 2.97** | +75% |
| success rate (inventory filled) | 0.0% | **4.0%** (peaks 15% at checkpoints) | — |
| invalid action ratio | 0.31 | 0.45 | +0.14 |
| idle ratio | 0.14 | **0.003** | −0.14 |
| DROP-action share | 14.3% | **0.0%** | learned to never drop |
| IDLE-action share | 14.1% | **0.3%** | learned to stop idling |

*("Trained" uses stochastic sampling from the policy distribution. Deterministic argmax
is reported further down — the comparison is instructive.)*

### Checkpoint progression (the cleanest learning story)

![checkpoint progression](runs/ppo_woodcutting_v2/plots/checkpoint_progression.png)

Each point is an independent 20-episode stochastic evaluation of the saved checkpoint
at that step — this is the honest quality curve for the policy, with zero overlap
between training-time rollouts and eval data. Return climbs monotonically-in-trend
from **+9 at 25k steps to +25 at 300k** (≈3× improvement) and trees-chopped grows
from 3.2 to 4.9 per episode. The non-monotonicity in success-rate (it oscillates
between 0% and 15%) is PPO policy churn at this training budget — reward is still
improving despite the success-rate jitter, which is what the blue-shaded 1-σ band on
the return plot captures.

### Episode return over training

![reward over time](runs/ppo_woodcutting_v2/plots/reward_over_time.png)

The blue curve is the rolling-mean episode return during training; the red dashed
line is the random-baseline return. Learning kicks in within the first ~25k steps and
the policy stabilizes above +15 for the remainder of training. The orange dots are
the trainer's built-in **deterministic** evals — see "Weaknesses" below for why they
stay flat at −9 even as the stochastic policy improves dramatically.

### Success rate and trees chopped

![success rate](runs/ppo_woodcutting_v2/plots/success_rate.png)

![trees per episode](runs/ppo_woodcutting_v2/plots/trees_chopped.png)

### Optimization diagnostics

![losses](runs/ppo_woodcutting_v2/plots/losses.png)

Policy loss hovers near zero (expected for PPO — the clipped objective is small per
update), value loss is bounded, and entropy anneals from ~1.9 (near-uniform) to ~1.5,
consistent with a policy that has committed to a strategy but still explores.

### What the policy learned

![action distribution](runs/ppo_woodcutting_v2/plots/action_distribution.png)

Quantitatively:

| action | random | trained (stochastic) |
| --- | --- | --- |
| MOVE_NORTH | 14.7% | 16.2% |
| MOVE_SOUTH | 13.9% | **19.3%** |
| MOVE_EAST | 14.2% | 12.4% |
| MOVE_WEST | 14.6% | 11.3% |
| INTERACT | 14.1% | **40.4%** |
| DROP | 14.3% | **0.0%** |
| IDLE | 14.1% | **0.3%** |

The trained agent:

1. **Recognizes INTERACT as the dominant-value action** — its share nearly tripled vs
   random.
2. **Eliminated useless actions** — DROP is extinct (0%), IDLE is essentially extinct
   (0.3%). This is a strong "the policy understood the task semantics" signal.
3. **Developed a directional navigation bias** — MOVE_SOUTH roughly doubles MOVE_WEST,
   suggesting the CNN learned a spatial feature that preferentially pulls the agent
   along one axis (the direction is an artifact of the random seed used to place trees
   in the observation frames the policy trained on).

## Interpretation — why PPO works here, and where it falls short

**Why PPO.** The task has a moderately sparse but dense-enough reward signal (+5 per log
with potential-based distance shaping) and a small discrete action space. PPO's clipped
objective gives stable policy improvement without the replay-buffer dynamics that make
DQN sensitive to reward scale and exploration-schedule tuning. The CNN backbone learns
a useful spatial representation even at 84×84 grayscale because the semantic units
(agent, tree, stump, HUD bar) are color-separated and tile-aligned.

**Evidence of genuine learning, not memorization.** Episodes start with a freshly
randomized tree layout every reset — the policy cannot have memorized positions. It is
learning a navigation-plus-interact behavior conditioned on current visual state.

**Current weaknesses (the honest failure modes):**

1. **The deterministic argmax policy collapses to 100% INTERACT** (see
   `eval_trained_det.json`). The top logit is usually INTERACT; movement only emerges
   through stochastic sampling. A well-trained policy would put INTERACT on top only
   when a tree is adjacent — not globally. This is the single biggest architectural
   tell: the shared-backbone actor doesn't yet condition strongly enough on "is a tree
   adjacent" vs "no tree adjacent" to flip the argmax between INTERACT and a MOVE.
2. **Success-rate plateau around 10–15%.** Filling inventory requires chopping 10 trees
   in 300 steps; the agent averages 4. The bottleneck is *post-chop navigation*: after
   a tree respawns (10 ticks), the agent doesn't efficiently relocate to the next
   closest tree.
3. **Adding an adjacency-bonus reward helped, but less than expected.** I ran a second
   training (`ppo_woodcutting_v2`) with an extra `+0.5` reward for transitioning
   `not adjacent -> adjacent`. It raised mean return from +14.3 to +18.0 and cut idle
   ratio further, but success rate did not meaningfully move. This is consistent with
   a *representation* bottleneck (the policy can't reliably distinguish adjacency
   states from the pixel observation), not a *reward* bottleneck.
4. **Invalid-action rate went up, not down**, relative to random (0.31 → 0.45). The
   agent is choosing INTERACT eagerly even when not adjacent — cheap under the current
   reward (−0.02 per invalid) but a symptom of weakness (1).

### Next improvements (in priority order)

1. **Recurrent policy head (GRU/LSTM).** The highest-leverage change. Gives the policy
   explicit "which tree am I walking toward" memory across frames and should fix the
   post-chop navigation gap that caps success rate.
2. **Larger input resolution + color channels.** 84×84 grayscale loses tile-level
   adjacency cues after CNN downsampling; 128×128 RGB would preserve them.
3. **Curriculum: `grid_size=8 → 16`.** Shorter distances mean the agent samples the
   chop reward more often early, separating "interact when adjacent" from "interact
   always" faster.
4. **Longer training + wider backbone.** The Nature-CNN at 300k steps is small-budget;
   a 2M-step run with a wider policy should close the success-rate gap on its own.
5. **DQN baseline** behind the same `BasePolicy` interface — useful sanity check that
   PPO's improvement isn't an artifact of the specific optimizer trajectory.

## Design choices

- **PPO over DQN.** PPO is the industry default for vision-based RL, handles dense/sparse
  rewards well, and composes cleanly with CNN policies. The `BasePolicy` interface makes
  DQN / SAC drop-in later.
- **Simulator-first.** Training on a live MMO is wall-clock prohibitive. The simulator
  replicates the observation and action semantics so the same policy checkpoints
  transfer to the real client unchanged.
- **Custom PPO implementation.** Transparent, hackable, and demonstrable at interview.
  A thin optional `sb3` baseline can be wired up for A/B comparison.

## Live OSRS evaluation

The same Gymnasium env, policy, and checkpoint format drive the real OSRS window
through a screen-capture + safety-gated input pipeline. Everything flows through the
existing `GameClient` interface — only the injected client differs between simulator
and live.

```
┌────────────┐   frames    ┌────────────┐   action    ┌────────────────┐
│ OSRS client│ ──────────▶ │ LiveOSRSClient │ ──▶ SafetyGate ──▶│ MouseKeyboardCtrl │
│  (game)    │   (mss)     │  (GameClient) │              │ (pynput)          │
└────────────┘             └────────────┘               └────────────────┘
      ▲                           │                              │ (blocked in
      │                           ▼                              │  dry-run mode)
      │                    Gymnasium env ──▶ PPO policy          │
      └──────────────────  cursor move / click  ─────────────────┘
```

### Install the live extra

```bash
pip install -e ".[live]"       # adds mss + pynput
```

On macOS, running live input additionally requires Screen Recording and Accessibility
permissions for your terminal. Grant them in `System Settings → Privacy & Security`.

### Safety layer (`SafetyGate`)

Every OS-level side effect — cursor move, click, keypress — passes through a single
`SafetyGate.approve` call. The gate enforces, in order:

1. **`enable_live_input` must be `true`.** Default is `false`, which means every
   action is *audit-logged but blocked at dispatch*. This lets you rehearse the whole
   stack against the live game with zero input side effects.
2. **Kill-switch file.** While the path exists, every subsequent approve is denied.
   Default `/tmp/osrs_rl_stop` — from another terminal, run `touch /tmp/osrs_rl_stop`
   to halt, `rm` to resume.
3. **Rate limit** (`max_actions_per_second`).
4. **Safe bounding box** (`safe_region_xywh`). Clicks or cursor moves whose target
   falls outside the bbox are denied. This is the single most important setting —
   make sure the box covers **only the area of the OSRS client you're comfortable
   having the agent click inside.**
5. **Audit log** on every attempted action (approved or denied), visible in the
   console and in the run log.

### Dry-run evaluation (safe default)

No input is ever sent. Use this to verify capture region, policy behavior, and the
audit log before flipping the live flag.

```bash
# 1. Edit configs/live.yaml: set capture_region_xywh and safe_region_xywh to match
#    your OSRS client window.
# 2. Run the trained checkpoint in dry-run mode.
osrs-eval \
    --checkpoint runs/ppo_woodcutting_v2/checkpoints/latest.pt \
    --live-config configs/live.yaml \
    --episodes 1
```

The console will print each approved action with a `[DRY]` prefix and the cursor
coordinates it *would have* moved to.

### Real-input evaluation (explicit opt-in)

After you have verified the capture region and audit log:

```bash
# In configs/live.yaml, set:  enable_live_input: true
# Prepare the kill switch in a second terminal (do not run the command yet):
#     touch /tmp/osrs_rl_stop       # emergency halt
#     rm    /tmp/osrs_rl_stop       # resume
osrs-eval \
    --checkpoint runs/ppo_woodcutting_v2/checkpoints/latest.pt \
    --live-config configs/live.yaml \
    --episodes 1
```

### What to expect from a sim-trained policy on real OSRS

Candidly: **don't expect woodcutting**. The policy was trained on 128×128 synthetic
frames whose pixel statistics (solid-color tiles, HUD bar) share almost nothing with
the real OSRS client. You will see the cursor make valid-shaped motions (bounded
inside the safe region, paced by the rate limit) and click occasionally, but there
is no reason for the policy to choose "click on a tree" reliably.

M5 is the **infrastructure milestone**, not the sim-to-real-success milestone —
closing the visual-domain gap is a dedicated training problem (domain randomization,
OSRS-frame fine-tuning, CV-based action decoding) that belongs in a separate
milestone.

## Roadmap

- [x] Scaffolding, action/reward abstractions
- [x] 2D simulator with rendered image observations
- [x] Gymnasium env + preprocessing pipeline
- [x] PPO (from scratch) + vectorized training + TensorBoard + checkpoints
- [x] Smoke tests for env and PPO
- [x] Evaluation harness + charts (reward / ep-length / success rate / action distribution)
- [x] Random-agent baseline + trained-agent comparison
- [x] Live OSRS client (screen capture + input) with safety gates + dry-run mode
- [ ] Closing the sim-to-real visual gap (domain randomization / OSRS fine-tuning)
- [ ] DQN baseline behind the same `BasePolicy` interface

## License

MIT.
