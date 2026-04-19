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
```

A laptop CPU run reaches a visibly-learning reward curve in under 30 minutes on the
default config. A single modern CPU should handle `num_envs=8` comfortably.

## Repository layout

```
src/osrs_rl/
├── env/              # Gymnasium env, GameClient abstraction, 2D simulator
├── vision/           # frame preprocessing (resize, grayscale, framestack)
├── rewards/          # composable reward components
├── agents/           # PPO (from scratch) + networks + rollout buffer
├── training/         # CLI entry point, trainer, checkpointing
├── evaluation/       # evaluation harness and metrics
└── utils/            # config, logging, seeding
configs/              # YAML configs (ppo_woodcutting.yaml is the MVP)
tests/                # env + PPO smoke tests
scripts/              # analysis + baselines
```

## Design choices

- **PPO over DQN.** PPO is the industry default for vision-based RL, handles dense/sparse
  rewards well, and composes cleanly with CNN policies. The `BasePolicy` interface makes
  DQN / SAC drop-in later.
- **Simulator-first.** Training on a live MMO is wall-clock prohibitive. The simulator
  replicates the observation and action semantics so the same policy checkpoints
  transfer to the real client unchanged.
- **Custom PPO implementation.** Transparent, hackable, and demonstrable at interview.
  A thin optional `sb3` baseline can be wired up for A/B comparison.

## Roadmap

- [x] Scaffolding, action/reward abstractions
- [x] 2D simulator with rendered image observations
- [x] Gymnasium env + preprocessing pipeline
- [x] PPO (from scratch) + vectorized training + TensorBoard + checkpoints
- [x] Smoke tests for env and PPO
- [ ] Evaluation harness + charts (reward / episode length / action distribution)
- [ ] Live OSRS client (screen capture + input) with safety gates
- [ ] Sim-to-real evaluation of sim-trained policies

## License

MIT.
