"""Concrete reward components and a factory for the woodcutting task."""

from __future__ import annotations

from osrs_rl.env.action_space import Action
from osrs_rl.env.game_client import GameState
from osrs_rl.rewards.base import CompositeReward, RewardComponent, WeightedComponent
from osrs_rl.utils.config import RewardConfig


class LogCollectionReward(RewardComponent):
    """+1 per log gained this step (dense primary signal)."""

    name = "log_collected"

    def compute(self, prev: GameState, action: Action, nxt: GameState) -> float:
        return float(nxt.inventory_count - prev.inventory_count)


class StepPenalty(RewardComponent):
    """Constant per-step penalty to discourage dithering."""

    name = "step_penalty"

    def compute(self, prev: GameState, action: Action, nxt: GameState) -> float:
        return 1.0


class InvalidActionPenalty(RewardComponent):
    """Penalty for actions the env could not execute (wall, nothing to chop, drop empty)."""

    name = "invalid_action"

    def compute(self, prev: GameState, action: Action, nxt: GameState) -> float:
        return 0.0 if nxt.last_action_valid else 1.0


class DistanceToTreeShaping(RewardComponent):
    """Potential-based shaping: reward is the decrease in distance to the nearest tree.

    Because this is a difference of potentials it does not change the optimal policy,
    but it accelerates learning by densifying the navigation signal.
    """

    name = "distance_shaping"

    def compute(self, prev: GameState, action: Action, nxt: GameState) -> float:
        if prev.nearest_tree_distance is None or nxt.nearest_tree_distance is None:
            return 0.0
        return float(prev.nearest_tree_distance - nxt.nearest_tree_distance)


class FullInventoryBonus(RewardComponent):
    """One-shot bonus the step inventory fills up (clear success signal)."""

    name = "full_inventory_bonus"

    def compute(self, prev: GameState, action: Action, nxt: GameState) -> float:
        return 1.0 if (nxt.inventory_full and not prev.inventory_full) else 0.0


def build_reward(cfg: RewardConfig) -> CompositeReward:
    """Factory that assembles the default woodcutting reward."""
    return CompositeReward(
        [
            WeightedComponent(LogCollectionReward(), cfg.log_collected),
            WeightedComponent(StepPenalty(), cfg.step_penalty),
            WeightedComponent(InvalidActionPenalty(), cfg.invalid_action_penalty),
            WeightedComponent(DistanceToTreeShaping(), cfg.distance_shaping),
            WeightedComponent(FullInventoryBonus(), cfg.full_inventory_bonus),
        ]
    )
