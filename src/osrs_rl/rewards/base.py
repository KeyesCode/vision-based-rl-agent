"""Composable reward interface.

A reward component computes a scalar from a transition ``(prev_state, action, next_state)``.
The :class:`CompositeReward` sums weighted components and returns a per-component breakdown
so training dashboards can visualize which signals are actually firing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from osrs_rl.env.action_space import Action
from osrs_rl.env.game_client import GameState


class RewardComponent(ABC):
    """Base class for a named reward signal."""

    name: str

    @abstractmethod
    def compute(self, prev: GameState, action: Action, nxt: GameState) -> float:
        """Return this component's raw (unweighted) contribution."""


@dataclass
class WeightedComponent:
    component: RewardComponent
    weight: float


class CompositeReward:
    """Sum of weighted reward components.

    Returns the total scalar plus a breakdown dict for logging.
    """

    def __init__(self, components: list[WeightedComponent]):
        self._components = components

    def compute(
        self, prev: GameState, action: Action, nxt: GameState
    ) -> tuple[float, dict[str, float]]:
        total = 0.0
        breakdown: dict[str, float] = {}
        for wc in self._components:
            raw = wc.component.compute(prev, action, nxt)
            contribution = raw * wc.weight
            total += contribution
            breakdown[wc.component.name] = contribution
        return total, breakdown

    @property
    def component_names(self) -> list[str]:
        return [wc.component.name for wc in self._components]
