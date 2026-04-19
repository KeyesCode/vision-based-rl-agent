"""Unit tests for composable reward components."""

from __future__ import annotations

from osrs_rl.env.action_space import Action, ActionType
from osrs_rl.env.game_client import GameState
from osrs_rl.rewards.components import (
    AdjacencyBonus,
    DistanceToTreeShaping,
    IdleActionPenalty,
    InvalidActionPenalty,
    LogCollectionReward,
)


def _state(inv=0, inv_full=False, valid=True, dist=5.0, step=0) -> GameState:
    return GameState(
        agent_xy=(0, 0),
        inventory_count=inv,
        inventory_full=inv_full,
        is_chopping=False,
        last_action_valid=valid,
        nearest_tree_distance=dist,
        step_index=step,
    )


def test_log_collection_rewards_delta_only():
    prev = _state(inv=3)
    nxt = _state(inv=4)
    assert LogCollectionReward().compute(prev, Action(ActionType.INTERACT), nxt) == 1.0
    assert LogCollectionReward().compute(prev, Action(ActionType.INTERACT), prev) == 0.0


def test_invalid_action_penalty_fires_on_invalid():
    prev = _state()
    nxt_valid = _state(valid=True)
    nxt_invalid = _state(valid=False)
    comp = InvalidActionPenalty()
    assert comp.compute(prev, Action(ActionType.INTERACT), nxt_valid) == 0.0
    assert comp.compute(prev, Action(ActionType.INTERACT), nxt_invalid) == 1.0


def test_idle_penalty_targets_idle_action_only():
    prev = _state()
    nxt = _state()
    comp = IdleActionPenalty()
    assert comp.compute(prev, Action(ActionType.IDLE), nxt) == 1.0
    assert comp.compute(prev, Action(ActionType.INTERACT), nxt) == 0.0


def test_distance_shaping_is_signed_delta():
    prev = _state(dist=5.0)
    closer = _state(dist=4.0)
    farther = _state(dist=6.0)
    comp = DistanceToTreeShaping()
    assert comp.compute(prev, Action(ActionType.MOVE_NORTH), closer) == 1.0
    assert comp.compute(prev, Action(ActionType.MOVE_NORTH), farther) == -1.0


def test_adjacency_bonus_fires_only_on_transition():
    not_adj = _state(dist=3.0)
    adj = _state(dist=1.0)
    comp = AdjacencyBonus()
    # Arrived this step -> fires.
    assert comp.compute(not_adj, Action(ActionType.MOVE_NORTH), adj) == 1.0
    # Already adjacent -> does not double-count.
    assert comp.compute(adj, Action(ActionType.IDLE), adj) == 0.0
    # Moved away -> no bonus.
    assert comp.compute(adj, Action(ActionType.MOVE_NORTH), not_adj) == 0.0
