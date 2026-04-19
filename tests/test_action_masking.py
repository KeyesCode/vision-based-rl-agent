"""Tests for inference-time action masking.

Protects the three correctness properties:

1. ``build_adjacency_mask(adjacent=False)`` zeroes the INTERACT slot and leaves
   every other slot at 1.
2. ``PPOActorCritic.act`` with a mask never returns a masked action under argmax,
   even when the raw actor head strongly prefers INTERACT.
3. ``RecurrentPPOActorCritic.act`` applies the same mask semantics at its argmax.
"""

from __future__ import annotations

import torch

from osrs_rl.agents.ppo import PPOActorCritic, RecurrentPPOActorCritic
from osrs_rl.env.action_space import ActionDecoder, ActionType, build_adjacency_mask


def test_build_adjacency_mask_zeroes_only_interact_when_not_adjacent():
    mask = build_adjacency_mask(adjacent_to_tree=False)
    assert len(mask) == ActionDecoder.n_actions()
    for i, v in enumerate(mask):
        expected = 0.0 if i == int(ActionType.INTERACT) else 1.0
        assert v == expected, f"slot {i}: expected {expected}, got {v}"


def test_build_adjacency_mask_is_all_ones_when_adjacent():
    mask = build_adjacency_mask(adjacent_to_tree=True)
    assert all(v == 1.0 for v in mask)


def _force_interact_preference(policy):
    """Drive the actor to output strongly positive logit on INTERACT, near-zero
    elsewhere — the exact pathology the mask is meant to override."""
    with torch.no_grad():
        policy.actor.bias.zero_()
        policy.actor.weight.zero_()
        policy.actor.bias[int(ActionType.INTERACT)] = 20.0


def test_feedforward_mask_forces_non_interact_argmax_when_not_adjacent():
    policy = PPOActorCritic(
        num_actions=ActionDecoder.n_actions(),
        in_channels=4,
        input_hw=(84, 84),
        feature_dim=64,
    )
    policy.eval()
    _force_interact_preference(policy)
    obs = torch.randint(0, 255, (1, 4, 84, 84), dtype=torch.uint8)

    # No mask -> argmax is INTERACT (confirms the preference we just installed).
    action_unmasked, _, _ = policy.act(obs, deterministic=True)
    assert int(action_unmasked.item()) == int(ActionType.INTERACT)

    # With not-adjacent mask -> argmax must NOT be INTERACT.
    mask = torch.as_tensor(
        build_adjacency_mask(adjacent_to_tree=False),
        dtype=torch.float32,
    ).unsqueeze(0)
    action_masked, _, _ = policy.act(obs, deterministic=True, mask=mask)
    assert int(action_masked.item()) != int(ActionType.INTERACT)


def test_feedforward_mask_is_no_op_when_all_allowed():
    policy = PPOActorCritic(
        num_actions=ActionDecoder.n_actions(),
        in_channels=4,
        input_hw=(84, 84),
        feature_dim=64,
    )
    policy.eval()
    _force_interact_preference(policy)
    obs = torch.randint(0, 255, (1, 4, 84, 84), dtype=torch.uint8)
    mask = torch.ones(1, ActionDecoder.n_actions())
    action, _, _ = policy.act(obs, deterministic=True, mask=mask)
    # All-ones mask must not alter argmax.
    assert int(action.item()) == int(ActionType.INTERACT)


def test_recurrent_mask_forces_non_interact_argmax_when_not_adjacent():
    policy = RecurrentPPOActorCritic(
        num_actions=ActionDecoder.n_actions(),
        in_channels=4,
        input_hw=(84, 84),
        feature_dim=64,
        hidden_size=16,
    )
    policy.eval()
    _force_interact_preference(policy)
    obs = torch.randint(0, 255, (1, 4, 84, 84), dtype=torch.uint8)
    hidden = policy.initial_hidden(1, torch.device("cpu"))
    starts = torch.zeros(1)

    action_unmasked, _, _, _ = policy.act(obs, hidden, starts, deterministic=True)
    assert int(action_unmasked.item()) == int(ActionType.INTERACT)

    mask = torch.as_tensor(
        build_adjacency_mask(adjacent_to_tree=False), dtype=torch.float32
    ).unsqueeze(0)
    action_masked, _, _, _ = policy.act(
        obs, hidden, starts, deterministic=True, mask=mask
    )
    assert int(action_masked.item()) != int(ActionType.INTERACT)
