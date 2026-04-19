"""Tests for the live client safety gate and dry-run path.

Screen capture is stubbed with an in-memory fake so these tests run without mss,
pynput, or real screen access — every CI worker can exercise the live code paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from osrs_rl.env.action_space import Action, ActionType
from osrs_rl.env.live.live_client import LiveOSRSClient
from osrs_rl.input_control.controller import MouseKeyboardController
from osrs_rl.input_control.safety import SafetyConfig, SafetyGate
from osrs_rl.utils.config import EnvConfig, LiveConfig


@dataclass
class FakeCapture:
    shape: tuple[int, int, int]

    def grab(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.uint8)

    def close(self) -> None:
        pass


def _build_client(live_cfg: LiveConfig) -> LiveOSRSClient:
    h = live_cfg.capture_region_xywh[3]
    w = live_cfg.capture_region_xywh[2]
    capture = FakeCapture(shape=(h, w, 3))
    safety = SafetyGate(
        SafetyConfig(
            enable_live_input=live_cfg.enable_live_input,
            safe_region_xywh=live_cfg.safe_region_xywh,
            max_actions_per_second=live_cfg.max_actions_per_second,
            kill_switch_file=live_cfg.kill_switch_file,
        )
    )
    controller = MouseKeyboardController(safety)
    return LiveOSRSClient(EnvConfig(), live_cfg, capture=capture, controller=controller)


def _live_cfg(**overrides) -> LiveConfig:
    base = LiveConfig(
        capture_region_xywh=(0, 0, 200, 200),
        safe_region_xywh=(50, 50, 100, 100),
        initial_cursor_xy=(100, 100),
        cursor_step_pixels=10,
        action_delay_seconds=0.0,
        enable_live_input=False,
        max_actions_per_second=0.0,  # disable rate limit for these tests
        kill_switch_file="/tmp/osrs_rl_stop_TESTS_DO_NOT_CREATE",
        hotkey_drop="q",
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_dry_run_never_instantiates_os_controllers():
    client = _build_client(_live_cfg())
    assert client._safety.dry_run is True
    # In dry-run the internal mouse/keyboard handles must stay None.
    assert client._controller._mouse is None
    assert client._controller._kb is None


def test_click_inside_safe_region_is_approved():
    client = _build_client(_live_cfg())
    state = client.reset()
    state = client.step(Action(ActionType.INTERACT))
    assert state.last_action_valid is True
    assert client._safety.stats()["approved"] >= 2  # initial move + click


def test_move_outside_safe_region_is_denied():
    client = _build_client(_live_cfg(cursor_step_pixels=200))  # one step jumps way out
    client.reset()
    state = client.step(Action(ActionType.MOVE_EAST))
    assert state.last_action_valid is False
    assert client._safety.stats()["denied_out_of_region"] >= 1


def test_kill_switch_denies_all_actions(tmp_path):
    ks = tmp_path / "stop"
    ks.touch()
    client = _build_client(_live_cfg(kill_switch_file=str(ks)))
    client.reset()  # reset calls move() which will be denied; state still returned
    state = client.step(Action(ActionType.INTERACT))
    assert state.last_action_valid is False
    assert client._safety.stats()["denied_kill_switch"] >= 1


def test_drop_is_noop_when_hotkey_empty():
    client = _build_client(_live_cfg(hotkey_drop=""))
    client.reset()
    state = client.step(Action(ActionType.DROP))
    assert state.last_action_valid is False  # no hotkey configured -> invalid


def test_live_mode_requires_pynput_when_not_dry_run():
    """If pynput is unavailable, flipping enable_live_input must fail fast."""
    # Build a controller-only test (don't build the full client since it needs capture).
    import osrs_rl.input_control.controller as ctrl

    if ctrl._HAS_PYNPUT:
        pytest.skip("pynput is installed — negative test cannot run")

    safety = SafetyGate(SafetyConfig(enable_live_input=True))
    with pytest.raises(ImportError):
        MouseKeyboardController(safety)
