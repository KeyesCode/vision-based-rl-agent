"""Mouse/keyboard controller — the only module that touches OS-level input APIs.

Every call goes through :class:`SafetyGate.approve` *before* dispatch. In dry-run mode
the approved action is logged but not sent to the OS, which means the entire pipeline
(capture → policy → decoded action → controller call) can be rehearsed against the
real OSRS window with zero side effects.

``pynput`` is an optional dependency and is only imported on live paths.
"""

from __future__ import annotations

import logging

from osrs_rl.input_control.safety import SafetyGate

try:
    from pynput import keyboard as _kb  # type: ignore
    from pynput import mouse as _mouse  # type: ignore
    _HAS_PYNPUT = True
    _pynput_error: Exception | None = None
except ImportError as _err:  # pragma: no cover - exercised only on bare installs
    _HAS_PYNPUT = False
    _pynput_error = _err
    _kb = None  # type: ignore
    _mouse = None  # type: ignore


class MouseKeyboardController:
    """Abstract input controller with built-in dry-run + safety gating."""

    def __init__(self, safety: SafetyGate, logger: logging.Logger | None = None):
        self._safety = safety
        self._logger = logger or logging.getLogger(__name__)
        if not safety.dry_run and not _HAS_PYNPUT:
            raise ImportError(
                "pynput is required when enable_live_input=true. "
                "Install with: pip install 'osrs-rl[live]'"
            ) from _pynput_error
        # Only instantiate the OS-level controllers when we actually intend to use them.
        self._mouse = _mouse.Controller() if _HAS_PYNPUT and not safety.dry_run else None
        self._kb = _kb.Controller() if _HAS_PYNPUT and not safety.dry_run else None

    @property
    def dry_run(self) -> bool:
        return self._safety.dry_run

    @property
    def safety(self) -> SafetyGate:
        return self._safety

    def move(self, x: int, y: int) -> bool:
        if not self._safety.approve("move", (x, y)):
            return False
        if self._mouse is not None:
            self._mouse.position = (x, y)
        return True

    def click(self, x: int, y: int) -> bool:
        if not self._safety.approve("click", (x, y)):
            return False
        if self._mouse is not None:
            self._mouse.position = (x, y)
            self._mouse.click(_mouse.Button.left, 1)
        return True

    def press_key(self, key: str) -> bool:
        if not self._safety.approve(f"key:{key}", None):
            return False
        if self._kb is not None:
            self._kb.press(key)
            self._kb.release(key)
        return True

    def wait(self) -> bool:
        """IDLE — no OS action, but still audit-logged."""
        mode = "DRY" if self.dry_run else "LIVE"
        self._logger.info(f"[{mode}] IDLE (no input)")
        return True
