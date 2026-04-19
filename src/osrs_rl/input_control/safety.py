"""Safety gate — every OS-level side effect passes through :meth:`approve`.

In dry-run mode (``enable_live_input=False``), the gate still performs every check and
emits the same audit log, but the controller must short-circuit before issuing the
actual OS call. This means a full rehearsal against the live game window is possible
with zero risk of sending input.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SafetyConfig:
    enable_live_input: bool = False
    safe_region_xywh: tuple[int, int, int, int] | None = None
    max_actions_per_second: float = 2.0
    kill_switch_file: str = "/tmp/osrs_rl_stop"


class SafetyGate:
    """Rate-limit, bbox-gate, and kill-switch check every live action attempt."""

    def __init__(self, cfg: SafetyConfig, logger: logging.Logger | None = None):
        self.cfg = cfg
        self._logger = logger or logging.getLogger(__name__)
        self._last_action_t = 0.0
        self._approved = 0
        self._denied_rate = 0
        self._denied_region = 0
        self._denied_killswitch = 0

    @property
    def dry_run(self) -> bool:
        return not self.cfg.enable_live_input

    def approve(self, desc: str, target_xy: tuple[int, int] | None = None) -> bool:
        if self._kill_switch_active():
            self._deny(desc, target_xy, reason="kill switch active")
            self._denied_killswitch += 1
            return False
        if self.cfg.max_actions_per_second > 0:
            now = time.monotonic()
            min_dt = 1.0 / self.cfg.max_actions_per_second
            if now - self._last_action_t < min_dt:
                self._deny(desc, target_xy, reason="rate limit")
                self._denied_rate += 1
                return False
        if target_xy is not None and self.cfg.safe_region_xywh is not None:
            if not self._in_safe_region(target_xy):
                self._deny(desc, target_xy, reason="outside safe region")
                self._denied_region += 1
                return False
        self._approve(desc, target_xy)
        self._last_action_t = time.monotonic()
        return True

    def _kill_switch_active(self) -> bool:
        return bool(self.cfg.kill_switch_file) and Path(self.cfg.kill_switch_file).exists()

    def _in_safe_region(self, xy: tuple[int, int]) -> bool:
        assert self.cfg.safe_region_xywh is not None
        x, y = xy
        l, t, w, h = self.cfg.safe_region_xywh
        return l <= x < l + w and t <= y < t + h

    def _approve(self, desc: str, target_xy: tuple[int, int] | None) -> None:
        self._approved += 1
        mode = "DRY" if self.dry_run else "LIVE"
        self._logger.info(f"[{mode}] APPROVE {desc} target={target_xy}")

    def _deny(self, desc: str, target_xy: tuple[int, int] | None, reason: str) -> None:
        self._logger.warning(f"[SAFETY] DENY {desc} target={target_xy} reason={reason}")

    def stats(self) -> dict[str, int]:
        return {
            "approved": self._approved,
            "denied_rate_limit": self._denied_rate,
            "denied_out_of_region": self._denied_region,
            "denied_kill_switch": self._denied_killswitch,
        }
