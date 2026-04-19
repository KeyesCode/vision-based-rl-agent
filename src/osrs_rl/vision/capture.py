"""Screen capture via ``mss``, returning RGB uint8 frames.

Separated from :mod:`osrs_rl.vision.preprocess` so that the preprocessing pipeline
(grayscale / resize / frame-stack) is shared verbatim between simulator frames and
live frames. ``mss`` is an optional dependency — import lazily so the core package
can be installed without it.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import mss  # type: ignore
    _import_error: Exception | None = None
except ImportError as _err:  # pragma: no cover - covered by the guard in ScreenCapture
    mss = None  # type: ignore
    _import_error = _err


@dataclass(frozen=True)
class CaptureRegion:
    """Absolute screen rectangle for screen capture."""

    left: int
    top: int
    width: int
    height: int

    def to_mss_dict(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


class ScreenCapture:
    """Thin wrapper over mss that returns an ``(H, W, 3)`` RGB uint8 frame."""

    def __init__(self, region: CaptureRegion):
        if mss is None:
            raise ImportError(
                "mss is required for live screen capture. "
                "Install with: pip install 'osrs-rl[live]'"
            ) from _import_error
        self._region = region
        self._mss = mss.mss()

    def grab(self) -> np.ndarray:
        img = self._mss.grab(self._region.to_mss_dict())
        # mss.ScreenShot is exposed via np.asarray as (H, W, 4) BGRA.
        arr = np.asarray(img, dtype=np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._mss.close()
