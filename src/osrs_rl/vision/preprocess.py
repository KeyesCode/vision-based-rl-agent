"""Frame preprocessing wrappers.

Custom implementations (as opposed to ``gym.wrappers.*``) keep the preprocessing
pipeline transparent in code review and ensure identical behavior across simulator
and live frames. Output is a channels-first uint8 tensor so the PPO policy can cast
and normalize in a single op.
"""

from __future__ import annotations

from collections import deque

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class GrayscaleObservation(gym.ObservationWrapper):
    """Convert ``(H, W, 3)`` uint8 RGB to ``(H, W)`` uint8 luminance."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, _ = env.observation_space.shape  # type: ignore[assignment]
        self.observation_space = Box(low=0, high=255, shape=(h, w), dtype=np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)


class ResizeObservation(gym.ObservationWrapper):
    """Resize ``(H, W)`` or ``(H, W, C)`` observations to a square ``size × size``."""

    def __init__(self, env: gym.Env, size: int):
        super().__init__(env)
        self.size = size
        shape = env.observation_space.shape  # type: ignore[union-attr]
        if len(shape) == 2:
            new_shape: tuple[int, ...] = (size, size)
        else:
            new_shape = (size, size, shape[2])
        self.observation_space = Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # INTER_AREA for downsampling is the standard choice for Atari-style pipelines.
        return cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)


class FrameStack(gym.ObservationWrapper):
    """Stack the last ``k`` observations along a new leading channel axis.

    Works for both grayscale ``(H, W)`` and RGB ``(H, W, C)`` inputs. Output is always
    channels-first: grayscale -> ``(k, H, W)``; RGB -> ``(k*C, H, W)``.
    """

    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        self.k = k
        self._frames: deque[np.ndarray] = deque(maxlen=k)
        shape = env.observation_space.shape  # type: ignore[union-attr]
        if len(shape) == 2:
            h, w = shape
            out_shape: tuple[int, ...] = (k, h, w)
        else:
            h, w, c = shape
            out_shape = (k * c, h, w)
        self.observation_space = Box(low=0, high=255, shape=out_shape, dtype=np.uint8)

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self._frames.append(obs)
        return self._stack(), info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._stack(), reward, terminated, truncated, info

    def observation(self, obs: np.ndarray) -> np.ndarray:  # pragma: no cover - unused
        return obs

    def _stack(self) -> np.ndarray:
        arr = np.stack(list(self._frames), axis=0)  # (k, H, W) or (k, H, W, C)
        if arr.ndim == 4:
            # (k, H, W, C) -> (k*C, H, W)
            k, h, w, c = arr.shape
            arr = arr.transpose(0, 3, 1, 2).reshape(k * c, h, w)
        return arr
