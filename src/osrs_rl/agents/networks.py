"""Policy networks.

The Nature-CNN is the canonical vision-RL backbone (Mnih et al. 2015) and remains the
strongest small-network baseline for 84×84 pixel inputs. Using it here makes results
directly comparable to the standard Atari PPO literature.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Orthogonal init — the default for PPO since Engstrom et al. 2020."""
    nn.init.orthogonal_(layer.weight, std)  # type: ignore[arg-type]
    nn.init.constant_(layer.bias, bias_const)  # type: ignore[arg-type]
    return layer


class NatureCNN(nn.Module):
    """Conv backbone returning a ``feature_dim``-d embedding for an ``(N, C, H, W)`` batch."""

    def __init__(self, in_channels: int, input_hw: tuple[int, int] = (84, 84), feature_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_hw)
            n_flat = self.conv(dummy).shape[1]
        self.fc = nn.Sequential(
            layer_init(nn.Linear(n_flat, feature_dim)),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))
