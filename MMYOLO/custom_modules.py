from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from ._bootstrap import ensure_local_ultralytics_repo

ensure_local_ultralytics_repo()

from ultralytics.nn.modules.conv import CBAM, ChannelAttention, Conv, SpatialAttention


class ChannelSlice(nn.Module):
    """Select a subset of channels while preserving BCHW layout."""

    def __init__(self, indices: int | Sequence[int]) -> None:
        super().__init__()
        if isinstance(indices, int):
            indices = [indices]
        if not indices:
            raise ValueError("ChannelSlice requires at least one channel index.")
        self.indices = tuple(int(index) for index in indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ChannelSlice expects BCHW input, got shape={tuple(x.shape)}")
        return x[:, self.indices, :, :]


class MultiInputStem(nn.Module):
    """Create one lightweight stem per input channel and fuse the resulting features."""

    def __init__(self, c1: int, c2: int, stem_channels: int = 32, k: int = 3, s1: int = 2, s2: int = 2) -> None:
        super().__init__()
        if c1 < 1:
            raise ValueError(f"MultiInputStem requires at least one input channel, got c1={c1}")
        if stem_channels < 1:
            raise ValueError(f"stem_channels must be positive, got stem_channels={stem_channels}")

        self.num_inputs = int(c1)
        self.slices = nn.ModuleList(ChannelSlice(index) for index in range(self.num_inputs))
        self.stems = nn.ModuleList(
            nn.Sequential(
                Conv(1, stem_channels, k, s1),
                Conv(stem_channels, stem_channels, k, s2),
            )
            for _ in range(self.num_inputs)
        )
        self.fuse = Conv(self.num_inputs * stem_channels, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [stem(channel_slice(x)) for channel_slice, stem in zip(self.slices, self.stems)]
        return self.fuse(torch.cat(features, 1))


__all__ = [
    "CBAM",
    "ChannelAttention",
    "ChannelSlice",
    "MultiInputStem",
    "SpatialAttention",
]
