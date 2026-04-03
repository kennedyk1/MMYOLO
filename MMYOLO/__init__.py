from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .channels import ChannelSelection, DEFAULT_CHANNEL_ORDER, resolve_channel_selection
from .custom_modules import CBAM, ChannelAttention, ChannelSlice, MultiInputStem, SpatialAttention
from .dataset import MultichannelNPYDataset
from .factory import (
    create_attention_yolo,
    create_multichannel_yolo,
    resolve_local_model_source,
    train_attention_yolo,
    train_multichannel_yolo,
    write_detection_data_yaml,
)
from .modeling import register_attention_modules
from .trainer import (
    AttentionDetectionTrainer,
    AttentionDetectionValidator,
    MultichannelDetectionTrainer,
    MultichannelDetectionValidator,
)


class MMYOLO:
    """SDK-like wrapper for the multichannel patch with optional attention architectures."""

    def __init__(
        self,
        model: str = "yolo26n.pt",
        architecture: str | None = None,
        dataset_type: str = DEFAULT_CHANNEL_ORDER,
        channel_order: str = DEFAULT_CHANNEL_ORDER,
        source_channels: int | None = None,
    ) -> None:
        self.model = model
        self.architecture = architecture
        self.dataset_type = dataset_type
        self.channel_order = channel_order
        self.source_channels = source_channels

    def build_model(self, nc: int | None = None, verbose: bool = True):
        return create_multichannel_yolo(
            model=self.model,
            architecture=self.architecture,
            dataset_type=self.dataset_type,
            nc=nc,
            channel_order=self.channel_order,
            verbose=verbose,
        )

    def create_data_yaml(
        self,
        dataset_root: str | Path,
        class_names: Iterable[str] | None = None,
        output_path: str | Path | None = None,
    ) -> Path:
        return write_detection_data_yaml(
            dataset_root=dataset_root,
            class_names=class_names,
            output_path=output_path,
            source_channels=self.source_channels,
        )

    def train(self, data: str | Path, **kwargs):
        return train_multichannel_yolo(
            data=data,
            model=self.model,
            architecture=self.architecture,
            dataset_type=self.dataset_type,
            channel_order=self.channel_order,
            source_channels=self.source_channels,
            **kwargs,
        )


MMYOLOAttention = MMYOLO


__all__ = [
    "CBAM",
    "ChannelAttention",
    "ChannelSelection",
    "ChannelSlice",
    "DEFAULT_CHANNEL_ORDER",
    "MMYOLO",
    "MMYOLOAttention",
    "MultiInputStem",
    "AttentionDetectionTrainer",
    "AttentionDetectionValidator",
    "MultichannelDetectionTrainer",
    "MultichannelDetectionValidator",
    "MultichannelNPYDataset",
    "SpatialAttention",
    "create_attention_yolo",
    "create_multichannel_yolo",
    "register_attention_modules",
    "resolve_channel_selection",
    "resolve_local_model_source",
    "train_attention_yolo",
    "train_multichannel_yolo",
    "write_detection_data_yaml",
]
