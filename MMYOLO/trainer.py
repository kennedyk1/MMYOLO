from __future__ import annotations

import math
import random
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ._bootstrap import ensure_local_ultralytics_repo
from .channels import DEFAULT_CHANNEL_ORDER, resolve_channel_selection
from .dataset import MultichannelNPYDataset
from .modeling import register_attention_modules

ensure_local_ultralytics_repo()
register_attention_modules()

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOCAL_RANK
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import strip_optimizer
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.utils.torch_utils import torch_distributed_zero_first


class AttentionDetectionValidator(DetectionValidator):
    """Validator for normalized multichannel `.npy` detection datasets."""

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks: dict | None = None,
        dataset_type: str = DEFAULT_CHANNEL_ORDER,
        channel_order: str = DEFAULT_CHANNEL_ORDER,
        source_channels: int | None = None,
        padding_value: float = 114.0 / 255.0,
    ) -> None:
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self.dataset_type = dataset_type
        self.channel_order = channel_order
        self.source_channels = source_channels or len(self.channel_order)
        self.padding_value = padding_value

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
        selection = resolve_channel_selection(self.dataset_type, self.channel_order)
        if self.data is not None:
            self.data["channels"] = selection.num_channels
            self.data["channel_names"] = list(selection.names)
        return MultichannelNPYDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=True,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=self.stride,
            pad=0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=1.0,
            dataset_type=self.dataset_type,
            channel_order=self.channel_order,
            source_channels=self.source_channels,
            padding_value=self.padding_value,
        )

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        return batch


class AttentionDetectionTrainer(DetectionTrainer):
    """Custom Ultralytics trainer for normalized multichannel `.npy` datasets."""

    METRIC_KEYS = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
    }

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        overrides = dict(overrides or {})
        register_attention_modules()
        self.dataset_type = overrides.pop("dataset_type", DEFAULT_CHANNEL_ORDER)
        self.channel_order = overrides.pop("channel_order", DEFAULT_CHANNEL_ORDER)
        source_channels = overrides.pop("source_channels", None)
        self.source_channels = int(source_channels) if source_channels is not None else None
        self.padding_value = float(overrides.pop("padding_value", 114.0 / 255.0))
        self.channel_selection = resolve_channel_selection(self.dataset_type, self.channel_order)
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.data["channels"] = self.channel_selection.num_channels
        self.data["channel_names"] = list(self.channel_selection.names)
        self.args.dataset_type = self.dataset_type
        self.args.channel_order = self.channel_order
        self.args.source_channels = self.source_channels
        self.args.padding_value = self.padding_value
        LOGGER.info(
            "MMYOLO attention patch active: "
            f"dataset_type={self.dataset_type}, "
            f"input_channels={self.data['channels']}, "
            f"indices={list(self.channel_selection.indices)}"
        )

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return MultichannelNPYDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            dataset_type=self.dataset_type,
            channel_order=self.channel_order,
            source_channels=self.source_channels,
            padding_value=self.padding_value,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float()
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            size = (
                random.randrange(
                    int(self.args.imgsz * (1.0 - self.args.multi_scale)),
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),
                )
                // self.stride
                * self.stride
            )
            scale_factor = size / max(imgs.shape[2:])
            if scale_factor != 1:
                new_shape = [
                    math.ceil(shape * scale_factor / self.stride) * self.stride for shape in imgs.shape[2:]
                ]
                imgs = nn.functional.interpolate(imgs, size=new_shape, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        validator_args = vars(copy(self.args)).copy()
        for key in ("dataset_type", "channel_order", "source_channels", "padding_value"):
            validator_args.pop(key, None)
        return AttentionDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=validator_args,
            _callbacks=self.callbacks,
            dataset_type=self.dataset_type,
            channel_order=self.channel_order,
            source_channels=self.source_channels,
            padding_value=self.padding_value,
        )

    def _run_final_split_eval(self, model: Path, split: str, plots: bool = False) -> dict[str, float] | None:
        """Run final evaluation on a specific split and return the core detection metrics."""
        if not self.data.get(split):
            return None

        self.validator.args.split = split
        self.validator.args.plots = plots
        self.validator.args.compile = False
        self.validator.dataloader = None
        self.validator.data = None

        metrics = self.validator(model=model)
        return {key: float(metrics[value]) for key, value in self.METRIC_KEYS.items()}

    def _write_model_info(self, output_path: Path, model: Path, val_metrics: dict[str, float], test_metrics: dict[str, float] | None) -> None:
        """Write a compact training summary with final validation and test metrics."""
        lines = [
            f"model_path: {model}",
            f"dataset_type: {self.dataset_type}",
            f"channel_order: {self.channel_order}",
            f"input_channels: {self.channel_selection.num_channels}",
            "",
            "[val]",
            f"P: {val_metrics['precision']:.6f}",
            f"R: {val_metrics['recall']:.6f}",
            f"mAP50: {val_metrics['map50']:.6f}",
            f"mAP50-95: {val_metrics['map50_95']:.6f}",
            "",
            "[test]",
        ]
        if test_metrics is None:
            lines.append("status: skipped (no test split defined in data.yaml)")
        else:
            lines.extend(
                [
                    f"P: {test_metrics['precision']:.6f}",
                    f"R: {test_metrics['recall']:.6f}",
                    f"mAP50: {test_metrics['map50']:.6f}",
                    f"mAP50-95: {test_metrics['map50_95']:.6f}",
                ]
            )

        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        LOGGER.info(f"Saved model info to {output_path}")

    def final_eval(self):
        """Run final validation with the custom validator using the selected channel count."""
        model = self.best if self.best.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if model:
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        if model:
            LOGGER.info(f"\nValidating {model}...")
            val_metrics = self._run_final_split_eval(model=model, split="val", plots=self.args.plots)
            if val_metrics is None:
                raise RuntimeError("Final evaluation requires a validation split in data.yaml.")
            self.metrics = dict(val_metrics)
            test_metrics = self._run_final_split_eval(model=model, split="test", plots=False)
            self._write_model_info(self.save_dir / "model_info.txt", model=model, val_metrics=val_metrics, test_metrics=test_metrics)
            self.run_callbacks("on_fit_epoch_end")


MultichannelDetectionTrainer = AttentionDetectionTrainer
MultichannelDetectionValidator = AttentionDetectionValidator
