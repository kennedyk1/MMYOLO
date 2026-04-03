from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Iterable

from ._bootstrap import ensure_local_ultralytics_repo
from .channels import DEFAULT_CHANNEL_ORDER, resolve_channel_selection
from .modeling import guess_scale_from_name, register_attention_modules

ensure_local_ultralytics_repo()

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML

from .dataset import infer_source_channels_from_dataset_root
from .trainer import AttentionDetectionTrainer


def resolve_local_model_source(model: str | Path) -> str:
    """Resolve custom MMYOLO YAMLs first, then fall back to the bundled Ultralytics clone."""
    model_path = Path(model)

    if model_path.exists():
        return str(model_path.resolve())

    workspace_relative = (Path(__file__).resolve().parents[1] / model_path).resolve()
    if workspace_relative.exists():
        return str(workspace_relative)

    custom_models_root = (Path(__file__).resolve().parent / "custom_models").resolve()
    custom_match = (custom_models_root / model_path).resolve()
    if custom_match.exists():
        return str(custom_match)

    if model_path.suffix.lower() not in {".yaml", ".yml"}:
        return str(model)

    cfg_root = Path(__file__).resolve().parents[1] / "ultralytics" / "ultralytics" / "cfg" / "models"
    direct_cfg_match = (cfg_root / model_path).resolve()
    if direct_cfg_match.exists():
        return str(direct_cfg_match)

    name_matches = sorted(cfg_root.rglob(model_path.name))
    if len(name_matches) == 1:
        return str(name_matches[0].resolve())
    if len(name_matches) > 1:
        available = ", ".join(str(match.relative_to(cfg_root)) for match in name_matches[:5])
        raise FileNotFoundError(
            f"Model YAML '{model}' is ambiguous inside {cfg_root}. "
            f"Matching files: {available}. Please use a more specific path."
        )

    custom_matches = sorted(custom_models_root.rglob(model_path.name))
    if len(custom_matches) == 1:
        return str(custom_matches[0].resolve())
    if len(custom_matches) > 1:
        available = ", ".join(str(match.relative_to(custom_models_root)) for match in custom_matches[:5])
        raise FileNotFoundError(
            f"Custom attention YAML '{model}' is ambiguous inside {custom_models_root}. "
            f"Matching files: {available}. Please use a more specific path."
        )

    raise FileNotFoundError(
        f"Model YAML '{model}' was not found. "
        f"Looked in the workspace, {custom_models_root}, and under {cfg_root}."
    )


def _infer_class_names(dataset_root: Path) -> list[str]:
    label_files = []
    for split in ("train", "val", "test"):
        split_dir = dataset_root / "labels" / split
        if split_dir.exists():
            label_files.extend(sorted(split_dir.glob("*.txt")))

    max_class = -1
    for label_file in label_files:
        if label_file.stat().st_size == 0:
            continue
        with open(label_file, encoding="utf-8") as file_obj:
            for row in file_obj:
                row = row.strip()
                if not row:
                    continue
                class_id = int(float(row.split()[0]))
                max_class = max(max_class, class_id)

    if max_class < 0:
        return ["class_0"]
    return [f"class_{index}" for index in range(max_class + 1)]


def write_detection_data_yaml(
    dataset_root: str | Path,
    class_names: Iterable[str] | None = None,
    output_path: str | Path | None = None,
    source_channels: int | None = None,
) -> Path:
    """Create a standard detection `data.yaml` for the NPY dataset layout produced by `download_dataset.py`."""
    dataset_root = Path(dataset_root).resolve()
    if output_path is None:
        output_path = dataset_root / "mid3k_npy_detect.yaml"
    output_path = Path(output_path).resolve()

    if class_names is None:
        names = _infer_class_names(dataset_root)
    else:
        names = [name.strip() for name in class_names if str(name).strip()]
        if not names:
            raise ValueError("class_names cannot be empty.")

    if source_channels is None:
        source_channels = infer_source_channels_from_dataset_root(dataset_root)

    data = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
        "channels": int(source_channels),
    }
    YAML.save(output_path, data)
    return output_path


def _load_custom_yaml_config(architecture: str | Path, scale_hint: str = "") -> tuple[dict, str]:
    architecture_path = resolve_local_model_source(architecture)
    architecture_cfg = YAML.load(architecture_path)
    architecture_cfg["yaml_file"] = architecture_path
    architecture_cfg["scale"] = scale_hint or guess_scale_from_name(architecture_path) or next(
        iter(architecture_cfg.get("scales", {"n": ()}).keys())
    )
    return architecture_cfg, architecture_path


def create_attention_yolo(
    model: str | Path = "yolo26n.pt",
    architecture: str | Path | None = None,
    dataset_type: str = DEFAULT_CHANNEL_ORDER,
    nc: int | None = None,
    channel_order: str = DEFAULT_CHANNEL_ORDER,
    verbose: bool = True,
) -> YOLO:
    """Create a YOLO detector with optional attention architecture overlays."""
    register_attention_modules()

    selection = resolve_channel_selection(dataset_type, channel_order)
    resolved_model = resolve_local_model_source(model)
    yolo = YOLO(resolved_model)
    if yolo.task != "detect":
        raise ValueError(f"This patch currently supports task='detect' only. Received task='{yolo.task}'.")

    if architecture is None:
        cfg = deepcopy(yolo.model.yaml)
        architecture_path = str(resolved_model)
    else:
        scale_hint = guess_scale_from_name(model) or guess_scale_from_name(architecture)
        cfg, architecture_path = _load_custom_yaml_config(architecture, scale_hint=scale_hint)

    if nc is None:
        nc = cfg.get("nc", yolo.model.yaml.get("nc"))

    patched_model = DetectionModel(cfg=cfg, ch=selection.num_channels, nc=nc, verbose=verbose)
    if Path(str(resolved_model)).suffix.lower() == ".pt":
        patched_model.load(yolo.model, verbose=verbose)

    if hasattr(yolo.model, "names"):
        patched_model.names = dict(yolo.model.names)
    patched_model.args = getattr(yolo.model, "args", {})
    yolo.model = patched_model
    yolo.overrides["model"] = str(architecture_path)
    yolo.overrides["task"] = "detect"
    yolo.overrides.pop("architecture", None)
    return yolo


def create_multichannel_yolo(
    model: str | Path = "yolo26n.pt",
    architecture: str | Path | None = None,
    dataset_type: str = DEFAULT_CHANNEL_ORDER,
    nc: int | None = None,
    channel_order: str = DEFAULT_CHANNEL_ORDER,
    verbose: bool = True,
) -> YOLO:
    """Backward-compatible alias that now supports optional attention architectures."""
    return create_attention_yolo(
        model=model,
        architecture=architecture,
        dataset_type=dataset_type,
        nc=nc,
        channel_order=channel_order,
        verbose=verbose,
    )


def train_attention_yolo(
    data: str | Path,
    model: str | Path = "yolo26n.pt",
    architecture: str | Path | None = None,
    dataset_type: str = DEFAULT_CHANNEL_ORDER,
    nc: int | None = None,
    channel_order: str = DEFAULT_CHANNEL_ORDER,
    source_channels: int | None = None,
    padding_value: float = 114.0 / 255.0,
    **train_kwargs,
):
    """Train a custom attention YOLO detector on the `.npy` dataset using the custom trainer."""
    yolo = create_attention_yolo(
        model=model,
        architecture=architecture,
        dataset_type=dataset_type,
        nc=nc,
        channel_order=channel_order,
        verbose=train_kwargs.pop("verbose", True),
    )
    yolo.overrides.pop("architecture", None)
    return yolo.train(
        data=str(data),
        trainer=AttentionDetectionTrainer,
        dataset_type=dataset_type,
        channel_order=channel_order,
        source_channels=source_channels,
        padding_value=padding_value,
        **train_kwargs,
    )


def train_multichannel_yolo(
    data: str | Path,
    model: str | Path = "yolo26n.pt",
    architecture: str | Path | None = None,
    dataset_type: str = DEFAULT_CHANNEL_ORDER,
    nc: int | None = None,
    channel_order: str = DEFAULT_CHANNEL_ORDER,
    source_channels: int | None = None,
    padding_value: float = 114.0 / 255.0,
    **train_kwargs,
):
    """Backward-compatible training entry point with optional custom architecture support."""
    return train_attention_yolo(
        data=data,
        model=model,
        architecture=architecture,
        dataset_type=dataset_type,
        nc=nc,
        channel_order=channel_order,
        source_channels=source_channels,
        padding_value=padding_value,
        **train_kwargs,
    )


__all__ = [
    "create_attention_yolo",
    "create_multichannel_yolo",
    "resolve_local_model_source",
    "train_attention_yolo",
    "train_multichannel_yolo",
    "write_detection_data_yaml",
]
