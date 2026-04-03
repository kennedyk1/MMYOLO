from __future__ import annotations

import gc
from pathlib import Path

import torch

from MMYOLO import MMYOLO, resolve_channel_selection, resolve_local_model_source


# =========================================================
# EXAMPLE: TRAIN MULTIPLE MODELS IN SEQUENCE
# Edit only this section.
# =========================================================
DATA = Path("MID-3K-NPY")
CLASS_NAMES = ["person"]

TRAIN_FROM_SCRATCH = True
MODEL_PRETRAINED = "yolo26n.pt"
MODEL_SCRATCH = "yolo26.yaml"
ARCHITECTURE = "yolo26_nstems_channelattention.yaml"

CHANNEL_ORDER = "RGBTDI"
DATASET_TYPES = [
    "RGB",
    "T",
    "D",
    "I",
    "RGBTDI",
    "RGBT",
]

EPOCHS = 100
PATIENCE = 20
IMGSZ = 640
BATCH = 8
WORKERS = 4
DEVICE = "0"

PROJECT = "runs/mmyolo"
RUN_NAME_PREFIX = "mid3k"
CONTINUE_ON_ERROR = False

TRAIN_KWARGS = {
    "pretrained": not TRAIN_FROM_SCRATCH,
    "plots": True,
    "save": True,
    "val": True,
    "patience": PATIENCE,
    "multi_scale": 0.0,
}


def resolve_model_source() -> str:
    if TRAIN_FROM_SCRATCH:
        return resolve_local_model_source(MODEL_SCRATCH)
    return str(MODEL_PRETRAINED)


def resolve_architecture_source() -> str:
    return resolve_local_model_source(ARCHITECTURE)


def resolve_data_source(wrapper: MMYOLO, data: Path) -> Path:
    data = data.resolve()
    if not data.exists():
        raise FileNotFoundError(f"Dataset path not found: {data}")

    if data.is_dir():
        return wrapper.create_data_yaml(
            dataset_root=data,
            class_names=CLASS_NAMES,
        )
    if data.suffix.lower() in {".yaml", ".yml"}:
        return data

    raise ValueError(f"`DATA` must point to an NPY dataset directory or a YAML file. Received: {data}")


def build_run_name(dataset_type: str) -> str:
    architecture_name = Path(ARCHITECTURE).stem.replace("yolo26_", "")
    return f"{RUN_NAME_PREFIX}_{architecture_name}_{dataset_type.lower()}"


def build_train_args(dataset_type: str) -> dict:
    train_args = {
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "device": DEVICE,
        "project": PROJECT,
        "name": build_run_name(dataset_type),
        **TRAIN_KWARGS,
    }
    if BATCH is not None:
        train_args["batch"] = BATCH
    if WORKERS is not None:
        train_args["workers"] = WORKERS
    return train_args


def cleanup_after_run() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_run_header(dataset_type: str, data_yaml: Path) -> None:
    selection = resolve_channel_selection(dataset_type, CHANNEL_ORDER)
    print("\n========== MMYOLO ==========")
    print(f"model_source   : {resolve_model_source()}")
    print(f"architecture   : {resolve_architecture_source()}")
    print(f"train_mode     : {'scratch' if TRAIN_FROM_SCRATCH else 'pretrained'}")
    print(f"dataset_type   : {dataset_type}")
    print(f"channel_order  : {CHANNEL_ORDER}")
    print(f"input_channels : {selection.num_channels}")
    print(f"data_yaml      : {data_yaml}")
    print(f"run_name       : {build_run_name(dataset_type)}")
    print(f"batch          : {BATCH if BATCH is not None else 'Ultralytics default'}")
    print(f"workers        : {WORKERS if WORKERS is not None else 'Ultralytics default'}")
    print(f"patience       : {PATIENCE}")
    print("============================")


def run_one_training(dataset_type: str, data_yaml: Path) -> None:
    wrapper = MMYOLO(
        model=resolve_model_source(),
        architecture=resolve_architecture_source(),
        dataset_type=dataset_type,
        channel_order=CHANNEL_ORDER,
    )
    print_run_header(dataset_type, data_yaml)
    wrapper.train(
        data=data_yaml,
        **build_train_args(dataset_type),
    )


def main() -> None:
    bootstrap = MMYOLO(
        model=resolve_model_source(),
        architecture=resolve_architecture_source(),
        dataset_type=DATASET_TYPES[0],
        channel_order=CHANNEL_ORDER,
    )
    data_yaml = resolve_data_source(bootstrap, DATA)

    results: list[tuple[str, str]] = []
    for dataset_type in DATASET_TYPES:
        try:
            run_one_training(dataset_type, data_yaml)
            results.append((dataset_type, "ok"))
        except Exception as exc:
            results.append((dataset_type, f"error: {exc}"))
            if not CONTINUE_ON_ERROR:
                raise
        finally:
            cleanup_after_run()

    print("\n========== SUMMARY ==========")
    for dataset_type, status in results:
        print(f"{dataset_type:<8} -> {status}")
    print("=============================")


if __name__ == "__main__":
    main()
