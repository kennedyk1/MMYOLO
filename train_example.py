from __future__ import annotations

from pathlib import Path

from MMYOLO import MMYOLO, resolve_channel_selection, resolve_local_model_source


# =========================================================
# EXAMPLE: TRAIN ONE MODEL
# Edit only this section.
# =========================================================
DATA = Path("MID-3K-NPY")
CLASS_NAMES = ["person"]

TRAIN_FROM_SCRATCH = True
MODEL_PRETRAINED = "yolo26n.pt"
MODEL_SCRATCH = "yolo26.yaml"
ARCHITECTURE = "yolo26_raw_channelattention.yaml"

DATASET_TYPE = "RGBTDI"
CHANNEL_ORDER = "RGBTDI"

EPOCHS = 100
PATIENCE = 20
IMGSZ = 640
BATCH = 8
WORKERS = 4
DEVICE = "0"

PROJECT = "runs/mmyolo"
NAME = "single_example"

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


def build_train_args() -> dict:
    train_args = {
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "device": DEVICE,
        "project": PROJECT,
        "name": NAME,
        **TRAIN_KWARGS,
    }
    if BATCH is not None:
        train_args["batch"] = BATCH
    if WORKERS is not None:
        train_args["workers"] = WORKERS
    return train_args


def main() -> None:
    wrapper = MMYOLO(
        model=resolve_model_source(),
        architecture=resolve_architecture_source(),
        dataset_type=DATASET_TYPE,
        channel_order=CHANNEL_ORDER,
    )
    data_yaml = resolve_data_source(wrapper, DATA)
    selection = resolve_channel_selection(DATASET_TYPE, CHANNEL_ORDER)

    print("========== MMYOLO ==========")
    print(f"model_source   : {resolve_model_source()}")
    print(f"architecture   : {resolve_architecture_source()}")
    print(f"train_mode     : {'scratch' if TRAIN_FROM_SCRATCH else 'pretrained'}")
    print(f"dataset_type   : {DATASET_TYPE}")
    print(f"channel_order  : {CHANNEL_ORDER}")
    print(f"input_channels : {selection.num_channels}")
    print(f"data_yaml      : {data_yaml}")
    print(f"batch          : {BATCH if BATCH is not None else 'Ultralytics default'}")
    print(f"workers        : {WORKERS if WORKERS is not None else 'Ultralytics default'}")
    print(f"patience       : {PATIENCE}")
    print("============================")

    wrapper.train(
        data=data_yaml,
        **build_train_args(),
    )


if __name__ == "__main__":
    main()
