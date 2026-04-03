# MMYOLO

Portuguese version: [README_PT.md](README_PT.md)

This repository keeps the local [`ultralytics/`](ultralytics/) clone untouched and implements the multichannel logic in [`MMYOLO/`](MMYOLO/).

The current `MMYOLO` package supports:

- normalized multichannel `.npy` datasets
- channel subsets such as `RGB`, `T`, `D`, `I`, `RGBT`, `RGBTDI`
- custom YOLO26 attention architectures
- raw `N`-channel training
- `N`-stem training with one lightweight stem per selected input channel

## Package layout

Core package:

- [`MMYOLO/__init__.py`](MMYOLO/__init__.py)
- [`MMYOLO/factory.py`](MMYOLO/factory.py)
- [`MMYOLO/trainer.py`](MMYOLO/trainer.py)
- [`MMYOLO/dataset.py`](MMYOLO/dataset.py)
- [`MMYOLO/modeling.py`](MMYOLO/modeling.py)
- [`MMYOLO/custom_modules.py`](MMYOLO/custom_modules.py)

Custom model YAMLs:

- [`MMYOLO/custom_models/yolo26_raw_channelattention.yaml`](MMYOLO/custom_models/yolo26_raw_channelattention.yaml)
- [`MMYOLO/custom_models/yolo26_raw_cbam.yaml`](MMYOLO/custom_models/yolo26_raw_cbam.yaml)
- [`MMYOLO/custom_models/yolo26_nstems_channelattention.yaml`](MMYOLO/custom_models/yolo26_nstems_channelattention.yaml)
- [`MMYOLO/custom_models/yolo26_nstems_cbam.yaml`](MMYOLO/custom_models/yolo26_nstems_cbam.yaml)

Examples:

- single run: [`train_example.py`](train_example.py)
- batch runs: [`train_batch.py`](train_batch.py)
- CLI entry: [`MMYOLO/train.py`](MMYOLO/train.py)
- quick usage sample: [`MMYOLO/example_usage.py`](MMYOLO/example_usage.py)
- package docs: [`MMYOLO/README.md`](MMYOLO/README.md)

## Attention variants

`MMYOLO` now includes the attention-oriented variants directly:

- `raw N + ChannelAttention`
- `raw N + CBAM`
- `N stems + ChannelAttention`
- `N stems + CBAM`

These variants are registered at runtime. No file inside [`ultralytics/`](ultralytics/) is edited.

## Installation

Create or activate your Python environment, then install the requirements:

```bash
pip install -r requirements.txt
```

## Typical workflow

1. Prepare the normalized `.npy` dataset with [`download_dataset.py`](download_dataset.py).
2. Pick one of the YAMLs in [`MMYOLO/custom_models/`](MMYOLO/custom_models/).
3. Train one run with [`train_example.py`](train_example.py) or several with [`train_batch.py`](train_batch.py).

## Python usage

```python
from MMYOLO import MMYOLO

wrapper = MMYOLO(
    model="yolo26n.pt",
    architecture="yolo26_nstems_channelattention.yaml",
    dataset_type="RGBTDI",
    channel_order="RGBTDI",
)

data_yaml = wrapper.create_data_yaml(
    dataset_root="MID-3K-NPY",
    class_names=["person"],
)

wrapper.train(
    data=data_yaml,
    epochs=100,
    imgsz=640,
    batch=8,
    workers=4,
    device="0",
    pretrained=True,
    project="runs/mmyolo",
    name="rgbtdi_nstems_channelattention",
)
```

## CLI usage

```bash
python MMYOLO/train.py \
  --data MID-3K-NPY \
  --model yolo26n.pt \
  --architecture yolo26_raw_cbam.yaml \
  --dataset-type RGBTDI \
  --channel-order RGBTDI \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --workers 4 \
  --device 0 \
  --pretrained
```

## Notes on pretrained weights

- raw architectures can reuse pretrained weights partially
- `n-stems` variants usually preserve more compatible weights than the raw attention variants created by inserting layers early
- all loading happens externally through `MMYOLO`, not by editing Ultralytics internals
