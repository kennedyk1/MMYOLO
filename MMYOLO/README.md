# MMYOLO Package

This package extends a local Ultralytics clone with multichannel `.npy` training and custom YOLO26 attention architectures, while keeping [`../ultralytics/`](../ultralytics/) untouched.

## Public API

Main entry points:

- [`MMYOLO`](__init__.py)
- [`create_multichannel_yolo`](factory.py)
- [`train_multichannel_yolo`](factory.py)
- [`write_detection_data_yaml`](factory.py)

Runtime registration:

- [`register_attention_modules`](modeling.py)
- [`ChannelSlice`](custom_modules.py)
- [`MultiInputStem`](custom_modules.py)
- `ChannelAttention`, `SpatialAttention`, `CBAM`

## Custom YOLO26 YAMLs

- [`custom_models/yolo26_raw_channelattention.yaml`](custom_models/yolo26_raw_channelattention.yaml)
- [`custom_models/yolo26_raw_cbam.yaml`](custom_models/yolo26_raw_cbam.yaml)
- [`custom_models/yolo26_nstems_channelattention.yaml`](custom_models/yolo26_nstems_channelattention.yaml)
- [`custom_models/yolo26_nstems_cbam.yaml`](custom_models/yolo26_nstems_cbam.yaml)

## How to use

```python
from MMYOLO import MMYOLO

wrapper = MMYOLO(
    model="yolo26n.pt",
    architecture="yolo26_nstems_cbam.yaml",
    dataset_type="RGBTDI",
    channel_order="RGBTDI",
)
```

If `architecture=None`, the package still works as a multichannel wrapper around the base YOLO architecture.  
If `architecture` points to one of the YAMLs above, the custom attention graph is used instead.

## Notes

- the package keeps backward-compatible names such as `create_multichannel_yolo` and `train_multichannel_yolo`
- attention blocks are registered only for the current Python process
- no source file under `ultralytics/` is modified
