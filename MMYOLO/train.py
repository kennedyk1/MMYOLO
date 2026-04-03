from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MMYOLO import DEFAULT_CHANNEL_ORDER, train_multichannel_yolo, write_detection_data_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO on normalized multichannel .npy tensors.")
    parser.add_argument("--data", required=True, help="Path to a ready data.yaml or to the NPY dataset root.")
    parser.add_argument("--model", default="yolo26n.pt", help="Base YOLO checkpoint or YAML.")
    parser.add_argument(
        "--architecture",
        default=None,
        help="Optional custom architecture YAML from MMYOLO/custom_models, e.g. yolo26_nstems_cbam.yaml.",
    )
    parser.add_argument("--dataset-type", default=DEFAULT_CHANNEL_ORDER, help="Examples: RGBTDI, RGBT, RGBTD, RGB, TDI.")
    parser.add_argument("--channel-order", default=DEFAULT_CHANNEL_ORDER, help="Order of channels stored in the .npy.")
    parser.add_argument("--source-channels", type=int, default=None, help="Total stored channels. Auto-inferred if omitted.")
    parser.add_argument("--classes", nargs="*", default=None, help="Optional class names for generated data.yaml.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights from --model if available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    if data_path.is_dir():
        data_yaml = write_detection_data_yaml(
            dataset_root=data_path,
            class_names=args.classes,
            source_channels=args.source_channels,
        )
    else:
        data_yaml = data_path

    train_multichannel_yolo(
        data=data_yaml,
        model=args.model,
        architecture=args.architecture,
        dataset_type=args.dataset_type,
        channel_order=args.channel_order,
        source_channels=args.source_channels,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
