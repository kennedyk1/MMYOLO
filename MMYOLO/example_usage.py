from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MMYOLO import MMYOLO


def main() -> None:
    mmyolo = MMYOLO(
        model="yolo26n.pt",
        architecture="yolo26_nstems_cbam.yaml",
        dataset_type="RGBTDI",
        channel_order="RGBTDI",
    )

    model = mmyolo.build_model(nc=1)
    print("Model created with input channels:", model.model.yaml["channels"])

    dataset_root = ROOT / "MID-3K-NPY"
    npy_exists = dataset_root.exists() and any(dataset_root.rglob("*.npy"))
    if npy_exists:
        data_yaml = mmyolo.create_data_yaml(
            dataset_root=dataset_root,
            class_names=["person"],
        )
        print("Data YAML:", data_yaml)
    else:
        print("NPY dataset not found yet in:", dataset_root)
        print("When it exists, generate the YAML with:")
        print("mmyolo.create_data_yaml(dataset_root='MID-3K-NPY', class_names=['person'])")

    # Example training:
    #
    # mmyolo.train(
    #     data=data_yaml,
    #     epochs=50,
    #     imgsz=640,
    #     batch=8,
    #     device="0",
    #     workers=4,
    #     project="runs/mmyolo",
    #     name="rgbtdi_nstems_cbam",
    #     pretrained=True,
    # )


if __name__ == "__main__":
    main()
