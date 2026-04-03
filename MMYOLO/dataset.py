from __future__ import annotations

import glob
import os
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ._bootstrap import ensure_local_ultralytics_repo
from .channels import DEFAULT_CHANNEL_ORDER, resolve_channel_selection

ensure_local_ultralytics_repo()

from ultralytics.data.augment import Compose, Format, LetterBox, RandomFlip
from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, check_file_speeds, get_hash, load_dataset_cache_file
from ultralytics.data.utils import save_dataset_cache_file, segments2boxes
from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM


def infer_source_channels_from_array_shape(shape: tuple[int, ...], expected_channels: int | None = None) -> int:
    """Infer the stored channel count from a 3D NPY tensor shape."""
    if len(shape) != 3:
        raise ValueError(f"Esperado array 3D, mas recebi shape={shape}")

    if expected_channels is not None:
        matched_dims = [dim for dim in (shape[0], shape[-1]) if dim == expected_channels]
        if len(matched_dims) == 1:
            return matched_dims[0]
        if len(matched_dims) == 2:
            return expected_channels

    small_dims = [dim for dim in (shape[0], shape[-1]) if dim <= 16]
    if len(small_dims) == 1:
        return small_dims[0]
    if len(small_dims) == 2 and small_dims[0] == small_dims[1]:
        return small_dims[0]

    raise ValueError(
        f"Não foi possível inferir automaticamente os canais do array com shape={shape}. "
        "Informe um channel_order compatível com o dataset."
    )


def infer_source_channels_from_npy(npy_path: str | Path, expected_channels: int | None = None) -> int:
    """Infer the stored channel count from a single `.npy` file."""
    npy_path = Path(npy_path)
    array = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    return infer_source_channels_from_array_shape(array.shape, expected_channels=expected_channels)


def infer_source_channels_from_dataset_root(dataset_root: str | Path, expected_channels: int | None = None) -> int:
    """Infer the stored channel count from the first available `.npy` file in a dataset root."""
    dataset_root = Path(dataset_root).resolve()
    search_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "images" / "test",
        dataset_root,
    ]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        first_match = next(search_dir.rglob("*.npy"), None)
        if first_match is not None:
            return infer_source_channels_from_npy(first_match, expected_channels=expected_channels)

    raise FileNotFoundError(f"Nenhum arquivo .npy encontrado para inferir source_channels em {dataset_root}")


def _parse_yolo_label_file(
    label_file: str | Path,
    keypoint: bool,
    num_cls: int,
    nkpt: int,
    ndim: int,
    single_cls: bool,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray | None, int, int, int, str]:
    """Parse a YOLO txt label file with the same output contract used by Ultralytics."""
    label_file = str(label_file)
    nm, nf, ne = 0, 0, 0
    msg = ""
    segments: list[np.ndarray] = []
    keypoints = None

    if os.path.isfile(label_file):
        nf = 1
        with open(label_file, encoding="utf-8") as file_obj:
            rows = [row.split() for row in file_obj.read().strip().splitlines() if row]
            if any(len(row) > 6 for row in rows) and not keypoint:
                classes = np.array([row[0] for row in rows], dtype=np.float32)
                segments = [np.array(row[1:], dtype=np.float32).reshape(-1, 2) for row in rows]
                labels = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), axis=1)
            else:
                labels = np.array(rows, dtype=np.float32)

        if nl := len(labels):
            if keypoint:
                expected_cols = 5 + nkpt * ndim
                if labels.shape[1] != expected_cols:
                    raise AssertionError(f"labels require {expected_cols} columns each")
                points = labels[:, 5:].reshape(-1, ndim)[:, :2]
            else:
                if labels.shape[1] != 5:
                    raise AssertionError(f"labels require 5 columns, {labels.shape[1]} columns detected")
                points = labels[:, 1:]

            if points.max() > 1.01:
                raise AssertionError(f"non-normalized or out of bounds coordinates {points[points > 1.01]}")
            if labels.min() < -0.01:
                raise AssertionError(f"negative class labels or coordinate {labels[labels < -0.01]}")

            max_cls = 0 if single_cls else labels[:, 0].max()
            if max_cls >= num_cls:
                raise AssertionError(
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )

            _, unique_indices = np.unique(labels, axis=0, return_index=True)
            if len(unique_indices) < nl:
                labels = labels[unique_indices]
                if segments:
                    segments = [segments[index] for index in unique_indices]
                msg = f"{label_file}: {nl - len(unique_indices)} duplicate labels removed"
        else:
            ne = 1
            labels = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
    else:
        nm = 1
        labels = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)

    if keypoint:
        keypoints = labels[:, 5:].reshape(-1, nkpt, ndim)
        if ndim == 2:
            kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
            keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)

    return labels[:, :5], segments, keypoints, nm, nf, ne, msg


def _verify_npy_label(args: tuple) -> list[Any]:
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, single_cls, source_channels = args
    nm, nf, ne, nc = 0, 0, 0, 0
    msg = ""
    segments: list[np.ndarray] = []
    keypoints = None
    try:
        image = np.load(im_file, mmap_mode="r", allow_pickle=False)
        if image.ndim != 3:
            raise AssertionError(f"image array must be 3D, got shape={image.shape}")

        if image.shape[-1] == source_channels:
            shape = (int(image.shape[0]), int(image.shape[1]))
        elif image.shape[0] == source_channels:
            shape = (int(image.shape[1]), int(image.shape[2]))
        else:
            raise AssertionError(
                f"expected {source_channels} source channels in either HWC or CHW layout, got shape={image.shape}"
            )
        if not ((shape[0] > 9) and (shape[1] > 9)):
            raise AssertionError(f"image size {shape} <10 pixels")

        labels, segments, keypoints, nm, nf, ne, msg = _parse_yolo_label_file(
            lb_file, keypoint=keypoint, num_cls=num_cls, nkpt=nkpt, ndim=ndim, single_cls=single_cls
        )
        return im_file, labels, shape, segments, keypoints, nm, nf, ne, nc, f"{prefix}{msg}" if msg else msg
    except Exception as exc:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/label: {exc}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


class MultichannelNPYDataset(YOLODataset):
    """Ultralytics-compatible detection dataset for HWC/CHW `.npy` tensors."""

    def __init__(
        self,
        *args,
        dataset_type: str = DEFAULT_CHANNEL_ORDER,
        channel_order: str = DEFAULT_CHANNEL_ORDER,
        source_channels: int | None = None,
        padding_value: float = 114.0 / 255.0,
        **kwargs,
    ):
        self.channel_selection = resolve_channel_selection(dataset_type, channel_order)
        self.channel_order = channel_order.upper().strip()
        self.source_channels = int(source_channels) if source_channels is not None else None
        self.padding_value = float(padding_value)
        super().__init__(*args, **kwargs)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """Read `.npy` files from the specified path."""
        try:
            files = []
            for path_like in img_path if isinstance(img_path, list) else [img_path]:
                path_obj = Path(path_like)
                if path_obj.is_dir():
                    files += glob.glob(str(path_obj / "**" / "*.npy"), recursive=True)
                elif path_obj.is_file():
                    if path_obj.suffix.lower() == ".txt":
                        with open(path_obj, encoding="utf-8") as file_obj:
                            parent = str(path_obj.parent) + os.sep
                            lines = file_obj.read().strip().splitlines()
                            files += [line.replace("./", parent) if line.startswith("./") else line for line in lines]
                    elif path_obj.suffix.lower() == ".npy":
                        files.append(str(path_obj))
                    else:
                        raise FileNotFoundError(
                            f"{self.prefix}{path_obj} is not a .npy file or .txt manifest. {FORMATS_HELP_MSG}"
                        )
                else:
                    raise FileNotFoundError(f"{self.prefix}{path_obj} does not exist")

            im_files = sorted(str(Path(path).resolve()) for path in files if str(path).lower().endswith(".npy"))
            if not im_files:
                raise AssertionError(f"{self.prefix}No .npy files found in {img_path}")
        except Exception as exc:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from exc

        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]
        check_file_speeds(im_files, prefix=self.prefix)
        if self.source_channels is None:
            self.source_channels = infer_source_channels_from_npy(im_files[0], expected_channels=len(self.channel_order))
        if self.source_channels != len(self.channel_order):
            raise ValueError(
                f"channel_order='{self.channel_order}' descreve {len(self.channel_order)} canais, "
                f"mas os arquivos .npy indicam {self.source_channels} canais salvos."
            )
        return im_files

    def cache_labels(self, path: Path = Path("./labels.cache")) -> dict:
        """Cache labels for `.npy` images without using PIL image verification."""
        cache = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be like 'kpt_shape: [17, 3]'."
            )

        self.label_files = [str(Path(im_file).with_suffix(".txt")).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}") for im_file in self.im_files]
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=_verify_npy_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                    repeat(self.source_channels),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    cache["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],
                            "bboxes": lb[:, 1:],
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")

        cache["hash"] = get_hash(self.label_files + self.im_files)
        cache["results"] = nf, nm, ne, nc, len(self.im_files)
        cache["msgs"] = msgs
        save_dataset_cache_file(self.prefix, path, cache, DATASET_CACHE_VERSION)
        return cache

    def get_labels(self) -> list[dict]:
        """Load cached labels using the same flow as Ultralytics, but with `.npy` verification."""
        self.label_files = [str(Path(im_file).with_suffix(".txt")).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}") for im_file in self.im_files]
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, exists = self.cache_labels(cache_path), False

        nf, nm, ne, nc, total = cache.pop("results")
        if exists and LOCAL_RANK in {-1, 0}:
            desc = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + desc, total=total, initial=total)
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))

        for key in ("hash", "version", "msgs"):
            cache.pop(key, None)
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored. {HELP_URL}"
            )
        self.im_files = [label["im_file"] for label in labels]

        lengths = ((len(label["cls"]), len(label["bboxes"]), len(label["segments"])) for label in labels)
        len_cls, len_boxes, len_segments = (sum(items) for items in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. Only boxes will be used."
            )
            for label in labels:
                label["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """Build a conservative multichannel-safe augmentation pipeline."""
        transforms = [
            LetterBox(
                new_shape=(self.imgsz, self.imgsz),
                scaleup=self.augment,
                padding_value=self.padding_value,
            )
        ]
        if self.augment:
            transforms.extend(
                [
                    RandomFlip(direction="vertical", p=hyp.flipud),
                    RandomFlip(direction="horizontal", p=hyp.fliplr),
                ]
            )
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=1.0,
            )
        )
        return Compose(transforms)

    def _load_hwc_image(self, index: int) -> np.ndarray:
        image_file = self.im_files[index]
        array = np.load(image_file, allow_pickle=False)
        if array.ndim != 3:
            raise ValueError(f"Esperado array 3D em {image_file}, mas recebi shape={array.shape}")

        target_channels = self.channel_selection.num_channels
        if array.shape[-1] == self.source_channels:
            hwc = array
        elif array.shape[0] == self.source_channels:
            hwc = np.transpose(array, (1, 2, 0))
        elif array.shape[-1] == target_channels:
            hwc = array
        elif array.shape[0] == target_channels:
            hwc = np.transpose(array, (1, 2, 0))
        else:
            raise ValueError(
                f"Não consegui interpretar os canais de {image_file}. "
                f"source_channels={self.source_channels}, selected_channels={target_channels}, shape={array.shape}"
            )

        if hwc.shape[-1] == self.source_channels:
            hwc = hwc[..., list(self.channel_selection.indices)]
        if np.issubdtype(hwc.dtype, np.integer):
            hwc = hwc.astype(np.float32) / float(np.iinfo(hwc.dtype).max)
        else:
            hwc = hwc.astype(np.float32, copy=False)

        if hwc.size and (hwc.min() < -1e-6 or hwc.max() > 1.0 + 1e-6):
            raise ValueError(
                f"Os arrays .npy devem estar normalizados em [0, 1]. "
                f"Arquivo {image_file} veio com min={float(hwc.min()):.5f}, max={float(hwc.max()):.5f}"
            )

        return np.ascontiguousarray(hwc)

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load and resize a `.npy` image preserving normalized float data in HWC order."""
        im, image_file = self.ims[i], self.im_files[i]
        if im is None:
            im = self._load_hwc_image(i)
            h0, w0 = im.shape[:2]
            if rect_mode:
                ratio = self.imgsz / max(h0, w0)
                if ratio != 1:
                    width = min(int(np.ceil(w0 * ratio)), self.imgsz)
                    height = min(int(np.ceil(h0 * ratio)), self.imgsz)
                    im = cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]

            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]
