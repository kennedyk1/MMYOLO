import csv
import json
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np


# =========================================================
# CONFIG
# =========================================================
RAW_OUTPUT_DIR = Path("MID-3K-dataset")
NPY_OUTPUT_DIR = Path("MID-3K-NPY")
TMP_DIR = RAW_OUTPUT_DIR / "_tmp_downloads"

REPOS = {
    "rgb": "https://github.com/kennedyk1/MID-3K-rgb/archive/refs/heads/main.zip",
    "thermal": "https://github.com/kennedyk1/MID-3K-thermal/archive/refs/heads/main.zip",
    "depth": "https://github.com/kennedyk1/MID-3K-depth/archive/refs/heads/main.zip",
    "intensity": "https://github.com/kennedyk1/MID-3K-intensity/archive/refs/heads/main.zip",
}

METAINFO_URL = "https://raw.githubusercontent.com/kennedyk1/MID-3K/main/metainfo.csv"

# Split desejado pelo usuário
SPLIT_RULES = {
    (4, 29): "val",
    (5, 7): "train",
    (5, 8): "val",
    (5, 9): "train",
    (5, 16): "test",
}


# =========================================================
# FILESYSTEM
# =========================================================
def create_raw_structure(base_dir: Path) -> None:
    for modality in REPOS.keys():
        for group in ["images", "labels"]:
            for split in ["train", "val", "test"]:
                (base_dir / modality / group / split).mkdir(parents=True, exist_ok=True)

    base_dir.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def create_npy_structure(base_dir: Path) -> None:
    for group in ["images", "labels"]:
        for split in ["train", "val", "test"]:
            (base_dir / group / split).mkdir(parents=True, exist_ok=True)

    base_dir.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dst: Path) -> None:
    print(f"[DOWNLOAD] {url}", flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dst)


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    print(f"[EXTRACT] {zip_path.name}", flush=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    extracted_folders = [p for p in extract_to.iterdir() if p.is_dir()]
    if len(extracted_folders) == 1:
        return extracted_folders[0]

    for p in extracted_folders:
        if "MID-3K" in p.name:
            return p

    raise RuntimeError(f"Não foi possível identificar a pasta extraída de {zip_path}")


# =========================================================
# METAINFO
# =========================================================
def read_metainfo(csv_path: Path) -> dict[str, str]:
    split_map = {}

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            file_id = row["FILE"].strip()
            month = int(row["MONTH"])
            day = int(row["DAY"])

            split = SPLIT_RULES.get((month, day))
            if split is None:
                raise ValueError(
                    f"Arquivo {file_id} possui data {day:02d}/{month:02d} fora das regras definidas."
                )

            split_map[file_id] = split

    return split_map


# =========================================================
# ORGANIZAÇÃO DO DATASET BRUTO
# =========================================================
def copy_modality_files(
    modality_name: str,
    extracted_repo_dir: Path,
    split_map: dict[str, str],
    output_dir: Path,
) -> None:
    src_images = extracted_repo_dir / "images"
    src_labels = extracted_repo_dir / "labels"

    if not src_images.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {src_images}")
    if not src_labels.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {src_labels}")

    image_files = sorted(src_images.glob("*.png"))
    label_files = sorted(src_labels.glob("*.txt"))

    print(
        f"[INFO] {modality_name}: {len(image_files)} imagens, {len(label_files)} labels",
        flush=True
    )

    missing_in_metainfo = []

    for img_path in image_files:
        stem = img_path.stem
        split = split_map.get(stem)

        if split is None:
            missing_in_metainfo.append(stem)
            continue

        dst = output_dir / modality_name / "images" / split / img_path.name
        shutil.copy2(img_path, dst)

    for lbl_path in label_files:
        stem = lbl_path.stem
        split = split_map.get(stem)

        if split is None:
            continue

        dst = output_dir / modality_name / "labels" / split / lbl_path.name
        shutil.copy2(lbl_path, dst)

    if missing_in_metainfo:
        print(
            f"[WARNING] {modality_name}: {len(missing_in_metainfo)} arquivos não encontrados no metainfo.csv",
            flush=True
        )
        print(f"Exemplo: {missing_in_metainfo[:10]}", flush=True)


def count_raw_files(base_dir: Path) -> None:
    print("\n========== RESUMO RAW ==========", flush=True)
    for modality in REPOS.keys():
        print(f"\n[{modality.upper()}]", flush=True)
        for split in ["train", "val", "test"]:
            n_img = len(list((base_dir / modality / "images" / split).glob("*.png")))
            n_lbl = len(list((base_dir / modality / "labels" / split).glob("*.txt")))
            print(f"  {split:<5} -> images: {n_img:<5} labels: {n_lbl:<5}", flush=True)


# =========================================================
# LEITURA E NORMALIZAÇÃO
# =========================================================
def load_png_unchanged(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Falha ao ler imagem: {path}")
    return img


def normalize_rgb(img: np.ndarray) -> np.ndarray:
    """
    Espera RGB com shape (H, W, 3).
    OpenCV lê colorido em BGR; convertemos para RGB.
    Retorna float32 em [0,1].
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"RGB inválido: shape={img.shape}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)
        vmax = float(img.max()) if img.size > 0 else 0.0
        if vmax > 1.0:
            raise ValueError(
                f"RGB com dtype inesperado e faixa > 1: dtype={img.dtype}, max={vmax}"
            )

    return img


def normalize_single_channel(img: np.ndarray, modality_name: str) -> np.ndarray:
    """
    Thermal / Depth / Intensity
    Retorna shape (H, W, 1), float32 em [0,1].

    Casos aceitos:
    - (H, W)
    - (H, W, 1)
    - (H, W, 3) com canais idênticos
    """
    if img.ndim == 3:
        if img.shape[2] == 1:
            img = img[:, :, 0]

        elif img.shape[2] == 3:
            c0 = img[:, :, 0]
            c1 = img[:, :, 1]
            c2 = img[:, :, 2]

            if np.array_equal(c0, c1) and np.array_equal(c0, c2):
                img = c0
            else:
                raise ValueError(
                    f"{modality_name} veio com 3 canais, mas eles não são idênticos. shape={img.shape}"
                )
        else:
            raise ValueError(
                f"{modality_name} deveria ser single-channel, mas shape={img.shape}"
            )

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)
        vmax = float(img.max()) if img.size > 0 else 0.0
        if vmax > 1.0:
            raise ValueError(
                f"{modality_name} com dtype inesperado e faixa > 1: dtype={img.dtype}, max={vmax}"
            )

    img = np.expand_dims(img, axis=-1)
    return img


# =========================================================
# STATS
# =========================================================
class RunningChannelStats:
    def __init__(self, num_channels: int):
        self.num_channels = num_channels
        self.count = 0
        self.sum = np.zeros(num_channels, dtype=np.float64)
        self.sum_sq = np.zeros(num_channels, dtype=np.float64)

    def update(self, arr_hwc: np.ndarray) -> None:
        if arr_hwc.ndim != 3 or arr_hwc.shape[2] != self.num_channels:
            raise ValueError(f"Esperado (H, W, {self.num_channels}), recebi {arr_hwc.shape}")

        flat = arr_hwc.reshape(-1, self.num_channels).astype(np.float64)
        self.sum += flat.sum(axis=0)
        self.sum_sq += (flat ** 2).sum(axis=0)
        self.count += flat.shape[0]

    def finalize(self):
        if self.count == 0:
            raise RuntimeError("Nenhum dado válido foi processado para calcular mean/std.")

        mean = self.sum / self.count
        var = self.sum_sq / self.count - mean ** 2
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        return mean.astype(np.float32), std.astype(np.float32)


# =========================================================
# BUILD NPY
# =========================================================
def build_npy_dataset(raw_base_dir: Path, npy_base_dir: Path) -> None:
    create_npy_structure(npy_base_dir)

    stats = RunningChannelStats(num_channels=6)

    skipped = {
        "missing_rgb": 0,
        "missing_thermal": 0,
        "missing_depth": 0,
        "missing_intensity": 0,
        "missing_label": 0,
        "shape_mismatch": 0,
        "other_errors": 0,
    }

    saved_counts = {"train": 0, "val": 0, "test": 0}

    for split in ["train", "val", "test"]:
        rgb_img_dir = raw_base_dir / "rgb" / "images" / split
        rgb_lbl_dir = raw_base_dir / "rgb" / "labels" / split
        rgb_files = sorted(rgb_img_dir.glob("*.png"))

        print(f"\n[{split}] total: {len(rgb_files)}", flush=True)

        for i, rgb_path in enumerate(rgb_files):
            stem = rgb_path.stem
            print(f"[{split}] Processando {i+1}/{len(rgb_files)}", flush=True)

            thermal_path = raw_base_dir / "thermal" / "images" / split / f"{stem}.png"
            depth_path = raw_base_dir / "depth" / "images" / split / f"{stem}.png"
            intensity_path = raw_base_dir / "intensity" / "images" / split / f"{stem}.png"
            rgb_label_path = rgb_lbl_dir / f"{stem}.txt"

            try:
                if not rgb_path.exists():
                    skipped["missing_rgb"] += 1
                    continue
                if not thermal_path.exists():
                    skipped["missing_thermal"] += 1
                    continue
                if not depth_path.exists():
                    skipped["missing_depth"] += 1
                    continue
                if not intensity_path.exists():
                    skipped["missing_intensity"] += 1
                    continue
                if not rgb_label_path.exists():
                    skipped["missing_label"] += 1
                    continue

                rgb_raw = load_png_unchanged(rgb_path)
                thermal_raw = load_png_unchanged(thermal_path)
                depth_raw = load_png_unchanged(depth_path)
                intensity_raw = load_png_unchanged(intensity_path)

                rgb = normalize_rgb(rgb_raw)
                thermal = normalize_single_channel(thermal_raw, "thermal")
                depth = normalize_single_channel(depth_raw, "depth")
                intensity = normalize_single_channel(intensity_raw, "intensity")

                h, w = rgb.shape[:2]
                if thermal.shape[:2] != (h, w) or depth.shape[:2] != (h, w) or intensity.shape[:2] != (h, w):
                    skipped["shape_mismatch"] += 1
                    print(
                        f"[WARNING] shape mismatch em {stem}: "
                        f"rgb={rgb.shape}, thermal={thermal.shape}, depth={depth.shape}, intensity={intensity.shape}",
                        flush=True
                    )
                    continue

                stacked = np.concatenate([rgb, thermal, depth, intensity], axis=-1).astype(np.float32)

                out_npy = npy_base_dir / "images" / split / f"{stem}.npy"
                np.save(out_npy, stacked)

                out_lbl = npy_base_dir / "labels" / split / f"{stem}.txt"
                shutil.copy2(rgb_label_path, out_lbl)

                stats.update(stacked)
                saved_counts[split] += 1

            except Exception as e:
                skipped["other_errors"] += 1
                print(f"[ERROR] {stem}: {e}", flush=True)
                continue

    mean_image, std_image = stats.finalize()

    print("\n========== RESUMO NPY ==========", flush=True)
    for split in ["train", "val", "test"]:
        n_img = len(list((npy_base_dir / "images" / split).glob("*.npy")))
        n_lbl = len(list((npy_base_dir / "labels" / split).glob("*.txt")))
        print(f"{split:<5} -> npy: {n_img:<5} labels: {n_lbl:<5}", flush=True)

    print("\n========== SKIPPED ==========", flush=True)
    for k, v in skipped.items():
        print(f"{k}: {v}", flush=True)

    channel_names = ["R", "G", "B", "Thermal", "Depth", "Intensity"]

    print("\n========== mean_image ==========", flush=True)
    print(mean_image.tolist(), flush=True)

    print("\n========== std_image ==========", flush=True)
    print(std_image.tolist(), flush=True)

    print("\n========== POR CANAL ==========", flush=True)
    for i, name in enumerate(channel_names):
        print(
            f"{name:<10} mean={float(mean_image[i]):.8f}  std={float(std_image[i]):.8f}",
            flush=True
        )

    stats_json = {
        "channel_order": channel_names,
        "shape_format": "(H, W, 6)",
        "mean_image": [float(x) for x in mean_image],
        "std_image": [float(x) for x in std_image],
        "saved_counts": saved_counts,
        "skipped": skipped,
    }

    with (npy_base_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2)

    print(f"\n[OK] stats.json salvo em: {npy_base_dir / 'stats.json'}", flush=True)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    create_raw_structure(RAW_OUTPUT_DIR)

    metainfo_path = RAW_OUTPUT_DIR / "metainfo.csv"
    if not metainfo_path.exists():
        download_file(METAINFO_URL, metainfo_path)
    else:
        print(f"[SKIP] metainfo já existe: {metainfo_path}", flush=True)

    split_map = read_metainfo(metainfo_path)
    print(f"[INFO] Entradas no metainfo.csv: {len(split_map)}", flush=True)

    for modality, zip_url in REPOS.items():
        zip_path = TMP_DIR / f"{modality}.zip"
        extract_dir = TMP_DIR / f"{modality}_extracted"

        if not zip_path.exists():
            download_file(zip_url, zip_path)
        else:
            print(f"[SKIP] zip já existe: {zip_path}", flush=True)

        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        repo_root = extract_zip(zip_path, extract_dir)

        copy_modality_files(
            modality_name=modality,
            extracted_repo_dir=repo_root,
            split_map=split_map,
            output_dir=RAW_OUTPUT_DIR,
        )

    count_raw_files(RAW_OUTPUT_DIR)
    build_npy_dataset(RAW_OUTPUT_DIR, NPY_OUTPUT_DIR)

    print("\nConcluído.", flush=True)
    print(f"RAW dataset : {RAW_OUTPUT_DIR.resolve()}", flush=True)
    print(f"NPY dataset : {NPY_OUTPUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()