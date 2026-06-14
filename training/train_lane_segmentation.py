from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from ultralytics import YOLO

# -----------------------------
# Config (lane_dataset only)
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "data" / "lane_dataset"

TRAIN_IMG_DIR = DATASET_DIR / "train" / "images"
TRAIN_LBL_DIR = DATASET_DIR / "train" / "labels"
VALID_IMG_DIR = DATASET_DIR / "valid" / "images"
VALID_LBL_DIR = DATASET_DIR / "valid" / "labels"
DATA_YAML = DATASET_DIR / "data.yaml"

PROJECT_DIR = ROOT_DIR / "runs"
RUN_NAME = "lane_seg_v5"

PRETRAINED_MODEL = ROOT_DIR / "models" / "lane_best.pt"
FALLBACK_MODEL = "yolov8n-seg.pt"
OUTPUT_MODEL = ROOT_DIR / "models" / "lane_best.pt"

DEFAULT_CLASS_NAMES = ["inline", "outline", "crosswalk"]
CLASS_MAP = {
    "inline": 0,
    "linline": 0,
    "outline": 1,
    "crosswalk": 2,
    "cross_walk": 2,
    "crosswlak": 2,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def convert_json_to_yolo(json_path: Path, out_txt_path: Path) -> bool:
    """Convert one LabelMe polygon JSON to YOLO segmentation txt."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] JSON load failed: {json_path.name} ({exc})")
        return False

    width = float(data.get("imageWidth", 0) or 0)
    height = float(data.get("imageHeight", 0) or 0)
    if width <= 0 or height <= 0:
        print(f"[WARN] Invalid image size in JSON: {json_path.name}")
        return False

    lines: List[str] = []
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        shape_type = str(shape.get("shape_type", "")).strip().lower()
        points = shape.get("points", [])

        if shape_type != "polygon":
            continue
        if label not in CLASS_MAP:
            continue
        if not isinstance(points, list) or len(points) < 3:
            continue

        cls_id = CLASS_MAP[label]
        coords: List[str] = []
        for p in points:
            if not isinstance(p, Sequence) or len(p) < 2:
                continue
            px = float(p[0])
            py = float(p[1])
            nx = min(1.0, max(0.0, px / width))
            ny = min(1.0, max(0.0, py / height))
            coords.extend((f"{nx:.6f}", f"{ny:.6f}"))

        if len(coords) >= 6:  # at least 3 points
            lines.append(f"{cls_id} " + " ".join(coords))

    if not lines:
        return False

    out_txt_path.write_text("\n".join(lines), encoding="utf-8")
    return True


def _iter_images(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    return (p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def ensure_labels_from_json(flat_image_paths: Sequence[Path]) -> int:
    """Create missing txt labels from same-name JSON files in lane_dataset root."""
    created = 0
    for img_path in flat_image_paths:
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            continue

        json_path = img_path.with_suffix(".json")
        if json_path.exists() and convert_json_to_yolo(json_path, txt_path):
            created += 1
    return created


def gather_pairs() -> List[Tuple[Path, Path]]:
    """
    Gather image-label pairs from lane_dataset only.

    Supports both structures:
    1) lane_dataset/*.jpg + *.txt (+ optional *.json)
    2) lane_dataset/images/*.jpg + lane_dataset/labels/*.txt
    """
    flat_images = list(_iter_images(DATASET_DIR))

    if flat_images:
        created = ensure_labels_from_json(flat_images)
        if created:
            print(f"[INFO] Generated {created} txt labels from JSON.")

        pairs: List[Tuple[Path, Path]] = []
        for img in flat_images:
            lbl = img.with_suffix(".txt")
            if lbl.exists():
                pairs.append((img, lbl))
        return pairs

    img_dir = DATASET_DIR / "images"
    lbl_dir = DATASET_DIR / "labels"
    if img_dir.exists() and lbl_dir.exists():
        pairs = []
        for img in _iter_images(img_dir):
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                pairs.append((img, lbl))
        return pairs

    return []


def _reset_split_dirs() -> None:
    for d in [TRAIN_IMG_DIR, TRAIN_LBL_DIR, VALID_IMG_DIR, VALID_LBL_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def split_dataset(pairs: Sequence[Tuple[Path, Path]], val_ratio: float, seed: int) -> Tuple[int, int]:
    if not pairs:
        raise RuntimeError("No image-label pairs found in lane_dataset")

    _reset_split_dirs()

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    n_total = len(shuffled)
    n_val = max(1, int(round(n_total * val_ratio))) if n_total > 1 else 1
    n_val = min(n_val, n_total)

    valid_pairs = shuffled[:n_val]
    train_pairs = shuffled[n_val:] if n_total > 1 else shuffled[:]

    for img, lbl in train_pairs:
        shutil.copy2(img, TRAIN_IMG_DIR / img.name)
        shutil.copy2(lbl, TRAIN_LBL_DIR / lbl.name)

    for img, lbl in valid_pairs:
        shutil.copy2(img, VALID_IMG_DIR / img.name)
        shutil.copy2(lbl, VALID_LBL_DIR / lbl.name)

    return len(train_pairs), len(valid_pairs)


def infer_num_classes(label_paths: Sequence[Path]) -> int:
    max_id = -1
    for path in label_paths:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                cls_id = int(line.split()[0])
                max_id = max(max_id, cls_id)
        except Exception:
            continue

    # fallback to known class config when parsing fails or empty labels
    if max_id < 0:
        return len(DEFAULT_CLASS_NAMES)
    return max_id + 1


def build_class_names(nc: int) -> List[str]:
    names = DEFAULT_CLASS_NAMES[:nc]
    while len(names) < nc:
        names.append(f"class_{len(names)}")
    return names


def write_data_yaml(nc: int, names: Sequence[str]) -> None:
    yaml_text = (
        f"path: {DATASET_DIR.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n\n"
        f"nc: {nc}\n"
        f"names: {list(names)}\n"
    )
    DATA_YAML.write_text(yaml_text, encoding="utf-8")


def resolve_model_path(user_model: str | None) -> str:
    if user_model:
        return user_model
    if PRETRAINED_MODEL.exists():
        return str(PRETRAINED_MODEL)
    return FALLBACK_MODEL


def train(args: argparse.Namespace) -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    pairs = gather_pairs()
    if not pairs:
        raise RuntimeError(
            "No usable samples found in lane_dataset. "
            "Expected image/txt pairs in lane_dataset root or images/labels subfolders."
        )

    train_count, valid_count = split_dataset(pairs, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[INFO] split done: train={train_count}, valid={valid_count}")

    label_paths = list(TRAIN_LBL_DIR.glob("*.txt")) + list(VALID_LBL_DIR.glob("*.txt"))
    nc = infer_num_classes(label_paths)
    names = build_class_names(nc)
    write_data_yaml(nc=nc, names=names)
    print(f"[INFO] data.yaml written: {DATA_YAML}")
    print(f"[INFO] classes (nc={nc}): {names}")

    model_path = resolve_model_path(args.model)
    print(f"[INFO] training model: {model_path}")

    model = YOLO(model_path)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        project=str(PROJECT_DIR),
        name=args.run_name,
        task="segment",
        exist_ok=True,
    )

    best_pt = PROJECT_DIR / args.run_name / "weights" / "best.pt"
    if best_pt.exists():
        OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, OUTPUT_MODEL)
        print(f"[DONE] best weight copied: {OUTPUT_MODEL}")

    print("[DONE] training completed")
    print(f"[DONE] results: {PROJECT_DIR / args.run_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lane segmentation using lane_dataset only")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--lrf", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=RUN_NAME)
    parser.add_argument("--model", type=str, default=None, help="Optional model path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
