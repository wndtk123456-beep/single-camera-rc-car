import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
장애물 감지 모델 학습 (YOLOv8n Detection)

── 데이터 구조 ───────────────────────────────────────────────────────────────
  data/object_dataset/
  └── 20260406_181559/
      ├── frame_000002.jpg
      ├── frame_000002.json   ← LabelMe rectangle 라벨
      ├── frame_000003.jpg
      ├── frame_000003.json
      └── ...

  - json이 없는 jpg는 학습에서 제외 (라벨 없음)
  - shape_type: "rectangle" → YOLO bbox 자동 변환
  - 클래스: CLASS_MAP 에 정의된 이름 → 클래스 ID 매핑

── 추가 데이터 폴더 ──────────────────────────────────────────────────────────
  DATA_DIRS 리스트에 폴더 경로 추가하면 여러 촬영분 합산 학습 가능

── 실행 ──────────────────────────────────────────────────────────────────────
  python object_train.py

── 결과 ──────────────────────────────────────────────────────────────────────
  runs/object_det_v1/weights/best.pt
  models/obstacle_best.pt  ← 통합 자율주행 코드에서 사용
"""

import json
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

# ── 설정 ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]

# 라벨링된 데이터 폴더 목록 (여러 개 추가 가능)
DATA_DIRS = [
    ROOT_DIR / "data" / "object_dataset" / "20260406_181559",
]

DATASET_DIR  = ROOT_DIR / "object_train_dataset"   # 변환 후 저장 위치 (자동 생성)
RUN_NAME     = "object_det_v1"
BASE_MODEL   = "yolov8n.pt"                  # 자동 다운로드
OUTPUT_MODEL = ROOT_DIR / "models" / "obstacle_best.pt"

# 클래스 이름 → ID 매핑 (소문자로 비교)
# test_auto_drive 에서 OBJECT_CLASS=0 이므로 장애물이 0번
CLASS_MAP = {
    "object":   0,
    "obstacle": 0,
}

# 학습 하이퍼파라미터
EPOCHS    = 100
IMGSZ     = 640
BATCH     = 16
LR0       = 0.001
LRF       = 0.1
PATIENCE  = 30
VAL_RATIO = 0.2
SEED      = 42
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

TRAIN_IMG = DATASET_DIR / "train" / "images"
TRAIN_LBL = DATASET_DIR / "train" / "labels"
VALID_IMG = DATASET_DIR / "valid" / "images"
VALID_LBL = DATASET_DIR / "valid" / "labels"
DATA_YAML = DATASET_DIR / "data.yaml"


def labelme_rect_to_yolo(json_path: Path) -> list[str] | None:
    """
    LabelMe rectangle JSON → YOLO detection 형식 변환.
    반환값: ["cls cx cy w h", ...] 또는 None (변환 불가)
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [WARN] JSON 읽기 실패: {json_path.name} ({e})")
        return None

    width  = float(data.get("imageWidth",  0) or 0)
    height = float(data.get("imageHeight", 0) or 0)
    if width <= 0 or height <= 0:
        print(f"  [WARN] 이미지 크기 정보 없음: {json_path.name}")
        return None

    lines = []
    for shape in data.get("shapes", []):
        label      = str(shape.get("label", "")).strip().lower()
        shape_type = str(shape.get("shape_type", "")).strip().lower()
        points     = shape.get("points", [])

        if shape_type != "rectangle":
            continue
        if label not in CLASS_MAP:
            continue
        if len(points) < 2:
            continue

        cls_id = CLASS_MAP[label]
        x1, y1 = float(points[0][0]), float(points[0][1])
        x2, y2 = float(points[1][0]), float(points[1][1])

        # 좌표 정규화
        x1_n = min(max(min(x1, x2) / width,  0.0), 1.0)
        y1_n = min(max(min(y1, y2) / height, 0.0), 1.0)
        x2_n = min(max(max(x1, x2) / width,  0.0), 1.0)
        y2_n = min(max(max(y1, y2) / height, 0.0), 1.0)

        cx = (x1_n + x2_n) / 2
        cy = (y1_n + y2_n) / 2
        w  = x2_n - x1_n
        h  = y2_n - y1_n

        if w > 0 and h > 0:
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines if lines else None


def collect_pairs() -> list[tuple[Path, list[str]]]:
    """모든 DATA_DIRS 에서 (이미지경로, yolo라인목록) 쌍 수집"""
    pairs = []
    total_json = 0
    total_skip = 0

    for src_dir in DATA_DIRS:
        if not src_dir.exists():
            print(f"[WARN] 폴더 없음, 스킵: {src_dir}")
            continue

        for img_path in sorted(src_dir.iterdir()):
            if not (img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS):
                continue

            json_path = img_path.with_suffix(".json")
            if not json_path.exists():
                continue  # 라벨 없는 이미지는 제외

            total_json += 1
            yolo_lines = labelme_rect_to_yolo(json_path)
            if yolo_lines is None:
                total_skip += 1
                continue

            pairs.append((img_path, yolo_lines))

    print(f"[INFO] JSON 파일: {total_json}개 / 변환 성공: {len(pairs)}개 / 스킵: {total_skip}개")
    return pairs


def build_dataset(pairs: list[tuple[Path, list[str]]]):
    """수집된 쌍을 train/valid 폴더에 복사 + txt 생성"""
    # 기존 폴더 초기화
    for d in [TRAIN_IMG, TRAIN_LBL, VALID_IMG, VALID_LBL]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    shuffled = list(pairs)
    random.Random(SEED).shuffle(shuffled)

    n_val   = max(1, int(round(len(shuffled) * VAL_RATIO)))
    valid_p = shuffled[:n_val]
    train_p = shuffled[n_val:]

    def copy_pair(img_path, yolo_lines, img_dir, lbl_dir):
        shutil.copy2(img_path, img_dir / img_path.name)
        (lbl_dir / img_path.stem).with_suffix(".txt").write_text(
            "\n".join(yolo_lines), encoding="utf-8"
        )

    for img, lines in train_p:
        copy_pair(img, lines, TRAIN_IMG, TRAIN_LBL)
    for img, lines in valid_p:
        copy_pair(img, lines, VALID_IMG, VALID_LBL)

    return len(train_p), len(valid_p)


def infer_num_classes() -> int:
    max_id = -1
    for lbl_dir in [TRAIN_LBL, VALID_LBL]:
        for txt in lbl_dir.glob("*.txt"):
            for line in txt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        max_id = max(max_id, int(line.split()[0]))
                    except ValueError:
                        pass
    return max(max_id + 1, max(CLASS_MAP.values()) + 1)


def build_class_names(nc: int) -> list[str]:
    id_to_name = {v: k for k, v in CLASS_MAP.items()}
    return [id_to_name.get(i, f"class_{i}") for i in range(nc)]


def write_data_yaml(nc: int, names: list[str]):
    yaml_text = (
        f"path: {DATASET_DIR.as_posix()}\n"
        "train: train/images\n"
        "val:   valid/images\n\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )
    DATA_YAML.write_text(yaml_text, encoding="utf-8")


def main():
    print("=" * 60)
    print("  장애물 감지 모델 학습 (YOLOv8n Detection)")
    print("=" * 60)

    # ── 1) 데이터 수집 및 변환 ────────────────────────────────────────────────
    pairs = collect_pairs()
    if not pairs:
        print("\n[ERROR] 변환 가능한 데이터가 없습니다.")
        print("  - DATA_DIRS 경로를 확인하세요.")
        print("  - JSON 파일이 있는지, shape_type이 rectangle인지 확인하세요.")
        print(f"  - CLASS_MAP에 등록된 라벨: {list(CLASS_MAP.keys())}")
        return

    # ── 2) train / valid 분리 ────────────────────────────────────────────────
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    train_count, valid_count = build_dataset(pairs)
    print(f"[INFO] 데이터셋 구성: train={train_count}장, valid={valid_count}장")

    # ── 3) data.yaml 작성 ────────────────────────────────────────────────────
    nc    = infer_num_classes()
    names = build_class_names(nc)
    write_data_yaml(nc, names)
    print(f"[INFO] 클래스 (nc={nc}): {names}")
    print(f"[INFO] data.yaml → {DATA_YAML}")

    # ── 4) 학습 ──────────────────────────────────────────────────────────────
    print(f"\n[INFO] 베이스 모델: {BASE_MODEL}")
    print(f"[INFO] epochs={EPOCHS}  imgsz={IMGSZ}  batch={BATCH}  lr0={LR0}\n")

    model = YOLO(BASE_MODEL)
    model.train(
        data      = str(DATA_YAML),
        epochs    = EPOCHS,
        imgsz     = IMGSZ,
        batch     = BATCH,
        lr0       = LR0,
        lrf       = LRF,
        optimizer = "AdamW",
        patience  = PATIENCE,
        project   = str(ROOT_DIR / "runs"),
        name      = RUN_NAME,
        task      = "detect",
        workers   = 0,        # Windows 멀티프로세싱 버그 방지
        mosaic    = 1.0,      # 소규모 데이터 augmentation
        exist_ok  = True,
    )

    # ── 5) 최종 가중치를 실행용 models 폴더로 복사 ────────────────────────────
    best_pt = ROOT_DIR / "runs" / RUN_NAME / "weights" / "best.pt"
    if best_pt.exists():
        OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, OUTPUT_MODEL)
        print(f"\n[DONE] best.pt → {OUTPUT_MODEL} 복사 완료")
    else:
        print(f"\n[WARN] best.pt를 찾을 수 없음: {best_pt}")

    print(f"[DONE] 학습 완료 → runs/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
