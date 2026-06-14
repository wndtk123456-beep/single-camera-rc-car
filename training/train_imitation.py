# 수동 주행 데이터로 기본 주행 모델을 학습합니다.


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
STEP 3 - 모방 학습 (Imitation Learning) 학습 코드

입력: collected_data/ 폴더의 프레임 + 키 조합 라벨
출력: MobileNetV3 기반 분류 모델 (카메라 프레임 → 키 조합)

특징:
  - MobileNetV3-Small (라즈베리파이 실시간 추론 가능)
  - 클래스 불균형 → 가중치 자동 계산
  - 수평 flip 금지 (좌/우 라벨이 달라서 flip 쓰면 안 됨)
  - 밝기/대비 augmentation만 적용
"""


import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ── 설정 ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT  = PROJECT_ROOT / "data" / "drive_dataset"
SAVE_PATH  = PROJECT_ROOT / "models" / "drive_best.pth"
IMG_SIZE   = 224
BATCH      = 32
EPOCHS     = 30
LR         = 1e-4
VAL_RATIO  = 0.15      # 전체 데이터의 15%를 validation으로 사용
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# 클래스 정의 (전체 8가지, 데이터 없는 클래스는 자동 제외)
ALL_CLASSES = ['w', 's', 'a', 'd', 'wa', 'wd', 'sa', 'sd']



# ── 데이터셋 ──────────────────────────────────────────────────────
class ImitationDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples      = samples   # [(jpg_path, key_str), ...]
        self.class_to_idx = class_to_idx
        self.transform    = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_str = self.samples[idx]
        img   = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[label_str]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_samples(data_root: str):
    """collected_data/ 하위 모든 세션에서 (jpg, key) 쌍 로드"""
    samples = []
    root = Path(data_root)
    for txt in sorted(root.glob("**/*.txt")):
        jpg = txt.with_suffix(".jpg")
        if not jpg.exists():
            continue
        lines = txt.read_text().strip().split('\n')
        if len(lines) < 2:
            continue
        key = lines[1].strip()
        if key and key != 'stop':
            samples.append((str(jpg), key))
    return samples


# ── 메인 ─────────────────────────────────────────────────────────
print(f"디바이스: {DEVICE}")
print(f"데이터 로드 중: {DATA_ROOT}")

samples = load_samples(DATA_ROOT)
if not samples:
    print("데이터 없음. collected_data/ 폴더 확인하세요.")
    exit(1)

SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# 실제 데이터에 존재하는 클래스만 사용
present = sorted(set(k for _, k in samples), key=lambda x: ALL_CLASSES.index(x) if x in ALL_CLASSES else 99)
class_to_idx = {c: i for i, c in enumerate(present)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
NUM_CLASSES  = len(present)

# 클래스 분포 출력
counts = Counter(k for _, k in samples)
print(f"\n클래스 분포 (총 {len(samples)}장):")
for c in present:
    pct = counts[c] / len(samples) * 100
    print(f"  {c:4s}: {counts[c]:5d}장  ({pct:.1f}%)")
print(f"  → 사용 클래스: {present}")

# 클래스 가중치 (불균형 보정)
class_weights = torch.tensor(
    [len(samples) / (NUM_CLASSES * counts[c]) for c in present],
    dtype=torch.float32
).to(DEVICE)
print(f"\n클래스 가중치: { {c: f'{w:.2f}' for c, w in zip(present, class_weights.cpu())} }")

# ── Transform ────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── 데이터 분할 ───────────────────────────────────────────────────
full_dataset = ImitationDataset(samples, class_to_idx, transform=train_transform)
val_size     = int(len(full_dataset) * VAL_RATIO)
train_size   = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))
val_ds.dataset = ImitationDataset(samples, class_to_idx, transform=val_transform)

print(f"\n학습: {train_size}장 / 검증: {val_size}장")

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

# ── 모델 ─────────────────────────────────────────────────────────
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── 학습 루프 ─────────────────────────────────────────────────────
best_val_acc = 0.0
print(f"\n학습 시작 (epochs={EPOCHS})")
print("-" * 55)

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss, train_correct = 0.0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item() * imgs.size(0)
        train_correct += (out.argmax(1) == labels).sum().item()

    # Validation
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            val_loss    += loss.item() * imgs.size(0)
            val_correct += (out.argmax(1) == labels).sum().item()

    train_acc = train_correct / train_size * 100
    val_acc   = val_correct   / val_size   * 100
    scheduler.step()

    marker = " ★" if val_acc > best_val_acc else ""
    print(f"Epoch {epoch:3d}/{EPOCHS}  "
          f"train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%{marker}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch'        : epoch,
            'model_state'  : model.state_dict(),
            'class_to_idx' : class_to_idx,
            'idx_to_class' : idx_to_class,
            'present_classes': present,
            'img_size'     : IMG_SIZE,
        }, SAVE_PATH)

print("-" * 55)
print(f"완료. 최고 val_acc={best_val_acc:.1f}%")
print(f"모델 저장: {SAVE_PATH}")

# 클래스별 정확도 출력
model.load_state_dict(torch.load(SAVE_PATH)['model_state'])
model.eval()
per_class_correct = Counter()
per_class_total   = Counter()
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(1)
        for pred, label in zip(preds.cpu(), labels.cpu()):
            cls = idx_to_class[label.item()]
            per_class_total[cls]   += 1
            per_class_correct[cls] += int(pred == label)

print("\n[클래스별 val 정확도]")
for c in present:
    total = per_class_total[c]
    if total == 0:
        continue
    acc = per_class_correct[c] / total * 100
    print(f"  {c:4s}: {per_class_correct[c]:4d}/{total:4d}  ({acc:.1f}%)")
