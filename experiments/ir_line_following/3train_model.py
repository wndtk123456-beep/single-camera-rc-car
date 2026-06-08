# PC 실행 - 모델 학습
# 실행 후 생성된 line_model.pkl을 라즈베리파이로 복사하세요

import json
import glob
import random
import numpy as np
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ===== 데이터 로드 (line_data_*.json 전부 합치기) =====
all_data = []
files = sorted(glob.glob('line_data_*.json'))

if not files:
    print('line_data_*.json 파일이 없습니다.')
    exit()

for fname in files:
    with open(fname, encoding='utf-8') as f:
        chunk = json.load(f)
    all_data.extend(chunk)
    print(f'  {fname}: {len(chunk)}건')

print(f'\n전체 데이터: {len(all_data)}건')
print('원본 분포:', dict(Counter(d['action'] for d in all_data)))

# ===== stop 제거 =====
data = [d for d in all_data if d['action'] != 'stop']
print(f'\nstop 제거 후: {len(data)}건')
print('분포:', dict(Counter(d['action'] for d in data)))

# ===== 클래스 균형 맞추기 (언더샘플링) =====
by_action = defaultdict(list)
for d in data:
    by_action[d['action']].append(d)

min_count = min(len(v) for v in by_action.values())
print(f'\n각 클래스당 {min_count}건으로 균형화')

balanced = []
for action, samples in by_action.items():
    balanced.extend(random.sample(samples, min_count))
random.shuffle(balanced)

# ===== 학습 =====
X = np.array([d['ir'] for d in balanced])
y = np.array([d['action'] for d in balanced])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f'\n학습 정확도: {model.score(X_train, y_train):.3f}')
print(f'테스트 정확도: {model.score(X_test, y_test):.3f}')
print()
print(classification_report(y_test, model.predict(X_test)))

# ===== 모든 패턴에 대한 룩업 테이블 생성 (32가지) =====
import itertools

lookup = {}
for bits in itertools.product([0, 1], repeat=5):
    ir = list(bits)
    action = model.predict([ir])[0]
    lookup[str(ir)] = action

print('\n=== IR 패턴 → 액션 룩업 테이블 ===')
for pattern, action in sorted(lookup.items()):
    print(f'  {pattern} → {action}')

with open('line_lookup.json', 'w', encoding='utf-8') as f:
    json.dump(lookup, f, ensure_ascii=False, indent=2)

print('\n룩업 테이블 저장 완료: line_lookup.json')
print('→ line_lookup.json 을 라즈베리파이로 복사 후 pi_auto.py 실행')
