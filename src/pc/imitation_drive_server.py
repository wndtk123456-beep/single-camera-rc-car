import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
STEP 4 - 모방 학습 자율주행

- 시작하자마자 자율주행
- 모델이 카메라 프레임 보고 알아서 모터 명령 결정
- SPACE : 일시정지 / 재개
- ESC   : 종료
"""

import socket
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── 설정 ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH  = PROJECT_ROOT / "models" / "drive_best.pth"
LISTEN_PORT = 5002
MIRROR_LR   = True
SPEED_SCALE = 0.7    # 속도 (0.7 = 70%)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────

# 키 조합 → 모터값
COMBO_MOTOR = {
    'w' : (-80, -80),
    's' : ( 80,  80),
    'a' : ( 20, -100),
    'd' : (-100,  20),
    'wa': (-80,  -20),
    'wd': (-20,  -80),
    'sa': ( 80,   20),
    'sd': ( 20,   80),
}

def to_motor_str(key_name: str) -> str:
    if key_name in COMBO_MOTOR:
        l, r = COMBO_MOTOR[key_name]
        return f'{int(l * SPEED_SCALE)},{int(r * SPEED_SCALE)}'
    return '0,0'

# ── 모델 로드 ─────────────────────────────────────────────────────
print(f"모델 로드 중: {MODEL_PATH}")
ckpt         = torch.load(MODEL_PATH, map_location=DEVICE)
idx_to_class = ckpt['idx_to_class']
IMG_SIZE     = ckpt.get('img_size', 224)
NUM_CLASSES  = len(idx_to_class)

model = models.mobilenet_v3_small()
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
model.load_state_dict(ckpt['model_state'])
model.eval().to(DEVICE)
print(f"클래스: {list(idx_to_class.values())}  |  속도: {int(SPEED_SCALE*100)}%")
print("자율주행 시작 | SPACE=일시정지 | ESC=종료\n")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict(frame_bgr):
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inp = transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(inp), dim=1)[0]
        idx   = probs.argmax().item()
    return idx_to_class[idx], float(probs[idx])

# ── 소켓 초기화 ───────────────────────────────────────────────────
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', LISTEN_PORT))
sock.settimeout(0.02)

# ── 상태 변수 ─────────────────────────────────────────────────────
paused     = False
addr       = None
disp       = None
last_motor = '0,0'
last_key   = 'stop'
last_conf  = 0.0

while True:
    got_frame = False

    # 프레임 수신
    try:
        data, addr = sock.recvfrom(1_000_000)
        if len(data) >= 4:
            size     = int.from_bytes(data[:4], 'little')
            img_data = data[4:4 + size]
            if len(img_data) == size:
                arr   = np.frombuffer(img_data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    if MIRROR_LR:
                        frame = cv2.flip(frame, 1)
                    got_frame = True
    except socket.timeout:
        pass

    # 키 입력
    key = cv2.waitKey(1) & 0xFF
    if key == 27:       # ESC
        break
    elif key == ord(' '):
        paused = not paused
        print(f"{'[일시정지]' if paused else '[재개]'}")

    # 자율주행: 새 프레임이 왔을 때만 추론
    if got_frame and not paused:
        last_key, last_conf = predict(frame)
        last_motor = to_motor_str(last_key)

    # 일시정지 시 정지 명령
    motor = '0,0' if paused else last_motor

    # 화면 표시
    if got_frame:
        disp = frame.copy()

    if disp is not None:
        status = "[PAUSE]" if paused else "[AUTO]"
        color  = (0, 165, 255) if paused else (0, 255, 0)
        cv2.putText(disp, f"{status}  {last_key}  ({last_conf*100:.0f}%)",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(disp, f"MOTOR: {motor}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(disp, "SPACE=Pause  ESC=Quit",
                    (10, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.imshow("Auto Drive", disp)

    # 모터 명령 전송
    if addr:
        sock.sendto(motor.encode(), addr)

# 종료 시 정지
if addr:
    sock.sendto(b'0,0', addr)
cv2.destroyAllWindows()
sock.close()
print("종료")
