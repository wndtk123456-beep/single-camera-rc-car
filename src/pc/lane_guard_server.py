import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
자율주행 - 모방학습 + 차선 안전장치

구조:
  1) 모방학습 모델 (GPU) → 매 프레임 추론 → 기본 주행 명령
  2) YOLOv8 (CPU, 별도 스레드) → inline 위치 감시 → 이탈 시 보정 오버라이드
     → 스레드 분리로 주행 루프 블로킹 없음

키 조작:
  SPACE : 일시정지 / 재개
  L     : 차선 안전장치 ON / OFF 토글
  ESC   : 종료
"""

import socket
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

# ── 설정 ──────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
IMITATION_MODEL = PROJECT_ROOT / "models" / "drive_best.pth"
LANE_MODEL      = PROJECT_ROOT / "models" / "lane_best.pt"
LISTEN_PORT     = 5002
MIRROR_LR       = True
SPEED_SCALE     = 0.8

LANE_THRESHOLD     = 130  # inline 중심 오프셋 임계값 (픽셀) - 크게 벗어날 때만 개입
MIN_MASK_PIX       = 300  # 마스크 최소 픽셀 수
INLINE_CLASS       = 0    # inline = class 0
DEVIATION_COUNT    = 3    # 연속 N프레임 이탈 시에만 보정 (순간 노이즈 방지)
CORRECTION_COOLDOWN = 0.5 # 보정 후 N초간 모방학습에 주도권 (과잉 개입 방지)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────

COMBO_MOTOR = {
    'w' : (-80,  -80),
    's' : ( 80,   80),
    'a' : ( 20, -100),
    'd' : (-100,  20),
    'wa': (-80,  -20),
    'wd': (-20,  -80),
    'sa': ( 80,   20),
    'sd': ( 20,   80),
}

def to_motor_str(key_name: str, scale: float = 1.0) -> str:
    if key_name in COMBO_MOTOR:
        l, r = COMBO_MOTOR[key_name]
        return f'{int(l * scale)},{int(r * scale)}'
    return '0,0'

# ── 모방학습 모델 로드 (GPU) ──────────────────────────────────────
print(f"모방학습 모델 로드: {IMITATION_MODEL}")
ckpt         = torch.load(IMITATION_MODEL, map_location=DEVICE)
idx_to_class = ckpt['idx_to_class']
IMG_SIZE     = ckpt.get('img_size', 224)
NUM_CLASSES  = len(idx_to_class)

imitation_model = models.mobilenet_v3_small()
imitation_model.classifier[-1] = nn.Linear(
    imitation_model.classifier[-1].in_features, NUM_CLASSES)
imitation_model.load_state_dict(ckpt['model_state'])
imitation_model.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── 차선 모델 로드 (CPU - GPU 경쟁 방지) ─────────────────────────
print(f"차선 모델 로드: {LANE_MODEL} (CPU)")
lane_model = YOLO(LANE_MODEL)
lane_model.to("cpu")
print(f"클래스: {list(idx_to_class.values())}  |  속도: {int(SPEED_SCALE*100)}%\n")

# ── 모방학습 추론 ─────────────────────────────────────────────────
def predict_imitation(frame_bgr):
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inp = transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(imitation_model(inp), dim=1)[0]
        idx   = probs.argmax().item()
    return idx_to_class[idx], float(probs[idx])

# ── 차선 안전장치 스레드 ──────────────────────────────────────────
_lane_input   = None          # 메인 → 스레드: 처리할 프레임
_lane_error   = None          # 스레드 → 메인: 오프셋 (None=정상/검출안됨)
_lane_status  = "대기 중"
_lane_lock    = threading.Lock()
_dev_count    = 0             # 연속 이탈 프레임 카운터 (스레드 내부 전용)
_lane_raw_error = None        # 시각화용 raw 오프셋 (보정 여부 무관, 항상 저장)

def lane_thread_func():
    global _lane_input, _lane_error, _lane_raw_error, _lane_status, _dev_count
    while True:
        with _lane_lock:
            frame = _lane_input
            _lane_input = None

        if frame is None:
            time.sleep(0.005)   # CPU 점유 방지
            continue

        h, w    = frame.shape[:2]
        results = lane_model(frame, verbose=False)[0]

        if results.masks is None:
            _dev_count = 0
            with _lane_lock:
                _lane_error     = None
                _lane_raw_error = None
                _lane_status    = "검출 안됨"
            continue

        combined = np.zeros((h, w), dtype=np.float32)
        for i, cls in enumerate(results.boxes.cls):
            if int(cls) == INLINE_CLASS:
                mask = results.masks.data[i].cpu().numpy()
                combined = np.maximum(combined, cv2.resize(mask, (w, h)))

        bottom = combined[h // 2:, :]
        if bottom.sum() < MIN_MASK_PIX:
            _dev_count = 0
            with _lane_lock:
                _lane_error     = None
                _lane_raw_error = None
                _lane_status    = "검출 안됨"
            continue

        cols = np.where(bottom > 0.5)[1]
        if len(cols) == 0:
            _dev_count = 0
            with _lane_lock:
                _lane_error     = None
                _lane_raw_error = None
                _lane_status    = "검출 안됨"
            continue

        error = int(np.mean(cols)) - w // 2

        if abs(error) <= LANE_THRESHOLD:
            # 정상 범위 → 카운터 리셋, 보정 없음
            _dev_count = 0
            with _lane_lock:
                _lane_error     = None
                _lane_raw_error = error
                _lane_status    = f"정상 (err={error:+d}px)"
        else:
            # 이탈 범위 → 카운터 증가, N프레임 연속 시에만 보정 신호
            _dev_count += 1
            if _dev_count >= DEVIATION_COUNT:
                with _lane_lock:
                    _lane_error     = error
                    _lane_raw_error = error
                    _lane_status    = f"보정필요 (err={error:+d}px, {_dev_count}f)"
            else:
                with _lane_lock:
                    _lane_error     = None
                    _lane_raw_error = error
                    _lane_status    = f"이탈감지중 ({_dev_count}/{DEVIATION_COUNT}f)"

threading.Thread(target=lane_thread_func, daemon=True).start()

# ── 방향 화살표 그리기 유틸 ──────────────────────────────────────
_ARROW_VECS = {
    'w':  ( 0.000, -1.000), 's':  ( 0.000,  1.000),
    'a':  (-1.000,  0.000), 'd':  ( 1.000,  0.000),
    'wa': (-0.707, -0.707), 'wd': ( 0.707, -0.707),
    'sa': (-0.707,  0.707), 'sd': ( 0.707,  0.707),
}

def draw_arrow(img, direction, cx, cy, size, color, thick=3):
    if direction not in _ARROW_VECS:
        return
    dx, dy = _ARROW_VECS[direction]
    pt1 = (int(cx - dx * size * 0.55), int(cy - dy * size * 0.55))
    pt2 = (int(cx + dx * size * 0.55), int(cy + dy * size * 0.55))
    cv2.arrowedLine(img, pt1, pt2, color, thick, tipLength=0.42)

# ── 소켓 초기화 ───────────────────────────────────────────────────
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', LISTEN_PORT))
sock.settimeout(0.02)

# ── 상태 변수 ─────────────────────────────────────────────────────
paused            = False
lane_guard        = True
addr              = None
disp              = None
last_motor        = '0,0'
last_key          = 'stop'
last_conf         = 0.0
lane_status       = "대기 중"
last_correction_t = 0.0   # 마지막 차선 보정 시각 (쿨다운용)
disp_raw_error    = None  # 시각화용 최신 raw 오프셋

print("자율주행 시작 | SPACE=일시정지 | L=차선안전장치토글 | ESC=종료\n")

while True:
    got_frame = False
    frame     = None

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

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord(' '):
        paused = not paused
        print(f"{'[일시정지]' if paused else '[재개]'}")
    elif key == ord('l'):
        lane_guard = not lane_guard
        print(f"차선 안전장치: {'ON' if lane_guard else 'OFF'}")

    if got_frame and frame is not None and not paused:
        # 1) 모방학습 추론 (GPU, 빠름)
        last_key, last_conf = predict_imitation(frame)
        motor = to_motor_str(last_key, SPEED_SCALE)

        # 2) 차선 스레드에 프레임 전달 (논블로킹)
        if lane_guard:
            with _lane_lock:
                _lane_input = frame.copy()

            # 스레드의 최신 결과 읽기
            with _lane_lock:
                error          = _lane_error
                raw_error      = _lane_raw_error
                lane_status    = _lane_status
            disp_raw_error = raw_error

            # 보정 조건: 연속 이탈 확인됨 + 쿨다운 경과
            now = time.time()
            if error is not None and (now - last_correction_t) > CORRECTION_COOLDOWN:
                if error > 0:
                    motor = to_motor_str('wd', SPEED_SCALE)
                else:
                    motor = to_motor_str('wa', SPEED_SCALE)
                last_correction_t = now
        else:
            lane_status = "안전장치 OFF"

        last_motor = motor

    motor = '0,0' if paused else last_motor

    if got_frame and frame is not None:
        disp = frame.copy()

    if disp is not None:
        h_img, w_img = disp.shape[:2]
        status      = "[PAUSE]" if paused else "[AUTO]"
        status_col  = (0, 165, 255) if paused else (0, 255, 0)
        guard_label = "LANE:ON " if lane_guard else "LANE:OFF"
        guard_col   = (0, 255, 255) if lane_guard else (120, 120, 120)
        if "보정" in lane_status:
            guard_col = (0, 80, 255)

        cv2.putText(disp, f"{status}  {last_key}  ({last_conf*100:.0f}%)",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_col, 2)
        cv2.putText(disp, f"MOTOR: {motor}",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 0), 2)
        cv2.putText(disp, f"{guard_label}  {lane_status}",
                    (10, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.65, guard_col, 2)

        # ── 차선 개입 시에만 시각화 ───────────────────────────────
        is_correcting = (time.time() - last_correction_t) < 0.4
        if is_correcting and disp_raw_error is not None:
            # 오프셋 바
            bar_y, bar_h_px = 114, 16
            bar_x1, bar_x2  = 10, w_img - 10
            cx_bar          = w_img // 2
            cv2.rectangle(disp, (bar_x1, bar_y), (bar_x2, bar_y + bar_h_px), (40, 40, 40), -1)
            tx1 = max(bar_x1, cx_bar - LANE_THRESHOLD)
            tx2 = min(bar_x2, cx_bar + LANE_THRESHOLD)
            cv2.rectangle(disp, (tx1, bar_y), (tx2, bar_y + bar_h_px), (0, 70, 0), -1)
            cv2.line(disp, (tx1, bar_y), (tx1, bar_y + bar_h_px), (0, 180, 0), 1)
            cv2.line(disp, (tx2, bar_y), (tx2, bar_y + bar_h_px), (0, 180, 0), 1)
            cv2.line(disp, (cx_bar, bar_y), (cx_bar, bar_y + bar_h_px), (180, 180, 180), 1)
            dot_x = int(cx_bar + disp_raw_error)
            dot_x = max(bar_x1 + 7, min(bar_x2 - 7, dot_x))
            cv2.circle(disp, (dot_x, bar_y + bar_h_px // 2), 7, (30, 30, 255), -1)
            cv2.putText(disp, "INLINE", (bar_x1 + 2, bar_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (130, 130, 130), 1)
            # 빨간 화살표 + 레이블
            arrow_cx = w_img // 2
            arrow_cy = h_img // 2 + 30
            corr_dir = 'wd' if disp_raw_error > 0 else 'wa'
            draw_arrow(disp, corr_dir, arrow_cx, arrow_cy, 60, (30, 30, 255), 5)
            cv2.putText(disp, "LANE OVERRIDE",
                        (arrow_cx - 80, arrow_cy + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 255), 2)

        cv2.putText(disp, "SPACE=Pause  L=LaneGuard  ESC=Quit",
                    (10, h_img - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        cv2.imshow("Auto Drive", disp)

    if addr:
        sock.sendto(motor.encode(), addr)

if addr:
    sock.sendto(b'0,0', addr)
cv2.destroyAllWindows()
sock.close()
print("종료")
