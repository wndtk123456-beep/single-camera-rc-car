import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
자율주행 - 모방학습 + 차선 안전장치 + 장애물 회피

우선순위: 횡단보도 정지 > 장애물 회피 > 차선 보정 > 모방학습

장애물 회피 방식:
  - YOLO 객체 감지 모델로 장애물 감지
  - 최근 OBJECT_CONFIRM_WINDOW 프레임 중 OBJECT_CONFIRM_RATIO 이상 감지 시 회피 시작
    (단일 프레임 오감지 방지, 방향은 감지 프레임들의 다수결로 결정)
  - bbox 위치 기준으로 회피 방향 결정 (좌 → 우회피, 우 → 좌회피)
  - 약한 강도(OBJECT_AVOID_SCALE)로 조향 → 차선이 사라지지 않는 범위
  - OBJECT_STEER_SEC 후 모방학습 복귀 → 차선유지가 자연스럽게 중앙 복귀

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
OBJECT_MODEL    = PROJECT_ROOT / "models" / "obstacle_best.pt"
LISTEN_PORT     = 5002
MIRROR_LR       = True
SPEED_SCALE     = 0.70
DISPLAY_FPS     = 30   # 화면 업데이트 주기 (낮출수록 루프가 빨라짐, 모터 제어는 영향 없음)

# 차선 안전장치
LANE_THRESHOLD      = 130
MIN_MASK_PIX        = 300
INLINE_CLASS        = 0
DEVIATION_COUNT     = 3
CORRECTION_COOLDOWN = 1.0

# 횡단보도
CROSSWALK_CLASS          = 2
CROSSWALK_STOP_SEC       = 3.0
CROSSWALK_COOLDOWN_SEC   = 8.0
CROSSWALK_CONF_THR       = 0.45

# 장애물 회피
OBJECT_CLASS         = 0       # 장애물 클래스 인덱스
OBJECT_CONF_THR      = 0.6   # 감지 최소 confidence
OBJECT_MIN_BOX_AREA  = 2000  # bbox 최소 면적(px²) - 이것보다 작으면 무시 (멀리 있는 물체, 노이즈 제거)
OBJECT_CONFIRM_FRAMES = 1    # 연속 N프레임 감지되어야 트리거 (오감지 방지, 높이면 느리게 반응)
OBJECT_STOP_SEC      = 0.0     # 감지 직후 브레이킹 시간 (0=즉시 조향, 속도 있는 채로 꺾는게 더 효과적)
OBJECT_STEER_SEC     = 0.9     # 회피 조향 유지 시간 (초)
OBJECT_FORWARD_SEC   = 0.9     # 회피 후 직진 시간 (초)
OBJECT_RETURN_SEC_A  = 1.2     # 'a'(좌복귀) 조향 시간 (초)
OBJECT_RETURN_SEC_D  = 1.2     # 'd'(우복귀) 조향 시간 (초)
OBJECT_POST_FWD_SEC  = 0.7     # 복귀 조향 후 직진 시간 (초)
OBJECT_COOLDOWN_SEC  = 4.0     # 복귀 완료 후 쿨다운 (초) - 재감지 방지
OBJECT_AVOID_SCALE_D = 0.6   # 'd'(우회피/우복귀) 조향 강도
OBJECT_AVOID_SCALE_A = 0.5   # 'a'(좌회피/좌복귀) 조향 강도

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────

COMBO_MOTOR = {
    'w' : (-90,  -90),
    's' : ( 90,   90),
    'a' : ( 30, -120),
    'd' : (-120,  30),
    'wa': (-100,  -30),
    'wd': (-30,  -100),
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

# ── 차선/장애물 모델 로드 (CPU) ───────────────────────────────────
print(f"차선 모델 로드: {LANE_MODEL} (CPU)")
lane_model = YOLO(LANE_MODEL)
lane_model.to("cpu")

print(f"장애물 모델 로드: {OBJECT_MODEL} (CPU)")
object_model = YOLO(OBJECT_MODEL)
object_model.to("cpu")

print(f"클래스: {list(idx_to_class.values())}  |  속도: {int(SPEED_SCALE*100)}%\n")

# ── 모방학습 추론 ─────────────────────────────────────────────────
def predict_imitation(frame_bgr):
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inp = transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(imitation_model(inp), dim=1)[0]
        idx   = probs.argmax().item()
    return idx_to_class[idx], float(probs[idx])

# ── 공유 변수 (메인 ↔ 스레드) ─────────────────────────────────────
_lane_input          = None
_lane_error          = None
_lane_raw_error      = None
_lane_status         = "대기 중"
_obj_confirm_count   = 0     # 스레드 내부: 연속 감지 프레임 카운터
_obj_confirm_side    = None  # 스레드 내부: 카운터 누적 중인 방향
_lane_lock           = threading.Lock()
_dev_count           = 0
_crosswalk_detected  = False
_object_detected     = False   # 장애물 감지 플래그
_object_side         = None    # 'left' or 'right'

# ── 차선 + 장애물 감지 스레드 ─────────────────────────────────────
def lane_thread_func():
    global _lane_input, _lane_error, _lane_raw_error, _lane_status
    global _dev_count, _crosswalk_detected, _object_detected, _object_side
    global _obj_confirm_count, _obj_confirm_side

    while True:
        with _lane_lock:
            frame = _lane_input
            _lane_input = None

        if frame is None:
            time.sleep(0.005)
            continue

        h, w = frame.shape[:2]

        # ── 장애물 감지 - imgsz=320으로 빠른 추론 ────────────────
        obj_results  = object_model(frame, verbose=False, imgsz=320)[0]
        found_this_frame = False   # 이번 프레임에서 유효한 감지가 있었는지

        for i, cls in enumerate(obj_results.boxes.cls):
            if int(cls) == OBJECT_CLASS:
                conf = float(obj_results.boxes.conf[i])
                if conf >= OBJECT_CONF_THR:
                    box      = obj_results.boxes.xyxy[i].cpu().numpy()
                    box_w    = box[2] - box[0]
                    box_h    = box[3] - box[1]
                    box_area = box_w * box_h
                    if box_area < OBJECT_MIN_BOX_AREA:
                        continue
                    # 이번 프레임에서 조건을 만족하는 감지 발견
                    found_this_frame = True
                    _obj_confirm_side = 'left' if (box[0] + box[2]) / 2 < w / 2 else 'right'
                    break

        if found_this_frame:
            # 연속 감지 카운터 증가
            _obj_confirm_count += 1
            if _obj_confirm_count >= OBJECT_CONFIRM_FRAMES:
                # N프레임 연속 감지 → 진짜 장애물로 판단, 트리거
                with _lane_lock:
                    _object_detected = True
                    _object_side     = _obj_confirm_side
                _obj_confirm_count = 0   # 트리거 후 카운터 리셋
        else:
            # 이번 프레임에서 감지 안 됨 → 카운터 리셋 (연속이 끊김)
            _obj_confirm_count = 0
            _obj_confirm_side  = None

        # ── 차선 감지 (lane model) - imgsz=320으로 빠른 추론 ──────
        lane_results = lane_model(frame, verbose=False, imgsz=320)[0]

        # 횡단보도 감지
        for i, cls in enumerate(lane_results.boxes.cls):
            if int(cls) == CROSSWALK_CLASS:
                if float(lane_results.boxes.conf[i]) >= CROSSWALK_CONF_THR:
                    with _lane_lock:
                        _crosswalk_detected = True
                    break

        if lane_results.masks is None:
            _dev_count = 0
            with _lane_lock:
                _lane_error     = None
                _lane_raw_error = None
                _lane_status    = "검출 안됨"
            continue

        combined = np.zeros((h, w), dtype=np.float32)
        for i, cls in enumerate(lane_results.boxes.cls):
            if int(cls) == INLINE_CLASS:
                combined = np.maximum(combined, cv2.resize(lane_results.masks.data[i].cpu().numpy(), (w, h)))

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
            _dev_count = 0
            with _lane_lock:
                _lane_error     = None
                _lane_raw_error = error
                _lane_status    = f"정상 (err={error:+d}px)"
        else:
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

# ── 방향 화살표 유틸 ──────────────────────────────────────────────
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
paused               = False
lane_guard           = True
addr                 = None
disp                 = None
last_motor           = '0,0'
last_key             = 'stop'
last_conf            = 0.0
lane_status          = "대기 중"
last_correction_t    = 0.0
last_display_t       = 0.0   # 마지막 화면 업데이트 시각 (display throttle용)
disp_raw_error       = None
crosswalk_stop_until     = 0.0
crosswalk_cooldown_until = 0.0

# 장애물 회피 상태
avoid_stop_until      = 0.0    # 이 시각까지 감지 직후 정지 (브레이킹)
avoid_end_t           = 0.0    # 이 시각까지 회피 조향 유지
avoid_fwd_end_t       = 0.0    # 이 시각까지 직진 (회피 후 전진)
avoid_ret_end_t       = 0.0    # 이 시각까지 복귀 조향 (반대 방향)
avoid_post_fwd_end_t  = 0.0    # 이 시각까지 복귀 후 직진
avoid_dir             = None   # 'a' or 'd'
avoid_ret_dir         = None   # 복귀 방향 ('a'→'d', 'd'→'a')
object_cooldown_until = 0.0    # 이 시각까지 재감지 무시

print("자율주행 시작 | SPACE=일시정지 | L=차선안전장치토글 | ESC=종료\n")

while True:
    got_frame = False
    frame     = None

    # 쌓인 UDP 패킷을 모두 드레인 → 가장 최신 프레임만 사용
    latest_data = None
    latest_addr = None
    sock.settimeout(0)          # 논블로킹으로 전환
    while True:
        try:
            d, a = sock.recvfrom(1_000_000)
            latest_data = d
            latest_addr = a
        except (socket.timeout, BlockingIOError):
            break
    sock.settimeout(0.02)       # 다시 원래 타임아웃으로

    if latest_data is None:
        # 버퍼가 비어있으면 최대 20ms 대기
        try:
            latest_data, latest_addr = sock.recvfrom(1_000_000)
        except socket.timeout:
            pass

    if latest_data is not None and len(latest_data) >= 4:
        addr     = latest_addr
        size     = int.from_bytes(latest_data[:4], 'little')
        img_data = latest_data[4:4 + size]
        if len(img_data) == size:
            arr   = np.frombuffer(img_data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                if MIRROR_LR:
                    frame = cv2.flip(frame, 1)
                got_frame = True

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
        # 1) 모방학습 추론
        last_key, last_conf = predict_imitation(frame)
        motor = to_motor_str(last_key, SPEED_SCALE)

        if lane_guard:
            # 2) 스레드에 프레임 전달
            with _lane_lock:
                _lane_input = frame.copy()

            # 3) 스레드 결과 읽기
            with _lane_lock:
                error          = _lane_error
                raw_error      = _lane_raw_error
                lane_status    = _lane_status
                cw_det         = _crosswalk_detected
                obj_det        = _object_detected
                obj_side       = _object_side
                if cw_det:
                    _crosswalk_detected = False
                if obj_det:
                    _object_detected = False

            disp_raw_error = raw_error
            now = time.time()

            # ── 횡단보도 감지 ──────────────────────────────────────
            if cw_det and now > crosswalk_cooldown_until:
                crosswalk_stop_until     = now + CROSSWALK_STOP_SEC
                crosswalk_cooldown_until = now + CROSSWALK_STOP_SEC + CROSSWALK_COOLDOWN_SEC
                print(f"[횡단보도] 감지! {CROSSWALK_STOP_SEC:.0f}초 정지 → 이후 {CROSSWALK_COOLDOWN_SEC:.0f}초 쿨다운")

            # ── 장애물 감지 → 정지 후 회피 시작 ──────────────────────
            if obj_det and now > object_cooldown_until and now > avoid_ret_end_t:
                avoid_dir     = 'd' if obj_side == 'left' else 'a'
                avoid_ret_dir = 'a' if avoid_dir == 'd' else 'd'

                # 복귀 방향에 따라 복귀 조향 시간 선택
                ret_sec = OBJECT_RETURN_SEC_D if avoid_ret_dir == 'd' else OBJECT_RETURN_SEC_A

                avoid_stop_until      = now + OBJECT_STOP_SEC
                avoid_end_t           = now + OBJECT_STOP_SEC + OBJECT_STEER_SEC
                avoid_fwd_end_t       = now + OBJECT_STOP_SEC + OBJECT_STEER_SEC + OBJECT_FORWARD_SEC
                avoid_ret_end_t       = now + OBJECT_STOP_SEC + OBJECT_STEER_SEC + OBJECT_FORWARD_SEC + ret_sec
                avoid_post_fwd_end_t  = now + OBJECT_STOP_SEC + OBJECT_STEER_SEC + OBJECT_FORWARD_SEC + ret_sec + OBJECT_POST_FWD_SEC
                object_cooldown_until = now + OBJECT_STOP_SEC + OBJECT_STEER_SEC + OBJECT_FORWARD_SEC + ret_sec + OBJECT_POST_FWD_SEC + OBJECT_COOLDOWN_SEC
                print(f"[장애물] {obj_side}쪽 → 정지 → '{avoid_dir}' → 직진 → '{avoid_ret_dir}'(복귀)")

            # ── 우선순위에 따른 모터 명령 결정 ────────────────────
            if now < crosswalk_stop_until:
                # 최우선: 횡단보도 정지
                motor = '0,0'
            elif now < avoid_stop_until:
                # 2순위-a: 장애물 감지 직후 브레이킹 정지
                motor = '0,0'
            elif now < avoid_end_t:
                # 2순위-b: 장애물 회피 조향
                avoid_scale = OBJECT_AVOID_SCALE_D if avoid_dir == 'd' else OBJECT_AVOID_SCALE_A
                motor = to_motor_str(avoid_dir, avoid_scale)
            elif now < avoid_fwd_end_t:
                # 2순위-c: 회피 후 직진
                motor = to_motor_str('w', SPEED_SCALE)
            elif now < avoid_ret_end_t:
                # 2순위-d: 직진 후 복귀 조향 (반대 방향)
                ret_scale = OBJECT_AVOID_SCALE_D if avoid_ret_dir == 'd' else OBJECT_AVOID_SCALE_A
                motor = to_motor_str(avoid_ret_dir, ret_scale)
            elif now < avoid_post_fwd_end_t:
                # 2순위-e: 복귀 조향 후 직진
                motor = to_motor_str('w', SPEED_SCALE)
            elif error is not None and (now - last_correction_t) > CORRECTION_COOLDOWN:
                # 3순위: 차선 이탈 보정
                if error > 0:
                    motor = to_motor_str('wd', SPEED_SCALE)
                else:
                    motor = to_motor_str('wa', SPEED_SCALE)
                last_correction_t = now
            # 4순위: 모방학습 (위에서 이미 motor에 설정됨)

        else:
            lane_status = "안전장치 OFF"

        last_motor = motor

    motor = '0,0' if paused else last_motor

    if got_frame and frame is not None:
        disp = frame.copy()

    # 화면 업데이트는 DISPLAY_FPS 주기로만 실행 (모터 전송과 무관)
    now_d = time.time()
    do_display = (now_d - last_display_t) >= (1.0 / DISPLAY_FPS)

    if do_display and disp is not None:
        last_display_t = now_d
        h_img, w_img   = disp.shape[:2]

        status     = "[PAUSE]" if paused else "[AUTO]"
        status_col = (0, 165, 255) if paused else (0, 255, 0)
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

        # ── 차선 보정 시각화 ──────────────────────────────────────
        is_correcting = (now_d - last_correction_t) < 0.4
        if is_correcting and disp_raw_error is not None:
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
            arrow_cx = w_img // 2
            arrow_cy = h_img // 2 + 30
            corr_dir = 'wd' if disp_raw_error > 0 else 'wa'
            draw_arrow(disp, corr_dir, arrow_cx, arrow_cy, 60, (30, 30, 255), 5)
            cv2.putText(disp, "LANE OVERRIDE",
                        (arrow_cx - 80, arrow_cy + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 255), 2)

        # ── 횡단보도 오버레이 ─────────────────────────────────────
        if now_d < crosswalk_stop_until:
            remain = crosswalk_stop_until - now_d
            ov_y1, ov_y2 = h_img // 2 - 45, h_img // 2 + 45
            cv2.rectangle(disp, (0, ov_y1), (w_img, ov_y2), (0, 0, 160), -1)
            cv2.rectangle(disp, (0, ov_y1), (w_img, ov_y2), (0, 0, 255), 3)
            cv2.putText(disp, f"CROSSWALK  STOP  {remain:.1f}s",
                        (w_img // 2 - 185, h_img // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        elif crosswalk_cooldown_until > now_d:
            remain_cd = int(crosswalk_cooldown_until - now_d)
            cv2.putText(disp, f"CW cooldown: {remain_cd}s",
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 170, 255), 1)

        # ── 장애물 회피 오버레이 ──────────────────────────────────
        if now_d < avoid_stop_until:
            remain_st = avoid_stop_until - now_d
            ov_y1, ov_y2 = h_img // 2 - 45, h_img // 2 + 45
            cv2.rectangle(disp, (0, ov_y1), (w_img, ov_y2), (0, 0, 140), -1)
            cv2.rectangle(disp, (0, ov_y1), (w_img, ov_y2), (0, 100, 255), 3)
            cv2.putText(disp, f"OBJECT  BRAKING  {remain_st:.1f}s",
                        (w_img // 2 - 185, h_img // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255), 2)
        elif now_d < avoid_end_t:
            remain_av = avoid_end_t - now_d
            ov_y1, ov_y2 = h_img // 2 - 45, h_img // 2 + 45
            cv2.rectangle(disp, (0, ov_y1), (w_img, ov_y2), (0, 100, 0), -1)
            cv2.rectangle(disp, (0, ov_y1), (w_img, ov_y2), (0, 255, 0), 3)
            dir_label = "RIGHT" if avoid_dir == 'd' else "LEFT"
            cv2.putText(disp, f"OBJECT  AVOID {dir_label}  {remain_av:.1f}s",
                        (w_img // 2 - 200, h_img // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255), 2)
        elif now_d < avoid_fwd_end_t:
            remain_fwd = avoid_fwd_end_t - now_d
            ov_y1, ov_y2 = h_img // 2 - 45, h_img // 2 + 45
            cv2.rectangle(disp, (0, ov_y1), (w_img, ov_y2), (100, 60, 0), -1)
            cv2.rectangle(disp, (0, ov_y1), (w_img, ov_y2), (0, 180, 255), 3)
            cv2.putText(disp, f"OBJECT  FORWARD  {remain_fwd:.1f}s",
                        (w_img // 2 - 185, h_img // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255), 2)
        elif now_d < avoid_ret_end_t:
            remain_ret = avoid_ret_end_t - now_d
            ov_y1, ov_y2 = h_img // 2 - 45, h_img // 2 + 45
            cv2.rectangle(disp, (0, ov_y1), (w_img, ov_y2), (80, 0, 80), -1)
            cv2.rectangle(disp, (0, ov_y1), (w_img, ov_y2), (255, 100, 255), 3)
            ret_label = "RIGHT" if avoid_ret_dir == 'd' else "LEFT"
            cv2.putText(disp, f"OBJECT  RETURN {ret_label}  {remain_ret:.1f}s",
                        (w_img // 2 - 205, h_img // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255), 2)
        elif now_d < avoid_post_fwd_end_t:
            remain_pfwd = avoid_post_fwd_end_t - now_d
            ov_y1, ov_y2 = h_img // 2 - 45, h_img // 2 + 45
            cv2.rectangle(disp, (0, ov_y1), (w_img, ov_y2), (100, 60, 0), -1)
            cv2.rectangle(disp, (0, ov_y1), (w_img, ov_y2), (0, 180, 255), 3)
            cv2.putText(disp, f"OBJECT  POST-FWD  {remain_pfwd:.1f}s",
                        (w_img // 2 - 185, h_img // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255), 2)
        elif object_cooldown_until > now_d:
            remain_oc = int(object_cooldown_until - now_d)
            cv2.putText(disp, f"OBJ cooldown: {remain_oc}s",
                        (10, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)

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
