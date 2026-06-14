"""
PC 데이터 수집 서버 (STEP 2: 모방 학습 데이터 수집)

역할:
  - 라즈베리파이 카메라 영상을 UDP로 수신
  - WASD 키로 RC카 조종 (W+A, W+D 동시 입력 지원)
  - 'R' 키로 녹화 시작/정지
  - 저장: collected_data/YYYYMMDD_HHMMSS/
      frame_000001.jpg  ← 카메라 프레임
      frame_000001.txt  ← 모터값(줄1) + 키조합(줄2)

라벨 파일 형식:
  -80,-30    ← left_motor,right_motor  (-100~100)
  wd         ← 키 조합 (w/s/a/d/wa/wd/sa/sd)

키 조작:
  W     - 전진        S     - 후진
  A     - 좌회전(제자리)  D     - 우회전(제자리)
  W+A   - 전진+좌     W+D   - 전진+우
  S+A   - 후진+좌     S+D   - 후진+우
  R     - 녹화 시작/정지
  ESC   - 종료
"""

import socket
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# ── 설정 ──────────────────────────────────────────────────────
LISTEN_PORT  = 5002
KEY_HOLD_SEC = 0.15          # 키 홀드 시간 (동시 입력 인식 여유 포함)
MIRROR_LR    = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVE_ROOT    = PROJECT_ROOT / "data" / "drive_dataset"
# ─────────────────────────────────────────────────────────────

# 키 코드
W = ord('w')
S = ord('s')
A = ord('a')
D = ord('d')
WASD = {W, S, A, D}

# 키 조합 → (left_motor, right_motor)
# 단독 키
COMBO_MOTOR = {
    frozenset([W])   : ((-80, -80), 'w'),    # 전진
    frozenset([S])   : (( 80,  80), 's'),    # 후진
    frozenset([A])   : (( 20, -100), 'a'),   # 좌회전 (제자리)
    frozenset([D])   : ((-100,  20), 'd'),   # 우회전 (제자리)
    # 전진+방향 (좁은 트랙 코너 대응)
    frozenset([W, A]): ((-80, -20), 'wa'),   # 전진+좌 (좌 느리게)
    frozenset([W, D]): ((-20, -80), 'wd'),   # 전진+우 (우 느리게)
    # 후진+방향
    frozenset([S, A]): (( 80,  20), 'sa'),   # 후진+좌
    frozenset([S, D]): (( 20,  80), 'sd'),   # 후진+우
}


def keys_to_motor(active_keys: set):
    """현재 활성 키 집합 → (motor_str, key_name)"""
    combo = frozenset(active_keys & WASD)
    if combo in COMBO_MOTOR:
        (left, right), name = COMBO_MOTOR[combo]
        return f'{left},{right}', name
    return '0,0', 'stop'


# ── 소켓 초기화 ───────────────────────────────────────────────
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', LISTEN_PORT))
sock.settimeout(0.02)
print(f'서버 대기 중... (포트 {LISTEN_PORT})')

# ── 상태 변수 ─────────────────────────────────────────────────
img        = None
addr       = None
recording  = False
save_dir   = None
frame_count = 0

# 키별 독립 만료 시각 (동시 입력 추적)
key_until = {W: 0.0, S: 0.0, A: 0.0, D: 0.0}


def start_recording():
    global recording, save_dir, frame_count
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(SAVE_ROOT) / ts
    save_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    recording = True
    print(f'[REC START] 저장 경로: {save_dir}')


def stop_recording():
    global recording
    recording = False
    print(f'[REC STOP] {frame_count}프레임 저장 완료 → {save_dir}')


# ── 메인 루프 ─────────────────────────────────────────────────
while True:
    got_frame = False

    # 프레임 수신
    try:
        data, addr = sock.recvfrom(1_000_000)
        if len(data) < 4:
            continue
        size     = int.from_bytes(data[:4], 'little')
        img_data = data[4:4 + size]
        if len(img_data) != size:
            continue
        img_arr = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        if MIRROR_LR:
            img = cv2.flip(img, 1)
        got_frame = True
    except socket.timeout:
        pass
    except Exception as e:
        print(f'수신 오류: {e}')
        continue

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    now = time.time()

    if key == 27:           # ESC → 종료
        if recording:
            stop_recording()
        break
    elif key == ord('r'):   # R → 녹화 토글
        if not recording:
            start_recording()
        else:
            stop_recording()
    elif key in WASD:       # WASD → 각 키 만료 시각 갱신
        key_until[key] = now + KEY_HOLD_SEC

    # 현재 활성 키 집합으로 모터 명령 결정
    active_keys = {k for k, until in key_until.items() if now <= until}
    motor, cur_key = keys_to_motor(active_keys)

    # 프레임 + 라벨 저장 (녹화 중 + 키가 눌린 프레임만)
    if recording and got_frame and img is not None and cur_key != 'stop':
        frame_count += 1
        name = f'frame_{frame_count:06d}'
        cv2.imwrite(str(save_dir / f'{name}.jpg'), img)
        (save_dir / f'{name}.txt').write_text(f'{motor}\n{cur_key}')

    # 화면 표시
    if img is not None:
        disp = img.copy()
        if recording:
            rec_text = f'[REC] {frame_count} frames'
            color    = (0, 0, 255)
        else:
            rec_text = '[STOP]  R = 녹화 시작'
            color    = (0, 200, 0)
        cv2.putText(disp, rec_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(disp, f'KEY: {cur_key}   MOTOR: {motor}',
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow('RC카 데이터 수집', disp)

    # 라즈베리파이로 모터 명령 전송
    if got_frame and addr:
        sock.sendto(motor.encode('utf-8'), addr)

cv2.destroyAllWindows()
sock.close()
