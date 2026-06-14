"""
라즈베리파이 클라이언트 (STEP 2: 모방 학습 데이터 수집)

역할:
  - Picamera2로 영상 캡처 → UDP로 PC 서버에 전송
  - PC 서버로부터 모터 명령 수신 → 모터 제어

모터 배선:
  motor_a (왼쪽): forward=GPIO18, backward=GPIO17
  motor_b (오른쪽): forward=GPIO22, backward=GPIO27

모터 명령 형식: "left,right"  (각각 -100 ~ 100)
  양수 → forward 핀 동작  (속도 = 값/100)
  음수 → backward 핀 동작 (속도 = 절댓값/100)
  0    → 정지

실행: python src/raspberry_pi/camera_motor_client.py
"""

import os
import socket
import threading
import time

import cv2
from gpiozero import Motor
from picamera2 import Picamera2


# ── 네트워크 설정 ──────────────────────────────────────────────
SERVER_IP    = os.getenv("RC_CAR_SERVER_IP", "192.168.0.6")
SERVER_PORT  = 5002
LOCAL_PORT   = 5001
JPEG_QUALITY = 35
MAX_UDP_PACKET  = 65507
HEADER_SIZE     = 4
MAX_JPEG_BYTES  = MAX_UDP_PACKET - HEADER_SIZE
# ─────────────────────────────────────────────────────────────

# ── 모터 초기화 ───────────────────────────────────────────────
motor_a = Motor(forward=18, backward=17)   # 왼쪽
motor_b = Motor(forward=22, backward=27)   # 오른쪽

# ── 카메라 초기화 ─────────────────────────────────────────────
print("카메라 초기화 중...")
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

_img = picam2.capture_array()


def capture_thread():
    global _img
    while True:
        _img = picam2.capture_array()


def apply_motor_command(motor_cmd: str):
    """
    "left,right" 문자열을 파싱해서 모터 제어
    값 범위: -100 ~ 100  (양수=전진방향, 음수=후진방향)
    """
    try:
        left_str, right_str = motor_cmd.split(",")
        left  = int(left_str.strip())
        right = int(right_str.strip())
    except Exception:
        left, right = 0, 0

    if left > 0:
        motor_a.forward(left / 100)
    elif left < 0:
        motor_a.backward(-left / 100)
    else:
        motor_a.stop()

    if right > 0:
        motor_b.forward(right / 100)
    elif right < 0:
        motor_b.backward(-right / 100)
    else:
        motor_b.stop()


# ── 캡처 스레드 + 소켓 초기화 ────────────────────────────────
threading.Thread(target=capture_thread, daemon=True).start()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", LOCAL_PORT))
sock.setblocking(False)

server_addr = (SERVER_IP, SERVER_PORT)
print(f"서버 연결 대기: {SERVER_IP}:{SERVER_PORT}")

last_send_error_ts = 0.0

# ── 메인 루프 ─────────────────────────────────────────────────
try:
    while True:
        # 프레임 인코딩 (수직 반전 적용)
        src = cv2.flip(_img, 0)
        ok, img_encoded = cv2.imencode(
            ".jpg", src,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not ok:
            continue

        img_bytes = img_encoded.tobytes()
        if len(img_bytes) > MAX_JPEG_BYTES:
            continue  # UDP 한계 초과 프레임은 스킵

        # 서버로 프레임 전송 (4바이트 크기 헤더 + JPEG)
        size_bytes = len(img_bytes).to_bytes(4, "little")
        try:
            sock.sendto(size_bytes + img_bytes, server_addr)
        except (BlockingIOError, TimeoutError):
            continue
        except OSError as e:
            now = time.time()
            if now - last_send_error_ts > 1.0:
                print(f"전송 오류: {e}")
                last_send_error_ts = now
            continue

        # 서버로부터 모터 명령 수신 (가장 최신 명령 사용)
        motor_cmd = "0,0"
        while True:
            try:
                data, _ = sock.recvfrom(1024)
                motor_cmd = data.decode("utf-8", errors="ignore")
            except BlockingIOError:
                break

        apply_motor_command(motor_cmd)
        time.sleep(0.005)

except KeyboardInterrupt:
    print("종료 중...")
finally:
    motor_a.stop()
    motor_b.stop()
    sock.close()
