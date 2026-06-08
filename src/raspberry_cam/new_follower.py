import socket
import threading
import time

import cv2
from gpiozero import Motor
from picamera2 import Picamera2


SERVER_IP = "192.168.0.65"
SERVER_PORT = 5002
LOCAL_PORT = 5001
MAX_UDP_PACKET = 65507
HEADER_SIZE = 4
MAX_JPEG_BYTES = MAX_UDP_PACKET - HEADER_SIZE
JPEG_QUALITY = 35


motor_a = Motor(forward=17, backward=18)
motor_b = Motor(forward=27, backward=22)

print("camera initializing...")
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(2)

_img = picam2.capture_array()


def capture_thread():
    global _img
    while True:
        _img = picam2.capture_array()


def apply_motor_command(motor_cmd: str):
    try:
        left_str, right_str = motor_cmd.split(",")
        left = int(left_str.strip())
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


print("capture thread start")
threading.Thread(target=capture_thread, daemon=True).start()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", LOCAL_PORT))
sock.setblocking(False)

server_addr = (SERVER_IP, SERVER_PORT)
print(f"udp client ready, server={SERVER_IP}:{SERVER_PORT}")
last_send_error_ts = 0.0

try:
    while True:
        src = cv2.flip(_img, 0)
        ok, img_encoded = cv2.imencode(
            ".jpg",
            src,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not ok:
            continue

        img_bytes = img_encoded.tobytes()
        if len(img_bytes) > MAX_JPEG_BYTES:
            # If a frame occasionally exceeds UDP limit, skip that frame.
            continue

        size_bytes = len(img_bytes).to_bytes(4, "little")

        try:
            sock.sendto(size_bytes + img_bytes, server_addr)
        except (BlockingIOError, TimeoutError):
            # Drop frame if socket is temporarily busy.
            continue
        except OSError as e:
            now = time.time()
            if now - last_send_error_ts > 1.0:
                print(f"send warning: {e}")
                last_send_error_ts = now
            continue

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
    print("stopping...")
finally:
    motor_a.stop()
    motor_b.stop()
    sock.close()
