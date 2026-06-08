import os
import socket
import time

import cv2
import numpy as np

USE_DETECTION = False
MIRROR_LEFT_RIGHT = True
KEY_HOLD_SEC = 0.10

if USE_DETECTION:
    import yolo_detector
    yolo_detector.load_model('last.onnx')


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 5002))
sock.settimeout(0.02)

print('server waiting...')

os.makedirs('saved', exist_ok=True)

num = 0
img = None
motor = '0,0'
addr = None
manual_motor = '0,0'
manual_until = 0.0

while True:
    got_frame = False
    try:
        data, addr = sock.recvfrom(1000000)

        if len(data) < 4:
            continue

        size = int.from_bytes(data[:4], 'little')
        img_data = data[4:4 + size]

        if len(img_data) != size:
            continue

        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            continue

        if MIRROR_LEFT_RIGHT:
            img = cv2.flip(img, 1)

        if USE_DETECTION:
            result = yolo_detector.inference(img)
            yolo_detector.draw_boxes(img, result)

        cv2.imshow('Received Image', img)
        got_frame = True

    except socket.timeout:
        pass
    except Exception as e:
        print(f'Error receiving data: {e}')
        continue

    key = cv2.waitKey(1) & 0xFF
    now = time.time()

    if key == 27:
        break
    elif key == ord('1'):
        if img is not None:
            cv2.imwrite(f'saved/image{num}.jpg', img)
            print(f'saved: saved/image{num}.jpg')
            num += 1
    elif key == ord('w'):
        manual_motor = '60,86'
        manual_until = now + KEY_HOLD_SEC
    elif key == ord('s'):
        manual_motor = '-60,-86'
        manual_until = now + KEY_HOLD_SEC
    elif key == ord('a'):
        manual_motor = '0,86'
        manual_until = now + KEY_HOLD_SEC
    elif key == ord('d'):
        manual_motor = '60,0'
        manual_until = now + KEY_HOLD_SEC

    if now <= manual_until:
        motor = manual_motor
    else:
        motor = '0,0'

    if got_frame and addr:
        sock.sendto(motor.encode('utf-8'), addr)

cv2.destroyAllWindows()
sock.close()
