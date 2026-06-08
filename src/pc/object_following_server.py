import os
import socket
from pathlib import Path

import cv2
import numpy as np

import yolo_detector


MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "last.onnx"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5002
CONFIDENCE_THRESHOLD = 0.5
CENTER_MARGIN_PX = 80
MAX_BOX_RATIO = 0.8


yolo_detector.load_model(str(MODEL_PATH))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SERVER_HOST, SERVER_PORT))
sock.settimeout(0.1)

print(f"PC inference server waiting on {SERVER_HOST}:{SERVER_PORT}")
os.makedirs("saved", exist_ok=True)

num = 0
motor = "0,0"
addr = None
img = None

try:
    while True:
        try:
            data, addr = sock.recvfrom(1000000)

            if len(data) < 4:
                print("Received packet is too short.")
                continue

            size = int.from_bytes(data[:4], "little")
            img_data = data[4:4 + size]

            if len(img_data) != size:
                print("Image payload size mismatch.")
                continue

            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                print("Image decode failed.")
                continue

            result = yolo_detector.inference(img)
            yolo_detector.draw_boxes(img, result)

            # Ignore boxes covering most of the frame because they tend to be
            # unstable for simple steering decisions.
            res_filtered = [
                r for r in result
                if r[4] > CONFIDENCE_THRESHOLD
                and (r[2] - r[0]) < img.shape[1] * MAX_BOX_RATIO
                and (r[3] - r[1]) < img.shape[0] * MAX_BOX_RATIO
            ]

            if res_filtered:
                best = max(res_filtered, key=lambda r: r[4])
                x1, y1, x2, y2, conf, cat = best

                img_center = img.shape[1] / 2
                box_center = (x1 + x2) / 2

                if box_center < img_center - CENTER_MARGIN_PX:
                    motor = "0,60"
                elif box_center > img_center + CENTER_MARGIN_PX:
                    motor = "60,0"
                else:
                    motor = "60,60"
            else:
                motor = "0,0"

            cv2.imshow("Received Image", img)

        except socket.timeout:
            pass
        except Exception as e:
            print(f"Error: {e}")
            continue

        key = cv2.waitKey(1)

        if key == 27:
            break
        elif key == ord("1") and img is not None:
            cv2.imwrite(f"saved/image{num}.jpg", img)
            print(f"saved: saved/image{num}.jpg")
            num += 1

        if addr:
            sock.sendto(motor.encode("utf-8"), addr)
finally:
    cv2.destroyAllWindows()
    sock.close()
