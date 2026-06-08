import socket
import cv2
import numpy as np
import os

import yolo_detector
yolo_detector.load_model('last.onnx')

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 5002))
sock.settimeout(0.1)

print("서버 대기 중...")

os.makedirs('saved', exist_ok=True)

num = 0
motor = '0,0'
addr = None
img = None

while True:
    try:
        data, addr = sock.recvfrom(1000000)

        if len(data) < 4:
            print('수신 데이터가 너무 짧음')
            continue

        size = int.from_bytes(data[:4], 'little')
        img_data = data[4:4+size]

        if len(img_data) != size:
            print('이미지 데이터 길이가 일치하지 않음')
            continue

        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            print('이미지 디코딩 실패')
            continue

        # YOLO 추론
        result = yolo_detector.inference(img)
        yolo_detector.draw_boxes(img, result)

        # 자동 추적 로직
        res_filtered = [r for r in result 
                if r[4] > 0.5
                and (r[2]-r[0]) < img.shape[1] * 0.8  # 너비가 화면의 80% 미만
                and (r[3]-r[1]) < img.shape[0] * 0.8] # 높이가 화면의 80% 미만

        if len(res_filtered) > 0:
            best = max(res_filtered, key=lambda r: r[4])
            x1, y1, x2, y2, conf, cat = best

            img_center = img.shape[1] / 2
            box_center = (x1 + x2) / 2

            if box_center < img_center - 80:
                motor = '0,60'
            elif box_center > img_center + 80:
                motor = '60,0'
            else:
                motor = '60,60'
        else:
            motor = '0,0'

        cv2.imshow('Received Image', img)

    except socket.timeout:
        pass
    except Exception as e:
        print(f'Error: {e}')
        continue

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == ord('1'):
        if img is not None:
            cv2.imwrite(f'saved/image{num}.jpg', img)
            print(f'저장: saved/image{num}.jpg')
            num += 1

    if addr:
        sock.sendto(motor.encode('utf-8'), addr)

cv2.destroyAllWindows()
sock.close()
