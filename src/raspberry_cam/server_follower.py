import socket
import cv2
import numpy as np

import yolo_detector
yolo_detector.load_model('carcan2.onnx')

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 5002))
sock.settimeout(1.0)

num = 0
while True:
    try:
        header, addr = sock.recvfrom(1000000)
        size = int.from_bytes(header[:4], 'little')
        print(size)

        img_array = np.frombuffer(header[4:4+size], dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        result = yolo_detector.inference(img)
        motor = "0,0"
        #바닥과의 거리를 이용해서 거리를 잰다.
        #거리가 가장 가까운 자동차 하나를 찾는다 (target)
        boxes = []
        scaled = result.copy()
        scaled[:, :4] = scaled[:, 0:4] * max(img.shape[:2]) / 640
    
        for x1, y1, x2, y2, conf, cat in scaled:
            if conf < 0.2:
                continue
            boxes.append((x1, y1, x2, y2))

        if len(boxes) > 0:
            sorted_boxes = sorted(boxes, key=lambda b: b[3], reverse=True)
            target_box = sorted_boxes[0]
    
            #target이 왼쪽 가운데 오른쪽에 있는지 확인
            x1, y1, x2, y2 = target_box
            center_x = (x1 + x2) / 2
            img_center_x = img.shape[1] / 2

            #거리에 따라서 전진하거나 멈춤
            dist = img.shape[0] - y2
            print(len(boxes), y2, dist, img.shape[0])
            if dist > img.shape[0] * 0.2:
                #이미지의 중앙으로부터 30%를 가운데 영역으로
                if center_x < img_center_x * 0.7:
                    motor = "0,80" 
                elif center_x > img_center_x * 1.3:
                    motor = "80,0" 
                else:
                    motor = "80,80"  # 전진
            else:
                motor = "0,0"  # 멈춤
            

        #왼쪽 오른쪽이면 - 좌우로 비틀기

        #가운데면 - 거리 맞추기 (가까워질 때까지 전진)

        dst = img.copy()
        yolo_detector.draw_boxes(dst, result)
        cv2.imshow('Received Image', dst)
        
        key = cv2.waitKey(10)

        
        if key == ord('q'):
            break
        elif key == ord('1'):
            cv2.imwrite(f'saved/image{num}.jpg', img)
            num += 1
        elif key == ord('a'):
            motor = "0,60"
        elif key == ord('d'):
            motor = "60,0"
        elif key == ord('w'):
            motor = "60,60"

        sock.sendto(motor.encode(), addr)

    except Exception as e:
        print(f"Error: {e}")
        continue

    



