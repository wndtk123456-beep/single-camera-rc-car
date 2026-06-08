import socket
import keyboard

import numpy as np

import ir_model

# UDP 연결 상태를 저장하지 않는다.
# 주소 적어서 메시지 보내면 끝.
# 소켓 생성
# 메시지 받기 or 보내기(항상 주소를 적어준다)

sock = socket. socket(socket.AF_INET, socket. SOCK_DGRAM)
print('소켓 생성')
sock.bind(('0.0.0.0', 5002))
# 송수신이 다 되는 만능 소켓 (udp라서)
# 보낼 때는 그때 그때 addr를 지정해준다
# 받을 때는 5002번으로 받는다.
while 1:
# 데이터 받아오기
    sensor, addr = sock.recvfrom(1024)
    sensor = sensor.decode().strip() #strip -> 공백이나 개행 없애기
    sensor = ''.join(reversed(sensor))
    print(' '.join(sensor))

    #극단적인 경우부터
    #"11111" - (경로이탈, 선이 없음) 앞으로 가기
    #"00000"' - (손으로 들었을 때) 멈춤

    p = 80
    
    print(sensor, end=" ")
    prob = ir_model.predict(sensor)
    label = np.array(prob) .argmax()

    print(label, prob)

    if sensor == "00000":
        print('선 이탈, 직진 복구')
        msg = f"{p},{p}"
    elif label == 0:
        msg = f"ø, {p}"
    elif label == 1:
        msg = f"{p},0"
    else:
        msg = f"{p}, {p}" 