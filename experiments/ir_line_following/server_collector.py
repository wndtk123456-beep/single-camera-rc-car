import socket
import keyboard
import json
import time

# UDP 연결 상태를 저장하지 않는다.
# 주소 적어서 메시지 보내면 끝.
# 소켓 생성
# 메시지 받기 or 보내기(항상 주소를 적어준다)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print('소켓 생성')
sock.bind(('0.0.0.0', 5002))
# 송수신이 다 되는 만능 소켓 (udp라서)
# 보낼 때는 그때 그때 addr를 지정해준다
# 받을 때는 5002번으로 받는다.


data = []
sensor, addr = sock.recvfrom(1024) 
sensor = sensor.decode().strip() #strip -> 공백이나 개행 없애기
sock.sendto("0,0".encode(), addr)
print('받은 내용', sensor)

while 1:
    # 데이터 받아오기
    
    print('받은 내용 : ', sensor)

    # 응답
    msg = None
    while msg is not None:
        if keyboard.is_pressed('a'):
            msg = "0,60"
            data.append((sensor, msg))
        elif keyboard.is_pressed('d'):
            msg = "60,0"
            data.append((sensor, msg))
        elif keyboard.is_pressed('w'):
            msg = "60,60"
            data.append((sensor, msg))
        #응급상황 제조기
        elif keyboard.is_pressed('j'):
            msg = "0,60"
            # data.append((sensor, msg))
        elif keyboard.is_pressed('l'):
            msg = "60,0"
            # data.append((sensor, msg))
        elif keyboard.is_pressed('i'):
            msg = "60,60"
            # data.append((sensor, msg))
        # 저장
        elif keyboard.is_pressed('1'):
            msg = "0,0"
            json.dump(data, open("ir_data.json", "w"), indent=1)
            print("saved.")

        else:
            pass
        
    if msg:
        begin = time.time()
        while 1:
            sensor, addr = sock.recvfrom(1024)
            sock.sendto(msg.encode(), addr)
            end = time.time()
            if end-begin > 0.1:
                break
        for _ in range(10):
            sensor, addr = sock.recvfrom(1024)
            sock.sendto("0,0".encode(), addr)






