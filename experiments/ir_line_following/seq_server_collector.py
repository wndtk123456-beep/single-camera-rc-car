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
print('ㅁㄴㅇㄻㄴㅇㄹ')
while 1:

    # 응답
    msg = None
    while msg is None:
        if keyboard.is_pressed('a'):
            while keyboard.is_pressed('a'): pass
            msg = "0,60"
            
        elif keyboard.is_pressed('d'):
            while keyboard.is_pressed('d'): pass
            msg = "60,0"
        elif keyboard.is_pressed('w'):
            while keyboard.is_pressed('w'): pass
            msg = "60,60"
        #응급상황 제조기
        elif keyboard.is_pressed('j'):
            msg = "0,60"

        elif keyboard.is_pressed('l'):
            msg = "60,0"

        elif keyboard.is_pressed('i'):
            msg = "60,60"

        # 저장
        elif keyboard.is_pressed('1'):
            # msg = "0,0"
            json.dump(data, open("seq_ir_data.json", "w"), indent=1)
            print("saved.")
        else:
            pass
    

    # 데이터 받아오기
    sensor, addr = sock.recvfrom(1024) 
    sensor = sensor.decode().strip() #strip -> 공백이나 개행 없애기
    sock.sendto("0,0".encode(), addr)
    
    data.append((sensor, msg))
    print('받은 내용 : ', sensor)
    sock.sendto(msg.encode(), addr)
    