import socket
import keyboard
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 3초 동안 데이터가 오지 않으면 대기를 풀고 예외를 발생시킵니다.
sock.settimeout(3.0)
print('소켓 생성')
sock.bind(('0.0.0.0', 5002))

while 1:
    print("메시지 대기 중 ...")
    try:
        data, addr = sock.recvfrom(1024) 
        msg = data.decode().strip()
        print('발신지 : ', addr)
        print('받은 내용 : ', msg)

        # 응답 (a d s w)
        if keyboard.is_pressed('w'):    # 전진
            msg = "80,80"
        elif keyboard.is_pressed('s'):  # 후진
            msg = "-80,-80"
        elif keyboard.is_pressed('a'):  # 좌회전
            msg = "0,80"
        elif keyboard.is_pressed('d'):  # 우회전
            msg = "80,0"
        else:                           # 정지
            msg = "0,0"
            
        sock.sendto(msg.encode(), addr)
        
        print("PC -> 라즈베리파이 응답 전송 완료")
        
    except socket.timeout:
        # PC가 라즈베리파이의 메시지를 받지 못하는 상황입니다.
        print("[문제 확인] 라즈베리파이로부터 데이터가 오지 않아 시간 초과되었습니다.")
    except Exception as e:
        print(f"[기타 에러] {e}")

    time.sleep(0.1)