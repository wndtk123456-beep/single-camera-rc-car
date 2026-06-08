import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print('소켓 생성')
sock.bind(('0.0.0.0', 5002))

last_msg = "0,0"

p = 45       # 직진 속도 (코너 대응을 위해 낮춤)
slow = 30    # 완만한 회전
turn = 45    # 급회전 바깥쪽 바퀴
reverse = -30  # 급회전 안쪽 바퀴 역회전 ← 핵심

while 1:
    data, addr = sock.recvfrom(1024)
    sensor = data.decode().strip()

    sensor = ''.join(reversed(sensor))
    print(' '.join(sensor))

    if sensor == "00000":
        msg = last_msg                      # 라인 잃으면 마지막 명령 유지

    elif sensor == "00001":
        msg = f"{reverse},{turn}"           # 좌회전 (안쪽 역회전으로 급선회)

    elif sensor in ("00011", "00110"):
        msg = f"{slow},{p}"                 # 조금 좌회전

    elif sensor == "00100":
        msg = f"{p},{p}"                    # 직진

    elif sensor in ("01000", "01100"):
        msg = f"{p},{slow}"                 # 조금 우회전

    elif sensor == "10000":
        msg = f"{turn},{reverse}"           # 우회전 (안쪽 역회전으로 급선회)

    elif sensor == "11111":
        msg = f"{slow},{slow}"              # 저속 직진

    else:
        msg = f"{p},{p}"

    last_msg = msg
    print("전송:", msg)
    sock.sendto(msg.encode(), addr)