import time
import socket
import serial
from gpiozero import Motor
lines=[]

# stem32
ser = serial.Serial('/dev/serial0', 9600, timeout=0.1)
ser.reset_input_buffer()
ser.reset_output_buffer()


# gpio 직접 제어
motor_a = Motor(forward=17, backward=18)
motor_b = Motor(forward=27, backward=22)


#pc와 통신
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 5001))
sock.settimeout(1.0)

# while
print('while...')
while True:
    ser.write('a\n'.encode())
    line = ser.readline().decode().strip()
    print('센서값:', line)
    lines.append(line)
    while len(lines) > 10:
        lines.pop(0)

    while True:
        try:
            data = "".join(lines[-6:]).encode()
            sock.sendto(data, ("192.168.0.65", 5002))
            data, addr = sock.recvfrom(1024)
            break
        except TimeoutError:
            pass

    left, right = data.decode().split(',')
    left = int(left.strip())
    right = int(right.strip())

    if left < 0:
        motor_a.backward(-left/100)
    else:
        motor_a.forward(left/100)

    if right < 0:
        motor_b.backward(-right/100)
    else:
        motor_b.forward(right/100)

    time.sleep(0.2)

    


# while
#데이터 수집
# pc에 요청
# 응답받고
# 모터 제어
# -> 여기서 0.1초 시간 보냄


