# 라즈베리파이 실행 - 데이터 수집 모드
# module_iris.py + module_motor.py 구조 참고

import serial
import socket
import time
import threading
from gpiozero import Motor

# ===== 시리얼 설정 (module_iris.py 와 동일) =====
ser = serial.Serial('/dev/serial0', 9600, timeout=0.1)
ser.reset_input_buffer()
ser.reset_output_buffer()

latest_ir = "11111"  # 기본값

def ir_reader():
    """IR 센서값을 시리얼로 읽어 latest_ir 갱신"""
    global latest_ir
    while True:
        ser.write('\n'.encode())
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if len(line) == 5 and all(c in '01' for c in line):
            latest_ir = line
        time.sleep(0.01)

threading.Thread(target=ir_reader, daemon=True).start()

# ===== 모터 설정 (pi_line.py 와 동일) =====
motor_a = Motor(forward=18, backward=17)
motor_b = Motor(forward=22, backward=27)

def control_motor(motor, value):
    """value: -100 ~ 100"""
    if value > 0:
        motor.forward(value / 100)
    elif value < 0:
        motor.backward(abs(value) / 100)
    else:
        motor.stop()

# ===== UDP 설정 (module_motor.py 와 동일) =====
PC_IP   = '192.168.0.65'
PC_PORT = 5002

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 5001))
sock.settimeout(0.5)

print('=== 데이터 수집 모드 (라즈베리파이) ===')
print(f'PC({PC_IP}:{PC_PORT})로 전송 시작')
print('Ctrl+C 로 종료')

try:
    while True:
        # IR 데이터를 PC로 전송
        sock.sendto((latest_ir + '\n').encode(), (PC_IP, PC_PORT))

        # PC에서 모터 명령 수신
        try:
            data, _ = sock.recvfrom(1024)
            cmd = data.decode().strip()
            left, right = cmd.split(',')
            left, right = int(left), int(right)
            control_motor(motor_a, left)
            control_motor(motor_b, right)
            print(f'\r명령: 좌={left:4d} 우={right:4d}  IR: {latest_ir}  ', end='', flush=True)
        except socket.timeout:
            motor_a.stop()
            motor_b.stop()

        time.sleep(0.02)  # 50Hz

except KeyboardInterrupt:
    print('\n종료')

finally:
    motor_a.stop()
    motor_b.stop()
    sock.close()
    ser.close()