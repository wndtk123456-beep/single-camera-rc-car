# 라즈베리파이 실행 - 자율 주행
# line_lookup.json 이 같은 폴더에 있어야 합니다

import serial
import time
import threading
import json
from gpiozero import Motor

# ===== 룩업 테이블 로드 =====
with open('line_lookup.json', encoding='utf-8') as f:
    lookup = json.load(f)
print('룩업 테이블 로드 완료')

# ===== 시리얼 (IR 센서) =====
ser = serial.Serial('/dev/serial0', 9600, timeout=0.1)
ser.reset_input_buffer()
ser.reset_output_buffer()

latest_ir = [1, 1, 1, 1, 1]

def ir_reader():
    global latest_ir
    while True:
        ser.write('\n'.encode())
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if len(line) == 5 and all(c in '01' for c in line):
            latest_ir = [int(c) for c in line]
        time.sleep(0.01)

threading.Thread(target=ir_reader, daemon=True).start()

# ===== 모터 설정 =====
motor_a = Motor(forward=18, backward=17)
motor_b = Motor(forward=22, backward=27)

def control_motor(motor, value):
    if value > 0:
        motor.forward(value / 100)
    elif value < 0:
        motor.backward(abs(value) / 100)
    else:
        motor.stop()

# 액션 → 모터 명령 (속도 조절: 0~100)
SPEED = 60  # 전체 속도 조절 (너무 빠르면 낮추세요)

ACTION_MAP = {
    'w': (-SPEED,     -SPEED),      # 전진
    's': ( SPEED,      SPEED),      # 후진
    'a': (-SPEED,  SPEED // 2),     # 우회전
    'd': ( SPEED // 2, -SPEED),     # 좌회전
}

print('=== 자율 주행 시작 ===')
print('Ctrl+C 로 종료')
time.sleep(1)  # IR 센서 안정화 대기

def line_position(ir):
    """라인을 감지한 센서들의 평균 위치 반환 (0=맨왼쪽 ~ 4=맨오른쪽, None=미감지)"""
    detecting = [i for i, v in enumerate(ir) if v == 0]
    if not detecting:
        return None
    return sum(detecting) / len(detecting)

try:
    while True:
        ir = latest_ir.copy()

        pos = line_position(ir)

        if pos is None:
            # 라인 미감지 → 정지
            action = 'stop'
        elif pos < 1.5:
            # 라인이 왼쪽 끝에 있음 → 왼쪽으로 틀어서 중앙으로
            action = 'd'
        elif pos > 2.5:
            # 라인이 오른쪽 끝에 있음 → 오른쪽으로 틀어서 중앙으로
            action = 'a'
        else:
            # 라인이 중앙 근처 → 모델 예측 사용
            action = lookup.get(str(ir), 'w')

        left, right = ACTION_MAP.get(action, (0, 0))
        control_motor(motor_a, left)
        control_motor(motor_b, right)

        print(f'\rIR: {ir}  pos: {pos}  예측: {action}  ', end='', flush=True)

        time.sleep(0.02)  # 50Hz

except KeyboardInterrupt:
    print('\n종료')

finally:
    motor_a.stop()
    motor_b.stop()
    ser.close()
