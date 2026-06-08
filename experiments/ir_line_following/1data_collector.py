# PC 실행 - 데이터 수집 모드
# server.py 구조 참고

import socket
import keyboard
import time
import threading
import json
from datetime import datetime

PI_ADDR    = None
latest_ir  = None
data_records = []
running    = True
lock       = threading.Lock()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 5002))
sock.settimeout(0.1)

print("=== 라인 트레이서 데이터 수집 ===")
print("W: 전진 | S: 후진 | A: 우회전 | D: 좌회전 | 없음: 정지 | Q: 저장 후 종료")
print("라즈베리파이 연결 대기 중...")

def receive_thread():
    """Pi에서 IR 데이터 수신 (형식: '11011')"""
    global PI_ADDR, latest_ir
    while running:
        try:
            data, addr = sock.recvfrom(1024)
            msg = data.decode().strip()

            # 5자리 0/1 문자열이면 IR 데이터
            if len(msg) == 5 and all(c in '01' for c in msg):
                if PI_ADDR is None:
                    PI_ADDR = addr
                    print(f'\n라즈베리파이 연결됨: {addr}')
                with lock:
                    latest_ir = [int(c) for c in msg]

        except socket.timeout:
            continue
        except Exception:
            break

threading.Thread(target=receive_thread, daemon=True).start()

try:
    while True:
        if keyboard.is_pressed('q'):
            print("\n종료 및 저장 중...")
            break

        if PI_ADDR is None:
            time.sleep(0.05)
            continue

        # 키보드 입력 감지 및 모터 명령 결정
        if keyboard.is_pressed('s'):
            cmd, action = '100,100', 's'
        elif keyboard.is_pressed('w'):
            cmd, action = '-100,-100', 'w'
        elif keyboard.is_pressed('d'):
            cmd, action = '-100,40', 'd'
        elif keyboard.is_pressed('a'):
            cmd, action = '40,-100', 'a'
        else:
            cmd, action = '0,0', 'stop'

        # 모터 명령 전송
        sock.sendto(cmd.encode(), PI_ADDR)

        # 현재 IR 값과 액션을 페어링해서 저장
        with lock:
            ir_snapshot = latest_ir.copy() if latest_ir is not None else None

        if ir_snapshot is not None:
            data_records.append({'ir': ir_snapshot, 'action': action})
            print(f'\rIR: {ir_snapshot}  액션: {action:5s}  수집: {len(data_records)}건    ',
                  end='', flush=True)

        time.sleep(0.02)  # 50Hz

finally:
    running = False
    sock.close()

    if data_records:
        filename = f'line_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_records, f, ensure_ascii=False, indent=2)
        print(f'\n\n저장 완료: {filename}  ({len(data_records)}건)')
    else:
        print('\n수집된 데이터가 없습니다.')