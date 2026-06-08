import module_iris
import threading
import time
import socket
from gpiozero import Motor

running = True

SERVER_IP = '192.168.0.65'
SERVER_PORT = 5002

# 오른쪽으로만 가면 왼쪽 출력을 줄여야 함
LEFT_TRIM = 0.08
RIGHT_TRIM = 0.00

# 무게 때문에 출발이 안 되면 최소 출력을 보장
MIN_POWER = 0.60

def apply_deadband(v):
    if v == 0:
        return 0.0

    sign = 1 if v > 0 else -1
    power = abs(v) / 100.0

    if power < MIN_POWER:
        power = MIN_POWER
    if power > 1.0:
        power = 1.0

    return sign * power

def apply_trim(lp, rp):
    if lp > 0:
        lp = max(0.0, lp - LEFT_TRIM)
    elif lp < 0:
        lp = min(0.0, lp + LEFT_TRIM)

    if rp > 0:
        rp = max(0.0, rp - RIGHT_TRIM)
    elif rp < 0:
        rp = min(0.0, rp + RIGHT_TRIM)

    return lp, rp

def drive_motor(motor, power):
    if power > 0:
        motor.forward(power)
    elif power < 0:
        motor.backward(-power)
    else:
        motor.stop()

def worker():
    global running

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 5001))
    sock.settimeout(0.1)

    motor_a = Motor(forward=17, backward=18)
    motor_b = Motor(forward=27, backward=22)

    prev_left = 0
    prev_right = 0

    while running:
        try:
            addr = (SERVER_IP, SERVER_PORT)
            msg = module_iris.data + '\n'
            sock.sendto(msg.encode(), addr)

            data, addr = sock.recvfrom(1024)
            msg = data.decode().strip()

            left, right = msg.split(',')
            left = int(left.strip())
            right = int(right.strip())

            lp = apply_deadband(left)
            rp = apply_deadband(right)

            lp, rp = apply_trim(lp, rp)

            # 정지 상태에서 출발할 때 잠깐 더 밀어주기
            if prev_left == 0 and left != 0:
                boost = min(1.0, abs(lp) + 0.10)
                if lp > 0:
                    motor_a.forward(boost)
                else:
                    motor_a.backward(boost)
                time.sleep(0.02)

            if prev_right == 0 and right != 0:
                boost = min(1.0, abs(rp) + 0.10)
                if rp > 0:
                    motor_b.forward(boost)
                else:
                    motor_b.backward(boost)
                time.sleep(0.02)

            drive_motor(motor_a, lp)
            drive_motor(motor_b, rp)

            prev_left = left
            prev_right = right

            print(f"cmd=({left},{right}) power=({lp:.2f},{rp:.2f})")

            time.sleep(0.01)

        except TimeoutError:
            motor_a.stop()
            motor_b.stop()

        except Exception as e:
            motor_a.stop()
            motor_b.stop()
            print("motor error:", e)

    motor_a.stop()
    motor_b.stop()
    sock.close()

if __name__ == '__main__':
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    input('엔터를 누르면 종료.')
    running = False
    th.join()
    print('끝.')