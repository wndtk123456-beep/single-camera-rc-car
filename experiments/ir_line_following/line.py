import serial
from gpiozero import Motor
from time import sleep

motor_a = Motor(forward=17, backward=18)
motor_b = Motor(forward=27, backward=22)

ser = serial.Serial('/dev/serial0', 9600, timeout=0.1)
ser.reset_input_buffer()

def control_motor(motor, value):
    speed = min(abs(value) / 100.0, 1.0)

    if value > 0:
        motor.forward(speed)
    elif value < 0:
        motor.backward(speed)
    else:
        motor.stop()

def set_motors(left, right):
    control_motor(motor_a, left)
    control_motor(motor_b, right)

last_dir = "C"

try:
    print("라인트레이서 시작")

    while True:
        pattern = ser.readline().decode("utf-8", errors="ignore").strip()

        if len(pattern) != 5:
            continue

        print("pattern =", pattern)

        if pattern == "00100":
            set_motors(35, 35)
            last_dir = "C"

        elif pattern in ["01100", "01000"]:
            set_motors(15, 35)
            last_dir = "L"

        elif pattern in ["11000", "10000"]:
            set_motors(0, 35)
            last_dir = "L"

        elif pattern in ["00110", "00010"]:
            set_motors(35, 15)
            last_dir = "R"

        elif pattern in ["00011", "00001"]:
            set_motors(35, 0)
            last_dir = "R"

        elif pattern in ["01110", "11111", "11100", "00111"]:
            set_motors(25, 25)

        elif pattern == "00000":
            if last_dir == "L":
                set_motors(-10, 25)
            elif last_dir == "R":
                set_motors(25, -10)
            else:
                set_motors(0, 0)

        else:
            set_motors(20, 20)

        sleep(0.01)

except KeyboardInterrupt:
    print("종료")

finally:
    motor_a.stop()
    motor_b.stop()
    ser.close()