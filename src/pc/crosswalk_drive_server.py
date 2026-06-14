import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
자율주행 - 모방학습 + 차선 안전장치

동작 원리:
  1) 모방학습 모델 (GPU)
     - 매 프레임마다 카메라 영상을 보고 W/A/S/D 키 조합을 예측
     - 예: "전진", "전진+좌회전" 등의 주행 명령 출력

  2) YOLOv8 차선 감지 (CPU, 별도 스레드)
     - 흰색 점선(inline)의 위치를 감시
     - 차선 중앙에서 많이 벗어나면 보정 명령으로 오버라이드(덮어쓰기)
     - 별도 스레드로 실행 → 주행 루프가 느려지지 않음

  3) 횡단보도 감지
     - 횡단보도 발견 시 정해진 시간 동안 정지

키 조작:
  SPACE : 일시정지 / 재개
  L     : 차선 안전장치 ON / OFF 토글
  ESC   : 종료
"""

import socket
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO


# =====================================================================
# 설정값 모음
# 동작을 바꾸고 싶으면 여기서만 수정하면 됩니다
# =====================================================================

# 모델 파일 경로
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
IMITATION_MODEL = PROJECT_ROOT / "models" / "drive_best.pth"        # 모방학습 모델 (WASD 예측)
LANE_MODEL      = PROJECT_ROOT / "models" / "lane_best.pt"          # 차선/횡단보도 감지 모델

# 네트워크 설정
LISTEN_PORT = 5002    # 라즈베리파이로부터 영상을 받는 UDP 포트 번호
MIRROR_LR   = True   # True = 화면 좌우 반전 (카메라 설치 방향에 따라 조절)

# 속도 설정
SPEED_SCALE = 0.8   # 전체 속도 배율 (1.0 = 100%, 0.5 = 절반 속도)

# 차선 안전장치 설정
LANE_THRESHOLD      = 130   # 차선이 중앙에서 이 픽셀 이상 벗어나면 이탈로 판단
MIN_MASK_PIX        = 300   # 차선 마스크 픽셀이 이것보다 적으면 "검출 안됨"으로 처리
INLINE_CLASS        = 0     # 흰색 점선(inline) 클래스 번호
DEVIATION_COUNT     = 3     # 몇 프레임 연속으로 이탈해야 보정을 시작할지 (노이즈 방지)
CORRECTION_COOLDOWN = 2.0   # 보정 후 몇 초 동안은 다시 보정하지 않을지 (과잉 개입 방지)

# 횡단보도 설정
CROSSWALK_CLASS        = 2     # 횡단보도 클래스 번호
CROSSWALK_STOP_SEC     = 3.0   # 횡단보도 발견 시 정지할 시간 (초)
CROSSWALK_COOLDOWN_SEC = 30.0  # 정지 후 이 시간 동안은 다시 감지하지 않음 (중복 감지 방지)
CROSSWALK_CONF_THR     = 0.45  # 이 값 이상 확신할 때만 횡단보도로 인정 (0.0 ~ 1.0)

# GPU 사용 여부 자동 결정 (GPU가 있으면 GPU, 없으면 CPU 사용)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# 모터 명령 테이블
# 키 조합 → (왼쪽 모터값, 오른쪽 모터값)
# 음수 = 전진 방향, 양수 = 후진 방향 (모터 배선에 따라 다름)
# =====================================================================
COMBO_MOTOR = {
    'w' : (-80,  -80),   # 전진
    's' : ( 80,   80),   # 후진
    'a' : ( 20, -100),   # 좌회전 (제자리)
    'd' : (-100,  20),   # 우회전 (제자리)
    'wa': (-80,  -20),   # 전진 + 좌회전
    'wd': (-20,  -80),   # 전진 + 우회전
    'sa': ( 80,   20),   # 후진 + 좌회전
    'sd': ( 20,   80),   # 후진 + 우회전
}


def to_motor_str(key_name, scale=1.0):
    """
    키 이름을 받아서 라즈베리파이로 보낼 모터 명령 문자열을 반환합니다.

    예시:
        to_motor_str('w', 1.0)  →  "-80,-80"
        to_motor_str('a', 0.5)  →  "10,-50"  (절반 속도)
    """
    # 해당 키가 테이블에 없으면 정지 명령 반환
    if key_name not in COMBO_MOTOR:
        return '0,0'

    left_power, right_power = COMBO_MOTOR[key_name]

    # 속도 배율 적용 후 정수로 변환
    left_scaled  = int(left_power  * scale)
    right_scaled = int(right_power * scale)

    # 라즈베리파이가 받을 수 있는 "왼쪽,오른쪽" 형식의 문자열 반환
    result = f'{left_scaled},{right_scaled}'
    return result


# =====================================================================
# 모방학습 모델 로드
# =====================================================================
print(f"모방학습 모델 로드 중: {IMITATION_MODEL}")

# 저장된 체크포인트(학습 결과) 파일 불러오기
checkpoint = torch.load(IMITATION_MODEL, map_location=DEVICE)

# 체크포인트에서 필요한 정보 꺼내기
idx_to_class = checkpoint['idx_to_class']        # 숫자 → 키이름 변환표 (예: {0:'w', 1:'a', ...})
IMG_SIZE     = checkpoint.get('img_size', 224)   # 모델 입력 이미지 크기 (기본값 224)
NUM_CLASSES  = len(idx_to_class)                 # 분류 클래스 수

# MobileNetV3-Small 모델 구조 생성 (ImageNet 학습된 구조)
imitation_model = models.mobilenet_v3_small()

# 마지막 분류 레이어를 우리 클래스 수에 맞게 교체
# (원래는 ImageNet의 1000개 클래스용이었으므로 우리 것으로 바꿈)
original_in_features = imitation_model.classifier[-1].in_features
imitation_model.classifier[-1] = nn.Linear(original_in_features, NUM_CLASSES)

# 저장된 가중치(학습 결과) 불러오기
imitation_model.load_state_dict(checkpoint['model_state'])

# 추론 모드로 전환 (드롭아웃 등 비활성화)
imitation_model.eval()

# GPU 또는 CPU로 이동
imitation_model.to(DEVICE)

# 이미지 전처리 파이프라인 정의
# 모델이 학습할 때 사용한 것과 동일한 전처리를 적용해야 올바른 예측 가능
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),                               # 크기 통일
    transforms.ToTensor(),                                                  # 0~255 → 0.0~1.0
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),   # ImageNet 기준 정규화
])


# =====================================================================
# 차선 감지 모델 로드 (CPU 사용)
# GPU를 모방학습 모델과 나눠 쓰면 느려지므로 차선 모델은 CPU에서 실행
# =====================================================================
print(f"차선 감지 모델 로드 중: {LANE_MODEL} (CPU)")
lane_model = YOLO(LANE_MODEL)
lane_model.to("cpu")

print(f"클래스 목록: {list(idx_to_class.values())}  |  속도: {int(SPEED_SCALE * 100)}%\n")


# =====================================================================
# 모방학습 추론 함수
# =====================================================================
def predict_imitation(frame_bgr):
    """
    카메라 프레임(BGR)을 입력받아 주행 명령(키 이름, 확신도)을 반환합니다.

    반환값:
        key_name   : 예측된 키 조합 문자열 (예: 'w', 'wa', 'd' 등)
        confidence : 모델의 확신도 (0.0 ~ 1.0)
    """
    # OpenCV BGR 이미지 → PIL RGB 이미지로 변환
    # (torchvision의 transform은 PIL 형식을 사용)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # 전처리 적용
    tensor = transform(pil_image)

    # 배치 차원 추가: [채널, H, W] → [1, 채널, H, W]
    # (모델은 한 번에 여러 이미지를 처리할 수 있는 배치 형태를 기대)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    # 추론 실행 (gradient 계산 불필요 → 메모리/속도 절약)
    with torch.no_grad():
        output = imitation_model(tensor)       # 원시 점수(logits) 출력
        probs  = torch.softmax(output, dim=1)  # 소프트맥스로 확률로 변환 (합계 = 1.0)
        probs  = probs[0]                       # 배치 차원 제거

    # 가장 높은 확률의 클래스 선택
    best_idx   = probs.argmax().item()
    key_name   = idx_to_class[best_idx]
    confidence = float(probs[best_idx])

    return key_name, confidence


# =====================================================================
# 차선 안전장치 스레드
# 메인 루프와 독립적으로 실행되어 차선 위치를 계속 감시합니다
# =====================================================================

# 스레드 간 데이터 공유 변수 (메인 ↔ 스레드)
_lane_input         = None        # 메인 → 스레드: 분석할 프레임 전달
_lane_error         = None        # 스레드 → 메인: 차선 오프셋 (None이면 정상 또는 미감지)
_lane_raw_error     = None        # 시각화용 오프셋 (보정 여부와 무관하게 항상 저장)
_lane_status        = "대기 중"   # 화면 표시용 상태 텍스트
_lane_lock          = threading.Lock()  # 공유 변수 동시 접근 방지용 잠금
_dev_count          = 0           # 연속 이탈 프레임 카운터 (스레드 내부에서만 사용)
_crosswalk_detected = False       # 스레드 → 메인: 횡단보도 감지 시 True로 설정


def lane_thread_func():
    """
    차선 감지 스레드 함수.
    계속 실행되면서 메인에서 전달된 프레임을 분석하고,
    차선 이탈 여부와 횡단보도 감지 결과를 공유 변수에 저장합니다.
    """
    global _lane_input, _lane_error, _lane_raw_error
    global _lane_status, _dev_count, _crosswalk_detected

    while True:

        # ─── 1단계: 처리할 프레임 가져오기 ──────────────────────────
        with _lane_lock:
            frame = _lane_input   # 메인이 전달한 프레임 꺼내기
            _lane_input = None    # 꺼냈으면 비워두기 (중복 처리 방지)

        # 새 프레임이 없으면 잠깐 쉬고 다시 시도
        if frame is None:
            time.sleep(0.005)
            continue

        # ─── 2단계: YOLO 모델로 차선/횡단보도 감지 ──────────────────
        h, w    = frame.shape[:2]
        results = lane_model(frame, verbose=False)[0]   # verbose=False = 로그 출력 억제

        # ─── 3단계: 횡단보도 감지 확인 ───────────────────────────────
        # 차선 감지 여부와 별개로 항상 체크
        for i, cls in enumerate(results.boxes.cls):
            class_id    = int(cls)
            detect_conf = float(results.boxes.conf[i])

            if class_id == CROSSWALK_CLASS and detect_conf >= CROSSWALK_CONF_THR:
                # 기준 이상 확신도로 횡단보도 감지 → 플래그 ON
                with _lane_lock:
                    _crosswalk_detected = True
                break   # 하나만 감지해도 충분

        # ─── 4단계: 차선 마스크가 없으면 "검출 안됨" 처리 ────────────
        if results.masks is None:
            _dev_count = 0   # 이탈 카운터 리셋
            with _lane_lock:
                _lane_error     = None
                _lane_raw_error = None
                _lane_status    = "검출 안됨"
            continue

        # ─── 5단계: inline 클래스 마스크만 하나로 합치기 ─────────────
        # 여러 개의 inline 마스크가 있을 수 있으므로 np.maximum으로 합침
        combined_mask = np.zeros((h, w), dtype=np.float32)
        for i, cls in enumerate(results.boxes.cls):
            if int(cls) == INLINE_CLASS:
                mask         = results.masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask, (w, h))
                combined_mask = np.maximum(combined_mask, mask_resized)

        # 화면 아래쪽 절반만 분석 (멀리 있는 차선보다 가까운 차선이 더 중요)
        bottom_half = combined_mask[h // 2:, :]

        # 마스크 픽셀이 너무 적으면 신뢰할 수 없으므로 무시
        if bottom_half.sum() < MIN_MASK_PIX:
            _dev_count = 0
            with _lane_lock:
                _lane_error     = None
                _lane_raw_error = None
                _lane_status    = "검출 안됨"
            continue

        # ─── 6단계: 차선 중심 위치 계산 ──────────────────────────────
        # 마스크가 있는 열(column) 번호들을 모두 가져옴
        lane_columns = np.where(bottom_half > 0.5)[1]

        if len(lane_columns) == 0:
            _dev_count = 0
            with _lane_lock:
                _lane_error     = None
                _lane_raw_error = None
                _lane_status    = "검출 안됨"
            continue

        # 차선 중심 X좌표 = 마스크 열들의 평균값
        lane_center_x   = int(np.mean(lane_columns))
        screen_center_x = w // 2

        # 오프셋 = 차선 중심 - 화면 중심
        # 양수(+): 차선이 오른쪽에 있음 → 차가 왼쪽으로 치우친 것 → 우회전 필요
        # 음수(-): 차선이 왼쪽에 있음  → 차가 오른쪽으로 치우친 것 → 좌회전 필요
        error = lane_center_x - screen_center_x

        # ─── 7단계: 이탈 여부 판단 및 공유 변수 업데이트 ─────────────
        if abs(error) <= LANE_THRESHOLD:
            # 허용 범위 내 → 정상, 이탈 카운터 리셋
            _dev_count = 0
            with _lane_lock:
                _lane_error     = None    # 보정 필요 없음
                _lane_raw_error = error   # 시각화용으로는 저장
                _lane_status    = f"정상 (err={error:+d}px)"
        else:
            # 허용 범위 초과 → 이탈 카운터 증가
            _dev_count += 1

            if _dev_count >= DEVIATION_COUNT:
                # DEVIATION_COUNT 프레임 연속 이탈 → 보정 신호 발생
                with _lane_lock:
                    _lane_error     = error   # 메인 루프에서 이 값으로 보정
                    _lane_raw_error = error
                    _lane_status    = f"보정필요 (err={error:+d}px, {_dev_count}f)"
            else:
                # 아직 연속 이탈 횟수 부족 → 대기 중
                with _lane_lock:
                    _lane_error     = None    # 아직 보정하지 않음
                    _lane_raw_error = error
                    _lane_status    = f"이탈감지중 ({_dev_count}/{DEVIATION_COUNT}f)"


# 차선 감지 스레드 시작
# daemon=True : 메인 프로그램이 종료되면 이 스레드도 자동으로 종료됨
lane_thread = threading.Thread(target=lane_thread_func, daemon=True)
lane_thread.start()


# =====================================================================
# 방향 화살표 시각화 유틸
# 화면에 현재 주행 방향을 화살표로 표시합니다
# =====================================================================

# 각 키 조합에 대한 방향 벡터 (dx, dy)
# 정규화된 값 (-1.0 ~ 1.0), 대각선은 √2/2 ≈ 0.707 사용
_ARROW_VECS = {
    'w':  ( 0.000, -1.000),   # 위 (전진)
    's':  ( 0.000,  1.000),   # 아래 (후진)
    'a':  (-1.000,  0.000),   # 왼쪽
    'd':  ( 1.000,  0.000),   # 오른쪽
    'wa': (-0.707, -0.707),   # 왼쪽 위 (전진+좌)
    'wd': ( 0.707, -0.707),   # 오른쪽 위 (전진+우)
    'sa': (-0.707,  0.707),   # 왼쪽 아래 (후진+좌)
    'sd': ( 0.707,  0.707),   # 오른쪽 아래 (후진+우)
}


def draw_arrow(img, direction, cx, cy, size, color, thick=3):
    """
    화면에 방향 화살표를 그립니다.

    매개변수:
        img       : 그릴 이미지
        direction : 방향 키 문자열 ('w', 'wa' 등)
        cx, cy    : 화살표 중심 좌표
        size      : 화살표 크기 (픽셀)
        color     : 색상 (B, G, R)
        thick     : 선 두께
    """
    # 알 수 없는 방향이면 그리지 않음
    if direction not in _ARROW_VECS:
        return

    dx, dy = _ARROW_VECS[direction]
    offset = size * 0.55

    # 중심에서 ±55% 위치에 시작점/끝점 설정
    pt1 = (int(cx - dx * offset), int(cy - dy * offset))   # 화살표 꼬리
    pt2 = (int(cx + dx * offset), int(cy + dy * offset))   # 화살표 머리

    cv2.arrowedLine(img, pt1, pt2, color, thick, tipLength=0.42)


# =====================================================================
# UDP 소켓 초기화
# 라즈베리파이로부터 카메라 영상을 받기 위한 소켓
# =====================================================================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', LISTEN_PORT))   # 모든 네트워크 인터페이스에서 수신
sock.settimeout(0.02)                  # 20ms 타임아웃 (블로킹 방지)


# =====================================================================
# 전역 상태 변수
# =====================================================================
paused     = False   # True = 일시정지 상태
lane_guard = True    # True = 차선 안전장치 활성화

addr = None          # 라즈베리파이의 IP:포트 (첫 프레임 수신 시 자동 설정)
disp = None          # 화면에 표시할 이미지 (마지막 유효 프레임 유지)

last_motor = '0,0'   # 마지막으로 전송한 모터 명령 (일시정지 해제 시 즉시 재사용)
last_key   = 'stop'  # 모방학습이 예측한 마지막 키
last_conf  = 0.0     # 모방학습 마지막 예측 확신도

lane_status       = "대기 중"   # 차선 상태 표시 텍스트
last_correction_t = 0.0          # 마지막 차선 보정 발생 시각 (쿨다운 계산용)
disp_raw_error    = None         # 시각화용 최신 차선 오프셋

crosswalk_stop_until     = 0.0   # 이 시각까지 횡단보도 정지 (time.time() 기준)
crosswalk_cooldown_until = 0.0   # 이 시각까지 횡단보도 재감지 무시

print("자율주행 시작 | SPACE=일시정지 | L=차선안전장치토글 | ESC=종료\n")


# =====================================================================
# 메인 루프 - 프레임 수신 → 추론 → 모터 제어
# =====================================================================
while True:

    # ─── 1단계: UDP 소켓에서 최신 프레임 수신 ────────────────────────
    # 소켓 버퍼에 여러 프레임이 쌓여 있을 수 있으므로
    # 모두 읽어서 가장 마지막 것만 사용 (처리 지연 방지)

    got_frame   = False   # 이번 루프에서 새 프레임을 받았는지 여부
    frame       = None

    latest_data = None    # 가장 최신 UDP 패킷 데이터
    latest_addr = None    # 가장 최신 UDP 패킷 발신자

    # 논블로킹으로 전환해서 버퍼에 있는 모든 패킷을 소진
    sock.settimeout(0)
    while True:
        try:
            data, sender = sock.recvfrom(1_000_000)
            latest_data  = data     # 계속 덮어쓰면서 마지막 것만 남김
            latest_addr  = sender
        except (socket.timeout, BlockingIOError):
            break   # 더 이상 받을 게 없으면 종료

    # 다시 타임아웃 있는 모드로 복구
    sock.settimeout(0.02)

    # 버퍼가 비어있었다면 최대 20ms 기다리면서 새 패킷 수신 시도
    if latest_data is None:
        try:
            latest_data, latest_addr = sock.recvfrom(1_000_000)
        except socket.timeout:
            pass   # 20ms 안에 안 오면 그냥 넘어감

    # ─── 2단계: 수신된 데이터를 이미지로 디코딩 ──────────────────────
    # 패킷 구조: [4바이트 크기 헤더] + [JPEG 데이터]
    if latest_data is not None and len(latest_data) >= 4:
        addr = latest_addr

        # 앞 4바이트에서 JPEG 데이터 크기 읽기 (little-endian 정수)
        jpeg_size = int.from_bytes(latest_data[:4], 'little')

        # 헤더 뒤의 JPEG 데이터 추출
        img_data = latest_data[4:4 + jpeg_size]

        # 크기가 올바를 때만 디코딩 (잘린 패킷 무시)
        if len(img_data) == jpeg_size:
            arr   = np.frombuffer(img_data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is not None:
                # 필요하면 좌우 반전 (카메라 설치 방향에 따라)
                if MIRROR_LR:
                    frame = cv2.flip(frame, 1)
                got_frame = True

    # ─── 3단계: 키보드 입력 처리 ──────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == 27:   # ESC 키 → 종료
        break

    elif key == ord(' '):   # SPACE 키 → 일시정지/재개 토글
        paused = not paused
        if paused:
            print("[일시정지]")
        else:
            print("[재개]")

    elif key == ord('l'):   # L 키 → 차선 안전장치 ON/OFF 토글
        lane_guard = not lane_guard
        if lane_guard:
            print("차선 안전장치: ON")
        else:
            print("차선 안전장치: OFF")

    # ─── 4단계: 새 프레임이 있고 일시정지 중이 아닐 때 → 주행 결정 ───
    if got_frame and frame is not None and not paused:

        # 4-1) 모방학습으로 기본 주행 명령 결정
        last_key, last_conf = predict_imitation(frame)
        motor = to_motor_str(last_key, SPEED_SCALE)

        # 4-2) 차선 안전장치가 켜져 있으면 차선 이탈/횡단보도 검사
        if lane_guard:

            # 차선 스레드에 분석할 프레임 전달 (복사본 전달 → 원본 보호)
            with _lane_lock:
                _lane_input = frame.copy()

            # 차선 스레드의 최신 분석 결과 읽기
            with _lane_lock:
                error            = _lane_error          # 차선 오프셋 (None이면 정상)
                raw_error        = _lane_raw_error       # 시각화용 오프셋
                lane_status      = _lane_status          # 상태 텍스트
                crosswalk_signal = _crosswalk_detected   # 횡단보도 감지 여부

                if crosswalk_signal:
                    _crosswalk_detected = False   # 읽었으니 플래그 리셋

            disp_raw_error = raw_error   # 시각화용으로 저장

            now = time.time()

            # 4-3) 횡단보도 감지 처리
            # 쿨다운 중이 아닐 때만 반응 (같은 횡단보도에서 여러 번 정지하지 않도록)
            if crosswalk_signal and now > crosswalk_cooldown_until:
                crosswalk_stop_until     = now + CROSSWALK_STOP_SEC
                crosswalk_cooldown_until = now + CROSSWALK_STOP_SEC + CROSSWALK_COOLDOWN_SEC
                print(f"[횡단보도] 감지! {CROSSWALK_STOP_SEC:.0f}초 정지 → 이후 {CROSSWALK_COOLDOWN_SEC:.0f}초 쿨다운")

            # 4-4) 최종 모터 명령 결정
            # 우선순위: 횡단보도 정지 > 차선 이탈 보정 > 모방학습 (기본값)

            if now < crosswalk_stop_until:
                # 우선순위 1: 횡단보도 정지 (최고 우선순위)
                motor = '0,0'

            elif error is not None and (now - last_correction_t) > CORRECTION_COOLDOWN:
                # 우선순위 2: 차선 이탈 보정
                # error > 0 → 차선이 오른쪽에 있음 → 오른쪽으로 조향
                # error < 0 → 차선이 왼쪽에 있음  → 왼쪽으로 조향
                if error > 0:
                    motor = to_motor_str('wd', SPEED_SCALE)   # 전진 + 우회전
                else:
                    motor = to_motor_str('wa', SPEED_SCALE)   # 전진 + 좌회전

                last_correction_t = now   # 보정 시각 기록 (쿨다운 시작)

            # else: 우선순위 3 = 모방학습 명령 그대로 사용 (위에서 이미 설정됨)

        else:
            # 차선 안전장치가 꺼져 있으면 상태 텍스트만 업데이트
            lane_status = "안전장치 OFF"

        last_motor = motor   # 일시정지 처리를 위해 마지막 명령 저장

    # ─── 5단계: 일시정지 중이면 정지 명령, 아니면 마지막 명령 사용 ────
    if paused:
        motor = '0,0'
    else:
        motor = last_motor

    # ─── 6단계: 화면 표시 ─────────────────────────────────────────────
    # 새 프레임이 있으면 표시용 이미지 업데이트
    if got_frame and frame is not None:
        disp = frame.copy()

    if disp is not None:
        h_img, w_img = disp.shape[:2]

        # 일시정지/자동주행 상태 색상 결정
        if paused:
            status_text  = "[PAUSE]"
            status_color = (0, 165, 255)   # 주황색
        else:
            status_text  = "[AUTO]"
            status_color = (0, 255, 0)     # 초록색

        # 차선 안전장치 상태 색상 결정
        if lane_guard:
            guard_label = "LANE:ON"
            guard_color = (0, 255, 255)    # 노란색
        else:
            guard_label = "LANE:OFF"
            guard_color = (120, 120, 120)  # 회색

        # 차선 보정 중이면 빨간색으로 강조
        if "보정" in lane_status:
            guard_color = (0, 80, 255)

        # 텍스트 오버레이 출력
        cv2.putText(disp,
                    f"{status_text}  {last_key}  ({last_conf * 100:.0f}%)",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

        cv2.putText(disp,
                    f"MOTOR: {motor}",
                    (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 0), 2)

        cv2.putText(disp,
                    f"{guard_label}  {lane_status}",
                    (10, 101),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, guard_color, 2)

        # ─── 차선 보정 중일 때만 시각화 표시 ─────────────────────────
        # 최근 0.4초 이내에 보정이 발생했으면 시각화 표시
        time_since_correction = time.time() - last_correction_t
        is_correcting = time_since_correction < 0.4

        if is_correcting and disp_raw_error is not None:

            # 차선 오프셋 바: 화면 상단에 차선 위치를 막대로 표시
            bar_y      = 114          # 바의 y 좌표
            bar_height = 16           # 바의 높이 (픽셀)
            bar_left   = 10
            bar_right  = w_img - 10
            bar_center = w_img // 2

            # 배경 (어두운 회색)
            cv2.rectangle(disp,
                          (bar_left,  bar_y),
                          (bar_right, bar_y + bar_height),
                          (40, 40, 40), -1)

            # 허용 범위 영역 (어두운 초록색)
            safe_left  = max(bar_left,  bar_center - LANE_THRESHOLD)
            safe_right = min(bar_right, bar_center + LANE_THRESHOLD)
            cv2.rectangle(disp,
                          (safe_left,  bar_y),
                          (safe_right, bar_y + bar_height),
                          (0, 70, 0), -1)

            # 허용 범위 경계선 (밝은 초록색)
            cv2.line(disp,
                     (safe_left, bar_y), (safe_left, bar_y + bar_height),
                     (0, 180, 0), 1)
            cv2.line(disp,
                     (safe_right, bar_y), (safe_right, bar_y + bar_height),
                     (0, 180, 0), 1)

            # 화면 중앙선 (회색)
            cv2.line(disp,
                     (bar_center, bar_y), (bar_center, bar_y + bar_height),
                     (180, 180, 180), 1)

            # 현재 차선 위치 점 (파란색)
            dot_x = bar_center + disp_raw_error
            dot_x = max(bar_left + 7, min(bar_right - 7, dot_x))   # 바 범위 안에 고정
            dot_x = int(dot_x)
            cv2.circle(disp, (dot_x, bar_y + bar_height // 2), 7, (30, 30, 255), -1)

            # 레이블
            cv2.putText(disp, "INLINE",
                        (bar_left + 2, bar_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (130, 130, 130), 1)

            # 보정 방향 화살표 및 레이블 표시
            arrow_cx = w_img // 2
            arrow_cy = h_img // 2 + 30

            if disp_raw_error > 0:
                correction_dir = 'wd'   # 오른쪽 보정
            else:
                correction_dir = 'wa'   # 왼쪽 보정

            draw_arrow(disp, correction_dir, arrow_cx, arrow_cy, 60, (30, 30, 255), 5)

            cv2.putText(disp, "LANE OVERRIDE",
                        (arrow_cx - 80, arrow_cy + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 255), 2)

        # ─── 횡단보도 상태 오버레이 ───────────────────────────────────
        now_display = time.time()

        if now_display < crosswalk_stop_until:
            # 정지 중 → 화면 중앙에 빨간 배너 표시
            remaining_stop = crosswalk_stop_until - now_display
            overlay_top    = h_img // 2 - 45
            overlay_bottom = h_img // 2 + 45

            # 배너 배경 (어두운 빨간색 채움)
            cv2.rectangle(disp,
                          (0, overlay_top),
                          (w_img, overlay_bottom),
                          (0, 0, 160), -1)

            # 배너 테두리 (밝은 빨간색)
            cv2.rectangle(disp,
                          (0, overlay_top),
                          (w_img, overlay_bottom),
                          (0, 0, 255), 3)

            # 정지 메시지 및 남은 시간
            cv2.putText(disp,
                        f"CROSSWALK  STOP  {remaining_stop:.1f}s",
                        (w_img // 2 - 185, h_img // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

        elif crosswalk_cooldown_until > now_display:
            # 쿨다운 중 → 화면 모서리에 작은 텍스트 표시
            remaining_cooldown = int(crosswalk_cooldown_until - now_display)
            cv2.putText(disp,
                        f"CW cooldown: {remaining_cooldown}s",
                        (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 170, 255), 1)

        # 키 조작 안내 (화면 하단)
        cv2.putText(disp,
                    "SPACE=Pause  L=LaneGuard  ESC=Quit",
                    (10, h_img - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        cv2.imshow("Auto Drive", disp)

    # ─── 7단계: 라즈베리파이로 모터 명령 전송 ─────────────────────────
    if addr:
        sock.sendto(motor.encode(), addr)


# =====================================================================
# 종료 처리
# =====================================================================

# RC카 정지 명령 전송
if addr:
    sock.sendto(b'0,0', addr)

cv2.destroyAllWindows()
sock.close()
print("종료")
