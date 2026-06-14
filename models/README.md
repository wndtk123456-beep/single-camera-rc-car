# 모델 파일

학습 데이터와 가중치는 용량 및 공개 범위를 고려해 저장소에 포함하지 않습니다.

실행 전 아래 파일을 이 폴더에 배치하세요.

| 파일 | 용도 | 생성 스크립트 |
| --- | --- | --- |
| `drive_best.pth` | MobileNetV3 모방학습 주행 | `training/train_imitation.py` |
| `lane_best.pt` | YOLOv8 차선·횡단보도 세그멘테이션 | `training/train_lane_segmentation.py` |
| `obstacle_best.pt` | YOLOv8 장애물 감지 | `training/train_obstacle_detection.py` |
