# crazyflie
Real-time nonlinear MPC for trajectory tracking on the Crazyflie - from simulation to actual flight tests

# Crazyflie NMPC Obstacle Avoidance

ROS2 기반 Crazyflie NMPC 장애물 회피 컨트롤러.
Safe Flight Corridor(SFC) 제약과 time-varying cutting plane으로
동적 장애물을 회피하며 레퍼런스 경로를 추종한다.

## 주요 기능
- SFC linear 제약 기반 NMPC (PANOC/OpEn 솔버)
- 동적 장애물 대응 TV cutting plane 생성
- 예측 경로 및 장애물 RViz 시각화

## 환경
- Ubuntu 22.04 / ROS2 Humble (버전 맞게 수정)
- Python 3.10, opengen (솔버 빌드용)
- (기타 의존 패키지)

## 빌드
```bash
cd ~/ros2_ws/src
git clone https://github.com/keoteom/crazyflie.git
cd ~/ros2_ws
colcon build --packages-select nmpc_uav_avoidance
source install/setup.bash
```

솔버는 최초 1회 생성 필요:
```bash
python3 build_solver.py
```

## 실행
```bash
ros2 launch nmpc_uav_avoidance (런치파일명).launch.py
```

## 노드 및 토픽
### nmpc_node
- 구독: `/sfc_coefficients`, (odometry 토픽), ...
- 발행: (제어 명령 토픽), `/nmpc/predicted_path`, (마커 토픽)

## 파라미터
| 이름 | 기본값 | 설명 |
|---|---|---|
| r_obs | 0.5 | 장애물 반경 |
| r_ego | 0.25 | 기체 반경 |
