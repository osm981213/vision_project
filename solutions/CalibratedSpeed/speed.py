import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# ✅ 이 코드는 "CCTV 영상"에서 차량을 탐지/추적하고,
#    ROI(관심영역) 안에서 차량이 이동한 "거리(미터)"를 계산
#    속도(km/h)를 추정해서 화면에 표시하는 코드.
#
# ✅ 핵심 흐름(한 줄 요약)
#   1) 모델 로드 → 2) CCTV 열기 → 3) 실제 폭/길이(미터) 입력
#   4) 화면에서 ROI 4점 클릭(사각형) → 5) 호모그래피(H) 계산
#   6) YOLO 추적(track) → 7) 픽셀좌표를 미터좌표로 변환 → 속도 계산
# ============================================================

# ============================================================
# 1) YOLO 모델 로드
# ============================================================
# - s_best.pt / m_best.pt 중 하나 선택해서 사용 가능.
model = YOLO("solutions/CalibratedSpeed/s_best.pt")  # yolo11s 으로 만든 best.pt 
# model = YOLO("solutions/CalibratedSpeed/m_best.pt") # yolo11m 으로 만든 best.pt

# ============================================================
# 2) 클래스 이름을 보기 쉽게 묶어주는 함수
# ============================================================
# - 모델이 출력하는 클래스가 '2 axle truck', '5 axle semi trailer'처럼 세분화되어 있으면
#   화면에 표시할 때 너무 길고 복잡함.
# - 그래서 키워드 기반으로 Truck/Bus/Car/Motorcycle 로 단순화해 표시함.
def get_group_label(model, cls_idx: int) -> str:
    # model.names는 보통 dict 또는 list 형태
    names = model.names
    if isinstance(names, dict):
        raw = names.get(int(cls_idx), str(cls_idx))
    else:
        raw = names[int(cls_idx)] if 0 <= int(cls_idx) < len(names) else str(cls_idx)

    # 비교하기 편하게 소문자/공백 형태로 정리
    name = str(raw).strip().lower().replace("_", " ").replace("-", " ")

    # 트럭/트레일러 계열 키워드 포함이면 Truck으로 통합
    truck_keywords = ["truck", "axle", "trailer", "semi", "lorry"]
    if any(k in name for k in truck_keywords):
        return "Truck"
    if "bus" in name:
        return "Bus"
    if "car" in name:
        return "Car"
    if "motorcycle" in name or "moto" in name or "bike" in name:
        return "Motocycle"

    # 어디에도 안 걸리면 원래 클래스명 그대로 사용
    return str(raw)

# ============================================================
# 3) CCTV 영상 열기
# ============================================================
video_path = 'https://stream6.bcits.go.kr/bucheon/TM090TC08P.stream/playlist.m3u8' # 소사역 앞
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# 디버깅용 출력(현재 영상의 해상도/FPS가 무엇으로 읽히는지 확인)
# print(f"읽어온 프레임 너비 (Width): {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} pixels")
# print(f"읽어온 프레임 높이 (Height): {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} pixels")
# print(f"읽어온 FPS (오류 가능): {cap.get(cv2.CAP_PROP_FPS)}")
# print("---------------------------------------------")

# ============================================================
# 4) FPS 고정(중요)
# ============================================================
# - 속도 = 이동거리 / 시간 이라서 시간(delta_t)을 우리가 고정해주면 더 안정적.
fps = 30.0
delta_t = 1.0 / fps   # 한 프레임당 시간(초)

# ============================================================
# 5) 실제 거리 입력(중요)
# ============================================================
# - 너는 ROI(사각형 영역)의 "실제 폭(W)"과 "실제 길이(L)"를 미터로 입력함.
# - 이 값을 기준으로 "픽셀 좌표 → 실제 미터 좌표" 변환을 만들 수 있음.

# 예시:
#   ROI 폭 = 26m, 깊이 = 78m
#   → 화면에서 찍은 ROI 사각형이 실제로는 26m × 78m 라고 가정(소사역 앞) 다른 곳은 계산을 해도 속도를 맞게 계산하는지 잘모르겠음
print("--- 1단계: 실제 측정 거리 입력 ---")
try:
    WIDTH_REAL_METERS = float(input("1. ROI 영역의 실제 폭(미터)을 입력하세요 (예: 26.0(소사역 앞만)): "))
    DEPTH_REAL_METERS = float(input("2. ROI 영역의 실제 깊이/길이(미터)를 입력하세요 (예: 78.0(소사역 앞만)): "))
except ValueError:
    print("오류: 유효한 숫자를 입력해야 합니다.")
    exit()

# ============================================================
# 6) ROI 4점 클릭
# ============================================================
# - 첫 프레임을 띄워서, 마우스로 ROI 사각형 꼭짓점 4개를 찍는다.
# - 순서:
#   1) 좌상단 → 2) 우상단 → 3) 우하단 → 4) 좌하단
#
# - 왜 4점을 찍냐?
#   CCTV는 원근감(멀수록 작아짐)이 있어서 픽셀당 실제 미터가 일정하지 않음.
#   그래서 "사각형 4점"을 이용해 원근을 보정하는 변환(호모그래피)을 만든다.
points = []
window_name = "Set ROI Points (4 Clicks) - Press 's' to start tracking"

def click_event(event, x, y, flags, param):
    # 마우스 왼쪽 클릭하면 점 저장
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))

            # 클릭한 점 표시(원 + 번호)
            cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame_copy, str(len(points)), (x + 10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, frame_copy)
            print(f"Point {len(points)} set: ({x}, {y})")

# ROI 설정용으로 첫 프레임 1장 읽기
success, frame = cap.read()
if not success:
    print("Error: Cannot read video frame for ROI setting.")
    exit()

# ROI를 다시 찍을 수도 있게 while로 감쌈
while True:
    frame_copy = frame.copy()
    points = []  # 재설정 위해 초기화

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)

    print("\n--- 2단계: 픽셀 좌표 설정 ---")
    print(f"현재 설정된 실제 거리: 폭={WIDTH_REAL_METERS}m, 깊이={DEPTH_REAL_METERS}m")
    print("마우스로 4개의 점을 순서대로 클릭하세요 (좌상단 -> 우상단 -> 우하단 -> 좌하단).")
    print("설정이 완료되면 키보드의 's'키를 누르거나, 재설정하려면 'r'키를 누르세요.")

    # 점 4개 찍을 때까지 대기
    while len(points) < 4:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break
        if key == ord('r'):
            points = []
            break

    # 마지막으로 s/r/q 입력 기다림
    key = cv2.waitKey(0) & 0xFF
    if key == ord('s') and len(points) == 4:
        break
    elif key == ord('r'):
        cv2.destroyWindow(window_name)
        continue
    elif key == ord('q'):
        exit()
    elif len(points) == 4:
        break
    else:
        print("경고: 4개의 점이 모두 설정되지 않았습니다. 재시도합니다.")
        cv2.destroyWindow(window_name)
        continue

cv2.destroyWindow(window_name)

if len(points) != 4:
    print("오류: 4개의 점이 모두 설정되지 않아 프로그램을 종료합니다.")
    exit()

# ============================================================
# 7) 호모그래피(H) 계산 = 픽셀좌표 → 실제좌표(미터) 변환 만들기
# ============================================================
# src_pts: 클릭한 픽셀 좌표(영상 위 사각형)
src_pts = np.float32(points)

# dst_pts: "실제 공간(미터)"에서의 사각형 좌표
# (0,0) ~ (W, L) 형태의 직사각형으로 정의
dst_pts = np.float32([
    [0, 0],
    [WIDTH_REAL_METERS, 0],
    [WIDTH_REAL_METERS, DEPTH_REAL_METERS],
    [0, DEPTH_REAL_METERS]
])

# H(호모그래피) = 픽셀 좌표를 미터 좌표로 바꾸는 변환 행렬
# cv2.perspectiveTransform()에서 사용됨
H, _ = cv2.findHomography(src_pts, dst_pts)

# ROI 영역 판정(ROI 안에 들어온 차량만 속도 계산)
ROI_POLYGON_3D = np.int32(src_pts).reshape((-1, 1, 2))

# ============================================================
# 8) 추적/속도 계산을 위한 저장소
# ============================================================
# object_tracks: 각 track_id에 대해 이전 프레임 좌표(픽셀) 저장
# object_speeds: track_id별로 마지막으로 계산된 속도 저장(깜빡임 방지)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # ROI 찍느라 1프레임 읽었으니 다시 처음으로
object_tracks = {}
object_speeds = {}

# ============================================================
# 9) 속도 계산 함수
# ============================================================
# - 같은 track_id(같은 차량) 기준으로
#   이전 위치 → 현재 위치 이동거리(미터)를 계산하고,
#   delta_t(프레임 간 시간)로 나눠 m/s → km/h로 변환
def calculate_speed(track_id, current_x, current_y):
    # 처음 등장한 차량이면 이전 좌표가 없으니 저장만 하고 속도는 None 반환
    if track_id not in object_tracks:
        object_tracks[track_id] = [current_x, current_y]
        return None

    prev_x, prev_y = object_tracks[track_id]
    object_tracks[track_id] = [current_x, current_y]

    # 픽셀 좌표(이전/현재)를 "미터 좌표"로 변환
    prev_coords_pixel = np.array([[[prev_x, prev_y]]], dtype='float32')
    prev_coords_real = cv2.perspectiveTransform(prev_coords_pixel, H)[0][0]

    curr_coords_pixel = np.array([[[current_x, current_y]]], dtype='float32')
    curr_coords_real = cv2.perspectiveTransform(curr_coords_pixel, H)[0][0]

    # 두 점 사이 거리(미터)
    distance_real = np.sqrt(
        (curr_coords_real[0] - prev_coords_real[0]) ** 2 +
        (curr_coords_real[1] - prev_coords_real[1]) ** 2
    )

    # 속도 = 거리 / 시간
    speed_mps = distance_real / delta_t
    speed_kmh = speed_mps * 3.6
    return speed_kmh

# ============================================================
# 10) 메인 루프: 프레임 읽기 → 추적 → ROI 내부만 속도 표시
# ============================================================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --------------------------------------------------------
    # YOLO track = "탐지 + 추적"을 한 번에 수행
    # --------------------------------------------------------
    results = model.track(
        frame,
        persist=True,                          # 이전 프레임의 트랙을 이어서 사용(추적 유지)
        tracker="solutions/CalibratedSpeed/bytetrack_speed.yaml",  # ByteTrack 설정 파일
        conf=0.35,                             # 이 값보다 낮은 박스는 버림(너무 높으면 많이 놓침)
        iou=0.5,                               # NMS 겹침 기준(중복 박스 제거 강도)
        imgsz=1280,                            # 입력 이미지 크기(크면 작은 차량 잡기 유리, 대신 느려짐)
        verbose=False,
        classes=[0,1,2,3,4,5,6,7,8,9,10,11,12] # 탐지할 클래스 인덱스(데이터셋 기준) 
    )

    # results[0].boxes.id 가 있어야 "추적 ID"가 나온 상태
    if results and results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)   # 박스 좌표(x1,y1,x2,y2)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int) # 각 박스의 추적 ID
        class_indices = results[0].boxes.cls.cpu().numpy().astype(int) # 클래스 번호

        for box, track_id, cls_idx in zip(boxes, track_ids, class_indices):
            x1, y1, x2, y2 = box

            # 속도 계산에 사용할 "대표 포인트"
            # - center_x: 박스 중앙 x
            # - bottom_y: 박스 하단 y (차량은 바닥(타이어)쪽이 실제 이동을 더 잘 반영)
            center_x = (x1 + x2) // 2
            bottom_y = y2

            # ------------------------------------------------
            # ROI 안에 들어온 객체만 처리
            # - ROI 밖은 속도 계산 안 함(노이즈/원근 오류 줄이기)
            # ------------------------------------------------
            point_to_test = np.array([center_x, bottom_y], dtype=np.float32)
            is_in_roi = cv2.pointPolygonTest(ROI_POLYGON_3D, point_to_test, False) >= 0
            if not is_in_roi:
                continue

            # 속도 계산(km/h)
            speed_kmh = calculate_speed(track_id, center_x, bottom_y)

            # 라벨을 트럭/버스/승용차/오토바이로 통합
            class_name = get_group_label(model, cls_idx)

            # 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 화면에 표시할 텍스트 만들기
            label = f'{class_name} ID {track_id}'

            # speed_kmh가 None이면 첫 프레임이라 계산 불가
            # 그래도 깜빡임 방지 위해 이전 속도가 있으면 계속 보여줌
            if speed_kmh is not None:
                object_speeds[track_id] = speed_kmh
                label += f': {speed_kmh:.1f} km/h'
            elif track_id in object_speeds:
                label += f': {object_speeds[track_id]:.1f} km/h'

            # 텍스트 표시
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --------------------------------------------------------
    # ROI 사각형(점으로 찍은 영역) 표시
    # --------------------------------------------------------
    cv2.polylines(frame, [ROI_POLYGON_3D], isClosed=True, color=(0, 255, 255), thickness=4)

    # 결과 화면 출력
    cv2.imshow("YOLOv8 Custom Speed Measurement", frame)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
