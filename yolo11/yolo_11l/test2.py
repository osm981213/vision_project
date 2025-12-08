import os
import json
import cv2
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("yolo11l.pt")

# 비디오 파일 로드
video_path = "https://strm3.spatic.go.kr/live/312.stream/playlist.m3u8"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise Exception("Error: Could not open video.")

# 선 좌표 수집 및 로드
def load_or_collect_points(frame):
    coordinates_file = "points.json"
    if os.path.exists(coordinates_file):
        with open(coordinates_file, "r") as f:
            return json.load(f)
    else:
        points = []
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Point Collection', param)

        cv2.imshow('Point Collection', frame)
        cv2.setMouseCallback('Point Collection', click_event, frame.copy())

        while len(points) < 4:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow('Point Collection')
        with open(coordinates_file, "w") as f:
            json.dump(points, f)
        return points

# 속도 계산 함수
def calculate_speed(time_taken, dist=25):
    return round((dist / time_taken) * 3.6, 1) if time_taken > 0 else 0

# 첫 프레임을 사용하여 좌표 수집
success, first_frame = cap.read()
if not success:
    raise Exception("Failed to read video")

points = load_or_collect_points(first_frame)
p1, p2, p3, p4 = points

# 트랙 히스토리 및 시간 저장
track_history = defaultdict(list)
vehicle_times = defaultdict(lambda: {'start': None, 'end': None})
vehicle_speeds = {}

# 비디오 처리
cv2.namedWindow('tracking', flags=cv2.WINDOW_AUTOSIZE)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 두 평행선 그리기
    cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 0), 2)
    cv2.line(frame, tuple(map(int, p3)), tuple(map(int, p4)), (0, 255, 0), 2)

    # YOLO 트래킹 수행
    results = model.track(frame, persist=True)

    # 검출된 차량 객체 처리
    for box, cls, track_id in zip(results[0].boxes.xywh.cpu(), results[0].boxes.cls.cpu().tolist(), results[0].boxes.id.int().cpu().tolist()):
        if cls not in [2, 3, 5, 7]:  # 차량 클래스 필터링
            continue

        x, y, w, h = box
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 차량의 트랙 히스토리 업데이트
        track = track_history[track_id]
        track.append((float(x), float(y)))
        if len(track) > 30:
            track.pop(0)

        # 트래킹 라인 그리기
        if len(track) > 1:
            points = np.array(track, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], False, (230, 230, 230), 2)

        # 차량이 선을 지나는지 확인 및 시간 기록
        y_pos = float(y)
        y_line1, y_line2 = (p1[1] + p2[1]) / 2, (p3[1] + p4[1]) / 2

        if track_id not in vehicle_speeds:
            if abs(y_pos - y_line1) < 5 and vehicle_times[track_id]['start'] is None:
                vehicle_times[track_id]['start'] = time.time()
            elif abs(y_pos - y_line2) < 5 and vehicle_times[track_id]['start'] is not None:
                vehicle_times[track_id]['end'] = time.time()
                time_taken = vehicle_times[track_id]['end'] - vehicle_times[track_id]['start']
                vehicle_speeds[track_id] = calculate_speed(time_taken)

        # 속도 표시
        if track_id in vehicle_speeds:
            cv2.putText(frame, f"ID: {track_id}, Speed: {vehicle_speeds[track_id]} km/h", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 프레임 출력
    cv2.imshow("tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
# 출처: https://42morrow.tistory.com/entry/교통-CCTV-영상-기반의-자동차-속도-측정 [AI 탐구노트:티스토리]


# from collections import defaultdict

# import cv2
# import numpy as np

# from ultralytics import YOLO

# model = YOLO("yolo11l.pt")
# video_path = "https://strm3.spatic.go.kr/live/312.stream/playlist.m3u8"
# cap = cv2.VideoCapture(video_path)
# track_history = defaultdict(lambda: [])

# while cap.isOpened():
#     success, frame = cap.read()
#     if success:
#         results = model.track(frame, persist=True)
#         boxes = results[0].boxes.xywh.cpu()
#         track_ids = results[0].boxes.id.int().cpu().tolist()
#         annotated_frame = results[0].plot()
#         for box, track_id in zip(boxes, track_ids):
#             x, y, w, h = box
#             track = track_history[track_id]
#             track.append((float(x), float(y)))
#             if len(track) > 30:
#                 track.pop(0)
#             points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#             cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
#         cv2.imshow("YOLO11 Tracking", annotated_frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         break
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from collections import defaultdict
# from ultralytics import YOLO
# from ultralytics.solutions import SpeedEstimator 

# # --- 1. 환경 및 비디오 설정 ---
# input_video_path = "https://strm3.spatic.go.kr/live/312.stream/playlist.m3u8" 
# output_video_path = "yolo_tracking_speed_result.mp4"

# cap = cv2.VideoCapture(input_video_path)
# if not cap.isOpened():
#     print("Error: Could not open video stream.")
#     exit()

# # 비디오 속성 가져오기
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)

# # 비디오 기록기 (VideoWriter) 초기화
# video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
# print(f"입력 비디오 W:{w}, H:{h}, FPS:{fps} / 속도 측정 및 경로 추적 시작...")

# # --- 2. YOLO 및 SpeedEstimator 초기화 ---
# # COCO 데이터셋 차량 관련 클래스 인덱스 (car, motorcycle, bus, truck)
# vehicle_classes = [2, 3, 5, 7] 

# # 속도 측정 영역(line_pts) 설정: 중앙 세로선 예시
# line_pts = [(360,1280), (360,320)] 

# # SpeedEstimator 객체 생성 
# speed_obj = SpeedEstimator(
#     model="yolo11l.pt",
#     fps=fps,
#     classes=vehicle_classes,
#     region=line_pts,
#     meter_per_pixel=0.005, # 픽셀 당 미터 값 (환경에 맞게 조정 필요)
#     max_speed=120,
#     show=False,
#     max_hist=3,
#     conf=0.5,
#     iou=0.5,
#     tracker="bytetrack.yaml"
# ) 

# # --- 3. 추적 경로 저장을 위한 defaultdict 초기화 ---
# track_history = defaultdict(lambda: [])

# # --- 4. 비디오 프레임 처리 루프 ---
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success: 
#         print("End of video stream or failed to read frame.")
#         break
    
#     # 4.1. ⭐️ 속도 계산 및 시각화 (SpeedEstimator 호출)
#     results = speed_obj(frame) 
#     annotated_frame = results.plot_im # SpeedEstimator가 그린 프레임 (속도, 경계 상자 포함)
    
#     # 4.2. ⭐️ 추적 경로 그리기 로직 통합 (에러 발생 부분 수정)
    
#     # 🚨 수정된 로직: 'boxes' 속성이 있는지 확인하고, 있다면 ID가 있는지 추가 확인
#     if hasattr(results, 'boxes') and results.boxes.id is not None:
#         boxes = results.boxes.xywh.cuda() # x, y, w, h
#         track_ids = results.boxes.id.int().cuda().tolist() # 추적 ID
        
#         for box, track_id in zip(boxes, track_ids):
#             x, y, w_box, h_box = box # 박스 정보
            
#             # 중심 좌표를 경로에 추가
#             center_x, center_y = float(x), float(y)
#             track = track_history[track_id]
#             track.append((center_x, center_y))
            
#             # 경로 길이 제한 (최대 30 프레임)
#             if len(track) > 30:
#                 track.pop(0)
            
#             # 경로를 cv2.polylines() 형식에 맞게 변환
#             points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            
#             # 추적 경로를 현재 프레임(annotated_frame)에 그리기
#             cv2.polylines(
#                 annotated_frame, 
#                 [points], 
#                 isClosed=False, 
#                 color=(0, 255, 255), # 청록색
#                 thickness=4 
#             )

#     # 4.3. 시각화 및 종료 조건
#     cv2.imshow("YOLO Tracking and Speed Estimation", annotated_frame)
    
#     # 출력 비디오에 프레임 쓰기
#     video_writer.write(annotated_frame)
    
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # --- 5. 종료 및 리소스 해제 ---
# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()
# print(f"처리 완료. 결과 파일: {output_video_path}")