# import os
# import json
# import cv2
# import numpy as np
# import time
# from collections import defaultdict
# from ultralytics import YOLO

# # YOLO 모델 로드
# model = YOLO("yolo11l.pt")
# # https://stream6.bcits.go.kr/bucheon/TM090TC08P.stream/playlist.m3u8 소사역 앞 cctv 경로 - 경찰청cctv
# # 비디오 파일 로드
# video_path = "https://stream6.bcits.go.kr/bucheon/TM090TC08P.stream/playlist.m3u8"
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     raise Exception("Error: Could not open video.")

# # 선 좌표 수집 및 로드
# def load_or_collect_points(frame):
#     coordinates_file = "points.json"
#     if os.path.exists(coordinates_file):
#         with open(coordinates_file, "r") as f:
#             return json.load(f)
#     else:
#         points = []
#         def click_event(event, x, y, flags, param):
#             if event == cv2.EVENT_LBUTTONDOWN:
#                 points.append((x, y))
#                 cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
#                 cv2.imshow('Point Collection', param)

#         cv2.imshow('Point Collection', frame)
#         cv2.setMouseCallback('Point Collection', click_event, frame.copy())

#         while len(points) < 4:
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cv2.destroyWindow('Point Collection')
#         with open(coordinates_file, "w") as f:
#             json.dump(points, f)
#         return points

# # 속도 계산 함수
# def calculate_speed(time_taken, dist=25):
#     return round((dist / time_taken) * 3.6, 1) if time_taken > 0 else 0

# # 첫 프레임을 사용하여 좌표 수집
# success, first_frame = cap.read()
# if not success:
#     raise Exception("Failed to read video")

# points = load_or_collect_points(first_frame)
# p1, p2, p3, p4 = points

# # 트랙 히스토리 및 시간 저장
# track_history = defaultdict(list)
# vehicle_times = defaultdict(lambda: {'start': None, 'end': None})
# vehicle_speeds = {}

# # 비디오 처리
# cv2.namedWindow('tracking', flags=cv2.WINDOW_AUTOSIZE)

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break

#     # 두 평행선 그리기
#     cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 0), 2)
#     cv2.line(frame, tuple(map(int, p3)), tuple(map(int, p4)), (0, 255, 0), 2)

#     # YOLO 트래킹 수행
#     results = model.track(frame, persist=True)

#     # 검출된 차량 객체 처리
#     for box, cls, track_id in zip(results[0].boxes.xywh.cpu(), results[0].boxes.cls.cpu().tolist(), results[0].boxes.id.int().cpu().tolist()):
#         if cls not in [2, 3, 5, 7]:  # 차량 클래스 필터링
#             continue

#         x, y, w, h = box
#         x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

#         # 바운딩 박스 그리기
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

#         # 차량의 트랙 히스토리 업데이트
#         track = track_history[track_id]
#         track.append((float(x), float(y)))
#         if len(track) > 30:
#             track.pop(0)

#         # 트래킹 라인 그리기
#         if len(track) > 1:
#             points = np.array(track, np.int32).reshape((-1, 1, 2))
#             cv2.polylines(frame, [points], False, (230, 230, 230), 2)

#         # 차량이 선을 지나는지 확인 및 시간 기록
#         y_pos = float(y)
#         y_line1, y_line2 = (p1[1] + p2[1]) / 2, (p3[1] + p4[1]) / 2

#         if track_id not in vehicle_speeds:
#             if abs(y_pos - y_line1) < 5 and vehicle_times[track_id]['start'] is None:
#                 vehicle_times[track_id]['start'] = time.time()
#             elif abs(y_pos - y_line2) < 5 and vehicle_times[track_id]['start'] is not None:
#                 vehicle_times[track_id]['end'] = time.time()
#                 time_taken = vehicle_times[track_id]['end'] - vehicle_times[track_id]['start']
#                 vehicle_speeds[track_id] = calculate_speed(time_taken)

#         # 속도 표시
#         if track_id in vehicle_speeds:
#             cv2.putText(frame, f"ID: {track_id}, Speed: {vehicle_speeds[track_id]} km/h", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # 프레임 출력
#     cv2.imshow("tracking", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
# # 출처: https://42morrow.tistory.com/entry/교통-CCTV-영상-기반의-자동차-속도-측정 [AI 탐구노트:티스토리]


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

# https://cctvsec.ktict.co.kr/6584/vgxiQJ+4oMaTCqkFKTCPGv9drhY+i7lsgzhHw8yoYAh1wSwWpFPUjxQfSIm4E3jQ  시화 이마트 사거리
# https://stream6.bcits.go.kr/bucheon/TM090TC08P.stream/playlist.m3u8 소사역 앞
# https://stream6.bcits.go.kr/bucheon/TM096TC04P.stream/playlist.m3u8 역곡남부역 사거리
# https://strm3.spatic.go.kr/live/302.stream/playlist.m3u8 숙대입구역

# import cv2
# import numpy as np
# from ultralytics import YOLO

# # 1. 모델 로드 및 설정
# model = YOLO('yolov8s.pt')

# # --- 2. 비디오 파일 설정 (ROI 설정을 위해 먼저 로드) ---
# video_path = 'https://stream6.bcits.go.kr/bucheon/TM090TC08P.stream/playlist.m3u8' 
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print(f"Error: Could not open video file {video_path}")
#     exit()
    
# # *******************************************************************
# # ** 현재 비디오의 프레임 속성 출력 (디버깅용)            **
# # *******************************************************************
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps_read = cap.get(cv2.CAP_PROP_FPS) # 실제 읽은 FPS 값

# print("--- 비디오 정보 ---")
# print(f"읽어온 프레임 너비 (Width): {frame_width} pixels")
# print(f"읽어온 프레임 높이 (Height): {frame_height} pixels")
# print(f"읽어온 FPS (Frames Per Second): {fps_read}")
# print("------------------")
# # *******************************************************************

# # 프레임 속도 (FPS)와 프레임 간 시간 간격 (델타 t) 계산
# fps = 30
# if fps == 0:
#     fps = 30 
# delta_t = 1.0 / fps 

# # --- 2-1. 마우스 이벤트 처리를 통한 동적 ROI 설정 ---

# # 마우스 클릭으로 선택된 점들을 저장할 리스트
# points = []
# window_name = "Set ROI Points (4 Clicks)"

# # 마우스 이벤트 콜백 함수
# def click_event(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # 점 4개를 초과하지 않도록 제한
#         if len(points) < 4:
#             points.append((x, y))
#             # 점을 표시 (임시)
#             cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
#             cv2.imshow(window_name, frame_copy)
#             print(f"Point {len(points)} set: ({x}, {y})")

# # 비디오의 첫 프레임을 읽어와 ROI 설정에 사용
# success, frame = cap.read()
# if not success:
#     print("Error: Cannot read video frame for ROI setting.")
#     exit()

# frame_copy = frame.copy()
# cv2.namedWindow(window_name)
# cv2.setMouseCallback(window_name, click_event)

# print("--- ROI 설정 단계 ---")
# print("비디오 프레임에 마우스 왼쪽 버튼으로 4개의 점을 순서대로 클릭하세요 (좌상단 -> 우상단 -> 우하단 -> 좌하단).")
# print("4개의 점을 모두 찍은 후 아무 키나 누르세요.")

# # ROI 설정 루프
# while len(points) < 4:
#     # 4개의 점이 찍히거나 키 입력이 있을 때까지 대기
#     cv2.imshow(window_name, frame_copy)
#     if cv2.waitKey(1) & 0xFF != 0xFF:
#         break # 점 4개를 찍지 않았어도 키 입력 시 루프 종료 가능

# cv2.destroyWindow(window_name)

# if len(points) != 4:
#     print("4개의 점이 모두 설정되지 않아 프로그램을 종료합니다.")
#     exit()

# # 마우스로 찍은 4개의 점을 src_pts로 설정
# src_pts = np.float32(points)

# # 2-2. 호모그래피 대상(dst_pts) 설정 및 행렬 계산

# # 사용자가 찍은 점에 맞춰 실제 세계의 폭과 깊이(미터)를 설정합니다.
# # 이 값은 사용자가 현장에 맞춰 수동으로 결정해야 합니다. (예: 폭 8m, 깊이 28m)
# WIDTH_REAL_METERS = 8.0   
# DEPTH_REAL_METERS = 28.0   

# dst_pts = np.float32([
#     [0, 0],              # 1. (0m, 0m) 좌상단
#     [WIDTH_REAL_METERS, 0],     # 2. (W m, 0m) 우상단
#     [WIDTH_REAL_METERS, DEPTH_REAL_METERS], # 3. (W m, L m) 우하단
#     [0, DEPTH_REAL_METERS]      # 4. (0m, L m) 좌하단
# ])

# # 호모그래피 행렬 (H) 계산
# H, _ = cv2.findHomography(src_pts, dst_pts)

# # ROI 필터링 및 시각화용 3차원 다각형을 전역에서 정의
# ROI_POLYGON_3D = np.int32(src_pts).reshape((-1, 1, 2))

# # 4. 추적 및 속도 저장을 위한 딕셔너리 초기화 (메인 루프용)
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 비디오를 처음으로 되감기
# object_tracks = {} 
# object_speeds = {} 

# # --- (calculate_speed 함수는 변경 없이 여기에 위치합니다) ---

# # 속도 계산 함수 (calculate_speed)
# def calculate_speed(track_id, current_x, current_y):
#     """
#     객체의 픽셀 위치 변화를 호모그래피를 사용하여 실제 세계 속도로 변환하여 계산합니다.
#     """
#     if track_id not in object_tracks:
#         object_tracks[track_id] = [current_x, current_y]
#         return None

#     prev_x, prev_y = object_tracks[track_id]
#     object_tracks[track_id] = [current_x, current_y]

#     # 이전 픽셀 위치를 실제 좌표(미터)로 변환
#     prev_coords_pixel = np.array([[[prev_x, prev_y]]], dtype='float32')
#     prev_coords_real = cv2.perspectiveTransform(prev_coords_pixel, H)[0][0]
    
#     # 현재 픽셀 위치를 실제 좌표(미터)로 변환
#     curr_coords_pixel = np.array([[[current_x, current_y]]], dtype='float32')
#     curr_coords_real = cv2.perspectiveTransform(curr_coords_pixel, H)[0][0]
    
#     # 실제 이동 거리 (유클리드 거리, 미터)
#     distance_real = np.sqrt(
#         (curr_coords_real[0] - prev_coords_real[0])**2 + 
#         (curr_coords_real[1] - prev_coords_real[1])**2
#     )
    
#     # ************ 디버깅 라인 추가 ************
#     if track_id == 1 and distance_real > 0: # ID 1번 차량의 이동 거리만 출력
#         print(f"ID {track_id} | Real Dist: {distance_real:.2f} m | FPS: {1/delta_t:.1f}")
#     # *****************************************

#     # 속도 계산 (미터/초) 및 Km/h 변환
#     speed_mps = distance_real / delta_t
#     speed_kmh = speed_mps * 3.6 

#     return speed_kmh

# # 5. 영상 처리 메인 루프
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break

#     # YOLOv8 탐지 및 ByteTrack 추적 수행
#     results = model.track(
#         frame, 
#         persist=True, 
#         tracker="bytetrack.yaml", 
#         verbose=False,
#         classes=[2, 3, 5, 7] 
#     )

#     # 탐지 및 추적 결과 처리
#     if results[0].boxes.id is not None:
#         boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
#         track_ids = results[0].boxes.id.cpu().numpy().astype(int)
#         class_indices = results[0].boxes.cls.cpu().numpy().astype(int)
        
#         for box, track_id, cls_idx in zip(boxes, track_ids, class_indices):
#             x1, y1, x2, y2 = box
            
#             # 객체 중심의 하단(바닥) 픽셀 좌표
#             center_x = (x1 + x2) // 2
#             bottom_y = y2 
            
#             # --- ROI 필터링 로직 ---
#             # 테스트할 점을 np.array(float32)로 명시적으로 변환
#             point_to_test = np.array([center_x, bottom_y], dtype=np.float32) 
            
#             # cv2.pointPolygonTest를 사용하여 점이 ROI 다각형 내부에 있는지 확인
#             is_in_roi = cv2.pointPolygonTest(ROI_POLYGON_3D, point_to_test, False) >= 0

#             if not is_in_roi: 
#                 continue 
            
#             # --- 필터링 끝: ROI 내 객체만 처리 ---
            
#             # 속도 계산
#             speed_kmh = calculate_speed(track_id, center_x, bottom_y)
            
#             # 결과 시각화
#             class_map = {2: 'Car', 3: 'Moto', 5: 'Bus', 7: 'Truck'}
#             class_name = class_map.get(cls_idx, 'Vehicle')
            
#             # 바운딩 박스 그리기 및 레이블 표시
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
#             label = f'{class_name} ID {track_id}'
#             if speed_kmh is not None:
#                 object_speeds[track_id] = speed_kmh 
#                 label += f': {speed_kmh:.1f} km/h'
#             elif track_id in object_speeds:
#                 label += f': {object_speeds[track_id]:.1f} km/h'
                
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     # 호모그래피 소스 영역 표시 (사용자가 방금 찍은 점들)
#     cv2.polylines(frame, [ROI_POLYGON_3D], isClosed=True, color=(0, 255, 255), thickness=4)

#     # 결과 프레임 표시
#     cv2.imshow("YOLOv8 + ByteTrack Speed Measurement", frame)

#     # 'q' 키를 누르면 루프를 종료
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from ultralytics import YOLO

# 1. 모델 로드 및 설정
model = YOLO('yolov8m.pt')

# --- 2. 비디오 파일 설정 (ROI 설정을 위해 먼저 로드) ---
video_path = 'https://strm3.spatic.go.kr/live/302.stream/playlist.m3u8' 

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# 프레임 속성 확인 (디버깅용)
print(f"읽어온 프레임 너비 (Width): {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} pixels")
print(f"읽어온 프레임 높이 (Height): {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} pixels")
print(f"읽어온 FPS (Frames Per Second, 오류 가능): {cap.get(cv2.CAP_PROP_FPS)}")
print("---------------------------------------------")

fps = 30.0 
delta_t = 1.0 / fps 

# --- 2-1. 거리 선계산 값 (폭과 깊이) 입력 받기 ---

print("--- 1단계: 실제 측정 거리 입력 ---")
try:
    WIDTH_REAL_METERS = float(input("1. ROI 영역의 실제 차선의 폭(미터)을 입력하세요 (예: 26.0): "))
    DEPTH_REAL_METERS = float(input("2. ROI 영역의 실제 차선의 깊이/길이(미터)를 입력하세요 (예: 78.0): "))
except ValueError:
    print("오류: 유효한 숫자를 입력해야 합니다.")
    exit()

# 2-2. 마우스 이벤트 처리를 통한 동적 ROI 픽셀 좌표 설정

points = []
window_name = "Set ROI Points (4 Clicks) - Press 's' to start tracking"

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame_copy, str(len(points)), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, frame_copy)
            print(f"Point {len(points)} set: ({x}, {y})")

success, frame = cap.read()
if not success:
    print("Error: Cannot read video frame for ROI setting.")
    exit()

while True:
    frame_copy = frame.copy()
    points = [] # 매 루프마다 초기화 (재설정 가능)
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)
    
    print("\n--- 2단계: 픽셀 좌표 설정 ---")
    print(f"현재 설정된 실제 거리: 폭={WIDTH_REAL_METERS}m, 깊이={DEPTH_REAL_METERS}m")
    print("마우스로 4개의 점을 순서대로 클릭하세요 (좌상단 -> 우상단 -> 우하단 -> 좌하단).")
    print("설정이 완료되면 키보드의 's'키를 누르거나, 재설정하려면 'r'키를 누르세요.")
    
    while len(points) < 4:
        # 's' 키나 'r' 키를 누르지 않는 한 4개의 점을 기다림
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break 
        if key == ord('r'):
            points = []
            break
            
    key = cv2.waitKey(0) & 0xFF # 4개 점 찍은 후 대기
    if key == ord('s') and len(points) == 4:
        break # 설정 완료
    elif key == ord('r'):
        cv2.destroyWindow(window_name)
        continue # 재시도
    elif key == ord('q'):
        exit()
    elif len(points) == 4:
        break # 4개 점 찍고 다른 키 누르면 일단 진행
    else:
        print("경고: 4개의 점이 모두 설정되지 않았습니다. 재시도합니다.")
        cv2.destroyWindow(window_name)
        continue

cv2.destroyWindow(window_name)

if len(points) != 4:
    print("오류: 4개의 점이 모두 설정되지 않아 프로그램을 종료합니다.")
    exit()

# 마우스로 찍은 4개의 점을 src_pts로 설정
src_pts = np.float32(points)

# 2-3. 호모그래피 행렬 계산 (실제 측정값 반영)

# 미리 입력받은 실제 폭과 깊이 값을 dst_pts에 사용
dst_pts = np.float32([
    [0, 0],              # 1. (0m, 0m)
    [WIDTH_REAL_METERS, 0],     # 2. (W m, 0m)
    [WIDTH_REAL_METERS, DEPTH_REAL_METERS], # 3. (W m, L m)
    [0, DEPTH_REAL_METERS]      # 4. (0m, L m)
])

# 호모그래피 행렬 (H) 계산
H, _ = cv2.findHomography(src_pts, dst_pts)

# ROI 필터링 및 시각화용 3차원 다각형을 전역에서 정의
ROI_POLYGON_3D = np.int32(src_pts).reshape((-1, 1, 2))

# 3. 추적 및 속도 저장을 위한 딕셔너리 초기화 (메인 루프용)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 비디오를 처음으로 되감기
object_tracks = {} 
object_speeds = {} 

# --- 4. 속도 계산 함수 (calculate_speed) ---
def calculate_speed(track_id, current_x, current_y):
    # ... (calculate_speed 함수는 이전과 동일)
    if track_id not in object_tracks:
        object_tracks[track_id] = [current_x, current_y]
        return None

    prev_x, prev_y = object_tracks[track_id]
    object_tracks[track_id] = [current_x, current_y]

    prev_coords_pixel = np.array([[[prev_x, prev_y]]], dtype='float32')
    prev_coords_real = cv2.perspectiveTransform(prev_coords_pixel, H)[0][0]
    
    curr_coords_pixel = np.array([[[current_x, current_y]]], dtype='float32')
    curr_coords_real = cv2.perspectiveTransform(curr_coords_pixel, H)[0][0]
    
    distance_real = np.sqrt(
        (curr_coords_real[0] - prev_coords_real[0])**2 + 
        (curr_coords_real[1] - prev_coords_real[1])**2
    )

    speed_mps = distance_real / delta_t
    speed_kmh = speed_mps * 3.6 

    return speed_kmh

# 5. 영상 처리 메인 루프 (이하 동일)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(
        frame, 
        persist=True, 
        tracker="bytetrack.yaml", 
        verbose=False,
        classes=[2, 3, 5, 7] 
    )

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_indices = results[0].boxes.cls.cpu().numpy().astype(int)
        
        for box, track_id, cls_idx in zip(boxes, track_ids, class_indices):
            x1, y1, x2, y2 = box
            
            center_x = (x1 + x2) // 2
            bottom_y = y2 
            
            # --- ROI 필터링 로직 ---
            point_to_test = np.array([center_x, bottom_y], dtype=np.float32) 
            is_in_roi = cv2.pointPolygonTest(ROI_POLYGON_3D, point_to_test, False) >= 0

            if not is_in_roi: 
                continue 
            
            # 속도 계산
            speed_kmh = calculate_speed(track_id, center_x, bottom_y)
            
            # 결과 시각화
            class_map = {2: 'Car', 3: 'Moto', 5: 'Bus', 7: 'Truck'}
            class_name = class_map.get(cls_idx, 'Vehicle')
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            label = f'{class_name} ID {track_id}'
            if speed_kmh is not None:
                object_speeds[track_id] = speed_kmh 
                label += f': {speed_kmh:.1f} km/h'
            elif track_id in object_speeds:
                label += f': {object_speeds[track_id]:.1f} km/h'
                
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 호모그래피 소스 영역 표시 
    cv2.polylines(frame, [ROI_POLYGON_3D], isClosed=True, color=(0, 255, 255), thickness=4)

    cv2.imshow("YOLOv8 + ByteTrack Speed Measurement", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()