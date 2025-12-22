from ultralytics import YOLO
import cv2
import time
import numpy as np
import os 
import sys 

# -------------------------------------------------------------
# --- 1. 주요 설정 값 ---
# -------------------------------------------------------------
SPEED_LIMIT = 50.0 

LINE_UPPER = 219 
LINE_LOWER = 300 
LINE_TOLERANCE = 3 
PIXEL_DISTANCE = abs(LINE_UPPER - LINE_LOWER)

DIST_UPWARD_M = 23.0 # 아래 -> 위 (23m)
DIST_DOWNWARD_M = 22.0 # 위 -> 아래 (22m)

# 1. 비디오 경로 설정
video_path = "https://211.57.45.101/media/L180130/chunklist.m3u8"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("FATAL ERROR: 비디오 스트림을 열 수 없습니다.")
    exit()

# 2. 모델 로드
model = YOLO("solutions/TOFSpeed/best.pt")

# 3. 캡처 폴더 설정
VIOLATION_DIR = "speed_violations"
if getattr(sys, 'frozen', False):
    script_dir = os.path.dirname(sys.executable)
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))

VIOLATION_PATH = os.path.join(script_dir, VIOLATION_DIR)
if not os.path.exists(VIOLATION_PATH):
    os.makedirs(VIOLATION_PATH)

# -------------------------------------------------------------
# --- 4. 데이터 및 함수 ---
# -------------------------------------------------------------
tracker_data = {} 

def calculate_speed_directional(time_in, time_out, direction_m):
    time_diff = time_out - time_in 
    if 0.05 < time_diff < 10.0:
        speed_kmh = (direction_m / time_diff) * 3.6
        if speed_kmh < 150: 
            return speed_kmh
    return 0

window_name = "YOLO Car Speed Tracker"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        time.sleep(1) 
        cap = cv2.VideoCapture(video_path)
        continue
    
    current_time = time.time() 
    frame_width = frame.shape[1] 
    results = model.track(frame, conf=0.3, persist=True, tracker="bytetrack.yaml", verbose=False)
    annotated_frame = frame.copy() 
    
    # 가이드 라인 표시
    cv2.line(annotated_frame, (0, LINE_UPPER), (frame_width, LINE_UPPER), (0, 255, 255), 2)
    cv2.line(annotated_frame, (0, LINE_LOWER), (frame_width, LINE_LOWER), (255, 0, 0), 2)
    
    if results[0].boxes.id is not None:
        detections = results[0].boxes.data.cpu().numpy()
        
        for detection in detections:
            if detection.size < 7: continue 

            x1, y1, x2, y2 = detection[:4].astype(int)
            track_id = int(detection[4])
            
            if track_id not in tracker_data:
                tracker_data[track_id] = {
                    'time_in': 0.0, 'time_out': 0.0, 'speed': 0.0, 
                    'status': 'Detect', 'start_y': 0, 'is_captured': False, 
                    'is_speeding': False, 'direction': None
                }

            # 방향 판별 및 기준점 설정
            if tracker_data[track_id]['direction'] == 'UP':
                tracking_point_y = y1
            elif tracker_data[track_id]['direction'] == 'DOWN':
                tracking_point_y = y2
            else:
                tracking_point_y = y1 if abs(y1 - LINE_LOWER) < abs(y2 - LINE_UPPER) else y2

            is_at_upper = (LINE_UPPER - LINE_TOLERANCE <= tracking_point_y <= LINE_UPPER + LINE_TOLERANCE)
            is_at_lower = (LINE_LOWER - LINE_TOLERANCE <= tracking_point_y <= LINE_LOWER + LINE_TOLERANCE)
            
            # 시작 지점 기록
            if tracker_data[track_id]['time_in'] == 0.0:
                if is_at_lower:
                    tracker_data[track_id].update({'time_in': current_time, 'start_y': LINE_LOWER, 'direction': 'UP'})
                elif is_at_upper:
                    tracker_data[track_id].update({'time_in': current_time, 'start_y': LINE_UPPER, 'direction': 'DOWN'})

            # 종료 지점 기록 및 계산
            if tracker_data[track_id]['time_in'] != 0.0 and tracker_data[track_id]['time_out'] == 0.0:
                is_passing_up = (is_at_upper and tracker_data[track_id]['direction'] == 'UP')
                is_passing_down = (is_at_lower and tracker_data[track_id]['direction'] == 'DOWN')
                
                if is_passing_up or is_passing_down:
                    tracker_data[track_id]['time_out'] = current_time
                    dist_m = DIST_UPWARD_M if is_passing_up else DIST_DOWNWARD_M 
                    speed_kmh = calculate_speed_directional(tracker_data[track_id]['time_in'], current_time, dist_m)
                    
                    tracker_data[track_id]['speed'] = speed_kmh
                    if speed_kmh > SPEED_LIMIT:
                        tracker_data[track_id]['is_speeding'] = True

                    # 과속 캡처
                    if speed_kmh > SPEED_LIMIT and not tracker_data[track_id]['is_captured']:
                        cap_frame = annotated_frame.copy()
                        cv2.rectangle(cap_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(cap_frame, f"VIOLATION: {speed_kmh:.1f}kmh", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        filename = f"V_{track_id}_{speed_kmh:.1f}kmh_{int(current_time*1000)}.jpg"
                        cv2.imwrite(os.path.join(VIOLATION_PATH, filename), cap_frame)
                        tracker_data[track_id]['is_captured'] = True

            # --- 화면 표시 로직 ---
            speed = tracker_data[track_id]['speed']
            is_speeding = tracker_data[track_id]['is_speeding']
            
            if is_speeding:
                color, label = (0, 0, 255), f"{speed:.1f} km/h (VIOLATION)"
            elif speed > 0:
                color, label = (0, 255, 0), f"{speed:.1f} km/h (PASS)"
            elif tracker_data[track_id]['time_in'] > 0:
                color, label = (0, 255, 255), f"ID {track_id}" # (Measuring...) 제거됨
            else:
                color, label = (255, 255, 255), f"ID {track_id}"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow(window_name, annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    
cap.release()
cv2.destroyAllWindows()