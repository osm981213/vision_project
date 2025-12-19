from ultralytics import YOLO
import cv2
import time
import numpy as np
import math 

# ----------------------------------------------------
# 1. 설정 및 초기화
# ----------------------------------------------------
video_path = "https://211.57.45.101/media/3030_video2/chunklist.m3u8"
cap = cv2.VideoCapture(video_path)
model = YOLO("solutions/TOFSpeed/best.pt")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30 
print(f"FPS 설정: {fps}")

# ----------------------------------------------------
# 2. 직진/회전 감지를 위한 변수 설정
# ----------------------------------------------------
vehicle_tracks = {}

# 전체 경로 저장 최대 길이 (판단 지연을 위해 넉넉히 설정)
MAX_TRACK_LENGTH = int(90) 

# 최초 상태 결정을 위한 최소 프레임 (0.3~0.5초)
SHORT_TRACKING_FRAMES = int(10) 

# 정지 판정 기준 (픽셀 변위)
MIN_MOVEMENT_DISPLACEMENT = 3 

# 회전 판단용 윈도우 사이즈
DIRECTION_WINDOW_SIZE = int(20) 

# 회전 판정 기준 각도 (10~15도 추천)
MIN_TURN_ANGLE = 5

# [중요] 회전 판정을 시작할 최소 경로 길이 (이 길이 전까진 STRAIGHT/STOPPED만 나옴)
# 이 값을 80~100 정도로 설정하면 차량이 충분히 주행한 뒤에야 회전 판정이 시작됩니다.
MIN_TRACKING_FRAMES = 90 

current_vehicle_state = {} 
recent_vehicle_buffer = {}
MAX_STATE_BUFFER_TIME = int(fps * 1) 
MAX_REAPPEAR_DISTANCE = 80 

# ----------------------------------------------------
# 3. 방향 (직진/회전) 감지 함수 (수정됨)
# ----------------------------------------------------
def check_direction_state(track_id, track_points):
    current_track_length = len(track_points)
    
    # --- 1단계: 데이터 부족 (PENDING) ---
    if current_track_length < SHORT_TRACKING_FRAMES: 
        return 'PENDING'
    
    # --- 2단계: 공통 정지 판단 (멈춤은 언제든 감지 가능) ---
    short_vec = np.array(track_points[-1]) - np.array(track_points[current_track_length - SHORT_TRACKING_FRAMES])
    short_disp = np.linalg.norm(short_vec)
    
    if short_disp < MIN_MOVEMENT_DISPLACEMENT * 1.5: 
        return 'STOPPED'
    
    # --- 3단계: 길이 기반 회전 판정 제한 ---
    # 경로 길이가 설정한 MIN_TRACKING_FRAMES보다 짧으면 회전 로직을 아예 타지 않고 STRAIGHT 반환
    if current_track_length < MIN_TRACKING_FRAMES:
        return 'STRAIGHT'
        
    # --- 4단계: 충분한 데이터 확보 후 회전 판단 ---
    start_idx = current_track_length - MIN_TRACKING_FRAMES
    mid_idx = current_track_length - DIRECTION_WINDOW_SIZE
    
    vector_a = np.array(track_points[mid_idx]) - np.array(track_points[start_idx])
    vector_b = np.array(track_points[-1]) - np.array(track_points[mid_idx])
    
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    if norm_a == 0 or norm_b == 0:
        return 'STRAIGHT'
        
    cos_angle = np.clip(np.dot(vector_a, vector_b) / (norm_a * norm_b), -1.0, 1.0)
    angle_deg = math.degrees(math.acos(cos_angle))
    cross_product_z = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
    
    if angle_deg < MIN_TURN_ANGLE:
        return 'STRAIGHT'
    else:
        # 외적 결과에 따른 방향 결정
        return 'RIGHT_TURN' if cross_product_z > 0 else 'LEFT_TURN'

# ----------------------------------------------------
# 4. 바운딩 박스 및 경로 그리기 함수
# ----------------------------------------------------
def draw_bbox_and_id(frame, bbox, track_id, color=(0, 255, 0), label_suffix="", draw_path=False):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {track_id} {label_suffix}"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if draw_path and track_id in vehicle_tracks:
        points = vehicle_tracks[track_id]
        if len(points) > 1:
            if "LEFT" in label_suffix:
                path_color = (255, 0, 0)
            elif "RIGHT" in label_suffix:
                path_color = (0, 165, 255)
            else:
                path_color = (150, 150, 150)
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], path_color, 1)

# ----------------------------------------------------
# 5. 메인 루프
# ----------------------------------------------------
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_count += 1
    if not success:
        time.sleep(1)
        cap = cv2.VideoCapture(video_path)
        continue
    
    # 추적 실행
    results = model.track(frame, persist=True, conf=0.4, tracker="bytetrack.yaml", imgsz=1024)
    
    annotated_frame = frame.copy()
    current_track_ids = set()
    
    color_map = {
        'STRAIGHT': (0, 255, 0), 'LEFT_TURN': (255, 0, 0), 'RIGHT_TURN': (0, 165, 255),
        'STOPPED': (128, 0, 128), 'PENDING': (150, 150, 150)
    }

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int) 
        
        for bbox, track_id in zip(bboxes, track_ids):
            current_track_ids.add(track_id)
            x1, y1, x2, y2 = bbox
            current_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            if track_id not in vehicle_tracks:
                vehicle_tracks[track_id] = []
                # 버퍼 복구 로직
                for old_pos, (old_state, _) in list(recent_vehicle_buffer.items()):
                    if np.linalg.norm(np.array(current_center) - np.array(old_pos)) < MAX_REAPPEAR_DISTANCE:
                        current_vehicle_state[track_id] = old_state
                        del recent_vehicle_buffer[old_pos]
                        break
                        
            vehicle_tracks[track_id].append(current_center)
            if len(vehicle_tracks[track_id]) > MAX_TRACK_LENGTH:
                vehicle_tracks[track_id].pop(0)

            # 상태 업데이트 (길이가 짧으면 TURN 판정 제외)
            state_type = check_direction_state(track_id, vehicle_tracks[track_id])
            
            if state_type and state_type != 'PENDING':
                current_vehicle_state[track_id] = state_type
            
            # 그리기
            state = current_vehicle_state.get(track_id, 'PENDING')
            color = color_map.get(state, (150, 150, 150))
            suffix = f"({state.replace('_TURN', '')})"
            draw_path = True if state != 'STOPPED' else False
            draw_bbox_and_id(annotated_frame, bbox, track_id, color, suffix, draw_path)

    # 트래킹 종료 및 버퍼 관리
    for tid in list(vehicle_tracks.keys()):
        if tid not in current_track_ids:
            if tid in current_vehicle_state and current_vehicle_state[tid] != 'PENDING':
                recent_vehicle_buffer[vehicle_tracks[tid][-1]] = (current_vehicle_state[tid], frame_count)
            del vehicle_tracks[tid]
            current_vehicle_state.pop(tid, None)

    # 버퍼 유효기간 체크
    recent_vehicle_buffer = {k: v for k, v in recent_vehicle_buffer.items() if frame_count - v[1] < MAX_STATE_BUFFER_TIME}

    # 카운트 UI
    y_pos = 30
    for s_name, s_color in color_map.items():
        count = sum(1 for v in current_vehicle_state.values() if v == s_name)
        cv2.putText(annotated_frame, f"{s_name}: {count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_color, 2)
        y_pos += 30

    cv2.namedWindow("yolo", cv2.WINDOW_NORMAL)
    cv2.imshow("yolo", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
        
cap.release()
cv2.destroyAllWindows()