from ultralytics import YOLO
import cv2
import time
import numpy as np
import os 
import sys 

# -------------------------------------------------------------
# --- 🚨🚨 1. 주요 설정 값 (고객님 최종 확정) 🚨🚨 ---
# -------------------------------------------------------------
SPEED_LIMIT = 50.0 # 과속 기준 (km/h)

LINE_UPPER = 219 
LINE_LOWER = 300 
LINE_TOLERANCE = 3 
PIXEL_DISTANCE = abs(LINE_UPPER - LINE_LOWER) # 81 픽셀

DIST_UPWARD_M = 23.0 # 아래 -> 위 (23m)
DIST_DOWNWARD_M = 22.0 # 위 -> 아래 (22m)

PPM_UPWARD = PIXEL_DISTANCE / DIST_UPWARD_M # 81 / 23 ≈ 3.52
PPM_DOWNWARD = PIXEL_DISTANCE / DIST_DOWNWARD_M # 81 / 22 ≈ 3.68

# 1. 비디오 경로 설정
video_path = "https://211.57.45.101/media/L180130/chunklist.m3u8"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("FATAL ERROR: 비디오 스트림/파일을 열 수 없습니다. 경로를 확인해 주세요.")
    exit()

# 2. 모델 로드 및 설정
model = YOLO("runs/detect/d1400b32i1024/weights/d1400b32i1024.pt")

# -------------------------------------------------------------
# --- 3. 캡처 폴더 생성 경로 설정 (스크립트 파일 위치 기준) ---
# -------------------------------------------------------------
VIOLATION_DIR = "speed_violations"

if getattr(sys, 'frozen', False):
    script_dir = os.path.dirname(sys.executable)
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))

VIOLATION_PATH = os.path.join(script_dir, VIOLATION_DIR)

if not os.path.exists(VIOLATION_PATH):
    os.makedirs(VIOLATION_PATH)
    print(f"✅ 과속 캡처 폴더 생성: {VIOLATION_PATH}")
# -------------------------------------------------------------

# -------------------------------------------------------------
# --- 💡 속도 측정 로직 초기화 ---
# -------------------------------------------------------------

tracker_data = {} 

# 계산 함수
def calculate_speed_directional(time_in, time_out, direction_m):
    time_diff = time_out - time_in 
    
    if time_diff > 0.05 and time_diff < 10.0:
        
        speed_mps = direction_m / time_diff
        speed_kmh = speed_mps * 3.6
        
        if speed_kmh < 150: 
            return speed_kmh, time_diff, direction_m
            
    return 0, time_diff, 0.0

# 4. 비디오 프레임 처리
window_name = "YOLO Car Speed Tracker (Finalized Settings with Capture)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 

print("---")
print(f"🚨 측정 라인: Y={LINE_UPPER} ~ Y={LINE_LOWER} (픽셀 거리: {PIXEL_DISTANCE}px, 오차: {LINE_TOLERANCE}px)")
print(f"   ⬆️ Upward (23m): PPM {PPM_UPWARD:.2f} | ⬇️ Downward (22m): PPM {PPM_DOWNWARD:.2f}")
print(f"   🚦 과속 기준: {SPEED_LIMIT} km/h")
print("---")


while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        time.sleep(1) 
        cap = cv2.VideoCapture(video_path)
        continue
    
    current_time = time.time() 
    frame_width = frame.shape[1] 
    
    results = model.track(frame, conf=0.5, persist=True, tracker="bytetrack.yaml", verbose=False)
    
    annotated_frame = frame.copy() 
    detections = results[0].boxes.data.cpu().numpy()
    
    # -------------------------------------------------------------
    # --- 캡처를 위한 기본 화면 정보 추가 ---
    # -------------------------------------------------------------
    
    # 가상 측정 라인 시각화 (캡처용)
    cv2.line(annotated_frame, (0, LINE_UPPER), (frame_width, LINE_UPPER), (0, 255, 255), 2) # Upper Line
    cv2.line(annotated_frame, (0, LINE_LOWER), (frame_width, LINE_LOWER), (255, 0, 0), 2) # Lower Line
    
    # 정보 표시 (방향별 PPM 정보 포함)
    info_text = f"UP: {PPM_UPWARD:.2f} (23m) | DOWN: {PPM_DOWNWARD:.2f} (22m) | Limit: {SPEED_LIMIT}km/h"
    cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    
    if detections.shape[0] > 0: 
        
        for detection in detections:
            if detection.size < 7: continue 

            x1, y1, x2, y2 = detection[:4].astype(int)
            track_id = int(detection[4])
            
            tracking_point_y = y2 

            if track_id not in tracker_data:
                # 'is_speeding' 필드를 추가하여, 한번 과속으로 측정되면 그 상태를 유지
                tracker_data[track_id] = {'time_in': 0.0, 'time_out': 0.0, 'speed': 0.0, 'status': 'Detect', 'start_y': 0, 'is_captured': False, 'is_speeding': False}

            
            # --- 7. 속도 측정 및 계산 (원래 로직) ---
            
            is_at_upper = (tracking_point_y >= LINE_UPPER - LINE_TOLERANCE and tracking_point_y <= LINE_UPPER + LINE_TOLERANCE)
            is_at_lower = (tracking_point_y >= LINE_LOWER - LINE_TOLERANCE and tracking_point_y <= LINE_LOWER + LINE_TOLERANCE)
            
            # Line In
            if tracker_data[track_id]['time_in'] == 0.0:
                if is_at_upper:
                    tracker_data[track_id]['time_in'] = current_time
                    tracker_data[track_id]['start_y'] = LINE_UPPER
                    tracker_data[track_id]['status'] = 'IN_UPPER'
                elif is_at_lower:
                    tracker_data[track_id]['time_in'] = current_time
                    tracker_data[track_id]['start_y'] = LINE_LOWER
                    tracker_data[track_id]['status'] = 'IN_LOWER'

            
            # Line Out & Calculation
            if tracker_data[track_id]['time_in'] != 0.0 and tracker_data[track_id]['time_out'] == 0.0:
                
                is_passing_up = (is_at_upper and tracker_data[track_id]['start_y'] == LINE_LOWER) 
                is_passing_down = (is_at_lower and tracker_data[track_id]['start_y'] == LINE_UPPER) 
                
                if is_passing_up or is_passing_down:
                    tracker_data[track_id]['time_out'] = current_time

                    direction_m = DIST_UPWARD_M if is_passing_up else DIST_DOWNWARD_M 
                    direction_str = "UPWARD (23m)" if is_passing_up else "DOWNWARD (22m)"

                    speed_kmh, time_diff, real_dist_m = calculate_speed_directional(
                        tracker_data[track_id]['time_in'], 
                        tracker_data[track_id]['time_out'],
                        direction_m
                    )
                    
                    tracker_data[track_id]['speed'] = speed_kmh
                    tracker_data[track_id]['status'] = 'DONE' 
                    
                    # 과속 여부 플래그 업데이트 (화면 유지 로직을 위해 추가)
                    if speed_kmh > SPEED_LIMIT:
                        tracker_data[track_id]['is_speeding'] = True
                        
                    
                    # --------------------------------------------------
                    # 💡💡💡 과속 캡처 로직 (화면 출력 로직 분리) 💡💡💡
                    # --------------------------------------------------
                    if speed_kmh > SPEED_LIMIT and not tracker_data[track_id]['is_captured']:
                        
                        # 🚨 1. 캡처 시점에 필요한 정보 그리기 (파일 저장용)
                        bbox_color = (0, 0, 255) # 빨간색
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bbox_color, 1) 
                        cv2.putText(annotated_frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 1) 
                        text = f"{speed_kmh:.1f} km/h VIOLATION!"
                        cv2.putText(annotated_frame, text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2) 


                        # 파일 이름 형식: VIOLATION_ID_SPEED_TIMESTAMP.jpg
                        timestamp = int(current_time * 1000)
                        filename = f"VIOLATION_{track_id}_{speed_kmh:.1f}kmh_{timestamp}.jpg"
                        filepath = os.path.join(VIOLATION_PATH, filename)
                        
                        # 🚨 2. 바운딩 박스가 그려진 상태의 프레임을 저장합니다.
                        cv2.imwrite(filepath, annotated_frame)
                        tracker_data[track_id]['is_captured'] = True 
                        print(f"📸 과속 캡처 저장: {filepath}")
                    
            
            # 8. **바운딩 박스 및 텍스트 표시 (화면 출력용 - 지속 유지 로직)**
            
            speed = tracker_data[track_id]['speed']
            is_speeding = tracker_data[track_id]['is_speeding'] # 새로 추가된 플래그 사용
            
            bbox_color = (255, 255, 255) # 기본값: 흰색
            text_to_display = f"ID {track_id}"
            
            
            if is_speeding: 
                # 💡 과속 차량: 측정 완료/캡처 여부 관계없이 빨간색으로 지속 표시
                bbox_color = (0, 0, 255)  # 빨간색
                text_to_display = f"{speed:.1f} km/h (VIOLATION)"
                text_color = (0, 0, 255)
                
            elif speed > 0:
                # 💡 정상 속도 측정 완료 차량: 녹색으로 표시
                bbox_color = (0, 255, 0)  # 녹색
                text_to_display = f"{speed:.1f} km/h (PASS)"
                text_color = (0, 255, 0)

            elif (min(LINE_UPPER, LINE_LOWER) <= tracking_point_y <= max(LINE_UPPER, LINE_LOWER)):
                # 💡 측정 라인 통과 중인 차량: 노란색으로 표시
                bbox_color = (0, 255, 255) # 노란색 (측정 중)
                text_to_display = f"ID {track_id} (Measuring...)"
                text_color = (0, 255, 255)

            else:
                # 💡 기타 차량 (탐지 중): 흰색
                text_color = (255, 255, 255)
                
            # 바운딩 박스 그리기
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bbox_color, 1) 
            # 트래커 ID 또는 상태 그리기
            cv2.putText(annotated_frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 1) 
            
            # 속도/위반 정보 표시 (ID보다 위에 표시)
            if speed > 0 or is_speeding: 
                cv2.putText(annotated_frame, text_to_display, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2) 
            
            
    # 12. 영상 결과 출력
    cv2.imshow(window_name, annotated_frame)
    
    key = cv2.waitKey(1) & 0xFF 
    
    if key == ord('q'):
        print("프로그램이 Q 키 입력으로 종료됩니다.")
        break
    
# 13. 자원 해제
cap.release()
cv2.destroyAllWindows()