# import cv2
# from ultralytics import YOLO
# from ultralytics.solutions import SpeedEstimator 
# # YOLO 모델은 SpeedEstimator 객체에 인수로 전달되므로, 여기서 로드할 필요는 없습니다.
# # model = YOLO("yolov11n.pt") 

# # 2. 입/출력 파일 경로 설정 
# input_video_path = "https://strm3.spatic.go.kr/live/312.stream/playlist.m3u8" 
# output_video_path = "speed_test_result.mp4"
# cap = cv2.VideoCapture(input_video_path)
# if not cap.isOpened(): exit()
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
# print(f"입력 비디오 W:{w}, H:{h}, FPS:{fps} / 속도 측정 시작...")


# # 4. ⭐️ 속도 측정기 (SpeedEstimator) 초기화 (모든 설정을 여기에 집중)
# # -----------------------------------------------------------------------------------------
# # COCO 데이터셋 차량 관련 클래스 인덱스
# vehicle_classes = [2, 3, 5, 7] 

# # 🚨 line_pts (region) 설정은 SpeedEstimator의 내부 추적 영역을 설정합니다.
# # 이 값은 초기화 시점에 인수로 전달합니다.
# # 중앙 라인: (w // 2, h)와 (w // 2, h // 4) 
# line_pts = [(360,1280), (360,320)] 

# # SpeedEstimator 객체 생성 시 모든 인수를 전달합니다.
# # SpeedEstimator가 model.track() 기능을 내부적으로 실행합니다.
# speed_obj = SpeedEstimator(
#     model="yolo11l.pt",
#     fps=fps,
#     classes=vehicle_classes,
#     region=line_pts,
#     meter_per_pixel=0.00006,
#     max_speed = 120,
#     show=True,
#     max_hist = 3,
#     conf= 0.3,
#     iou = 0.5,
#     tracker = "bytetrack.yaml"
# ) 
# # -----------------------------------------------------------------------------------------


# # 5. 비디오 프레임 처리 루프
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success: break
    
#     # 6. ⭐️ 속도 계산 및 시각화 (최종 API 호출)
#     # 🚨 SpeedEstimator 객체 자체를 원본 프레임만 가지고 호출합니다.
#     #    객체가 내부적으로 추적(Tracking)과 속도 계산을 모두 처리합니다.
#     results = speed_obj(frame) 
    
#     # 7. 시각화 (결과의 plot_im 속성 사용)
#     # SpeedEstimator는 SolutionResults 객체를 반환하며, 이미지 데이터는 plot_im에 담겨있습니다.
#     annotated_frame = results.plot_im
    
#     # 8. 출력 비디오에 프레임 쓰기
#     video_writer.write(annotated_frame)

# # 9. 종료 및 리소스 해제
# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()
# print(f"처리 완료. 결과 파일: {output_video_path}")

import cv2
import numpy as np

# 1. 사용할 CCTV 영상 파일 경로
video_path = 'https://stream6.bcits.go.kr/bucheon/TM090TC08P.stream/playlist.m3u8' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# 첫 번째 프레임을 읽어와서 픽셀 선택에 사용
success, frame = cap.read()
if not success:
    print("Error: Failed to read the first frame.")
    exit()

# 2. 전역 변수 설정
selected_points = []
window_name = "Select 4 Source Points (src_pts)"

# 3. 마우스 이벤트 핸들러 함수
def get_points(event, x, y, flags, param):
    """마우스 클릭 이벤트를 처리하고 픽셀 좌표를 저장합니다."""
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) < 4:
            # 클릭된 픽셀 좌표 저장
            selected_points.append((x, y))
            
            # 시각적 피드백: 클릭한 위치에 작은 원 그리기
            cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
            
            # 현재까지 선택된 점의 개수 표시
            text = f"Point {len(selected_points)}: ({x}, {y})"
            cv2.putText(frame_copy, text, (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            print(f"Selected Point {len(selected_points)}: ({x}, {y})")
            
            # 4개의 점이 모두 선택되면 다각형을 그려 시각화
            if len(selected_points) == 4:
                # 4개의 점을 연결하여 직사각형 영역 표시 (노란색)
                pts = np.array(selected_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame_copy, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
                print("\n--- 4개 점 선택 완료 ---")
                print("Press 'q' to finish and see the final src_pts array.")

# 4. 마우스 이벤트 리스너 설정 및 프레임 표시
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, get_points)

print("--- 4개의 소스 포인트(src_pts)를 순서대로 클릭하세요 ---")
print("1. 좌상단 (Upper-Left)")
print("2. 우상단 (Upper-Right)")
print("3. 좌하단 (Lower-Left)")
print("4. 우하단 (Lower-Right)")

while True:
    # 원본 프레임을 복사하여 마크를 그립니다.
    frame_copy = frame.copy()
    
    # 선택된 점들을 기반으로 시각화 (선택이 진행 중일 때)
    if 1 < len(selected_points) <= 4:
        for i in range(len(selected_points)):
            cv2.circle(frame_copy, selected_points[i], 5, (0, 255, 0), -1)
        
        # 현재까지의 점들을 연결
        if len(selected_points) == 4:
            pts = np.array(selected_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame_copy, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    cv2.imshow(window_name, frame_copy)
    
    # 'q' 키를 누르거나 4개의 점이 모두 선택되면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q') or len(selected_points) == 4:
        break

cv2.destroyAllWindows()
cap.release()

# 5. 최종 src_pts 배열 출력
if len(selected_points) == 4:
    # np.float32 형식으로 변환하여 출력
    final_src_pts = np.float32(selected_points)
    print("\n--- 최종 src_pts 배열 (코드로 사용) ---")
    print(f"src_pts = {final_src_pts}")
else:
    print("4개의 점이 모두 선택되지 않았습니다. 다시 실행해주세요.")