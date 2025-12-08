import cv2
from ultralytics import YOLO
from ultralytics.solutions import SpeedEstimator 
# YOLO 모델은 SpeedEstimator 객체에 인수로 전달되므로, 여기서 로드할 필요는 없습니다.
# model = YOLO("yolov11n.pt") 

# 2. 입/출력 파일 경로 설정 
input_video_path = "https://strm3.spatic.go.kr/live/312.stream/playlist.m3u8" 
output_video_path = "speed_test_result.mp4"
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened(): exit()
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
print(f"입력 비디오 W:{w}, H:{h}, FPS:{fps} / 속도 측정 시작...")


# 4. ⭐️ 속도 측정기 (SpeedEstimator) 초기화 (모든 설정을 여기에 집중)
# -----------------------------------------------------------------------------------------
# COCO 데이터셋 차량 관련 클래스 인덱스
vehicle_classes = [2, 3, 5, 7] 

# 🚨 line_pts (region) 설정은 SpeedEstimator의 내부 추적 영역을 설정합니다.
# 이 값은 초기화 시점에 인수로 전달합니다.
# 중앙 라인: (w // 2, h)와 (w // 2, h // 4) 
line_pts = [(360,1280), (360,320)] 

# SpeedEstimator 객체 생성 시 모든 인수를 전달합니다.
# SpeedEstimator가 model.track() 기능을 내부적으로 실행합니다.
speed_obj = SpeedEstimator(
    model="yolo11l.pt",
    fps=fps,
    classes=vehicle_classes,
    region=line_pts,
    meter_per_pixel=0.00006,
    max_speed = 120,
    show=True,
    max_hist = 3,
    conf= 0.3,
    iou = 0.5,
    tracker = "bytetrack.yaml"
) 
# -----------------------------------------------------------------------------------------


# 5. 비디오 프레임 처리 루프
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    # 6. ⭐️ 속도 계산 및 시각화 (최종 API 호출)
    # 🚨 SpeedEstimator 객체 자체를 원본 프레임만 가지고 호출합니다.
    #    객체가 내부적으로 추적(Tracking)과 속도 계산을 모두 처리합니다.
    results = speed_obj(frame) 
    
    # 7. 시각화 (결과의 plot_im 속성 사용)
    # SpeedEstimator는 SolutionResults 객체를 반환하며, 이미지 데이터는 plot_im에 담겨있습니다.
    annotated_frame = results.plot_im
    
    # 8. 출력 비디오에 프레임 쓰기
    video_writer.write(annotated_frame)

# 9. 종료 및 리소스 해제
cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"처리 완료. 결과 파일: {output_video_path}")