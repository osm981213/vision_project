import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# ✅ YOLO 차량 탐지/추적 + ROI(4점 클릭) + 호모그래피 기반 속도(km/h) 추정
# ✅ 너는 "점(ROI 4점)"은 직접 찍고,
#    "실제 거리(폭/깊이 26,78 같은 값)" 입력은 안 하도록(프리셋 자동) 만들어둠.
# ============================================================

# =========================
# 0) 사용자 설정
# =========================
MODEL_PATH = "solutions/CalibratedSpeed/m_best.pt"
# MODEL_PATH = "solutions/CalibratedSpeed/s_best.pt"

VIDEO_PATH = "https://strm1.spatic.go.kr/live/5.stream/playlist.m3u8"               # 수색교 교차로
# VIDEO_PATH = "https://stream6.bcits.go.kr/bucheon/TM090TC08P.stream/playlist.m3u8"  # 소사역 앞

# ByteTrack 설정 파일(✅ 직접 만들어둔 yaml 경로를 넣어)
TRACKER_YAML = "solutions/CalibratedSpeed/bytetrack_speed.yaml"


# FPS 고정 (캡쳐가 FPS를 이상하게 주는 경우가 있어서)
FPS_FIXED = 30.0
DELTA_T = 1.0 / FPS_FIXED

# YOLO 트래킹 파라미터
CONF = 0.15
IOU = 0.45
IMGSZ = 640
MAX_DET = 300

# 데이터셋 클래스(0~12) 유지
CLASSES = list(range(13))

# =========================
# 1) CCTV별 실제거리(폭/깊이) 프리셋 (✅ 입력 안 받음)
# =========================
PRESET_METERS = {
    "https://stream6.bcits.go.kr/bucheon/TM090TC08P.stream/playlist.m3u8": (26.0, 78.0),  # 소사역 앞(예시)
    "https://strm1.spatic.go.kr/live/5.stream/playlist.m3u8": (30.3, 80.0),                # 수색교 교차로(예시)
}
DEFAULT_W, DEFAULT_D = 26.0, 78.0


# =========================
# 2) 라벨 통합(Truck/Bus/Car/Motorcycle)
# =========================
def get_group_label(model, cls_idx: int) -> str:
    names = model.names
    if isinstance(names, dict):
        raw = names.get(int(cls_idx), str(cls_idx))
    else:
        raw = names[int(cls_idx)] if 0 <= int(cls_idx) < len(names) else str(cls_idx)

    name = str(raw).strip().lower().replace("_", " ").replace("-", " ")

    truck_keywords = ["truck", "axle", "trailer"]
    if any(k in name for k in truck_keywords):
        return "Truck"
    if "bus" in name:
        return "Bus"
    if "car" in name:
        return "Car"
    if "motorcycle" in name or "moto" in name or "bike" in name:
        return "Motorcycle"

    return str(raw)


# =========================
# 3) 스트림 열기(끊기면 재오픈)
# =========================
def open_capture(url: str, retry_sec: float = 1.0, max_retry: int = 30):
    cap = cv2.VideoCapture(url)
    tries = 0
    while not cap.isOpened():
        tries += 1
        if tries >= max_retry:
            raise RuntimeError(f"[ERROR] 스트림 열기 실패: {url}")
        print(f"[WARN] 스트림 열기 실패 → 재시도 {tries}/{max_retry} ...")
        time.sleep(retry_sec)
        cap.release()
        cap = cv2.VideoCapture(url)
    return cap


# =========================
# 4) ROI 4점 찍는 함수 (좌상→우상→우하→좌하)
# =========================
def select_roi_4points(frame):
    """
    - 마우스로 4점 클릭해서 ROI 사각형 잡기
    - r: 리셋 / s: 확정 / q: 종료
    """
    win = "ROI: Click 4 points (TL->TR->BR->BL) | [r]=reset [s]=save [q]=quit"
    points = []
    draw = frame.copy()

    def redraw():
        nonlocal draw
        draw = frame.copy()

        # 점 표시
        for i, (x, y) in enumerate(points, start=1):
            cv2.circle(draw, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(draw, str(i), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 선 표시(찍은 만큼)
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(draw, points[i], points[i + 1], (0, 255, 255), 2)

        # 4점이면 닫아서 표시
        if len(points) == 4:
            cv2.polylines(draw, [np.array(points, dtype=np.int32)], True, (0, 255, 255), 3)

        cv2.imshow(win, draw)

    def on_mouse(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((int(x), int(y)))
                redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(win)
            raise SystemExit
        if key == ord('r'):
            points = []
            redraw()
        if key == ord('s'):
            if len(points) == 4:
                cv2.destroyWindow(win)
                return points
            else:
                print("[WARN] 4점을 다 찍어야 저장됩니다.")


# =========================
# 5) 메인
# =========================
def main():
    # ✅ bytetrack yaml 자동 생성 제거했으므로, 파일 존재 체크만 함
    if not os.path.exists(TRACKER_YAML):
        raise FileNotFoundError(
            f"[ERROR] tracker yaml 파일이 없습니다: {TRACKER_YAML}\n"
            f"       bytetrack_speed.yaml 파일을 해당 경로에 만들어두거나 경로를 수정하세요."
        )

    model = YOLO(MODEL_PATH)

    # ✅ 실제거리(폭/깊이) 자동 적용
    W, D = PRESET_METERS.get(VIDEO_PATH, (DEFAULT_W, DEFAULT_D))
    if VIDEO_PATH not in PRESET_METERS:
        print("[WARN] 이 CCTV는 프리셋이 없음 → DEFAULT 값 사용됨.")
        print("       PRESET_METERS에 (폭,깊이) 추가하면 입력 없이 자동 적용 가능!")
    print(f"[AUTO] 실제거리 적용: WIDTH={W}m, DEPTH={D}m")

    cap = open_capture(VIDEO_PATH)

    # 첫 프레임 읽고 ROI 4점 선택
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("[ERROR] 첫 프레임 읽기 실패(스트림 문제일 수 있음).")

    roi_points = select_roi_4points(first)  # ✅ 너가 직접 점 찍는 부분
    src_pts = np.float32(roi_points)

    # 실제 공간(미터) 좌표(직사각형)
    dst_pts = np.float32([
        [0, 0],
        [W, 0],
        [W, D],
        [0, D]
    ])

    # 호모그래피(픽셀→미터)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    ROI_POLYGON = np.int32(src_pts).reshape((-1, 1, 2))

    # 다시 스트림 시작(안정적으로)
    cap.release()
    cap = open_capture(VIDEO_PATH)

    # 트랙별 이전 위치 / 속도 저장 (끊겨도 표시 유지)
    prev_pos = {}          # track_id -> (x,y)
    last_speed = {}        # track_id -> speed_kmh
    speed_ema = {}         # track_id -> smoothed speed
    EMA_ALPHA = 0.25       # 0~1 (낮을수록 더 부드러움)

    # 스트림 끊김 대비
    fail_count = 0
    FAIL_REOPEN_N = 30

    while True:
        ok, frame = cap.read()
        if not ok:
            fail_count += 1
            print(f"[WARN] 프레임 수신 실패 ({fail_count}/{FAIL_REOPEN_N})")
            time.sleep(0.05)

            if fail_count >= FAIL_REOPEN_N:
                print("[WARN] 스트림 재오픈 시도...")
                cap.release()
                cap = open_capture(VIDEO_PATH)
                fail_count = 0
            continue

        fail_count = 0

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_YAML,
            conf=CONF,
            iou=IOU,
            imgsz=IMGSZ,
            max_det=MAX_DET,
            verbose=False,
            classes=CLASSES,
        )

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), tid, cls_idx, conf_v in zip(boxes, ids, clss, confs):
                cx = (x1 + x2) // 2
                by = y2  # 박스 바닥점(차량 이동 포인트로 더 안정적)

                # ROI 안에 있는지 체크
                if cv2.pointPolygonTest(ROI_POLYGON, (float(cx), float(by)), False) < 0:
                    continue

                # 속도 계산: 이전 픽셀 -> 현재 픽셀을 미터좌표로 변환 후 거리 계산
                if tid in prev_pos:
                    px, py = prev_pos[tid]

                    prev_real = cv2.perspectiveTransform(
                        np.array([[[px, py]]], dtype=np.float32), H
                    )[0][0]
                    curr_real = cv2.perspectiveTransform(
                        np.array([[[cx, by]]], dtype=np.float32), H
                    )[0][0]

                    dist_m = float(np.hypot(curr_real[0] - prev_real[0], curr_real[1] - prev_real[1]))
                    
                    if dist_m < (3.0 / 3.6 *DELTA_T):
                        speed_kmh = 0.0
                    else:
                        speed_kmh = (dist_m / DELTA_T) * 3.6
                        prev_pos[tid] = (cx, by)

                    # EMA(부드럽게)
                    if tid not in speed_ema:
                        speed_ema[tid] = speed_kmh
                    else:
                        speed_ema[tid] = (1 - EMA_ALPHA) * speed_ema[tid] + EMA_ALPHA * speed_kmh

                    last_speed[tid] = speed_ema[tid]

                prev_pos[tid] = (cx, by)

                class_name = get_group_label(model, cls_idx)

                # 박스
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # ✅ 라벨 순서: 차종 -> ID -> 속도 -> conf
                if tid in last_speed:
                    label = f"{class_name} ID {tid} {last_speed[tid]:.1f} km/h conf {conf_v:.2f}"
                else:
                    label = f"{class_name} ID {tid} --.- km/h conf {conf_v:.2f}"

                cv2.putText(frame, label, (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ROI 표시
        cv2.polylines(frame, [ROI_POLYGON], True, (0, 255, 255), 4)

        # 현재 설정값 표시(디버깅)
        cv2.putText(frame, f"conf={CONF} iou={IOU} imgsz={IMGSZ} fps={FPS_FIXED}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("YOLO11 Speed Measurement (ROI Click Required)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # 실행 중 ROI 다시 찍고 싶으면 r
        if key == ord('r'):
            print("[INFO] ROI 재설정 시작")
            roi_points = select_roi_4points(frame)
            src_pts = np.float32(roi_points)
            H, _ = cv2.findHomography(src_pts, dst_pts)
            ROI_POLYGON = np.int32(src_pts).reshape((-1, 1, 2))
            prev_pos.clear()
            last_speed.clear()
            speed_ema.clear()
            print("[INFO] ROI 재설정 완료")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
