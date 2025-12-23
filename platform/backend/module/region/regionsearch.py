import base64
from queue import Queue
import threading
import time
import cv2
from ultralytics import YOLO

from .domain.region import Region
from .domain.detection import Detection
from collections import defaultdict

from model.model_registry import ModelMeta

from .vehiclecounter import VehicleCounter
from ..utils.draw_boxes import plot_detections, draw_regions



class CCTVProcessor:
    def resolve_video_path(self, upload_dir: str, video_id: str) -> str:
        p = upload_dir / video_id
        if not p.exists():
            raise FileNotFoundError(f"video not found: {video_id}")
        return str(p)
    
    def __init__(self):
        self.loadedModel = None
        self.modelMeta = None
        self.cap = None
        self.regions = []
        self.running = False
        self.mode = "track"  # "detect" or "track"
        self.modelclasses = []
        self.time_out = 30  # seconds
        self.time_out_msg = "30초 동안 프레임이 수신되지 않아 종료되었습니다."
        self.firstLoad = True
        
        
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)

        self.inference_thread = None

    def load_model(self, modelTarget='yolo11s', custom_weights=None):
        try:
            if custom_weights and custom_weights.strip():
                self.loadedModel = YOLO(f"model/{custom_weights}.pt")
            else:
                self.loadedModel = YOLO(f"model/{modelTarget}.pt")
            print("Model loaded")
            return True
        except Exception as e:
            print("Error loading model:", e)
            return False

    def open_source(self, source_type, source, upload_dir):
        try:
            if self.cap:
                self.cap.release()

            if source_type == "rtsp":
                self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
            elif source_type == "file":
                # source는 video_id
                path = self.resolve_video_path(upload_dir, source)
                self.cap = cv2.VideoCapture(path)

            elif source_type == "url":
                self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)

            if not self.cap.isOpened():
                print("Failed to open video source")
                return False

            print("Video source opened")
            return True
        except Exception as e:
            print("Error opening source:", e)
            return False

    def point_in_region(self, x, y, region):
        return (region['x'] <= x <= region['x'] + region['w'] and
                region['y'] <= y <= region['y'] + region['h'])

    # --------------------------
    # Inference Worker Thread
    # --------------------------
    def inference_worker(self, counter: VehicleCounter):
        class_names = self.modelMeta.classes
        self.modelclasses = []
        for k in class_names.keys():
            self.modelclasses.append( int(k) )
        delay = 0
        timewait = 0
        if self.cap is not None:
            delay = 1.0 / self.cap.get(cv2.CAP_PROP_FPS)  # ms 단위
            self.cap.get(cv2.CAP_PROP_FPS)
            print("FPS:", self.cap.get(cv2.CAP_PROP_FPS))
            print("Delay between frames (s):", delay)
        while self.running:
            if self.cap is None:
                time.sleep(0.01)
                continue
            # if self.frame_queue.empty():
            #     # # timeout 처리 30초 지나도 프레임이 안들어오면 out
            #     # timewait += 0.01
            #     # if timewait >= self.time_out:
            #     #     print(self.time_out_msg)
            #     #     self.running = False
            #     #     break
            #     # time.sleep(0.01)
            #     continue
            # if self.frame_queue.full():
            #     # print("Frame queue full, skipping frame queue size:", self.frame_queue.qsize())
            #     # emptying the queue to get the latest frame
            #     self.frame_queue.get()
            #     continue  # 최신 프레임만 유지 (이전 프레임 버림)
            frame = None
            try:
                frame = self.frame_queue.get(timeout=0.01)
            except:
                continue

            startTime = time.time()
            # frame = self.frame_queue.get()
            resized_w = 640
            resized_h = 360

            # Resize to lower resolution for speed
            resized = cv2.resize(frame, (resized_w, resized_h))
            plotted_frame = resized.copy()
            detections = []
            
            # 추론
            # if self.mode == "detect":
            #     inference_strategy = PredictInference(self.model)
            #     vehicle_counter = PredictVehicleCounter()
            # else:
            #     inference_strategy = TrackInference(self.model)
            #     vehicle_counter = TrackVehicleCounter()
            
            # detections = inference_strategy.infer(resized, self.regions)
            # vehicle_counter.update(detections)
            
            # self.plot_detections(plotted_frame, detections)
                

            # 추론 전략 선택
            if self.regions:
                for region in self.regions:                    
                    x1, y1, x2, y2 = int(region['x']), int(region['y']), int(region['x'] + region['w']), int(region['y'] + region['h'])
                    results = None
                    # region 단위로 추론
                    regionFrame = resized[y1:y2, x1:x2]
                    if self.mode == "track":
                        results = self.loadedModel.track(
                            regionFrame,                # FIXED SIZE
                            persist=True,
                            classes= self.modelclasses,
                            workers=1,  # to avoid deadlock
                            device=0,
                            verbose=False
                        )
                    else:
                        results = self.loadedModel.predict(
                            regionFrame,                # FIXED SIZE
                            classes= self.modelclasses,
                            conf=self.modelMeta.conf,
                            worker=1,  # to avoid deadlock
                            device=0,
                            verbose=False
                        )
                    
                    
                    # save frame to disk for debug
                    # cv2.imwrite(f"uploaded_videos/debug_region_{region['id']}.jpg", regionFrame)
                    # cv2.imwrite(f"uploaded_videos/debug_fullframe_{region['id']}.jpg", results[0].plot())
                    
                    if results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                        classes = results[0].boxes.cls.cpu().numpy().astype(int)

                        for box, track_id, cls in zip(boxes, track_ids, classes):
                            x1, y1, x2, y2 = box
                            
                            # Adjust box coordinates to full resized frame
                            x1 += region['x']
                            y1 += region['y']
                            x2 += region['x']
                            y2 += region['y']
                            
                            #to int
                            x1 = int(x1)
                            y1 = int(y1)
                            x2 = int(x2)
                            y2 = int(y2)
                            
                            vehicle_class = class_names.get( str(cls), "unknown")

                            color = {
                                'car': (0, 255, 0),
                                'bus': (0, 0, 255),
                                'truck': (0, 165, 255),
                                'motorcycle': (255, 200, 0)
                            }.get(vehicle_class, (255, 255, 255))

                            # bbox
                            cv2.rectangle(plotted_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                            # label
                            cv2.putText(plotted_frame, f"{vehicle_class} #{track_id}",
                                        (int(x1), int(y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                            # region counting
                            counter.add_vehicle(region["id"], vehicle_class, track_id)
                            # print("vehicle_class:", vehicle_class, "track_id:", track_id, "region_id:", region["id"])
                            # cx = (x1 + x2) / 2
                            # cy = (y1 + y2) / 2

                            # if (region['x'] <= cx <= region['x'] + region['w'] and
                            #     region['y'] <= cy <= region['y'] + region['h']):
                            #     counter.add_vehicle(region["id"], vehicle_class, track_id)
                            detections.append(
                                Detection(
                                    x1=int(x1), y1=int(y1),
                                    x2=int(x2), y2=int(y2),
                                    cls=vehicle_class,
                                    track_id=track_id,
                                    region_id=region['id']
                                )
                            )
                            

                        
            else:
                results = None
                # 전체 프레임 단위로 추론 
                if self.mode == "track":
                    results = self.loadedModel.track(
                        resized,
                        persist=True,
                        classes=self.modelclasses,
                        verbose=False
                    )
                else:
                    results = self.loadedModel.predict(
                        resized,
                        classes=self.modelclasses,
                        verbose=False
                    )
                
                # Get original frame size
                orig_h, orig_w = frame.shape[:2]   # 예: 1080, 1920
                # print(orig_h, orig_w , "")

                # Calculate scaling factors
                scale_x = orig_w / resized_w
                scale_y = orig_h / resized_h
                # print(scale_x, scale_y , "scale")
                

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)

                    for box, track_id, cls in zip(boxes, track_ids, classes):
                        x1, y1, x2, y2 = box
                        vehicle_class = class_names.get( str(cls), "unknown")

                        color = {
                            'car': (0, 255, 0),
                            'bus': (0, 0, 255),
                            'truck': (0, 165, 255),
                            'motorcycle': (255, 200, 0)
                        }.get(vehicle_class, (255, 255, 255))

                        # bbox
                        cv2.rectangle(plotted_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                        # label
                        cv2.putText(plotted_frame, f"{vehicle_class} #{track_id}",
                                    (int(x1), int(y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        # region counting
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2

                        if len(self.regions) == 0:
                            # print("vehicle_class:", vehicle_class, "track_id:", track_id)
                            counter.add_vehicle("global", vehicle_class, track_id)
                        else:
                            for region in self.regions:
                                if (region['x'] <= cx <= region['x'] + region['w'] and
                                    region['y'] <= cy <= region['y'] + region['h']):
                                    counter.add_vehicle(region["id"], vehicle_class, track_id)

            # region 그리기 (same resized canvas 기준)
            for region in self.regions:
                cv2.rectangle(
                    plotted_frame,
                    (int(region['x']), int(region['y'])),
                    (int(region['x'] + region['w']), int(region['y'] + region['h'])),
                    (0, 255, 255), 2
                )
                cv2.putText(plotted_frame, f"Region {region['id']}",
                            (int(region['x']), int(region['y']) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            

            # Update counter with current detections (not accumulated)
            counter.update_current_vehicles(detections)
            
            # Encode frame to base64
            # _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
            _, buffer = cv2.imencode(".jpg", plotted_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            # _, buffer = cv2.imencode(".jpg", results[0].plot(), [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            # Push result to queue
            if self.result_queue.full():
                self.result_queue.get()

            self.result_queue.put({
                "frame": frame_base64,
                "detections": detections,
                "orig_size": (orig_w, orig_h),
                "resized_size": (resized_w, resized_h)
            })
            
            endTime = time.time()
            elapsed = endTime - startTime
            
            # if not self.firstLoad and elapsed > delay * 2: # 2배 이상 걸리면 본 프레임을 따라가기 위해 스킵
            #     skip_count = int(elapsed / delay) - 1
            #     skip_count = max(skip_count, 1)  # 최소 1개는 스킵
            #     print(f"[WARNING] Inference time {elapsed:.3f}s exceeds frame delay {delay:.3f}s, skipping {skip_count} frames")
            #     for _ in range(skip_count):
            #         self.cap.grab()   # 프레임 건너뛰기
            # # 첫 로딩은 오래 걸리므로 패스
            # self.firstLoad = False
                    
                    
    # regionsearch.py (CCTVProcessor 내부)
    
    
    # dev=True 일 때 쓰기 좋은 "predict 전용" inference_worker
    # - track_id 없음 (항상 None)
    # - region crop별로 predict 수행
    # - 결과는 기존 result_queue 포맷 그대로 유지
    # - counter.update_current_vehicles(detections)만 사용 (add_vehicle 미사용)

    import base64
    import time
    import cv2

    def inference_worker_predict(self, counter: "VehicleCounter"):
        if self.loadedModel is None or self.modelMeta is None:
            print("inference_worker_predict: model or modelMeta is None")
            return

        # modelMeta.classes: {"2":"car", "3":"motorcycle", ...} 형태라고 가정
        class_names = self.modelMeta.classes

        # YOLO classes 인자에 넣을 int 리스트 만들기
        self.modelclasses = []
        for k in class_names.keys():
            try:
                self.modelclasses.append(int(k))
            except:
                pass

        resized_w, resized_h = 640, 360
        conf = getattr(self.modelMeta, "conf", None)

        while self.running:
            # 프레임 받아오기 (최신 프레임만 쓰려면 main에서 이미 drop하고 있다고 가정)
            try:
                frame = self.frame_queue.get(timeout=0.01)
            except:
                continue

            start_time = time.time()

            # resize (프론트/region 좌표가 640x360 기준으로 넘어온다고 가정)
            resized = cv2.resize(frame, (resized_w, resized_h))
            plotted_frame = resized.copy()
            detections = []

            # regions 없으면 global로 전체 프레임을 1개 region으로 처리
            regions = self.regions if self.regions else [{
                "id": "global",
                "x": 0,
                "y": 0,
                "w": resized_w,
                "h": resized_h
            }]

            # region 단위 predict
            for region in regions:
                rx1 = int(region.get("x", 0))
                ry1 = int(region.get("y", 0))
                rx2 = int(rx1 + int(region.get("w", 0)))
                ry2 = int(ry1 + int(region.get("h", 0)))

                # clamp
                rx1 = max(0, min(rx1, resized_w - 1))
                ry1 = max(0, min(ry1, resized_h - 1))
                rx2 = max(0, min(rx2, resized_w))
                ry2 = max(0, min(ry2, resized_h))
                if rx2 <= rx1 or ry2 <= ry1:
                    continue

                crop = resized[ry1:ry2, rx1:rx2]
                if crop.size == 0:
                    continue

                # predict
                kwargs = {
                    "classes": self.modelclasses,
                    "verbose": False
                }
                if conf is not None:
                    kwargs["conf"] = conf

                results = self.loadedModel.predict(crop, **kwargs)
                if not results or results[0].boxes is None:
                    continue

                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    continue

                xyxys = boxes.xyxy.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)

                # 결과를 full resized 좌표계로 되돌리고 detections 구성
                for (bx1, by1, bx2, by2), cls_id in zip(xyxys, clss):
                    x1 = int(bx1 + rx1)
                    y1 = int(by1 + ry1)
                    x2 = int(bx2 + rx1)
                    y2 = int(by2 + ry1)

                    vehicle_class = class_names.get(str(cls_id), "unknown")

                    # draw bbox + label (track_id 없음)
                    color = {
                        "car": (0, 255, 0),
                        "bus": (0, 0, 255),
                        "truck": (0, 165, 255),
                        "motorcycle": (255, 200, 0),
                    }.get(vehicle_class, (255, 255, 255))

                    cv2.rectangle(plotted_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        plotted_frame,
                        f"{vehicle_class}",
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )

                    detections.append(
                        Detection(
                            x1=x1, y1=y1,
                            x2=x2, y2=y2,
                            cls=vehicle_class,
                            track_id=None,
                            region_id=region["id"]
                        )
                    )

            # region 박스 그리기
            for region in self.regions:
                cv2.rectangle(
                    plotted_frame,
                    (int(region["x"]), int(region["y"])),
                    (int(region["x"] + region["w"]), int(region["y"] + region["h"])),
                    (0, 255, 255), 2
                )
                cv2.putText(
                    plotted_frame,
                    f"Region {region['id']}",
                    (int(region["x"]), max(0, int(region["y"]) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )

            # 현재 프레임 기준 카운트 업데이트 (track_id 필요 없음)
            counter.update_current_vehicles(detections)

            # encode & push
            ok, buffer = cv2.imencode(".jpg", plotted_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except:
                    pass

            self.result_queue.put({
                "frame": frame_base64,
                "detections": detections,
                "orig_size": (frame.shape[1], frame.shape[0]),
                "resized_size": (resized_w, resized_h),
                "elapsed": time.time() - start_time
            })


    # --------------------------
    # Inference Wroker Thread 추론 버전 2
    # --------------------------
    def inference_worker_classv(self, counter: VehicleCounter):
        class_names = self.modelMeta.classes
        self.modelclasses = []
        for k in class_names.keys():
            self.modelclasses.append( int(k) )
        delay = 0
        # if self.cap is not None:
        #     delay = 1.0 / self.cap.get(cv2.CAP_PROP_FPS)  # ms 단위
        #     self.cap.get(cv2.CAP_PROP_FPS)
        #     print("FPS:", self.cap.get(cv2.CAP_PROP_FPS))
        #     print("Delay between frames (s):", delay)
        while self.running:
            if self.cap is None:
                time.sleep(0.01)
                continue
            if self.frame_queue.empty():
                continue
            startTime = time.time()
            frame = self.frame_queue.get()
            resized_w = 640
            resized_h = 360

            # Resize to lower resolution for speed
            resized = cv2.resize(frame, (resized_w, resized_h))
            plotted_frame = resized.copy()
            detections = []
            inference_strategy = None
            vehicle_counter = None
            
            
            # 추론
            if self.mode == "detect":
                inference_strategy = PredictInference(self.loadedModel, self.modelMeta)
                vehicle_counter = PredictVehicleCounter()
                detections = inference_strategy.infer(resized, self.regions)
                vehicle_counter.update(detections)
            else:
                inference_strategy = TrackInference(self.loadedModel, self.modelMeta)
                vehicle_counter = TrackVehicleCounter()
                detections = inference_strategy.infer(resized, self.regions)
                vehicle_counter.update(detections)
                
            # 시각화
            self.plot_detections(plotted_frame, detections)
            draw_regions(plotted_frame, self.regions)
            
            # Update counter with current detections (not accumulated)
            counter.update_current_vehicles(detections)
            
            # Encode frame to base64
            _, buffer = cv2.imencode(".jpg", plotted_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            # Push result to queue
            if self.result_queue.full():
                self.result_queue.get()
            self.result_queue.put({
                "frame": frame_base64,
                "detections": detections,
                "orig_size": (frame.shape[1], frame.shape[0]),
                "resized_size": (resized_w, resized_h)
            })
            
            endTime = time.time()
            elapsed = endTime - startTime
            if elapsed > delay * 2: # 2배 이상 걸리면 본 프레임을 따라가기 위해 스킵
                skip_count = int(elapsed / delay) - 1
                skip_count = max(skip_count, 1)  # 최소 1개는 스킵
                print(f"[WARNING] Inference time {elapsed:.3f}s exceeds frame delay {delay:.3f}s, skipping {skip_count} frames")
                for _ in range(skip_count):
                    self.cap.grab()   # 프레임 건너뛰기
                    
    # --------------------------
    # Inference Thread Starter
    # --------------------------
    def start_inference_thread_classv(self, counter: VehicleCounter):
        if self.inference_thread and self.inference_thread.is_alive():
            return
        
        self.inference_thread = threading.Thread(
            target=self.inference_worker_classv,
            args=(counter,),
            daemon=True
        )        
        self.inference_thread.start()
        print("Inference thread started")
        
    def start_inference_thread(self, counter: VehicleCounter):
        if self.inference_thread and self.inference_thread.is_alive():
            return
        
        self.inference_thread = threading.Thread(
            target=self.inference_worker,
            args=(counter,),
            daemon=True
        )        
        self.inference_thread.start()
        print("Inference thread started")
        
    # regionsearch.py (CCTVProcessor 내부)
    # dev=True일 때 predict worker를 쓰도록 start 함수 하나 추가 (선택)

    def start_inference_thread_auto(self, counter: "VehicleCounter"):
        if self.inference_thread and self.inference_thread.is_alive():
            return

        use_predict = bool(getattr(self.modelMeta, "dev", False))
        print("use_predict", use_predict)

        target = self.inference_worker if use_predict else self.inference_worker

        self.inference_thread = threading.Thread(
            target=target,
            args=(counter,),
            daemon=True
        )
        self.inference_thread.start()
        print("Inference thread started:", "predict" if use_predict else "track")


    def stop_inference_thread(self):
            """현재 inference thread를 안전하게 종료"""
            if self.inference_thread and self.inference_thread.is_alive():
                self.running = False  # worker 루프 종료 신호
                try:
                    self.inference_thread.join(timeout=1.0)
                except:
                    pass

            self.inference_thread = None

            # queue 정리 (중요)
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    break

            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except:
                    break
        
    # --------------------------
    # Detection Plotter
    # --------------------------
        
    def plot_detections(self, plotted_frame, detections):
        for d in detections:
            color = {
                'car': (0, 255, 0),
                'bus': (0, 0, 255),
                'truck': (0, 165, 255),
                'motorcycle': (255, 200, 0)
            }.get(d.cls, (255, 255, 255))

            cv2.rectangle(plotted_frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)

            label = d.cls
            if d.track_id is not None:
                label += f" #{d.track_id}"

            cv2.putText(
                plotted_frame,
                label,
                (d.x1, d.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
    def setModelMeta(self, model_meta: ModelMeta):
        self.modelMeta = model_meta

        

# --------------------------
# Predict Inference Strategy
# --------------------------
class PredictInference:
    def __init__(self, loadedModel, model_meta: ModelMeta):
        self.loadedModel = loadedModel
        self.model_meta = model_meta
        
    def infer(self, frame, regions):
        detections = []

        if not regions:
            regions = [{
                "id": "global",
                "x": 0,
                "y": 0,
                "w": frame.shape[1],
                "h": frame.shape[0]
            }]
        
        for region in regions:
            rx1, ry1 = int(region['x']), int(region['y'])
            rx2 = rx1 + int(region['w'])
            ry2 = ry1 + int(region['h'])

            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue

            results = self.loadedModel.predict(crop, classes=[2,3,5,7], verbose=False)
            boxes = results[0].boxes
            if not boxes:
                continue

            for box, cls in zip(boxes.xyxy.cpu().numpy(),
                                boxes.cls.cpu().numpy().numpy()):
                bx1, by1, bx2, by2 = map(int, box)
                detections.append(
                    Detection(
                        x1=bx1+rx1, y1=by1+ry1,
                        x2=bx2+rx1, y2=by2+ry1,
                        cls=self.class_names[cls],
                        region_id=region['id']
                    )
                )
        return detections
    
class PredictVehicleCounter(VehicleCounter):
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))

    def update(self, detections):
        for d in detections:
            self.counts[d.region_id][d.cls] += 1

    def get_stats(self):
        return self.counts

# --------------------------
# Track Inference Strategy
# --------------------------
class TrackInference:
    def __init__(self, loadedModel, model_meta: ModelMeta):
        self.loadedModel = loadedModel
        self.model_meta = model_meta

    def infer(self, frame, regions):
        detections = []
        results = None
        regionFrame = None
        classKeys = self.model_meta.classes.keys()
        for k in classKeys:
            classKeys = int(k)
            break

        if not regions:
            results = self.loadedModel.track(
                frame,
                persist=True,
                classes=[classKeys],
                conf= self.model_meta.conf,
                verbose=False
            )
            boxes = results[0].boxes
            if boxes is None or boxes.id is None:
                return detections

            for box, cls, tid in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.cls.cpu().numpy().astype(int),
                boxes.id.cpu().numpy().astype(int)
            ):
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2

                region_id = None
                for r in regions:
                    if r['x'] <= cx <= r['x']+r['w'] and r['y'] <= cy <= r['y']+r['h']:
                        region_id = r['id']
                        break

                detections.append(
                    Detection(
                        x1=int(box[0]), y1=int(box[1]),
                        x2=int(box[2]), y2=int(box[3]),
                        cls=self.model_meta.classes[cls],
                        track_id=tid,
                        region_id=region_id
                    )
                )
        else:
            for region in regions:
                rx1, ry1 = int(region['x']), int(region['y'])
                rx2 = rx1 + int(region['w'])
                ry2 = ry1 + int(region['h'])

                regionFrame = frame[ry1:ry2, rx1:rx2]
                if regionFrame.size == 0:
                    continue

                results = self.loadedModel.track(
                    regionFrame,
                    persist=True,
                    classes=[classKeys],
                    conf= self.model_meta.conf,
                    max_det=100,
                    verbose=False
                )
                boxes = results[0].boxes
                if boxes is None or boxes.id is None:
                    continue
                for box, cls, tid in zip(
                    boxes.xyxy.cpu().numpy(),
                    boxes.cls.cpu().numpy().astype(int),
                    boxes.id.cpu().numpy().astype(int)
                ):
                    bx1, by1, bx2, by2 = map(int, box)

                    detections.append(
                        Detection(
                            x1=bx1+rx1, y1=by1+ry1,
                            x2=bx2+rx1, y2=by2+ry1,
                            cls=self.model_meta.classes[str(cls)],
                            track_id=tid,
                            region_id=region['id']
                        )
                    )
        

        # save frame to disk for debug
        if self.model_meta.dev:
            for region in regions:
                cv2.imwrite(f"uploaded_videos/debug_region_{region['id']}.jpg", regionFrame)
                cv2.imwrite(f"uploaded_videos/debug_fullframe_{region['id']}.jpg", results[0].plot())

        
        return detections
    
class TrackVehicleCounter(VehicleCounter):
    def __init__(self):
        self.seen = set()
        self.counts = defaultdict(lambda: defaultdict(int))

    def update(self, detections):
        for d in detections:
            key = (d.region_id, d.track_id)
            if key in self.seen:
                continue
            self.seen.add(key)
            self.counts[d.region_id][d.cls] += 1

    def get_stats(self):
        return self.counts

