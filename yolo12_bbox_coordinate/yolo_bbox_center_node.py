#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2


class Yolo3DCenterNode(Node):
    def __init__(self):
        super().__init__('yolo_3d_center_node')

        # ───────── topic 설정 ─────────
        self.image_topic = '/camera/camera/color/image_raw'
        self.depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
        self.camera_info_topic = '/camera/camera/color/camera_info'

        # ───────── 기본 객체 ─────────
        self.bridge = CvBridge()
        self.model = YOLO("/home/jwg/inference/best.pt")
        self.device = 'cpu'

        # ───────── 버퍼 및 상태 ─────────
        self.frame_buffer = []
        self.detection_buffer = []
        self.depth_buffer = []
        self.max_frames = 40
        self.frame_count = 0
        self.got_image = False
        self.processed = False
        self.finished = False          # ★ 처리 완료 플래그

        self.camera_info = None

        # ───────── 구독자 ─────────
        self.create_subscription(Image,  self.image_topic,      self.image_callback, 10)
        self.create_subscription(Image,  self.depth_topic,      self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        # ───────── 타이머 ─────────
        self.timer = self.create_timer(5.0, self.check_topic_status)

        self.get_logger().info("Yolo 3D Center Node Started")

    # ────────────────────────────────────────────
    # 콜백
    # ────────────────────────────────────────────
    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info("Camera info received.")

    def check_topic_status(self):
        self.get_logger().info(
            f"RGB 프레임 수: {self.frame_count}, Depth 프레임 수: {len(self.depth_buffer)}"
        )
        if not self.got_image:
            self.get_logger().warn(f"이미지 수신 대기 중: '{self.image_topic}'")
        elif (not self.processed and
              self.frame_count >= self.max_frames and
              len(self.depth_buffer) >= self.max_frames):
            self.get_logger().info("RGB와 Depth 프레임 수 충분. 처리 시작.")
            self.process_best_frame()
            self.processed = True

    def depth_callback(self, msg):
        if len(self.depth_buffer) < self.max_frames:
            try:
                depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                self.depth_buffer.append(depth_img.copy())
            except Exception as e:
                self.get_logger().error(f"Depth 변환 실패: {e}")

    def image_callback(self, msg):
        if self.frame_count >= self.max_frames or self.camera_info is None:
            return
        self.got_image = True
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image 변환 실패: {e}")
            return

        try:
            results = self.model(frame, imgsz=640, device=self.device)[0]
            detections = sv.Detections.from_ultralytics(results)
        except Exception as e:
            self.get_logger().error(f"YOLO 추론 실패/Detection 변환 실패: {e}")
            return

        self.frame_buffer.append(frame.copy())
        self.detection_buffer.append(detections)
        self.frame_count += 1
        self.get_logger().info(f"프레임 수신: {self.frame_count}/{self.max_frames}")

    # ────────────────────────────────────────────
    # 핵심 처리 함수
    # ────────────────────────────────────────────
    def process_best_frame(self):
        CONFIDENCE_THRESHOLD = 0.6

        if len(self.frame_buffer) < self.max_frames or len(self.depth_buffer) < self.max_frames:
            self.get_logger().warn("RGB 또는 Depth 프레임 부족. 처리 중단.")
            return

        # ── 가장 객체 많은 프레임 선택 ──
        best_index = max(
            range(len(self.detection_buffer)),
            key=lambda i: len(self.detection_buffer[i]),
            default=-1
        )
        if best_index == -1:
            self.get_logger().warn("감지된 객체가 없습니다.")
            return

        best_frame      = self.frame_buffer[best_index]
        best_depth      = self.depth_buffer[best_index]
        best_detections = self.detection_buffer[best_index]

        # ── intrinsic ──
        fx, fy = self.camera_info.k[0], self.camera_info.k[4]
        cx, cy = self.camera_info.k[2], self.camera_info.k[5]

        box_annotator   = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        labels, riped_3d_infos = [], []
        annotated = best_frame.copy()
        filt_xyxy, filt_ids, filt_conf = [], [], []

        for i, class_id in enumerate(best_detections.class_id):
            conf = best_detections.confidence[i]
            if conf < CONFIDENCE_THRESHOLD:
                continue

            class_name = self.model.names[int(class_id)]
            x1, y1, x2, y2 = best_detections.xyxy[i]
            cx_px, cy_px  = int((x1+x2)/2), int((y1+y2)/2)

            labels.append(f"ID={len(riped_3d_infos)+1}")
            filt_xyxy.append([x1, y1, x2, y2])
            filt_ids.append(class_id)
            filt_conf.append(conf)

            if class_name.lower() == 'riped':
                if 0 <= cx_px < best_depth.shape[1] and 0 <= cy_px < best_depth.shape[0]:
                    depth = best_depth[cy_px, cx_px].astype(np.float32) / 1000.0
                    if depth == 0.0:
                        win = best_depth[max(0, cy_px-1):cy_px+2,
                                         max(0, cx_px-1):cx_px+2]
                        nonzero = win[win > 0]
                        if nonzero.size:
                            depth = np.mean(nonzero)/1000.0
                    if 0.1 < depth < 10.0:
                        X = (cx_px - cx) * depth / fx
                        Y = (cy_px - cy) * depth / fy
                        Z = depth
                        riped_3d_infos.append(("riped", (cx_px, cy_px), (X, Y, Z)))

                        cv2.circle(annotated, (cx_px, cy_px), 5, (0,255,0), -1)
                        cv2.putText(annotated, f"({cx_px},{cy_px})",
                                    (cx_px+5, cy_px-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                        cv2.putText(annotated, f"ID={len(riped_3d_infos)}",
                                    (cx_px-10, cy_px-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        # ── 어노테이션 마무리 ──
        dets = sv.Detections(
            xyxy=np.array(filt_xyxy),
            class_id=np.array(filt_ids),
            confidence=np.array(filt_conf)
        ) if filt_xyxy else sv.Detections.empty()

        annotated = box_annotator.annotate(scene=annotated, detections=dets)
        annotated = label_annotator.annotate(scene=annotated, detections=dets, labels=labels)

        # 카메라 중심축 시각화
        cam_c = (int(cx), int(cy))
        cv2.arrowedLine(annotated, cam_c, (cam_c[0]+50, cam_c[1]), (0,0,255), 2)
        cv2.arrowedLine(annotated, cam_c, (cam_c[0], cam_c[1]+50), (0,255,0), 2)
        cv2.circle(annotated, cam_c, 3, (255,0,0), -1)

        cv2.imwrite("best_detection_frame.jpg", annotated)

        # 로그 출력
        if riped_3d_infos:
            self.get_logger().info("=== riped 객체 중심 및 3D 좌표 ===")
            for _, (u,v), (x,y,z) in riped_3d_infos:
                self.get_logger().info(f"riped @ ({u},{v}) → X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
        else:
            self.get_logger().info("3D 좌표를 계산할 riped 객체가 없습니다.")

        # ── ★ 처리 완료 플래그 켜기 ──
        self.finished = True
        self.get_logger().info("처리 완료! 노드 정리 대기 중...")

# ────────────────────────────────────────────
# main
# ────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)

    node = Yolo3DCenterNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    try:
        # finished 될 때까지만 spin
        while rclpy.ok() and not node.finished:
            executor.spin_once(timeout_sec=0.1)
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()
