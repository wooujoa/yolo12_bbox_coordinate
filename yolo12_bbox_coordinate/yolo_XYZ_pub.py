import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

from vision_msgs.msg import DetectedCrop, DetectedCropArray


class Yolo3DCenterNode(Node):
    def __init__(self):
        super().__init__('yolo_3d_center_node')

        self.image_topic = '/camera/camera/color/image_raw'
        self.depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
        self.camera_info_topic = '/camera/camera/color/camera_info'

        self.bridge = CvBridge()
        self.model = YOLO("/home/jwg/inference/best3.pt")
        self.device = 'cpu'

        self.frame_buffer = []
        self.detection_buffer = []
        self.depth_buffer = []
        self.max_frames = 40
        self.frame_count = 0
        self.got_image = False

        self.camera_info = None
        self.processed = False  # 전송 여부

        self.publisher_ = self.create_publisher(DetectedCropArray, '/detected_crops', 10)

        self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.get_logger().info("Yolo 3D Center Node Started")
        self.timer = self.create_timer(5.0, self.check_topic_status)

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info("Camera info received.")

    def check_topic_status(self):
        self.get_logger().info(f"RGB 프레임 수: {self.frame_count}, Depth 프레임 수: {len(self.depth_buffer)}")
        if not self.got_image:
            self.get_logger().warn(f"이미지 수신 대기 중: '{self.image_topic}'")
        elif not self.processed and self.frame_count >= self.max_frames and len(self.depth_buffer) >= self.max_frames:
            self.get_logger().info("RGB와 Depth 프레임 수 충분. 처리 시작.")
            self.process_best_frame()

    def depth_callback(self, msg):
        if len(self.depth_buffer) < self.max_frames:
            try:
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                self.depth_buffer.append(depth_image.copy())
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
            self.get_logger().error(f"YOLO 추론 실패 또는 Detection 변환 실패: {e}")
            return

        self.frame_buffer.append(frame.copy())
        self.detection_buffer.append(detections)
        self.frame_count += 1
        self.get_logger().info(f"프레임 수신: {self.frame_count}/{self.max_frames}")

    def process_best_frame(self):
        CONFIDENCE_THRESHOLD = 0.6

        if len(self.frame_buffer) < self.max_frames or len(self.depth_buffer) < self.max_frames:
            self.get_logger().warn("RGB 또는 Depth 프레임 부족. 처리 중단.")
            return

        max_objects = 0
        best_index = -1
        for i, det in enumerate(self.detection_buffer):
            if len(det) > max_objects:
                max_objects = len(det)
                best_index = i

        if best_index == -1:
            self.get_logger().warn("감지된 객체가 없습니다.")
            return

        best_depth = self.depth_buffer[best_index]
        best_detections = self.detection_buffer[best_index]

        """
        #camera_info의 내부 파라미터
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        """
        # 수동 보정된 내부 파라미터
        fx = 590.000000
        fy = 590.000000
        cx = 320.000000
        cy = 240.000000
        

        riped_3d_infos = []

        for i, class_id in enumerate(best_detections.class_id):
            conf = best_detections.confidence[i]
            if conf < CONFIDENCE_THRESHOLD:
                continue

            class_name = self.model.names[int(class_id)]
            x1, y1, x2, y2 = best_detections.xyxy[i]
            cx_px = int((x1 + x2) / 2)
            cy_px = int((y1 + y2) / 2)

            if class_name.lower() == 'riped':
                if 0 <= cx_px < best_depth.shape[1] and 0 <= cy_px < best_depth.shape[0]:
                    depth = best_depth[cy_px, cx_px].astype(np.float32) / 1000.0
                    if 0.1 < depth < 10.0:
                        X = (cx_px - cx) * depth / fx
                        Y = (cy_px - cy) * depth / fy
                        Z = depth
                        riped_3d_infos.append((X, Y, Z))
                        self.get_logger().info(f"[riped] 픽셀=({cx_px},{cy_px}) → 3D=(X={X:.2f}, Y={Y:.2f}, Z={Z:.2f})")

        if riped_3d_infos:
            crop_array_msg = DetectedCropArray()
            crop_array_msg.header = Header()
            crop_array_msg.header.stamp = self.get_clock().now().to_msg()
            crop_array_msg.total_objects = len(riped_3d_infos)

            for i, (x, y, z) in enumerate(riped_3d_infos):
                crop = DetectedCrop()
                crop.id = i + 1
                crop.x = float(x * 100)  # m → cm
                crop.y = float(y * 100)  # m → cm
                crop.z = float(z * 100)  # m → cm
                crop_array_msg.objects.append(crop)
                self.get_logger().info(f"[Publish] ID={crop.id}, X={crop.x:.2f}, Y={crop.y:.2f}, Z={crop.z:.2f}")

            self.publisher_.publish(crop_array_msg)
            self.get_logger().info(f"DetectedCropArray 메시지 전송 완료 (총 {crop_array_msg.total_objects}개)")

            self.processed = True  # 전송이 실제로 이루어졌을 때만 처리 완료로 표시
        else:
            self.get_logger().warn("riped 객체가 없어 메시지 전송 생략됨.")


def main(args=None):
    rclpy.init(args=args)
    node = Yolo3DCenterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
