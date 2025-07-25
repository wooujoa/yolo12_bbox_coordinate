import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from std_msgs.msg import Header, Bool  ## Bool 메시지 타입 추가
from cv_bridge import CvBridge
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import gc  # 메모리 수거

from vision_msgs.msg import DetectedCrop, DetectedCropArray


class Yolo3DCenterNode(Node):
    def __init__(self):
        super().__init__('yolo_3d_center_node')

        self.image_topic = '/camera/camera/color/image_raw/compressed'
        self.depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
        self.camera_info_topic = '/camera/camera/color/camera_info'
        ## 활성화 토픽 추가
        self.activate_topic = '/yolo_activate'

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
        self.processed = False
        
        ## 활성화 상태 플래그 추가 - 초기값은 False (비활성화)
        self.activated = False

        self.publisher_ = self.create_publisher(DetectedCropArray, '/detected_crops', 10)

        ## YOLO 활성화 신호를 받는 subscriber 추가
        self.create_subscription(Bool, self.activate_topic, self.activate_callback, 10)
        self.create_subscription(CompressedImage, self.image_topic, self.compressed_image_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        ## 시작 메시지 변경 - 활성화 대기 상태임을 명시
        self.get_logger().info("Yolo 3D Center Node Started - Waiting for activation signal...")
        self.timer = self.create_timer(5.0, self.check_topic_status)

    ## 새로 추가된 함수 - YOLO 활성화 신호 받는 콜백
    def activate_callback(self, msg):
        """YOLO 활성화 신호 받는 콜백"""
        if msg.data and not self.activated:
            self.activated = True
            self.get_logger().info("YOLO 노드 활성화됨! 프레임 수집 시작...")
            # 기존 버퍼 초기화 (혹시 모를 이전 데이터 제거)
            self.reset_buffers()
        elif not msg.data:
            self.activated = False
            self.get_logger().info("YOLO 노드 비활성화됨")

    ## 새로 추가된 함수 - 버퍼 초기화
    def reset_buffers(self):
        """버퍼 초기화"""
        self.frame_buffer.clear()
        self.detection_buffer.clear()
        self.depth_buffer.clear()
        self.frame_count = 0
        self.processed = False
        self.got_image = False

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info("Camera info received.")

    def check_topic_status(self):
        ## 활성화 상태 체크 추가 - 비활성화 상태면 대기 메시지만 출력
        if not self.activated:
            self.get_logger().info("활성화 대기 중... (/yolo_activate 토픽 신호 필요)")
            return
            
        self.get_logger().info(f"RGB 프레임 수: {self.frame_count}, Depth 프레임 수: {len(self.depth_buffer)}")
        if not self.got_image:
            self.get_logger().warn(f"이미지 수신 대기 중: '{self.image_topic}'")
        elif not self.processed and self.frame_count >= self.max_frames and len(self.depth_buffer) >= self.max_frames:
            self.get_logger().info("RGB와 Depth 프레임 수 충분. 처리 시작.")
            self.process_best_frame()

    def depth_callback(self, msg):
        ## 활성화 상태 체크 추가 - 비활성화 상태면 처리하지 않음
        if not self.activated:
            return
            
        if len(self.depth_buffer) < self.max_frames:
            try:
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                self.depth_buffer.append(depth_image)  # copy() 제거
            except Exception as e:
                self.get_logger().error(f"Depth 변환 실패: {e}")

    def compressed_image_callback(self, msg):
        ## 활성화 상태 체크 추가 - 비활성화 상태면 처리하지 않음
        if not self.activated:
            return
            
        if self.frame_count >= self.max_frames or self.camera_info is None:
            return
        self.got_image = True

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f"Compressed Image 디코딩 실패: {e}")
            return

        try:
            results = self.model(frame, imgsz=640, device=self.device)[0]
            detections = sv.Detections.from_ultralytics(results)
            self.get_logger().info(f"[YOLO] 감지 객체 수: {len(detections)}")
        except Exception as e:
            self.get_logger().error(f"YOLO 추론 실패 또는 Detection 변환 실패: {e}")
            return

        self.frame_buffer.append(frame)  # copy 제거
        self.detection_buffer.append(detections)
        self.frame_count += 1

        self.get_logger().debug(f"버퍼 상태: frame_buffer={len(self.frame_buffer)}, "
                                f"detection_buffer={len(self.detection_buffer)}, "
                                f"depth_buffer={len(self.depth_buffer)}")

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

        fx = 590.0
        fy = 590.0
        cx = 320.0
        cy = 240.0

        riped_3d_infos = []

        depths = []
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
                        depths.append((i, class_id, cx_px, cy_px, depth))

        # 최소 Z값 (가장 가까운 객체의 depth)
        if not depths:
            self.get_logger().warn("riped 객체가 없어 메시지 전송 생략됨.")
            return

        min_z = min(d[-1] for d in depths)  # 가장 가까운 z값

        # 최종 객체 정보 생성
        riped_3d_infos = []
        for idx, class_id, cx_px, cy_px, depth in depths:
            X = (cx_px - cx) * min_z / fx  # 가장 작은 Z값 사용
            Y = (cy_px - cy) * min_z / fy  # 가장 작은 Z값 사용
            Z = depth                     # 원래 depth 사용
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
                crop.x = float(x)
                crop.y = float(y)
                crop.z = float(z)
                crop_array_msg.objects.append(crop)
                self.get_logger().info(f"[Publish] ID={crop.id}, X={crop.x:.2f}, Y={crop.y:.2f}, Z={crop.z:.2f}")

            self.publisher_.publish(crop_array_msg)
            self.get_logger().info(f"DetectedCropArray 메시지 전송 완료 (총 {crop_array_msg.total_objects}개)")
        else:
            self.get_logger().warn("riped 객체가 없어 메시지 전송 생략됨.")

        # 메모리 해제 및 GC
        self.frame_buffer.clear()
        self.detection_buffer.clear()
        self.depth_buffer.clear()
        self.frame_count = 0
        self.processed = False
        ## 처리 완료 후 비활성화 상태로 변경 (기존에는 activated 변수 없었음)
        self.activated = False  # 처리 완료 후 비활성화

        self.get_logger().info("버퍼 초기화 완료. 프레임/메모리 해제됨.")
        gc.collect()
        self.get_logger().debug("GC 강제 실행 완료.")
        
        ## 메시지 변경 및 노드 종료 제거 - 계속 실행되도록 수정
        self.get_logger().info("처리 완료. 다음 활성화 신호 대기 중...")
        # rclpy.shutdown()  ## 기존 노드 종료 코드 제거

def main(args=None):
    rclpy.init(args=args)
    node = Yolo3DCenterNode()
    ## 노드가 계속 실행되어 여러 번 활성화 신호를 받을 수 있도록 유지
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()