#!/usr/bin/env python3
"""
Subscribe to /detected_crops (DetectedCropArray) and print each object's 3-D position.
"""

import rclpy
from rclpy.node import Node
from vision_msgs.msg import DetectedCropArray        # ↳ 패키지 이름이 다르면 수정하세요


class CropListenerNode(Node):
    def __init__(self):
        super().__init__('crop_listener_node')

        # QoS: 센서 데이터가 아니므로 기본 Depth 10 Reliability RELIABLE 그대로 사용
        self.subscription = self.create_subscription(
            DetectedCropArray,
            '/detected_crops',
            self.listener_callback,
            10
        )
        self.get_logger().info("CropListenerNode 시작. '/detected_crops' 수신 대기 중…")

    def listener_callback(self, msg: DetectedCropArray):
        total = msg.total_objects
        stamp = msg.header.stamp
        self.get_logger().info(
            f"\n[{stamp.sec}.{stamp.nanosec:09d}] DetectedCropArray 수신 "
            f"(총 {total}개)\n" + "-" * 45
        )

        for crop in msg.objects:
            self.get_logger().info(
                f"  ▸ ID {crop.id:02d} │ "
                f"X: {crop.x:6.3f} m, "
                f"Y: {crop.y:6.3f} m, "
                f"Z: {crop.z:6.3f} m"
            )


def main(args=None):
    rclpy.init(args=args)
    node = CropListenerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
