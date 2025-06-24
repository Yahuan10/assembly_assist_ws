import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class FakeCameraNode(Node):
    def __init__(self):
        super().__init__('fake_camera_node')
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        self.get_logger().info("Fake camera node started.")

    def timer_callback(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Fake Camera Frame", (50, 240), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FakeCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
