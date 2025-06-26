import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        self.get_logger().info("Initializing USB camera...")
        self.cap = cv2.VideoCapture(0,cv2.CAP_V4L2)  # 0 表示第一个摄像头（UGREEN）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera")
            return

        self.get_logger().info("Camera started. Publishing images to /camera/image_raw")

        self.timer = self.create_timer(0.03, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(ros_image)

    def destroy_node(self):
        self.get_logger().info("Releasing camera...")
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
