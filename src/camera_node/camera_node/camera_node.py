import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
from pypylon import pylon

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        self.get_logger().info("Initializing Basler camera...")
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()

        self.camera.Width.SetValue(1280)
        self.camera.Height.SetValue(720)
        self.camera.ExposureTime.SetValue(8000.0)

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.get_logger().info("Camera started. Publishing images to /camera/image_raw")

        timer_period = 0.03
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        if self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                img = grab_result.Array
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)

                ros_image = self.bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
                ros_image.header.stamp = self.get_clock().now().to_msg()
                self.publisher.publish(ros_image)
            grab_result.Release()

    def destroy_node(self):
        self.get_logger().info("Shutting down camera...")
        self.camera.StopGrabbing()
        self.camera.Close()
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
