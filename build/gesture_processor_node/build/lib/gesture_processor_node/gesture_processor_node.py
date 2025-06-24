import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import mediapipe as mp

class GestureProcessorNode(Node):
    def __init__(self):
        super().__init__('gesture_processor_node')
        self.subscriber = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(Bool, '/gesture/confirm', 10)
        self.bridge = CvBridge()
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7
        )

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb_image)

            confirmed = False
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[0]
                    middle_finger_tip = hand_landmarks.landmark[12]
                    if middle_finger_tip.y < wrist.y:
                        confirmed = True

            self.publisher.publish(Bool(data=confirmed))
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = GestureProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
