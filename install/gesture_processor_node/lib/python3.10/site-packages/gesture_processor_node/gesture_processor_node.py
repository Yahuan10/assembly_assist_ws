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
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.confirmed_frames = 0
        self.required_confirm_frames = 5  # N consecutive frames
        self.last_published = False

        self.get_logger().info("GestureProcessorNode with filter started.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb_image)

            confirmed = False
            hand_label = None

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    if hand_label == 'Left':  # Only respond to left hand
                        wrist = hand_landmarks.landmark[0]
                        tips = [4, 8, 12, 16, 20]
                        fingers_up = 0
                        for idx in tips:
                            tip = hand_landmarks.landmark[idx]
                            if tip.y < wrist.y:
                                fingers_up += 1
                        if fingers_up >= 4:
                            confirmed = True
                    self.mp_drawing.draw_landmarks(cv_image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # Frame filtering
            if confirmed:
                self.confirmed_frames += 1
            else:
                self.confirmed_frames = 0

            if self.confirmed_frames >= self.required_confirm_frames:
                if not self.last_published:
                    self.publisher.publish(Bool(data=True))
                    self.get_logger().info("✋ Confirmed left hand gesture detected and published.")
                    self.last_published = True
            else:
                if self.last_published:
                    self.publisher.publish(Bool(data=False))
                    self.get_logger().info("✋ Gesture lost. Publishing False.")
                    self.last_published = False

            cv2.putText(cv_image, f"Confirmed Frames: {self.confirmed_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if confirmed else (0, 0, 255), 2)
            cv2.putText(cv_image, f"Hand: {hand_label}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.imshow("Gesture Debug", cv_image)
            cv2.waitKey(1)

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
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()