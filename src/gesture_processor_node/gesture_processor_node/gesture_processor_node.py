import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import time

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

        self.last_hand_time = time.time()
        self.action_published = False      

        self.get_logger().info("GestureProcessorNode no-hand 2s trigger started.")
        
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb_image)

            detected_hand = False

            if results.multi_hand_landmarks:
                detected_hand = True
                self.last_hand_time = time.time()
                self.action_published = False

                
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(cv_image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            
            if not detected_hand:
                if (time.time() - self.last_hand_time) > 2.0 and not self.action_published:
                    self.publisher.publish(Bool(data=True))
                    self.get_logger().info(" No hand for 2s, action triggered and published True!")
                    self.action_published = True

           
            cv2.putText(cv_image, 
                        f"Last Hand: {int(time.time() - self.last_hand_time)}s ago", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if detected_hand else (0,0,255), 2)
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