import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32

class StateManagerNode(Node):
    def __init__(self):
        super().__init__('state_manager_node')
        self.subscription = self.create_subscription(Bool, '/gesture/confirm', self.gesture_callback, 10)
        self.publisher = self.create_publisher(Int32, '/assembly/step', 10)
        self.current_step = 0
        self.total_steps = 5  # can be made parameterized later
        self.confirmed_last_frame = False
        self.get_logger().info("State manager node started.")

    def gesture_callback(self, msg):
        if msg.data and not self.confirmed_last_frame:
            self.current_step += 1
            if self.current_step <= self.total_steps:
                self.get_logger().info(f"Confirmed gesture! Advancing to step {self.current_step}")
                self.publisher.publish(Int32(data=self.current_step))
            else:
                self.get_logger().info("All steps completed.")
        self.confirmed_last_frame = msg.data

def main(args=None):
    rclpy.init(args=args)
    node = StateManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
