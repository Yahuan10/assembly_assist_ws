import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from PIL import Image, ImageTk
import tkinter as tk
import os

class ProjectionNode(Node):
    def __init__(self):
        super().__init__('projection_node')
        self.subscription = self.create_subscription(Int32, '/assembly/step', self.step_callback, 10)
        self.image_label = None
        self.root = tk.Tk()

        self.root.attributes('-fullscreen', True)
        self.root.configure(background='black')
        self.root.bind('<Escape>', lambda e: self.root.destroy())

        self.image_label = tk.Label(self.root, bg='black')
        self.image_label.pack(expand=True)

        self.get_logger().info("Projection node launched (Tkinter full screen)")
        self.root.after(100, self.spin_ros)
        self.root.mainloop()

    def spin_ros(self):
        rclpy.spin_once(self, timeout_sec=0.01)
        self.root.after(10, self.spin_ros)

    def step_callback(self, msg):
        step = msg.data
        image_path = os.path.join("instructions", f"step_{step}.jpg")
        self.get_logger().info(f"Displaying {image_path}")
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img = img.resize((self.root.winfo_screenwidth(), self.root.winfo_screenheight()), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
        else:
            self.get_logger().warn(f"Image not found: {image_path}")

def main(args=None):
    rclpy.init(args=args)
    node = ProjectionNode()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
