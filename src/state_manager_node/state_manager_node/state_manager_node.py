#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This node manages the assembly workflow state transitions based on gesture confirmation.
It controls UR5 robotic arm movements and Robotiq gripper actions according to predefined steps.

该节点根据手势确认信号管理装配流程的状态转换。
按照预定义步骤控制UR5机械臂运动和Robotiq夹爪动作。

Features:
- Simple state machine implementation base on steps.yaml
- UR5 joint position control via topic
- Robotiq gripper control via official message type
- Gesture confirmation input handling

功能特性：
- 简单状态机实现基于steps.yaml
- 通过话题控制UR5关节位置
- 使用官方消息类型控制Robotiq夹爪
- 手势确认输入处理
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from robotiq_2f_gripper_msgs.msg import RobotiqGripperCommand
import yaml
import time  # 添加time模块导入

class StateManagerNode(Node):
    def __init__(self):
        super().__init__('state_manager_node')
        
        # Load steps from YAML
        self.steps = self.load_steps("/home/jason77/assembly_assist_ws/config/steps.yaml")
        self.current_step = 0
        
        # Subscribers
        self.gesture_sub = self.create_subscription(
            Bool, '/gesture/confirm', self.gesture_callback, 10)

        # Publishers
        self.gripper_pub = self.create_publisher(
            RobotiqGripperCommand, '/output', 10)
        self.ur5_pub = self.create_publisher(
            String, '/joint_group_position_controller/command', 10)
        self.step_pub = self.create_publisher(
            String, '/assembly/step', 10)  # New publisher

        self.current_state = "IDLE"
        self.get_logger().info("State Manager Node is running.")

    def load_steps(self, file_path):
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load steps.yaml: {e}")
            return {'steps': []}

    def gesture_callback(self, msg):
        if not msg.data:
            return

        if self.current_step >= len(self.steps['steps']):
            self.get_logger().info("All steps completed.")
            return

        step_data = self.steps['steps'][self.current_step]
        
        self.get_logger().info(f"Executing step {self.current_step}: {step_data}")  # 新增日志输出
        
        # Publish current step name to projection node
        step_msg = String()
        step_msg.data = step_data.get('projection_text', f"Step {self.current_step}")
        self.step_pub.publish(step_msg)
        
        # Handle different types of steps based on available fields
        self.get_logger().info(f"=== 开始执行步骤 {self.current_step} ===")
        self.get_logger().info(f"完整步骤配置: {step_data}")
        
        if 'pick_pose' in step_data and 'place_pose' in step_data':
            self.get_logger().info(f"1. 移动到拾取位置: {step_data['pick_pose']}")
            # Send UR5 to pick pose (example format conversion)
            pick_position = step_data['pick_pose']['position']
            pick_orientation = step_data['pick_pose'].get('orientation', {"x":0.0, "y":0.0, "z":0.0, "w":1.0})
            pick_pose_str = f"[{pick_position['x']}, {pick_position['y']}, {pick_position['z']}, {pick_orientation['x']}, {pick_orientation['y']}, {pick_orientation['z']}, {pick_orientation['w']}]"
            
            ur5_msg = String()
            ur5_msg.data = pick_pose_str
            self.ur5_pub.publish(ur5_msg)
            
            # Simulate gripper action after delay
            if 'gripper' in step_data:
                self.get_logger().info(f"2. 执行夹爪动作: {step_data['gripper']}")
                self.send_gripper_command(step_data['gripper'])
                
            # Optional: add delay before place
            if 'gripper_delay' in step_data:
                self.get_logger().info(f"3. 等待延迟 {step_data['gripper_delay']} 秒")
                time.sleep(step_data['gripper_delay'])
                
            self.get_logger().info(f"4. 移动到放置位置: {step_data['place_pose']}")
            # Send UR5 to place pose
            place_position = step_data['place_pose']['position']
            place_orientation = step_data['place_pose'].get('orientation', {"x":0.0, "y":0.0, "z":0.0, "w":1.0})
            place_pose_str = f"[{place_position['x']}, {place_position['y']}, {place_position['z']}, {place_orientation['x']}, {place_orientation['y']}, {place_orientation['z']}, {place_orientation['w']}]"
            
            ur5_msg = String()
            ur5_msg.data = place_pose_str
            self.ur5_pub.publish(ur5_msg)
            
        elif 'gripper' in step_data:
            self.get_logger().info(f"1. 执行独立夹爪动作: {step_data['gripper']}")
            self.send_gripper_command(step_data['gripper'])
            
        # Handle delay-only steps (like assembly_delay)
        if 'gripper_delay' in step_data:
            self.get_logger().info(f"2. 等待延迟 {step_data['gripper_delay']} 秒")
            time.sleep(step_data['gripper_delay'])
            
        self.get_logger().info(f"=== 步骤 {self.current_step} 完成 ===\n")
        self.current_step += 1

    def send_ur5_command(self, command):
        msg = String()
        if command == "MOVE_TO_PICK_POSE":
            msg.data = "[0.0, -1.57, 1.57, -1.57, -1.57, 0.0]"
        elif command == "MOVE_TO_PLACE_POSE":
            msg.data = "[0.5, -1.2, 1.0, -0.5, -1.57, 0.0]"
        self.ur5_pub.publish(msg)

    def send_gripper_command(self, command):
        msg = RobotiqGripperCommand()
        if command == "OPEN":
            msg.command = "OPEN"
        elif command == "CLOSE":
            msg.command = "CLOSE"
        self.gripper_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    state_manager = StateManagerNode()
    rclpy.spin(state_manager)
    rclpy.shutdown()

if __name__ == '__main__':
    main()