#!/usr/bin/env python
"""
机器人TCP位置监控和状态机状态跟踪脚本
用于实时获取UR5机械臂TCP位置并在OpenCV中可视化
同时监控SMACH状态机的状态变化
"""

import rospy
import tf
import numpy as np
import cv2
import threading
import time
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from moveit_commander import MoveGroupCommander
import sys

class RobotTCPVisualizer:
    """
    机器人TCP位置可视化类
    """
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('robot_tcp_visualizer', anonymous=True)
        
        # 初始化MoveIt组
        try:
            self.move_group = MoveGroupCommander("manipulator")
        except Exception as e:
            rospy.logwarn("无法初始化MoveIt组: %s", e)
            self.move_group = None
        
        # TF监听器
        self.tf_listener = tf.TransformListener()
        
        # 当前TCP位置
        self.current_tcp_position = None
        self.current_tcp_orientation = None
        
        # 历史轨迹点
        self.trajectory_points = []
        
        # OpenCV可视化参数
        self.window_width = 800
        self.window_height = 600
        self.workspace_min = np.array([-0.5, -0.5, 0.0])  # [x, y, z]
        self.workspace_max = np.array([0.5, 0.5, 0.8])    # [x, y, z]
        
        # SMACH状态
        self.current_state = "未知"
        
        # 创建OpenCV窗口
        cv2.namedWindow('Robot TCP Visualization', cv2.WINDOW_AUTOSIZE)
        
        # 启动各个线程
        self.start_threads()
        
    def start_threads(self):
        """
        启动所有监控线程
        """
        # TCP位置更新线程
        tcp_thread = threading.Thread(target=self.update_tcp_position)
        tcp_thread.daemon = True
        tcp_thread.start()
        
        # SMACH状态监听线程
        state_thread = threading.Thread(target=self.listen_smach_states)
        state_thread.daemon = True
        state_thread.start()
        
        # 可视化更新线程
        viz_thread = threading.Thread(target=self.visualization_loop)
        viz_thread.daemon = True
        viz_thread.start()
        
        rospy.loginfo("所有监控线程已启动")
        
    def get_tcp_position_tf(self):
        """
        通过TF获取TCP位置
        """
        try:
            # 等待变换
            self.tf_listener.waitForTransform("base", "ee_link", rospy.Time(), rospy.Duration(1.0))
            
            # 获取变换
            (trans, rot) = self.tf_listener.lookupTransform("base", "ee_link", rospy.Time(0))
            return np.array(trans), np.array(rot)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("TF变换错误: %s", e)
            return None, None
    
    def get_tcp_position_moveit(self):
        """
        通过MoveIt获取TCP位置
        """
        if self.move_group is None:
            return None, None
            
        try:
            current_pose = self.move_group.get_current_pose().pose
            position = np.array([
                current_pose.position.x,
                current_pose.position.y,
                current_pose.position.z
            ])
            orientation = np.array([
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w
            ])
            return position, orientation
        except Exception as e:
            rospy.logwarn("MoveIt获取位置错误: %s", e)
            return None, None
    
    def update_tcp_position(self):
        """
        持续更新TCP位置
        """
        rate = rospy.Rate(30)  # 30Hz更新频率
        while not rospy.is_shutdown():
            # 优先使用TF获取位置
            pos, rot = self.get_tcp_position_tf()
            
            # 如果TF失败，尝试使用MoveIt
            if pos is None:
                pos, rot = self.get_tcp_position_moveit()
            
            if pos is not None:
                self.current_tcp_position = pos
                self.current_tcp_orientation = rot
                
                # 添加到轨迹点（限制点数量）
                self.trajectory_points.append(pos.copy())
                if len(self.trajectory_points) > 500:  # 最多保存500个点
                    self.trajectory_points.pop(0)
            
            rate.sleep()
    
    def listen_smach_states(self):
        """
        监听SMACH状态变化
        """
        def state_callback(msg):
            self.current_state = msg.data
            rospy.loginfo("SMACH状态更新: %s", self.current_state)
        
        # 订阅SMACH状态主题
        rospy.Subscriber('/smach/container_status', String, state_callback)
        rospy.loginfo("开始监听SMACH状态")
    
    def world_to_image(self, point):
        """
        将世界坐标转换为图像坐标
        """
        # 归一化到0-1范围
        normalized = (point - self.workspace_min) / (self.workspace_max - self.workspace_min)
        
        # 转换为图像坐标 (Y轴翻转)
        u = int(normalized[0] * self.window_width)
        v = int((1 - normalized[1]) * self.window_height)
        
        return (u, v)
    
    def draw_workspace(self, image):
        """
        绘制工作空间边界
        """
        # 绘制边界
        cv2.rectangle(image, (0, 0), (self.window_width-1, self.window_height-1), (100, 100, 100), 2)
        
        # 绘制坐标轴标签
        cv2.putText(image, 'X+', (self.window_width-50, self.window_height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(image, 'Y+', (self.window_width//2, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 绘制工作空间信息
        cv2.putText(image, f'Workspace: X[{self.workspace_min[0]:.1f}, {self.workspace_max[0]:.1f}]', 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(image, f'Y[{self.workspace_min[1]:.1f}, {self.workspace_max[1]:.1f}]', 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(image, f'Z[{self.workspace_min[2]:.1f}, {self.workspace_max[2]:.1f}]', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return image
    
    def draw_trajectory(self, image):
        """
        绘制轨迹
        """
        # 绘制历史轨迹
        for i in range(1, len(self.trajectory_points)):
            pt1 = self.world_to_image(self.trajectory_points[i-1])
            pt2 = self.world_to_image(self.trajectory_points[i])
            # 根据时间渐变颜色（从蓝到绿）
            color_ratio = i / len(self.trajectory_points)
            color = (int(255 * (1 - color_ratio)), int(255 * color_ratio), 255)
            cv2.line(image, pt1, pt2, color, 2)
    
    def draw_tcp_position(self, image):
        """
        绘制当前TCP位置
        """
        if self.current_tcp_position is not None:
            current_img_point = self.world_to_image(self.current_tcp_position)
            
            # 绘制TCP位置
            cv2.circle(image, current_img_point, 8, (0, 0, 255), -1)
            cv2.circle(image, current_img_point, 12, (0, 0, 255), 2)
            
            # 绘制坐标信息
            coord_text = f"X: {self.current_tcp_position[0]:.3f}"
            cv2.putText(image, coord_text, (current_img_point[0]+15, current_img_point[1]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            coord_text = f"Y: {self.current_tcp_position[1]:.3f}"
            cv2.putText(image, coord_text, (current_img_point[0]+15, current_img_point[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            coord_text = f"Z: {self.current_tcp_position[2]:.3f}"
            cv2.putText(image, coord_text, (current_img_point[0]+15, current_img_point[1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_state_info(self, image):
        """
        绘制状态信息
        """
        # 绘制SMACH状态
        cv2.putText(image, f"SMACH State: {self.current_state}", 
                   (10, self.window_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 绘制时间戳
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(image, f"Time: {timestamp}", 
                   (self.window_width - 150, self.window_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def render(self):
        """
        渲染图像
        """
        # 创建黑色背景
        image = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        
        # 绘制工作空间
        image = self.draw_workspace(image)
        
        # 绘制轨迹
        self.draw_trajectory(image)
        
        # 绘制TCP位置
        self.draw_tcp_position(image)
        
        # 绘制状态信息
        self.draw_state_info(image)
        
        # 显示图像
        cv2.imshow('Robot TCP Visualization', image)
        cv2.waitKey(1)
    
    def visualization_loop(self):
        """
        可视化循环
        """
        rate = rospy.Rate(30)  # 30Hz刷新率
        while not rospy.is_shutdown():
            self.render()
            rate.sleep()
    
    def run(self):
        """
        运行主循环
        """
        rospy.loginfo("机器人TCP可视化监控已启动")
        rospy.loginfo("按Ctrl+C退出程序")
        
        try:
            # 保持节点运行
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("程序被用户中断")
        finally:
            # 清理资源
            cv2.destroyAllWindows()

def main():
    """
    主函数
    """
    try:
        # 创建可视化器实例
        visualizer = RobotTCPVisualizer()
        
        # 运行可视化器
        visualizer.run()
        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("程序执行出错: %s", e)

if __name__ == '__main__':
    main()