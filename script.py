#!/usr/bin/env python
# filepath: /Users/yujiangtao/my_project/PJ_Auto/src/schledluer/src/script.py

import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import DisplayTrajectory
from trajectory_msgs.msg import JointTrajectory
import threading
import time
from datetime import datetime, timedelta
import tf2_ros
import tf2_geometry_msgs

class TrajectoryPredictor:
    def __init__(self, h_matrix=None):
        rospy.init_node('trajectory_predictor', anonymous=True)
        
        # 🔧 使用标定后的H矩阵
        if h_matrix is not None:
            self.h_matrix = h_matrix
        else:
            # 默认H矩阵（需要替换为你的实际标定结果）
            self.h_matrix = np.array([
                [800.0,  0.0,   512.0],
                [0.0,    -800.0, 384.0],
                [0.0,    0.0,    1.0]
            ], dtype=np.float32)
        
        # 🔧 画布大小设置为1024x768
        self.window_width = 1024
        self.window_height = 768
        
        # 当前状态
        self.current_state = "Start"
        self.current_tcp_pose = None
        self.target_pose = None
        self.trajectory_points = []
        self.interpolated_trajectory = []
        
        # 从scheduler代码中提取的关键位置点
        self.key_positions = self.load_key_positions()
        
        # 状态轨迹映射
        self.state_trajectories = self.create_state_trajectory_mapping()
        
        # 显示设置
        self.window_name = "机械臂轨迹预测和方向指示"
        
        # 颜色设置
        self.colors = {
            'background': (40, 40, 40),
            'current_pos': (0, 255, 0),        # 绿色 - 当前位置
            'target_pos': (0, 0, 255),         # 红色 - 目标位置
            'predicted_path': (255, 165, 0),   # 橙色 - 预测路径
            'direction_arrow': (0, 255, 255),  # 黄色 - 方向箭头
            'text_white': (255, 255, 255),
            'grid': (80, 80, 80),
            'workspace_boundary': (128, 128, 128)  # 灰色 - 工作空间边界
        }
        
        self.setup_ros_communication()
        self.start_prediction_thread()
        self.start_display()

    def set_h_matrix(self, h_matrix):
        """🔧 设置标定后的H矩阵"""
        self.h_matrix = h_matrix.astype(np.float32)
        rospy.loginfo("H矩阵已更新")

    def tcp_to_screen_coords(self, tcp_position):
        """🚀 使用H矩阵将TCP坐标转换为屏幕坐标"""
        try:
            # 将TCP坐标转换为齐次坐标 [x, y, 1]
            tcp_homogeneous = np.array([
                tcp_position[0],  # X坐标
                tcp_position[1],  # Y坐标  
                1.0               # 齐次坐标
            ], dtype=np.float32)
            
            # 应用H矩阵变换
            screen_homogeneous = self.h_matrix @ tcp_homogeneous
            
            # 转换回非齐次坐标
            if screen_homogeneous[2] != 0:
                screen_x = int(screen_homogeneous[0] / screen_homogeneous[2])
                screen_y = int(screen_homogeneous[1] / screen_homogeneous[2])
            else:
                screen_x, screen_y = 0, 0
            
            # 限制在画布范围内
            screen_x = max(0, min(screen_x, self.window_width - 1))
            screen_y = max(0, min(screen_y, self.window_height - 1))
            
            return (screen_x, screen_y)
            
        except Exception as e:
            rospy.logwarn(f"TCP坐标转换错误: {e}")
            return (self.window_width // 2, self.window_height // 2)

    def screen_to_tcp_coords(self, screen_x, screen_y):
        """🔧 将屏幕坐标转换回TCP坐标（用于调试）"""
        try:
            # 使用H矩阵的逆矩阵
            h_inv = np.linalg.inv(self.h_matrix)
            
            screen_homogeneous = np.array([screen_x, screen_y, 1.0], dtype=np.float32)
            tcp_homogeneous = h_inv @ screen_homogeneous
            
            if tcp_homogeneous[2] != 0:
                tcp_x = tcp_homogeneous[0] / tcp_homogeneous[2]
                tcp_y = tcp_homogeneous[1] / tcp_homogeneous[2]
            else:
                tcp_x, tcp_y = 0.0, 0.0
                
            return (tcp_x, tcp_y)
            
        except Exception as e:
            rospy.logwarn(f"屏幕坐标转换错误: {e}")
            return (0.0, 0.0)

    def load_key_positions(self):
        """从scheduler代码中提取关键位置点"""
        return {
            # 电机位置 (16个)
            'motors': [
                [0.263, 0.115, 0.194, 0.018, 0.999, -0.010, 0.009],
                [0.263, 0.068, 0.194, 0.018, 0.999, -0.010, 0.009],
                [0.263, 0.021, 0.194, 0.018, 0.999, -0.010, 0.009],
                [0.263, -0.026, 0.194, 0.018, 0.999, -0.010, 0.009],
                [0.343, 0.115, 0.194, 0.018, 0.999, -0.010, 0.009],
                [0.343, 0.068, 0.194, 0.018, 0.999, -0.010, 0.009],
                [0.343, 0.021, 0.194, 0.018, 0.999, -0.010, 0.009],
                [0.343, -0.026, 0.194, 0.018, 0.999, -0.010, 0.009],
                [0.423, 0.115, 0.194, 0.018, 0.999, -0.010, 0.009],
                [0.423, 0.068, 0.194, 0.018, 0.999, -0.010, 0.009],
                [0.423, 0.021, 0.194, 0.018, 0.999, -0.010, 0.009],
                [0.423, -0.026, 0.194, 0.018, 0.999, -0.010, 0.009],
                [0.505, 0.115, 0.194, 0.018, 0.999, -0.010, 0.009],
                [0.505, 0.068, 0.194, 0.018, 0.999, -0.010, 0.009],
                [0.505, 0.021, 0.194, 0.018, 0.999, -0.010, 0.009],
                [0.505, -0.026, 0.194, 0.018, 0.999, -0.010, 0.009],
            ],
            
            # 过渡点
            'transitions': {
                'over_motor': [0.327, 0.0, 0.355, -0.003, -0.999, 0.002, 0.0007],
                'over_gb0_1': [0.439, -0.271, 0.215, -0.001, -0.696, 0.718, 0.001],
                'over_gb0_2': [0.439, -0.210, 0.300, -0.001, -0.696, 0.718, 0.001],
                'over_gb1_1': [0.438, -0.331, 0.391, 0.703, 0.710, 0.002, 0.005],
                'over_gb1_2': [0.438, -0.460, 0.391, 0.703, 0.710, 0.002, 0.005],
                'over_gb1_3': [0.438, -0.460, 0.354, 0.703, 0.710, 0.002, 0.005],
            },
            
            # PCB位置
            'pcb1': [
                [0.631, -0.139, 0.168, 0.703, 0.710, 0.002, 0.005],
                [0.686, -0.139, 0.168, 0.703, 0.710, 0.002, 0.005],
                [0.741, -0.139, 0.168, 0.703, 0.710, 0.002, 0.005],
                [0.796, -0.139, 0.168, 0.703, 0.710, 0.002, 0.005],
            ],
            
            # 人类交接位置
            'handover': [0.011, -0.407, 0.434, 0.659, 0.133, -0.040, 0.738],
            
            # 电池位置
            'battery': [
                [0.606, -0.019, 0.154, 0.999, 0.012, -0.012, 0.022],
                [0.606, 0.103, 0.154, 0.999, 0.012, -0.012, 0.022],
                [0.707, -0.019, 0.154, 0.999, 0.012, -0.012, 0.022],
                [0.707, 0.101, 0.154, 0.999, 0.012, -0.012, 0.022],
            ]
        }

    def create_state_trajectory_mapping(self):
        """创建状态到轨迹的映射"""
        return {
            'MPickUp': {
                'sequence': [
                    {'type': 'joint', 'description': '移动到准备位置'},
                    {'type': 'move', 'target': 'transitions.over_motor', 'description': '移动到电机上方'},
                    {'type': 'pick', 'target': 'motors.0', 'description': '抓取电机'},
                    {'type': 'joint', 'description': '移动到交接准备位置'},
                ],
                'duration': 14.0
            },
            'MHoldHD': {
                'sequence': [
                    {'type': 'move', 'target': 'handover', 'description': '移动到交接位置'},
                    {'type': 'wait', 'duration': 8.0, 'description': '等待人类接收'},
                    {'type': 'joint', 'description': '返回安全位置'},
                ],
                'duration': 18.0
            },
            'MPositioning': {
                'sequence': [
                    {'type': 'move', 'target': 'transitions.over_gb0_1', 'description': '移动到定位区域'},
                    {'type': 'gripper', 'action': 'open', 'description': '释放电机'},
                    {'type': 'move', 'target': 'transitions.over_gb0_2', 'description': '移动到安全位置'},
                ],
                'duration': 5.0
            },
            'PCB1PickUpAndPositioning': {
                'sequence': [
                    {'type': 'joint', 'description': '移动到PCB1区域'},
                    {'type': 'pick', 'target': 'pcb1.0', 'description': '抓取PCB1'},
                    {'type': 'move', 'target': 'transitions.over_gb1_1', 'description': '过渡点1'},
                    {'type': 'move', 'target': 'transitions.over_gb1_2', 'description': '过渡点2'},
                    {'type': 'move', 'target': 'transitions.over_gb1_3', 'description': '定位点'},
                    {'type': 'gripper', 'action': 'open', 'description': '放置PCB1'},
                ],
                'duration': 13.0
            }
        }

    def setup_ros_communication(self):
        """设置ROS通信"""
        # 订阅当前TCP位置
        self.tcp_subscriber = rospy.Subscriber('/move_group/display_planned_path', 
                                              DisplayTrajectory, 
                                              self.trajectory_callback)
        
        # 订阅当前TCP pose
        self.pose_subscriber = rospy.Subscriber('/ur5/tcp_pose', 
                                               Pose, 
                                               self.tcp_pose_callback)
        
        # 订阅SMACH状态
        self.state_subscriber = rospy.Subscriber('/smach/container_status', 
                                                String, 
                                                self.state_callback)

    def trajectory_callback(self, msg):
        """轨迹规划回调 - 获取MoveIt规划的轨迹"""
        if msg.trajectory and len(msg.trajectory) > 0:
            trajectory = msg.trajectory[0].joint_trajectory
            self.trajectory_points = self.convert_joint_trajectory_to_cartesian(trajectory)

    def tcp_pose_callback(self, msg):
        """TCP位置回调"""
        self.current_tcp_pose = [
            msg.position.x, msg.position.y, msg.position.z,
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        ]

    def state_callback(self, msg):
        """状态变化回调"""
        new_state = msg.data.strip()
        if new_state != self.current_state:
            self.current_state = new_state
            self.predict_next_trajectory()

    def convert_joint_trajectory_to_cartesian(self, joint_trajectory):
        """将关节轨迹转换为笛卡尔轨迹（简化版本）"""
        cartesian_points = []
        
        if len(joint_trajectory.points) > 0:
            start_pos = self.current_tcp_pose if self.current_tcp_pose else [0, 0, 0, 0, 0, 0, 1]
            end_pos = self.target_pose if self.target_pose else start_pos
            
            num_points = len(joint_trajectory.points)
            for i in range(num_points):
                t = i / (num_points - 1) if num_points > 1 else 0
                interpolated_pos = [
                    start_pos[j] + t * (end_pos[j] - start_pos[j]) for j in range(3)
                ]
                interpolated_pos.extend([0, 0, 0, 1])
                cartesian_points.append(interpolated_pos)
        
        return cartesian_points

    def predict_next_trajectory(self):
        """预测下一段轨迹"""
        if self.current_state not in self.state_trajectories:
            return
        
        state_info = self.state_trajectories[self.current_state]
        sequence = state_info['sequence']
        
        self.interpolated_trajectory = []
        current_pos = self.current_tcp_pose if self.current_tcp_pose else [0, 0, 0, 0, 0, 0, 1]
        
        for step in sequence:
            if step['type'] == 'move' or step['type'] == 'pick':
                target_key = step.get('target', '')
                target_pos = self.get_position_from_key(target_key)
                
                if target_pos:
                    interpolated_points = self.interpolate_trajectory(current_pos, target_pos, 20)
                    for point in interpolated_points:
                        self.interpolated_trajectory.append({
                            'position': point[:3],
                            'orientation': point[3:7],
                            'description': step['description'],
                            'type': step['type']
                        })
                    current_pos = target_pos

    def get_position_from_key(self, key):
        """从关键字获取位置"""
        if not key:
            return None
        
        keys = key.split('.')
        data = self.key_positions
        
        try:
            for k in keys:
                if k.isdigit():
                    data = data[int(k)]
                else:
                    data = data[k]
            return data
        except (KeyError, IndexError, TypeError):
            return None

    def interpolate_trajectory(self, start_pos, end_pos, num_points):
        """在两点间插值生成轨迹"""
        trajectory = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            interpolated = [
                start_pos[j] + t * (end_pos[j] - start_pos[j]) for j in range(len(start_pos))
            ]
            trajectory.append(interpolated)
        return trajectory

    def start_prediction_thread(self):
        """启动预测线程"""
        def prediction_loop():
            rate = rospy.Rate(5)
            while not rospy.is_shutdown():
                if self.current_state != "Start":
                    self.predict_next_trajectory()
                rate.sleep()
        
        thread = threading.Thread(target=prediction_loop)
        thread.daemon = True
        thread.start()

    def create_display_image(self):
        """创建显示图像"""
        img = np.full((self.window_height, self.window_width, 3), 
                      self.colors['background'], dtype=np.uint8)
        
        # 绘制各个部分
        self.draw_title(img)
        self.draw_workspace_boundary(img)
        self.draw_trajectory_overlay(img)
        self.draw_direction_indicators(img)
        self.draw_next_targets_overlay(img)
        self.draw_coordinate_info(img)
        
        return img

    def draw_title(self, img):
        """绘制标题"""
        title = f"机械臂轨迹预测 - 当前状态: {self.current_state}"
        cv2.putText(img, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, self.colors['text_white'], 2)

    def draw_workspace_boundary(self, img):
        """🔧 绘制工作空间边界"""
        # 定义工作空间的TCP坐标边界
        workspace_tcp_coords = [
            [-0.5, -0.7],  # 左下角
            [0.8, -0.7],   # 右下角  
            [0.8, 0.5],    # 右上角
            [-0.5, 0.5],   # 左上角
        ]
        
        # 转换为屏幕坐标
        screen_coords = []
        for tcp_coord in workspace_tcp_coords:
            screen_coord = self.tcp_to_screen_coords(tcp_coord)
            screen_coords.append(screen_coord)
        
        # 绘制工作空间边界
        if len(screen_coords) >= 4:
            pts = np.array(screen_coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, self.colors['workspace_boundary'], 2)
            
            # 添加标签
            cv2.putText(img, "工作空间", (screen_coords[0][0], screen_coords[0][1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['workspace_boundary'], 1)

    def draw_trajectory_overlay(self, img):
        """🚀 绘制轨迹覆盖层（直接在1024x768画布上）"""
        # 绘制当前位置
        if self.current_tcp_pose:
            current_screen = self.tcp_to_screen_coords(self.current_tcp_pose[:2])
            cv2.circle(img, current_screen, 12, self.colors['current_pos'], -1)
            cv2.circle(img, current_screen, 15, self.colors['text_white'], 2)
            cv2.putText(img, "当前位置", (current_screen[0] + 20, current_screen[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['current_pos'], 2)
        
        # 绘制预测轨迹
        prev_screen = None
        for i, point in enumerate(self.interpolated_trajectory):
            screen_pos = self.tcp_to_screen_coords(point['position'][:2])
            
            # 绘制轨迹线
            if prev_screen is not None:
                cv2.line(img, prev_screen, screen_pos, self.colors['predicted_path'], 3)
            
            # 绘制轨迹点
            cv2.circle(img, screen_pos, 5, self.colors['predicted_path'], -1)
            
            # 每10个点标注一个描述
            if i % 10 == 0 and point.get('description'):
                cv2.putText(img, point['description'][:8], 
                           (screen_pos[0] + 10, screen_pos[1] + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_white'], 1)
            
            prev_screen = screen_pos

    def draw_direction_indicators(self, img):
        """绘制方向指示箭头"""
        if len(self.interpolated_trajectory) < 2:
            return
        
        # 绘制方向箭头（每8个点绘制一个箭头）
        for i in range(0, len(self.interpolated_trajectory) - 1, 8):
            current_point = self.interpolated_trajectory[i]['position']
            next_point = self.interpolated_trajectory[i + 1]['position']
            
            # 转换到屏幕坐标
            current_screen = self.tcp_to_screen_coords(current_point[:2])
            next_screen = self.tcp_to_screen_coords(next_point[:2])
            
            # 计算方向向量
            direction = np.array(next_screen) - np.array(current_screen)
            length = np.linalg.norm(direction)
            
            if length > 10:  # 只绘制足够长的箭头
                direction = direction / length * 30
                arrow_end = tuple((np.array(current_screen) + direction).astype(int))
                cv2.arrowedLine(img, current_screen, arrow_end, 
                               self.colors['direction_arrow'], 3, tipLength=0.4)

    def draw_next_targets_overlay(self, img):
        """绘制接下来的目标位置覆盖层"""
        # 在右侧显示目标信息
        info_x = self.window_width - 350
        info_y = 80
        
        # 绘制半透明背景
        overlay = img.copy()
        cv2.rectangle(overlay, (info_x - 10, info_y - 10), 
                     (self.window_width - 10, info_y + 300), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        cv2.putText(img, "接下来的目标:", (info_x, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text_white'], 2)
        
        # 显示接下来的目标
        y_offset = 30
        shown_targets = set()
        
        for i, point in enumerate(self.interpolated_trajectory[:50]):
            if point['type'] in ['move', 'pick'] and point['description'] not in shown_targets:
                target_text = f"• {point['description']}"
                cv2.putText(img, target_text, (info_x, info_y + y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['predicted_path'], 1)
                
                # 显示坐标
                pos_text = f"  [{point['position'][0]:.3f}, {point['position'][1]:.3f}]"
                cv2.putText(img, pos_text, (info_x, info_y + y_offset + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_white'], 1)
                
                # 在主视图中标出目标位置
                target_screen = self.tcp_to_screen_coords(point['position'][:2])
                cv2.circle(img, target_screen, 8, self.colors['target_pos'], 2)
                cv2.putText(img, str(len(shown_targets) + 1), 
                           (target_screen[0] - 5, target_screen[1] + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['target_pos'], 1)
                
                shown_targets.add(point['description'])
                y_offset += 40
                
                if len(shown_targets) >= 6:
                    break

    def draw_coordinate_info(self, img):
        """🔧 绘制坐标信息"""
        info_y = self.window_height - 80
        
        # H矩阵信息
        h_info = f"H矩阵: [{self.h_matrix[0,0]:.1f}, {self.h_matrix[0,1]:.1f}, {self.h_matrix[0,2]:.1f}]"
        cv2.putText(img, h_info, (20, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_white'], 1)
        
        # 当前TCP坐标
        if self.current_tcp_pose:
            tcp_info = f"TCP: [{self.current_tcp_pose[0]:.3f}, {self.current_tcp_pose[1]:.3f}, {self.current_tcp_pose[2]:.3f}]"
            cv2.putText(img, tcp_info, (20, info_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['current_pos'], 1)
            
            # 对应的屏幕坐标
            screen_coords = self.tcp_to_screen_coords(self.current_tcp_pose[:2])
            screen_info = f"屏幕坐标: [{screen_coords[0]}, {screen_coords[1]}]"
            cv2.putText(img, screen_info, (20, info_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['current_pos'], 1)

    def start_display(self):
        """启动显示循环"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
        while not rospy.is_shutdown():
            try:
                display_img = self.create_display_image()
                cv2.imshow(self.window_name, display_img)
                
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):  # 保存截图
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"trajectory_prediction_{timestamp}.png", display_img)
                    rospy.loginfo("截图已保存")
                elif key == ord('c'):  # 校准模式 - 显示鼠标坐标对应的TCP坐标
                    self.calibration_mode(display_img)
                    
            except Exception as e:
                rospy.logwarn(f"显示错误: {e}")
                break
        
        cv2.destroyAllWindows()

    def calibration_mode(self, img):
        """🔧 校准模式 - 点击屏幕显示对应的TCP坐标"""
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                tcp_coords = self.screen_to_tcp_coords(x, y)
                rospy.loginfo(f"屏幕坐标 ({x}, {y}) 对应TCP坐标: ({tcp_coords[0]:.3f}, {tcp_coords[1]:.3f})")
                
                # 在图像上显示
                cv2.circle(img, (x, y), 5, (0, 255, 255), -1)
                cv2.putText(img, f"TCP:({tcp_coords[0]:.3f},{tcp_coords[1]:.3f})", 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.imshow(self.window_name, img)
        
        cv2.setMouseCallback(self.window_name, mouse_callback)
        rospy.loginfo("校准模式：点击屏幕查看对应TCP坐标，按任意键退出")

def main():
    """主函数"""
    # 🔧 在这里设置你的实际H矩阵
    # 示例H矩阵（需要替换为你的标定结果）
    h_matrix = np.array([
        [-944.336237, -87.4102614, -4.95230579],   # 缩放和平移X
        [-62.4513781,  874.191558, 134.016380],  # 缩放和平移Y（负号表示Y轴翻转）
        [-0.00962922264,    -0.142556057,     1.0]     # 齐次坐标
    ], dtype=np.float32)

    try:
        predictor = TrajectoryPredictor(h_matrix)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()