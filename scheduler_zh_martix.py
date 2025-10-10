#!/usr/bin/env python
"""
机械臂装配调度器 (Robot Assembly Scheduler)
===========================================================
功能：使用SMACH状态机控制UR5机械臂完成电子产品装配任务
包含：电机抓取、PCB定位、电池安装等完整装配流程
===========================================================
"""

# ===== 基础Python库导入 =====
import numpy as np # type: ignore     # 数值计算库，用于矩阵运算和数组操作
import rospy # type: ignore           # ROS Python客户端库
import moveit_commander # type: ignore # MoveIt运动规划库
import sys                            # 系统相关功能
import smach # type: ignore           # 状态机库，用于构建复杂的机器人行为
import smach_ros # type: ignore       # SMACH与ROS的集成
import tf   # type: ignore            # 坐标变换库
import math                           # 数学函数库
import copy                           # 深拷贝工具
import time                           # 时间相关功能
import csv                            # CSV文件处理
import threading                      # 多线程支持
import cv2
# ===== ROS消息和服务导入 =====
from schledluer.msg import robot_msgs                                              # 自定义机器人消息
from tf.transformations import quaternion_from_euler  # type: ignore              # 欧拉角转四元数
from geometry_msgs.msg import Pose, PoseStamped , PointStamped # type: ignore    # 几何消息类型
from moveit_msgs.msg import Grasp, PlaceLocation # type: ignore                   # MoveIt抓取和放置消息
from moveit_commander.move_group import MoveGroupCommander # type: ignore         # MoveIt运动组控制器
from moveit_commander import PlanningSceneInterface # type: ignore               # MoveIt场景规划接口
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg # type: ignore # Robotiq夹爪输出控制
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_input as inputMsg # type: ignore   # Robotiq夹爪输入状态
from moveit_msgs.msg import RobotTrajectory # type: ignore                        # 机器人轨迹消息
from trajectory_msgs.msg import JointTrajectoryPoint # type: ignore              # 关节轨迹点
from std_msgs.msg import String # type: ignore                                    # 标准字符串消息
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_input  as inputMsg # type: ignore # 夹爪状态输入（重复导入）
#test
# ===== 系统常量定义 ===== 
# TCP(Tool Center Point)到人体工学位置的变换参数 - 四元数表示
tcp_to_hum = [-0.0017881569928987558, -0.6960133488624333, 0.7180244453880938, 0.0017653682307088938]

# ===== 机器人关键位置定义 =====
# 机器人基准位置(Home位置) - [x, y, z, qx, qy, qz, qw]
rb_arm_home = np.array([-0.28531283917512756,  0.08176575019716574, 0.3565888897535509, 
                        0.021838185570339213, -0.9997536365149914, 0.0006507883874787611, 0.003916171666392069])

# ===== 电机抓取位置数组 (16个位置) =====
# 工作台上电机的抓取位置，按4x4网格排列
# 每个位置包含：[x坐标, y坐标, z坐标(抓取高度), 四元数姿态(qx, qy, qz, qw)]
rb_arm_on_m =  [
    # 第一行电机位置 (Y=0.115附近)
    np.array([0.2631105225136129,    0.11513901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),  # 电机1
    np.array([0.2631105225136129,    0.06813901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),  # 电机2
    np.array([0.2631105225136129,    0.02113901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),  # 电机3
    np.array([0.2631105225136129,    -0.02613901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]), # 电机4
    
    # 第二行电机位置 (X=0.343附近)
    np.array([0.3431105225136129,    0.11513901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),  # 电机5
    np.array([0.3431105225136129,    0.06813901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),  # 电机6
    np.array([0.3431105225136129,    0.02113901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),  # 电机7
    np.array([0.3431105225136129,    -0.02613901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]), # 电机8
    
    # 第三行电机位置 (X=0.423附近)
    np.array([0.4231105225136129,    0.11513901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),  # 电机9
    np.array([0.4231105225136129,    0.06813901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),  # 电机10
    np.array([0.4231105225136129,    0.02113901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),  # 电机11
    np.array([0.4231105225136129,    -0.02613901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]), # 电机12
    
    # 第四行电机位置 (X=0.505附近)
    np.array([0.5051105225136129,    0.11513901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),  # 电机13
    np.array([0.5051105225136129,    0.06813901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),  # 电机14
    np.array([0.5051105225136129,    0.02113901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),  # 电机15
    np.array([0.5051105225136129,    -0.02613901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008])  # 电机16
]

# 人类交接位置 - 固定的静态位置
rb_arm_on_hum_static = np.array([0.01127138298740326, -0.40789791168606154, 0.4347020900402719,
                                0.65967278113823, 0.13322073168864898, -0.04031615244060301, 0.7385517357139446])

# 过渡位置 - 用于安全移动的中转点
rb_arm_transition = np.array([0.22048980978459626, -0.11962800779329041, 0.22232535871506093,
                             -0.00519597519482744, -0.7000337195214675, 0.7140685181262056, 0.005651972731604554])

# ===== 各工作区域的过渡位置 =====
# 电机抓取区域上方的过渡点
rb_arm_transition_over_m = np.array([0.32755193192480295, 0, 0.3552028979677898,
                                    -0.002982105237080432, -0.9999915258909946, 0.00274986658972347, 0.0007024445132438654])

# PCB1区域上方的过渡点
rb_arm_transition_over_pcb1 = np.array([0.7371109279194257, -0.12405534656551466, 0.3564804416147143,
                                       0.701903624069778, 0.7119699919919052, 0.017634125210474614, 0.010911949817505392])

# PCB2区域上方的过渡点
rb_arm_transition_over_pcb2 = np.array([0.39299783753064255, -0.25037007326362604, 0.4098002793824048,
                                       0.9994996999243878, 0.018132610811126923, -0.01488422170868633, 0.021213632889197198])

# ===== 装配区域过渡轨迹点 =====
# GB0区域(电机装配区域)的过渡点
rb_arm_transition_over_gb0_1 = np.array([0.43920883565114404, -0.27118297223348553, 0.21533919567733978,
                                        -0.0017450344372635439, -0.6960370290652399, 0.7180016795804389, 0.0017312263041814983])
rb_arm_transition_over_gb0_2 = np.array([0.43920912923516957, -0.21053279915028705, 0.3005243278363052,
                                        -0.0017450344372635439, -0.6960370290652399, 0.7180016795804389, 0.0017312263041814983])

# GB1区域(PCB1装配区域)的过渡轨迹 - 三个序列点
rb_arm_transition_over_gb1_1 = np.array([0.4384443484397782, -0.33196635637380156, 0.3911058561909323,
                                        0.703591897260684, 0.7105805074782172, 0.0027980514341484353, 0.005094645157629108])
rb_arm_transition_over_gb1_2 = np.array([0.4384443484397782, -0.46021622516842065, 0.3910704893338057,
                                        0.703591897260684, 0.7105805074782172, 0.0027980514341484353, 0.005094645157629108])
rb_arm_transition_over_gb1_3 = np.array([0.4384443484397782, -0.46018886151503624, 0.35487302405715704,
                                        0.703591897260684, 0.7105805074782172, 0.0027980514341484353, 0.005094645157629108])

# GB2区域(电池装配区域)的过渡轨迹 - 三个序列点
rb_arm_transition_over_gb2_1 = np.array([0.5486854170473805, -0.3885145028949433, 0.3814376455406984,
                                        -0.0020243784347149197, 0.9996642271967776, 0.02193613735553668, 0.013643336576571479])
rb_arm_transition_over_gb2_2 = np.array([0.5486854170473805, -0.4685145028949433, 0.3514376455406984,
                                        -0.0020243784347149197, 0.9996642271967776, 0.02193613735553668, 0.013643336576571479])
rb_arm_transition_over_gb2_3 = np.array([0.5486854170473805, -0.27057393609447655, 0.35141967557281056,
                                        -0.0020243784347149197, 0.9996642271967776, 0.02193613735553668, 0.013643336576571479])

# GB3区域(PCB2装配区域)的过渡轨迹 - 三个序列点
rb_arm_transition_over_gb3_1 = np.array([0.6402808547359244, -0.26790083190492066, 0.38354439061807685,
                                        0.019159378644141387, 0.999622466840548, -0.005839393784188026, 0.018808069486830555])
rb_arm_transition_over_gb3_2 = np.array([0.6402808547359244, -0.46790083190492066, 0.38354439061807685,
                                        0.019159378644141387, 0.999622466840548, -0.005839393784188026, 0.018808069486830555])
rb_arm_transition_over_gb3_3 = np.array([0.6402808547359244, -0.46790083190492066, 0.35478830422197494,
                                        0.010290457772605947, 0.9997533010888775, -0.005696452057469483, 0.018841281131592193])

# ===== PCB组件抓取位置 =====
# PCB1的抓取位置数组 (4个位置)
rb_arm_on_pcb1 = [
    np.array([0.6316488317010515, -0.13953502575569454, 0.16890158973568933,  # PCB1位置1
             0.703591897260684, 0.7105805074782172, 0.0027980514341484353, 0.005094645157629108]),
    np.array([0.6866488317010515, -0.13953502575569454, 0.16890158973568933,  # PCB1位置2
             0.703591897260684, 0.7105805074782172, 0.0027980514341484353, 0.005094645157629108]),
    np.array([0.7416488317010515, -0.13953502575569454, 0.16890158973568933,  # PCB1位置3
             0.703591897260684, 0.7105805074782172, 0.0027980514341484353, 0.005094645157629108]),
    np.array([0.7966488317010515, -0.13953502575569454, 0.16890158973568933,  # PCB1位置4
             0.703591897260684, 0.7105805074782172, 0.0027980514341484353, 0.005094645157629108])
]

# PCB2的抓取位置数组 (目前只定义了1个位置)
rb_arm_on_pcb2 = [
    np.array([0.4165396170280912, -0.2591341631409675, 0.15891605521745683,   # PCB2位置1
             0.7090144359375834, 0.7047068297621026, 0.006839549337615578, 0.02529889886172804]),
    np.array([]),  # 预留位置2
    np.array([]),  # 预留位置3
    np.array([])   # 预留位置4
]

# ===== 电池抓取位置 =====
# 电池的抓取位置数组 (4个位置，2x2布局)
rb_arm_on_battery = [
    np.array([0.6064043378480363, -0.019193581297668794, 0.15491817631772764,  # 电池位置1 (左下)
             0.9995912737424026, 0.012216905158772546, -0.012814390858456945, 0.022446025779846696]),
    np.array([0.6064043378480363,  0.103193581297668794, 0.15491817631772764,  # 电池位置2 (左上)
             0.9995912737424026, 0.012216905158772546, -0.012814390858456945, 0.022446025779846696]),
    np.array([0.7079493583489366, -0.019193581297668794, 0.15491817631772764,  # 电池位置3 (右下)
             0.9995912737424026, 0.012216905158772546, -0.012814390858456945, 0.022446025779846696]),
    np.array([0.7079493583489366,  0.101393581297668794, 0.15491817631772764,  # 电池位置4 (右上)
             0.9995912737424026, 0.012216905158772546, -0.012814390858456945, 0.022446025779846696])
]

# ===== 人体工学参数常量 =====
# 基于DIN 33402-2标准的人体尺寸数据 (男女平均值)
forearmlenghdin = 0.2688      # 前臂长度 (m) - 标准值
upperarmlenghtdin = 0.342     # 上臂长度 (m) - 标准值

# 人体工学范围值
forearmlenghdin_max = 0.355   # 前臂最大长度 (m)
forearmlenghdin_min = 0.187   # 前臂最小长度 (m)

upperarmlenghtdin_max = 0.405 # 上臂最大长度 (m)
upperarmlenghtdin_min = 0.285 # 上臂最小长度 (m)

tcp_coversion = 0.2           # TCP转换参数

# ===== 安全坐标定义 =====
savety_koord_1 = np.array([ 0.20,  0.0, 0.6])   # 安全坐标1 - 高位安全点
savety_koord_2 = np.array([-0.24, -0.7, 0.04])  # 安全坐标2 - 低位安全点

user = ""  # 用户标识符

use_built_in_rb_control = False  # 是否使用内置机器人控制

# ===== 机器人控制类 =====
class RobotControl:
    """
    机器人控制类 - 负责UR5机械臂的所有运动控制
    功能包括：
    - MoveIt运动规划和执行  
    - 夹爪控制
    - 轨迹规划
    - 安全监控
    """
    
    def __init__(self, group_name):
        """
        初始化机器人控制系统
        Args:
            group_name (str): MoveIt规划组名称 (通常为 'manipulator')
        """
        
        if not use_built_in_rb_control:       
            # 初始化MoveIt组件和夹爪控制节点
            self.group_name = group_name                        # 存储规划组名称
            self.move_group = MoveGroupCommander(self.group_name)  # MoveIt运动规划接口
            self.gripper_controller = GripperController()          # 夹爪控制器
            self.scene = PlanningSceneInterface()                  # 场景规划接口
            self.robot = moveit_commander.RobotCommander()         # 机器人状态接口

            rospy.sleep(2)  # 短暂等待，确保场景初始化完成

            # ===== 添加碰撞检测对象到MoveIt场景 =====
            # 获取规划坐标框架
            planning_frame = self.move_group.get_planning_frame()
            rospy.loginfo("规划坐标框架: %s", planning_frame)

            # 添加工作台面作为碰撞对象
            Tisch = PoseStamped()
            Tisch.header.frame_id = planning_frame
            Tisch.pose.position.x = 0.0      # 台面中心X坐标
            Tisch.pose.position.y = 0.0      # 台面中心Y坐标
            Tisch.pose.position.z = -0.09    # 台面高度（机器人基座下方）
            
            self.scene.add_box("Tisch", Tisch, size=(3, 2, 0.05))  # 添加台面：长3m，宽2m，厚0.05m
            rospy.loginfo("工作台面已添加到规划场景中。")

            # 添加左侧墙壁作为碰撞对象
            Wand_links = PoseStamped()
            Wand_links.header.frame_id = planning_frame 
            Wand_links.pose.position.x = -0.37  # 左墙位置
            Wand_links.pose.position.y = 0.00   # 墙壁中心Y坐标
            Wand_links.pose.position.z = 0.00   # 墙壁底部Z坐标
            
            self.scene.add_box("Wand_links", Wand_links, size=(0.05, 3, 3))  # 添加左墙：厚0.05m，宽3m，高3m
            rospy.loginfo("左侧墙壁已添加到规划场景中")

            # 添加后方墙壁作为碰撞对象
            Wand_hinten = PoseStamped()
            Wand_hinten.header.frame_id = planning_frame 
            Wand_hinten.pose.position.x = 0.00   # 后墙中心X坐标
            Wand_hinten.pose.position.y = 0.34   # 后墙位置
            Wand_hinten.pose.position.z = 0.00   # 后墙底部Z坐标
            
            self.scene.add_box("Wand_hinten", Wand_hinten, size=(3, 0.05, 3))  # 添加后墙：宽3m，厚0.05m，高3m
            rospy.loginfo("后方墙壁已添加到规划场景中")

            # 添加天花板作为碰撞对象
            Decke = PoseStamped()
            Decke.header.frame_id = planning_frame  
            Decke.pose.position.x = 0.0    # 天花板中心X坐标
            Decke.pose.position.y = 0.0    # 天花板中心Y坐标
            Decke.pose.position.z = 0.92   # 天花板高度
            
            self.scene.add_box("Decke", Decke, size=(3, 2, 0.05))  # 添加天花板：长3m，宽2m，厚0.05m
            rospy.loginfo("天花板已添加到规划场景中。")

            # 添加支架底板作为碰撞对象
            Halter_Grundplatte = PoseStamped()
            Halter_Grundplatte.header.frame_id = planning_frame  
            Halter_Grundplatte.pose.position.x = 2*0.28   # 支架X位置
            Halter_Grundplatte.pose.position.y = -0.59    # 支架Y位置
            Halter_Grundplatte.pose.position.z = -0.04    # 支架Z位置
            
            self.scene.add_box("Halter_Grundplatte", Halter_Grundplatte, size=(0.60, 0.22, 0.22))  # 支架尺寸
            rospy.loginfo("支架底板已添加到规划场景中。")

            # 获取末端执行器链接信息
            eef_link = self.move_group.get_end_effector_link()
            rospy.loginfo("末端执行器链接: %s", eef_link)

            # ===== 设置运动参数 =====
            self.move_group.set_max_velocity_scaling_factor(0.1)      # 设置最大速度比例为10%
            self.move_group.set_max_acceleration_scaling_factor(0.1)  # 设置最大加速度比例为10%

        # ===== 用户输入初始化 =====
        while True:
            user = input('请输入用户名(按Enter使用默认值): ')
            if (user == ""):
                user = "test"  # 默认用户名
                print(f"用户名: {user}")
            break
        
    def convert_to_pose(self, koords):
        """
        将1x7数组转换为Pose对象
        Args:
            koords: 包含位置和姿态的数组 [x, y, z, qx, qy, qz, qw]
        Returns:
            Pose: ROS Pose消息对象
        """
        target_pose = Pose()
        target_pose.position.x = koords[0]     # X位置
        target_pose.position.y = koords[1]     # Y位置
        target_pose.position.z = koords[2]     # Z位置
        target_pose.orientation.x = koords[3]  # 四元数X
        target_pose.orientation.y = koords[4]  # 四元数Y
        target_pose.orientation.z = koords[5]  # 四元数Z
        target_pose.orientation.w = koords[6]  # 四元数W
        return target_pose
    
    def convert_to_koords(self, pose= Pose()):
        """
        将Pose对象转换为1x7数组
        Args:
            pose: ROS Pose消息对象
        Returns:
            list: 包含位置和姿态的数组 [x, y, z, qx, qy, qz, qw]
        """
        koords = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        koords[0] = pose.position.x       # X位置
        koords[1] = pose.position.y       # Y位置
        koords[2] = pose.position.z       # Z位置
        koords[3] = pose.orientation.x    # 四元数X
        koords[4] = pose.orientation.y    # 四元数Y
        koords[5] = pose.orientation.z    # 四元数Z
        koords[6] = pose.orientation.w    # 四元数W
        return koords

    def move_to_target(self, target_pose, speed):
        """
        将机器人移动到目标位置(点到点运动)
        Args:
            target_pose: 目标位置和姿态
            speed: 运动速度(1-100的百分比)
        Returns:
            bool: 运动是否成功
        """
        if not use_built_in_rb_control:
            # 使用外部机器人控制
            command = [{'type':'p2p','pose':target_pose}] 
            self.publish_rb_cmds(command)
            return True
        
        # 使用MoveIt进行运动规划
        self.move_group.set_max_velocity_scaling_factor(speed / 100.0)  # 设置速度比例
        self.move_group.set_pose_target(target_pose)                     # 设置目标位置
        rospy.loginfo("移动机器人到: x={}, y={}, z={}".format(target_pose.position.x, target_pose.position.y, target_pose.position.z))
        # rospy.loginfo("示例坐标: 0.2662104568594572, -0.35661957908057046, 0.24265798894634866 | 姿态: 0.0050765060764118896, -0.8027125907596652, 0.5948306511336113, 0.042464363811632704")
        success = self.move_group.go(wait=True)  # 执行运动并等待完成
        if success:
            rospy.loginfo("运动成功!")
            return True
        else:
            rospy.logwarn("运动失败!")
            self.move_group.stop()                    # 停止机器人运动
            self.move_group.clear_pose_targets()      # 清除目标位置
            return False

    def move_to_target_carth(self, target_pose, speed):
        """
        将机器人沿直线移动到目标位置(笛卡尔运动)
        Args:
            target_pose: 目标位置和姿态
            speed: 运动速度(1-100的百分比)
        Returns:
            bool: 运动是否成功
        """
        if not use_built_in_rb_control:
            # 使用外部机器人控制的笛卡尔运动
            command = [{'type':'cartesian','pose':target_pose}]
            self.publish_rb_cmds(command)
            return True

        # 使用MoveIt进行笛卡尔路径规划
        self.move_group.set_max_velocity_scaling_factor(speed / 100.0)
        waypoints = []  # 路径点列表
        waypoints.append(target_pose)                                        # 添加目标位置到路径点
        self.move_group.set_planning_time(10.0)                             # 设置规划时间限制
        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.05)  # 计算笛卡尔路径，步长0.05m
        rospy.loginfo("直线移动机器人到: x={}, y={}, z={}".format(target_pose.position.x, target_pose.position.y, target_pose.position.z))

        success = self.move_group.execute(plan, wait=True)  # 执行规划的路径
        if success:
            rospy.loginfo("直线运动成功!")
            return True
        else:
            rospy.logwarn("直线运动失败!")
            self.move_group.stop()                    # 停止机器人运动
            self.move_group.clear_pose_targets()      # 清除目标位置
            return False
        
    def move_to_target_carth_plan(self, waypoints, speed):
        """
        沿多个路径点进行笛卡尔运动
        Args:
            waypoints: 路径点列表，包含多个目标位置
            speed: 运动速度(1-100的百分比)
        Returns:
            bool: 运动是否成功
        """
        if not use_built_in_rb_control:
            # 使用外部机器人控制，逐个发送路径点
            for pose in waypoints:
                command = [{'type':'cartesian','pose':pose}]
                self.publish_rb_cmds(command)
                return True

        # 使用MoveIt进行多点笛卡尔路径规划
        self.move_group.set_max_velocity_scaling_factor(speed / 100.0)      # 设置速度比例

        self.move_group.set_planning_time(10.0)                             # 设置规划时间限制
        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.05)  # 计算经过所有路径点的笛卡尔路径
        
        success = self.move_group.execute(plan, wait=True)  # 执行完整路径
        if success:
            rospy.loginfo("多点路径运动成功!")
            return True
        else:
            rospy.logwarn("多点路径运动失败!")
            self.move_group.stop()                    # 停止机器人运动
            self.move_group.clear_pose_targets()      # 清除目标位置
            return False 

    def move_to_taget_plan(self, waypoints, speed):
        """
        按序列移动到一系列路径点(点到点运动)
        Args:
            waypoints: 路径点列表，机器人将依次访问这些点
            speed: 运动速度(1-100的百分比)
        Returns:
            bool: 所有路径点是否都成功到达
        """
        if not use_built_in_rb_control:
            # 使用外部机器人控制，逐个发送路径点
            for pose in waypoints:
                command = [{'type':'p2p','pose':pose}]
                self.publish_rb_cmds(command)
                return True
            
        # 使用MoveIt逐个规划和执行路径点
        self.move_group.set_max_velocity_scaling_factor(speed / 100.0)
        for i, waypoint in enumerate(waypoints):
            self.move_group.set_pose_target(waypoint)       # 设置当前路径点为目标
            plan = self.move_group.plan()                   # 规划到该点的路径
            if plan[0]:                                     # 如果规划成功
                rospy.loginfo(f"执行路径点 {i+1}...")
                self.move_group.execute(plan[1], wait=True) # 执行规划的路径
            else:
                rospy.logwarn(f"无法到达路径点 {i+1}!")
                return False
        return True

    def move_to_joint_goal(self, joint_goal, speed):
        """
        将机器人移动到指定的关节角度
        Args:
            joint_goal: 目标关节角度 (弧度制的6个关节角度)
            speed: 运动速度(1-100的百分比)
        Returns:
            bool: 运动是否成功
        """
        if not use_built_in_rb_control:
            command = [{'type':'joint','joints':joint_goal}]
            rospy.loginfo(type(command))
            rospy.loginfo(command)
            self.publish_rb_cmds(command)
            return True

        self.move_group.set_max_velocity_scaling_factor(speed / 100.0)
        success = self.move_group.go(joint_goal, wait=True)
        if success:
            rospy.loginfo("Bewegung erfolgreich!")
            return True
        else:
            rospy.logwarn("Bewegung fehlgeschlagen!")
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            return False

    def stop_robot(self):
        """
        停止机器人运动
        立即停止当前执行的运动指令，用于紧急停止或重新规划
        """
        # 停止机器人的移动
        self.move_group.stop()
        rospy.loginfo("机器人已停止！")

    def reset_robot(self):
        """
        重置机器人到初始位置
        将机器人移动到预定义的"home"位置，通常用于初始化或结束任务时
        Returns:
            bool: 移动是否成功 (隐式返回)
        """
        # 设置机器人回到预定义的home位置
        self.move_group.set_named_target("home")
        self.move_group.go(wait=True)  # 等待移动完成
        rospy.loginfo("机器人已重置到'Home'位置！")

    def handover_to_hum(self, speed):
        """
        执行向人类交接物品的动作
        根据计算出的肩膀位置，规划机器人的交接路径
        Args:
            speed: 运动速度 (1-100的百分比)
        Returns:
            bool: 交接动作是否成功完成
        """
        # 执行机器人向人类移动的动作
        handover_pose_end = Pose()
        try:
            # 计算基于肩膀位置的最佳交接位置
            handover_pose_end = self.point_inside(self.calc_handover_position_schoulder())
        except:
            rospy.logwarn("无法计算交接位置！")
            return False

        # 设置交接的起始位置(在结束位置前方10cm)
        handover_pose_start = copy.deepcopy(handover_pose_end)
        handover_pose_start.position.y = handover_pose_end.position.y + 0.1

        # 如果使用外部机器人控制系统
        if not use_built_in_rb_control:
            command = [{'type': 'cartesian', 'pose': handover_pose_start},
                      {'type': 'cartesian', 'pose': handover_pose_end}]
            self.publish_rb_cmds(command)
            return True
        
        rospy.loginfo("移动机器人到位置：x={}, y={}, z={}".format(
            handover_pose_start.position.x, 
            handover_pose_start.position.y, 
            handover_pose_start.position.z))

        # 先移动到准备位置，再移动到最终交接位置
        if not self.move_to_target_carth(handover_pose_start, speed):
            return False
        if not self.move_to_target_carth(handover_pose_end, speed):
            return False
        return True

    def calc_handover_position_schoulder(self):
        """
        计算基于肩膀坐标的最符合人体工程学的交接位置
        通过追踪人的肩膀位置来确定最舒适的物品交接点
        Returns:
            Pose: 计算出的交接位置姿态
        """
        # 创建TF广播器和监听器，用于坐标系变换
        broadcaster = tf.TransformBroadcaster()
        listener = tf.TransformListener()  

        # 尝试两次获取人体测量数据，每次最多尝试10秒
        for i in range(2):
            for sek in range(10):
                hm = get_Hum_mertics()  # 获取人体测量数据
                
                # 检查是否检测到完整的人体关键点(肩膀、肘部、手部)并且在正常范围内
                if (not(all(x == 0 for x in hm.shoulderkoords)) and 
                    not(all(x == 0 for x in hm.elbowkoords)) and 
                    not(all(x == 0 for x in hm.handkoords)) and 
                    hm.inside_norm_upper and hm.inside_norm_fore):
                    
                    rospy.loginfo("检测到肩膀、肘部和手部")
                    rospy.loginfo("前臂长度: %s", hm.forearmlenght)
                    rospy.loginfo("上臂长度: %s", hm.uperarmlenght)
                    rospy.loginfo("肩膀坐标: %s", hm.shoulderkoords)

                    # 根据人体尺寸计算交接位置的偏移量
                    # y轴：前臂长度 + TCP转换系数
                    # z轴：负上臂长度(向下偏移)
                    translation = [0, (hm.forearmlenght + tcp_coversion), -hm.uperarmlenght]
                    break

                # 如果只检测到肩膀和肘部，且上臂在正常范围内
                elif (not(all(x == 0 for x in hm.shoulderkoords)) and 
                      not(all(x == 0 for x in hm.elbowkoords)) and 
                      hm.inside_norm_upper):
                    
                    rospy.loginfo("检测到肩膀和肘部")
                    rospy.loginfo("前臂长度: %s", hm.forearmlenght)
                    rospy.loginfo("上臂长度: %s", hm.uperarmlenght)
                    rospy.loginfo("肩膀坐标: %s", hm.shoulderkoords)

                    # 使用标准前臂长度(DIN标准)计算交接位置
                    translation = [0, (forearmlenghdin + tcp_coversion), -hm.uperarmlenght]
                    break

                # 如果只检测到肩膀
                elif not(all(x == 0 for x in hm.shoulderkoords)):
                    rospy.loginfo("检测到肩膀")
                    rospy.loginfo("前臂长度: %s", hm.forearmlenght)
                    rospy.loginfo("上臂长度: %s", hm.uperarmlenght)
                    rospy.loginfo("肩膀坐标: %s", hm.shoulderkoords)

                    # 使用标准DIN人体尺寸进行计算
                    translation = [0, (forearmlenghdin + tcp_coversion), -upperarmlenghtdin]
                    break

                else:
                    rospy.loginfo(f"未检测到人体关键点，尝试: {sek}/10")
                    time.sleep(1)  # 等待1秒后重试

            try:    
                # 创建交接点的坐标信息
                handover_point = PointStamped()
                handover_point.header.frame_id = "right_shoulder"  # 基于右肩坐标系
                handover_point.header.stamp = rospy.Time.now() 
                handover_point.point.x = translation[0]     # x轴偏移
                handover_point.point.y = -translation[1]    # y轴偏移(负值表示向机器人方向)
                handover_point.point.z = translation[2]     # z轴偏移(向下)

                time.sleep(3)  # 等待坐标系稳定
                # 等待从base到right_shoulder的坐标变换可用
                listener.waitForTransform("base", "right_shoulder", rospy.Time(0), rospy.Duration(1.0))
                    
                # 广播交接位置的坐标变换
                broadcaster.sendTransform(
                    (handover_point.point.x, handover_point.point.y, handover_point.point.z),
                    (0.0, 0.0, 0.0, 1.0),  # 四元数表示的旋转(无旋转)
                    rospy.Time.now(),
                    "handover_position",    # 目标坐标系名称
                    "right_shoulder"        # 源坐标系名称
                )

                # 等待从base到handover_position的坐标变换可用
                listener.waitForTransform("base", "handover_position", rospy.Time(0), rospy.Duration(1.0))
                # 获取交接位置在base坐标系下的坐标
                hand_over_position_koords, _ = listener.lookupTransform("base", "handover_position", rospy.Time(0))

                # 将坐标转换为机器人可识别的姿态格式
                # 注意：坐标转换时x、y轴取负值以适应机器人坐标系
                hand_over_position = self.convert_to_pose(np.array([
                    -hand_over_position_koords[0],  # x坐标(取负值)
                    -hand_over_position_koords[1],  # y坐标(取负值)  
                    hand_over_position_koords[2],   # z坐标
                    tcp_to_hum[0],                  # 四元数x
                    tcp_to_hum[1],                  # 四元数y
                    tcp_to_hum[2],                  # 四元数z
                    tcp_to_hum[3]                   # 四元数w
                ]))
                
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logwarn(f"坐标变换出错: {e}")

            hm = None  # 清理变量
            return hand_over_position
        
    def reset_robot(self):
        """
        重置机器人到home位置的备用方法
        """
        self.move_group.set_named_target("home")
        self.move_group.go(wait=True)
        rospy.loginfo("机器人已重置到'Home'位置！")

    def point_inside(self, pose):
        """
        检查交接位置是否在安全区域矩形范围内
        Args:
            pose: 要检查的位置姿态
        Returns:
            Pose: 调整后的安全位置姿态
        """
        # 获取位置坐标
        point = [pose.position.x, pose.position.y, pose.position.z]
        
        # 计算安全区域的边界范围
        xmin, xmax = sorted([savety_koord_1[0], savety_koord_2[0]])
        ymin, ymax = sorted([savety_koord_1[1], savety_koord_2[1]])
        zmin, zmax = sorted([savety_koord_1[2], savety_koord_2[2]])
        
        # 检查点是否在安全区域外
        if (point[0] < xmin or point[0] > xmax or 
            point[1] < ymin or point[1] > ymax or 
            point[2] < zmin or point[2] > zmax):
            rospy.logwarn(f"点位于安全区域外")

        # 将坐标限制在安全区域内
        pose.position.x = max(xmin, min(point[0], xmax))  # 限制x坐标
        pose.position.y = max(ymin, min(point[1], ymax))  # 限制y坐标
        pose.position.z = max(zmin, min(point[2], zmax))  # 限制z坐标

        return pose

    def pick_up(self, target):
        """
        执行拾取物品的动作
        Args:
            target: 目标位置的坐标数组 [x, y, z, qx, qy, qz, qw]
        Returns:
            bool: 拾取动作是否成功
        """
        # 创建目标位置上方的安全点(高10cm)
        over_target = target.copy()
        over_target[2] = over_target[2] + 0.1  # z轴向上偏移10cm
        
        # 如果使用外部机器人控制系统
        if not use_built_in_rb_control:
            over_target_pose = self.convert_to_pose(over_target)
            target_pose = self.convert_to_pose(target)

            # 定义拾取动作序列：移动到上方->下降拾取->上升到安全位置
            command = [{'type': 'p2p', 'pose': over_target_pose},      # 点对点移动到上方
                      {'type': 'cartesian', 'pose': target_pose},      # 笛卡尔下降到目标
                      {'type': 'cartesian', 'pose': over_target_pose}] # 笛卡尔上升到安全位置
            
            self.publish_rb_cmds(command)
            return True

        # 使用内置MoveIt控制系统执行拾取序列
        # 1. 移动到目标上方的安全位置
        if not self.move_to_target(self.convert_to_pose(over_target), 5):
            rospy.logwarn("无法移动到目标上方")
            return False
        
        # 2. 打开夹爪准备拾取
        if not self.gripper_controller.send_gripper_command('open'):
            rospy.logwarn("无法打开夹爪")
            return False

        # 3. 笛卡尔直线下降到目标位置
        if not self.move_to_target_carth(self.convert_to_pose(target), 10):
            rospy.logwarn("无法下降到目标位置")
            return False
        
        # 4. 关闭夹爪抓取物品
        if not self.gripper_controller.send_gripper_command('close'):
            rospy.logwarn("无法关闭夹爪")
            return False

        # 5. 笛卡尔直线上升到安全位置
        if not self.move_to_target_carth(self.convert_to_pose(over_target), 5):
            rospy.logwarn("无法上升到安全位置")
            return False
        
        return True
    
    def pick_up_plan(self, target):
        """
        带路径规划的拾取动作
        相比pick_up方法，这个方法会进行更详细的轨迹规划
        Args:
            target: 目标位置的坐标数组 [x, y, z, qx, qy, qz, qw]
        Returns:
            list: 包含拾取动作序列的指令列表
        """
        # 创建目标位置上方的安全点(高10cm)
        over_target = target.copy()
        over_target[2] = over_target[2] + 0.1
        
        # 返回详细的拾取动作序列
        return [
            {"type": "cartesian", "pose": self.convert_to_pose(over_target)},  # 移动到上方
            {"type": "gripper", "action": 'open'},                             # 打开夹爪
            {"type": "cartesian", "pose": self.convert_to_pose(target)},       # 下降到目标
            {"type": "gripper", "action": 'close'},                            # 关闭夹爪
            {"type": "cartesian", "pose": self.convert_to_pose(over_target)}   # 上升到安全位置
        ]
        
    def planner(self, command_list):
        """
        机器人轨迹规划器，用于执行各种运动指令序列
        Args:
            command_list: 包含运动指令的列表
        Returns:
            bool: 轨迹规划和执行是否成功
        """
        trajectory = RobotTrajectory()
        joint_trajectory = trajectory.joint_trajectory
        joint_trajectory.joint_names = self.move_group.get_joints()
        waypoints = []
        
        # 遍历指令列表，处理不同类型的运动
        for command in command_list:
            try:
                # 笛卡尔空间运动
                if command['type'] == 'cartesian':
                    waypoints.append(command['pose'])  # 添加路径点

                # 关节空间运动
                elif command['type'] == 'joint':
                    self.move_group.set_joint_value_target(command['joints'])
                    plan = self.move_group.plan()
                    if plan and plan[0]:
                        if not self.move_group.execute(plan[1], wait=True):
                            rospy.logwarn("关节运动执行失败")
                            return False
                    else:
                        rospy.logwarn("关节运动无法规划")
                        return False
                    
                # 点对点运动到指定姿态
                elif command['type'] == 'p2p':
                    self.move_group.set_pose_target(command['pose'])
                    plan = self.move_group.plan()
                    if plan and plan[0]:
                        if not self.move_group.execute(plan[1], wait=True):
                            rospy.logwarn("点对点运动执行失败")
                            return False
                    else:
                        rospy.logwarn("点对点运动无法规划")
                        return False
                    
                    # 如果有待执行的笛卡尔路径点，先执行完成
                    if waypoints:
                        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.01, True)
                        if fraction >= 1.0:  # 修正逻辑：fraction >= 1.0 表示成功
                            self.move_group.execute(plan, wait=True)
                        else:
                            rospy.logwarn("笛卡尔运动无法完全规划")
                            return False
                        waypoints = []  # 清空路径点列表
                    
                    # 执行夹爪指令
                    if not self.gripper_controller.send_gripper_command(command['action']):
                        rospy.logwarn(f"夹爪指令执行失败: {command['action']}")
                        return False
                        
            except Exception as e:
                rospy.logerr(f"处理指令时出错: {e}")
                return False
        
        # 执行剩余的笛卡尔路径点
        if waypoints:
            (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.01, True)
            if fraction >= 1.0:  # 修正逻辑
                if not self.move_group.execute(plan, wait=True):
                    rospy.logwarn("笛卡尔路径执行失败")
                    return False 
            else:
                rospy.logwarn("笛卡尔运动无法完全规划")
                return False
        return True

    def publish_rb_cmds(self, commands):
        """
        发布机器人指令到外部控制系统
        Args:
            commands: 包含机器人指令的列表
        """
        cmd_nr = 0
        for cmd in commands:
            msg = robot_msgs()
            command_pub = rospy.Publisher('/robot/command', robot_msgs, queue_size=10)

            rospy.loginfo(f"指令类型: {type(cmd)}")
            msg.nr = cmd_nr  # 指令序号
            msg.type = cmd["type"]  # 指令类型
            
            # 关节空间指令
            if cmd["type"] == "joint":
                msg.joints = list(cmd["joints"])

            # 笛卡尔空间指令
            elif cmd["type"] == "cartesian":
                msg.pose = cmd["pose"]
            
            # 点对点运动指令
            elif cmd["type"] == "p2p":
                msg.pose = cmd["pose"]

            rospy.loginfo(f"发布指令: {msg}")

            command_pub.publish(msg)  # 发布指令
            cmd_nr += 1  # 递增指令序号
            rospy.sleep(1)  # 等待1秒确保指令发送完成

# ====== 夹爪控制器 ======

class GripperController:
    def __init__(self):
        """
        初始化夹爪控制器
        设置ROS发布者和指令消息结构
        """
        # 初始化夹爪控制器
        self.pub = rospy.Publisher('Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=10)
        self.command = outputMsg.Robotiq2FGripper_robot_output()

    def statusInterpreter(status):
        """
        根据状态变量的当前值生成状态描述字符串
        Args:
            status: 夹爪状态消息
        Returns:
            str: 状态描述字符串
        """
        rospy.Subscriber("Robotiq2FGripperRobotInput", inputMsg.Robotiq2FGripper_robot_input, status)
        output = ''
        
        # gSTA - 夹爪状态
        if(status.gSTA == 0):
            output = '夹爪处于复位状态(或自动释放状态)。如果夹爪已激活，请查看故障状态\n'
        if(status.gSTA == 1):
            output = '激活进行中\n'
        if(status.gSTA == 2):
            output = '未使用\n'
        if(status.gSTA == 3):
            output = '激活状态'

        # gOBJ - 物体检测状态
        if(status.gOBJ == 0):
            output = '手指正在运动中(仅在gGTO = 1时有意义)\n'
        if(status.gOBJ == 1):
            output = '手指在张开时因接触而停止\n'
        if(status.gOBJ == 2):
            output = '关闭状态'
        if(status.gOBJ == 3):
            output = '打开状态'
    
        # gFLT - 故障状态
        if(status.gFLT == 0x05):
            output = '优先级故障：动作延迟，必须在动作前完成初始化\n'
        if(status.gFLT == 0x07):
            output = '优先级故障：动作前必须设置激活位\n'
        if(status.gFLT == 0x09):
            output = '轻微故障：通信芯片未就绪(可能正在启动)\n'   
        if(status.gFLT == 0x0B):
            output = '轻微故障：自动释放进行中\n'
        if(status.gFLT == 0x0E):
            output = '重大故障：过流保护触发\n'
        if(status.gFLT == 0x0F):
            output = '重大故障：自动释放完成\n'
        rospy.loginfo(output)
        return output

    def send_gripper_command(self, action_type):
        """
        向夹爪发送控制指令
        Args:
            action_type: 动作类型 ('open', 'close', 'activate', 'reset')
        Returns:
            bool: 指令发送是否成功
        """
        # 张开夹爪
        if action_type == 'open':
            self.command.rPR = 0  # 位置请求：0 = 完全张开
        # 关闭夹爪
        elif action_type == 'close':
            self.command.rPR = 255  # 位置请求：255 = 完全关闭
        # 激活夹爪
        elif action_type == 'activate':
            self.command.rACT = 1   # 激活夹爪
            self.command.rGTO = 1   # 前往请求位置
            self.command.rSP = 255  # 速度：最大速度
            self.command.rFR = 150  # 力度：中等力度
        # 停用夹爪
        elif action_type == 'deactivate':
            self.command.rACT = 0   # 停用夹爪
            
        self.pub.publish(self.command)  # 发布指令
        rospy.sleep(2)  # 等待指令执行
        return action_type == self.statusInterpreter  # 验证指令是否成功执行

# ====== 人体数据获取 ======

class get_Hum_mertics:
    """
    人体测量数据获取类
    用于追踪和存储人体关键点位置及尺寸信息
    """
    def __init__(self):
        """
        初始化人体追踪数据
        设置各关键点的初始坐标和尺寸参数
        """
        # 人体尺寸数据
        self.uperarmlenght = 0.0    # 上臂长度
        self.forearmlenght = 0.0    # 前臂长度
        
        # 左侧身体关键点坐标 [x, y, z]
        self.shoulderkoords = [0.0, 0.0, 0.0]  # 肩膀坐标
        self.elbowkoords =    [0.0, 0.0, 0.0]  # 肘部坐标
        self.handkoords =     [0.0, 0.0, 0.0]  # 手部坐标
        
        # 右侧身体关键点坐标
        self.rightshoulderkoords = [0.0, 0.0, 0.0]  # 右肩坐标
        self.rightelbowkoords =    [0.0, 0.0, 0.0]  # 右肘坐标
        self.righthandkoords =     [0.0, 0.0, 0.0]  # 右手坐标
        
        # 左侧身体关键点坐标
        self.leftshoulderkoords = [0.0, 0.0, 0.0]   # 左肩坐标
        self.leftelbowkoords =    [0.0, 0.0, 0.0]   # 左肘坐标
        self.lefthandkoords =     [0.0, 0.0, 0.0]   # 左手坐标
        
        # 尺寸范围检查标志
        self.inside_norm_upper = True  # 上臂长度是否在正常范围内
        self.inside_norm_fore  = True  # 前臂长度是否在正常范围内
        
        self.calc_arm_lenght()  # 计算手臂长度
        self.stop_event = threading.Event()  # 停止事件，用于线程控制

    def camera_listener(self):
        """
        摄像头监听器，从TF系统读取肩膀、肘部和手部的坐标变换
        实时更新人体关键点的三维坐标信息
        """
        try:
            time = rospy.Time(0)  # 获取最新的变换时间
            listener = tf.TransformListener()

            # 等待各关键点的坐标变换可用
            listener.waitForTransform("base", "shoulder", time, rospy.Duration(1.0))
            listener.waitForTransform("base", "elbow",    time, rospy.Duration(1.0))
            listener.waitForTransform("base", "hand",     time, rospy.Duration(1.0))

            # 获取各关键点在base坐标系下的位置
            shoulder_trans, _ = listener.lookupTransform("base", "shoulder",  time)
            elbow_trans,    _ = listener.lookupTransform("base", "elbow",     time)
            hand_trans,     _ = listener.lookupTransform("base", "hand",      time)

            # 更新关键点坐标
            self.shoulderkoords = [shoulder_trans[0], shoulder_trans[1], shoulder_trans[2]]
            self.elbowkoords =    [elbow_trans[0], elbow_trans[1], elbow_trans[2]]
            self.handkoords =     [hand_trans[0], hand_trans[1], hand_trans[2]]

        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF变换错误: {e}")
    
    def camera_listener_arms(self):
        """
        双臂摄像头监听器，分别读取左右臂的肩膀、肘部和手部坐标变换
        用于更精确的双臂追踪和人体姿态识别
        """
        try:
            time = rospy.Time(0)  # 获取最新变换时间
            listener = tf.TransformListener()

            # 等待右臂关键点的坐标变换
            listener.waitForTransform("base", "right_shoulder", time, rospy.Duration(1.0))
            listener.waitForTransform("base", "right_elbow",    time, rospy.Duration(1.0))
            listener.waitForTransform("base", "right_hand",     time, rospy.Duration(1.0))

            # 获取右臂关键点位置
            right_shoulder_trans, _ = listener.lookupTransform("base", "right_shoulder",  time)
            right_elbow_trans,    _ = listener.lookupTransform("base", "right_elbow",     time)
            right_hand_trans,     _ = listener.lookupTransform("base", "right_hand",      time)

            # 更新右臂关键点坐标
            self.rightshoulderkoords = [right_shoulder_trans[0], right_shoulder_trans[1], right_shoulder_trans[2]]
            self.rightelbowkoords =    [right_elbow_trans[0],    right_elbow_trans[1],    right_elbow_trans[2]]
            self.righthandkoords =     [right_hand_trans[0],     right_hand_trans[1],     right_hand_trans[2]]

            # 等待左臂关键点的坐标变换
            listener.waitForTransform("base", "left_shoulder", time, rospy.Duration(1.0))
            listener.waitForTransform("base", "left_elbow",    time, rospy.Duration(1.0))
            listener.waitForTransform("base", "left_hand",     time, rospy.Duration(1.0))

            # 获取左臂关键点位置
            left_shoulder_trans, _ = listener.lookupTransform("base", "left_shoulder",  time)
            left_elbow_trans,    _ = listener.lookupTransform("base", "left_elbow",     time)
            left_hand_trans,     _ = listener.lookupTransform("base", "left_hand",      time)

            # 更新左臂关键点坐标
            self.leftshoulderkoords = [left_shoulder_trans[0], left_shoulder_trans[1], left_shoulder_trans[2]]
            self.leftelbowkoords =    [left_elbow_trans[0],    left_elbow_trans[1],    left_elbow_trans[2]]
            self.lefthandkoords =     [left_hand_trans[0],     left_hand_trans[1],     left_hand_trans[2]]

        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF变换错误: {e}")

    def calc_arm_lenght(self):
        """
        计算上臂和前臂的长度
        通过摄像头获取关键点坐标，计算各关节间的欧几里得距离
        """
        # 获取关键点坐标数据
        self.camera_listener()
        self.camera_listener_arms()
        
        # 计算左臂的上臂和前臂长度
        self.uperarmlenght = self.calc_euclidean_distance(self.leftshoulderkoords,  self.leftelbowkoords)
        self.forearmlenght = self.calc_euclidean_distance(self.lefthandkoords,      self.leftelbowkoords)
        
        # 检查尺寸是否在正常范围内
        self.is_inside_norm()

        # 将测量数据保存到CSV文件
        with open('armlaengen.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f'{user}{rospy.Time.now()} 上臂长度: {self.uperarmlenght}'])
            writer.writerow([f'{user}{rospy.Time.now()} 前臂长度: {self.forearmlenght}'])

    def calc_euclidean_distance(self, point1, point2):
        """
        计算两点间的欧几里得距离
        Args:
            point1: 第一个点的坐标 [x, y, z]
            point2: 第二个点的坐标 [x, y, z]
        Returns:
            float: 两点间的距离
        """
        distance = 0.0
        for i in range(len(point1)):
            distance += (point2[i] - point1[i]) ** 2
        return math.sqrt(distance)
    
    def calc_angel(self, point1, point2, point3):
        """
        计算由三个点构成的角度
        Args:
            point1, point2, point3: 三个点的坐标，point2为顶点
        Returns:
            float: 角度(弧度制)
        """
        vec1 = point1 - point2  # 向量1
        vec2 = point2 - point3  # 向量2

        # 使用向量夹角公式计算角度
        angelrad = np.arccos(np.dot(vec1, vec2) / (np.sqrt((vec1*vec1).sum()) * np.sqrt((vec2*vec2).sum())))
        # 将弧度转换为角度
        angle = angelrad * 360 / 2 / np.pi
        return angle

    def is_inside_norm(self):
        """
        检查上臂和前臂长度是否在第5到第95百分位数范围内
        根据DIN标准的人体尺寸数据进行验证
        """
        # 检查上臂长度是否在正常范围内
        if (self.uperarmlenght <= upperarmlenghtdin_max) and (self.uperarmlenght >= upperarmlenghtdin_min):
            self.inside_norm_upper = True
        else:
            rospy.logwarn(f"上臂尺寸超出第5到第95百分位数范围")
            self.inside_norm_upper = False

        # 检查前臂长度是否在正常范围内
        if (self.forearmlenght <= forearmlenghdin_max) and (self.forearmlenght >= forearmlenghdin_min):
            self.inside_norm_fore = True
        else:
            rospy.logwarn(f"前臂尺寸超出第5到第95百分位数范围")
            self.inside_norm_fore = False
    
    def get_arm_angels(self):
        """
        持续获取双臂的肘部角度
        在后台线程中运行，实时计算并记录左右手臂的肘部弯曲角度
        """
        while not self.stop_event.is_set():
            # 获取双臂关键点坐标
            self.camera_listener_arms()

            # 转换为numpy数组便于计算
            right_shoulder = np.array([self.rightshoulderkoords[0], self.rightshoulderkoords[1], self.rightshoulderkoords[2]])
            right_elbow = np.array([self.rightelbowkoords[0], self.rightelbowkoords[1], self.rightelbowkoords[2]])
            right_hand = np.array([self.righthandkoords[0], self.righthandkoords[1], self.righthandkoords[2]])

            left_shoulder = np.array([self.leftshoulderkoords[0], self.leftshoulderkoords[1], self.leftshoulderkoords[2]])
            left_elbow = np.array([self.leftelbowkoords[0], self.leftelbowkoords[1], self.leftelbowkoords[2]])
            left_hand = np.array([self.lefthandkoords[0], self.lefthandkoords[1], self.lefthandkoords[2]])

            # 计算左右手臂的肘部角度
            right_angle = self.calc_angel(right_shoulder, right_elbow, right_hand)
            left_angle = self.calc_angel(left_shoulder, left_elbow, left_hand)

            # 实时显示角度信息
            print(f'右臂角度: {right_angle} 左臂角度: {left_angle}', end='\r') 
            
            # 将角度数据保存到CSV文件
            with open('armlaengen.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f'{user} {rospy.Time.now()} 右肘角度: {right_angle}'])
                writer.writerow([f'{user} {rospy.Time.now()} 左肘角度: {left_angle}'])

# 创建机器人控制实例
robot_control = RobotControl("manipulator")
    
# ---- 画布与区域（与 Initial 一致）----
WIDTH, HEIGHT = 1024, 768
CENTER = (WIDTH // 2, HEIGHT // 2)
RECTS = [
    (0, 0, 18200, 14000, "Bound"),
    (3071.6, 1514.1, 4400, 600, "PCB1"),
    (-3126.8, 1463.4, 7100, 1200, "PCB2"),
    (-6619.7, -1977.6, 2900, 3600, "PCB3"),
    (952.28, 4894.7, 3200, 4200, "Motor"),
    (-4815.2, 4892.9, 5600, 4200, "Battery"),
    (-6619.7, -5577.6, 2900, 1800, "Handover"),
]
RECT_W, RECT_H = 18200, 14000
SCALE = min(WIDTH / RECT_W, HEIGHT / RECT_H)

# ===== 单应性矩阵（从vis.py集成）=====
# 用于将机器人世界坐标(米)精确转换为屏幕像素坐标
H = np.array([
    [7.35135136e+02, 0., 2.03448646e+02],
    [0., 1.02857148e+03, 1.71085702e+02],
    [0., 0., 1.00000000e+00]
])

def world_to_screen(wx, wy):
    """
    将工作区域坐标(毫米制)转换为屏幕像素坐标
    用于绘制RECTS中定义的工作区域矩形
    Args:
        wx, wy: 工作区域坐标 (毫米制单位)
    Returns:
        tuple: 屏幕像素坐标 (px, py)
    """
    return (int(CENTER[0] + wx * SCALE), int(CENTER[1] - wy * SCALE))

def world_to_screen_homography(x, y):
    """
    使用单应性矩阵将机器人世界坐标(米)转换为屏幕像素坐标
    Args:
        x, y: 世界坐标 (米制单位)
    Returns:
        tuple: 屏幕像素坐标 (px, py)
    """
    # 构造齐次坐标
    point = np.array([x, y, 1.0])
    
    # 应用单应性变换
    pixel = H @ point
    
    # 归一化齐次坐标
    pixel /= pixel[2]
    
    return int(round(pixel[0])), int(round(pixel[1]))

def draw_rect(img, cx, cy, w, h, color, name=""):
    pt1 = world_to_screen(cx - w/2, cy - h/2)
    pt2 = world_to_screen(cx + w/2, cy + h/2)
    cv2.rectangle(img, pt1, pt2, color, 2, lineType=cv2.LINE_AA)
    if name:
        cv2.putText(img, name, (pt1[0], pt1[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# ---- 状态→区域映射 & 状态流向（你可按工艺修改）----
STATE_TO_RECT = {
    "MPickUp": "Motor",
    "MHoldHD": "Handover",
    "MPositioning": "Bound",
    "PCB1PickUpAndPositioning": "PCB1",
    "PCB2PickUpAndPositioning": "PCB2",
    "BatteryPickUpAndPositioning": "Battery",
}
NEXT_OF = {
    "MPickUp": "MHoldHD",
    "MHoldHD": "MPositioning",
    "MPositioning": "PCB1PickUpAndPositioning",
    "PCB1PickUpAndPositioning": "PCB2PickUpAndPositioning",
    "PCB2PickUpAndPositioning": "BatteryPickUpAndPositioning",
    "BatteryPickUpAndPositioning": "Finish",
}

RECT_INDEX = { name: i for i, (_,_,_,_,name) in enumerate(RECTS) }

# ---- 预览轨迹：由当前状态名生成 Pose 列表（只显示，不执行）----
def _safe_pose(arr7):
    try:
        return robot_control.convert_to_pose(arr7)
    except Exception:
        p = Pose(); p.position.z = 0.3; p.orientation.w = 1.0
        return p

def preview_MPickUp():
    poses = []
    try:
        over_m = _safe_pose(rb_arm_transition_over_m)
        t0 = rb_arm_on_m[0] if isinstance(rb_arm_on_m, list) and len(rb_arm_on_m)>0 else rb_arm_transition_over_m
        over = np.array(t0).copy(); over[2] = t0[2] + 0.10
        poses += [over_m, _safe_pose(over), _safe_pose(t0), _safe_pose(over)]
    except: pass
    return poses

def preview_MHoldHD():
    poses = []
    try:
        start = _safe_pose(rb_arm_transition)
        end   = robot_control.point_inside(_safe_pose(rb_arm_on_hum_static))
        poses += [start, end]
    except: pass
    return poses

def preview_MPositioning():
    poses = []
    try:
        poses += [_safe_pose(rb_arm_transition_over_gb0_1),
                  _safe_pose(rb_arm_transition_over_gb0_2)]
    except: pass
    return poses

def preview_PCB1():
    poses = []
    try:
        t = rb_arm_on_pcb1[0]
        over = np.array(t).copy(); over[2] = t[2] + 0.10
        poses += [_safe_pose(over), _safe_pose(t),
                  _safe_pose(rb_arm_transition_over_gb1_1),
                  _safe_pose(rb_arm_transition_over_gb1_2),
                  _safe_pose(rb_arm_transition_over_gb1_3)]
    except: pass
    return poses

def preview_PCB2():
    poses = []
    try:
        t = None
        for cand in rb_arm_on_pcb2:
            if isinstance(cand, np.ndarray) and cand.size == 7:
                t = cand; break
        if t is not None:
            over = np.array(t).copy(); over[2] = t[2] + 0.10
            poses += [_safe_pose(over), _safe_pose(t)]
        poses += [_safe_pose(rb_arm_transition_over_gb3_1),
                  _safe_pose(rb_arm_transition_over_gb3_2),
                  _safe_pose(rb_arm_transition_over_gb3_3)]
    except: pass
    return poses

def preview_Battery():
    poses = []
    try:
        t = rb_arm_on_battery[0]
        over = np.array(t).copy(); over[2] = t[2] + 0.10
        poses += [_safe_pose(over), _safe_pose(t),
                  _safe_pose(rb_arm_transition_over_gb2_1),
                  _safe_pose(rb_arm_transition_over_gb2_2)]
    except: pass
    return poses

PREVIEW_BUILDERS = {
    "MPickUp": preview_MPickUp,
    "MHoldHD": preview_MHoldHD,
    "MPositioning": preview_MPositioning,
    "PCB1PickUpAndPositioning": preview_PCB1,
    "PCB2PickUpAndPositioning": preview_PCB2,
    "BatteryPickUpAndPositioning": preview_Battery,
}

# ---- GUI 共享状态（线程安全）----
_gui_lock = threading.Lock()
_gui_current_state = "-"
_gui_next_state = "-"
_gui_highlight_idx = None
_gui_preview_poses = []

def gui_set_state(state_name: str, next_name: str):
    global _gui_current_state, _gui_next_state, _gui_highlight_idx, _gui_preview_poses
    with _gui_lock:
        _gui_current_state = state_name
        _gui_next_state = next_name
        rect_name = STATE_TO_RECT.get(state_name, None)
        _gui_highlight_idx = RECT_INDEX.get(rect_name, None)
        builder = PREVIEW_BUILDERS.get(state_name, lambda: [])
        _gui_preview_poses = builder() or []

def gui_clear():
    global _gui_highlight_idx, _gui_preview_poses, _gui_current_state, _gui_next_state
    with _gui_lock:
        _gui_highlight_idx = None
        _gui_preview_poses = []
        _gui_current_state = "-"
        _gui_next_state = "-"

# ---- 绘制：轨迹 + 高亮 + 右下角状态牌 ----
def _pose_to_xy(p: Pose):
    return float(p.position.x), float(p.position.y)

def _draw_trajectory(img, poses, color=(0,0,255)):
    pts = []
    for p in poses:
        x, y = _pose_to_xy(p)
        # 使用单应性矩阵转换机器人坐标(米)到屏幕像素坐标
        # 不需要再经过 world_to_screen了吗？
        pts.append(world_to_screen_homography(x, y))
    if len(pts) >= 2:
        for i in range(len(pts)-1):
            cv2.line(img, pts[i], pts[i+1], color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(img, pt, 3, color, -1, cv2.LINE_AA)

def _draw_state_footer(img):
    with _gui_lock:
        cur, nxt = _gui_current_state, _gui_next_state
    label = f"Current: {cur}    Next: {nxt}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    pad = 10
    x = WIDTH - tw - pad*2 - 20
    y = HEIGHT - th - pad*2 - 20  # 右下角
    cv2.rectangle(img, (x, y), (x + tw + pad*2, y + th + pad*2), (0,0,0), 2)
    cv2.rectangle(img, (x+2, y+2), (x + tw + pad*2 - 2, y + th + pad*2 - 2), (255,255,255), -1)
    cv2.putText(img, label, (x+pad, y + th + int(pad*0.8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)

def _visualizer_thread():
    cv2.namedWindow("Robot Trajectory Visualizer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Trajectory Visualizer", WIDTH, HEIGHT)
    while True:
        img = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
        with _gui_lock:
            hi = _gui_highlight_idx
            preview = list(_gui_preview_poses)
        # 矩形 & 高亮
        for i, (x, y, w, h, name) in enumerate(RECTS):
            color = (0,255,0) if hi != i else (0,0,255)
            draw_rect(img, x, y, w, h, color, name)
        # 轨迹
        if preview:
            _draw_trajectory(img, preview)
        # 右下角状态
        _draw_state_footer(img)

        cv2.imshow("Robot Trajectory Visualizer", img)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:  # ESC 退出
            break
        if cv2.getWindowProperty("Robot Trajectory Visualizer", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyWindow("Robot Trajectory Visualizer")

# 启动 GUI 线程
threading.Thread(target=_visualizer_thread, daemon=True).start()

# =========================
#  包装 SMACH 状态：进入该状态时刷新 GUI
# =========================
class GuiProxyState(smach.State):
    """包装原状态：进入时更新 GUI，再调用原状态的 execute。"""
    def __init__(self, wrapped_state: smach.State, state_name: str):
        # 尝试拿到原状态的 outcomes；失败则要求调用方传 transitions 的 keys 一致
        try:
            outcomes = list(wrapped_state.get_registered_outcomes())
        except Exception:
            outcomes = list(getattr(wrapped_state, "_outcomes", [])) or ['succeeded','aborted']
        smach.State.__init__(self, outcomes=outcomes, input_keys=getattr(wrapped_state,'_input_keys',[]), output_keys=getattr(wrapped_state,'_output_keys',[]))
        self._wrapped = wrapped_state
        self._state_name = state_name

    def execute(self, userdata):
        # 进入状态 → 显示该状态的轨迹、高亮、Current/Next
        gui_set_state(self._state_name, NEXT_OF.get(self._state_name, "-"))
        # 调用原状态逻辑
        outcome = self._wrapped.execute(userdata)
        return outcome

def add_gui_state(sm, label: str, state_obj: smach.State, transitions: dict):
    """替代 smach.StateMachine.add：用 GUI 包装一下原状态。"""
    smach.StateMachine.add(label, GuiProxyState(state_obj, label), transitions=transitions)

# ############################ 初始化SMACH状态机 ############################

class Start(smach.State):
    """
    起始状态类
    定义装配流程的入口点，根据不同的装配任务跳转到相应状态
    """
    def __init__(self):
        smach.State.__init__(self, 
                           outcomes=['MPickUp', 'MHoldHD', 'MPositioning', 
                                   'PCB1PickUpAndPositioning', 'PCB2PickUpAndPositioning', 
                                   'BatteryPickUpAndPositioning', 'succeeded_end', 'test'])
    
    def execute(self, userdata):
        rospy.loginfo(f"执行状态: {self.__class__.__name__}")

        # 显示主菜单
        print("\n--- 主菜单 ---")
        print("1. MPickUp (手机拾取)")
        print("2. MHoldHD (手机支架)")  
        print("3. MPositioning (手机定位)")
        print("4. PCB1PickUpAndPositioning (PCB1拾取和定位)")
        print("5. PCB2PickUpAndPositioning (PCB2拾取和定位)")
        print("6. BatteryPickUpAndPositioning (电池拾取和定位)")
        print("7. Beenden (结束)")
        print("8. Test (测试)")

        start = input("请选择一个状态: ")
        while True:
            if start == "1" or start == "":
                print("\n你选择了手机拾取状态")
                return 'MPickUp'
            elif start == "2":
                print("\n你选择了手机支架状态")
                return 'MHoldHD'
            elif start == "3":
                print("\n你选择了手机定位状态")
                return 'MPositioning'
            elif start == "4":
                print("\n你选择了PCB1拾取和定位状态")
                return 'PCB1PickUpAndPositioning'
            elif start == "5":
                print("\n你选择了PCB2拾取和定位状态")
                return 'PCB2PickUpAndPositioning'
            elif start == "6":
                print("\n你选择了电池拾取和定位状态")
                return 'BatteryPickUpAndPositioning'
            elif start == "7":
                print("\n你选择了中止操作")
                return 'succeeded_end'
            elif start == "8":
                print("\n你选择了测试模式")
                return 'test'
            else:
                print(f"\n选项 {start} 不存在，请重新选择")

class MPickUp(smach.State):
    """
    手机拾取状态类
    负责执行拾取手机的操作序列
    """
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'succeeded_with_HD', 'aborted'])
        self.counter = 0  # 计数器，用于跟踪执行次数
    
    def execute(self, userdata):
        """
        执行手机拾取操作
        Returns:
            str: 执行结果状态 ('succeeded', 'succeeded_with_HD', 'aborted')
        """
        rospy.loginfo(f"执行状态: {self.__class__.__name__}")
        
        # 拾取手机(Motor1)
        ### 以下代码段用于测试时可以注释掉
        #if not robot_control.gripper_controller.send_gripper_command('activate'):
        #    return 'aborted'
        #return 'succeeded_with_HD'

        '''调试代码块用于测试'''
        while True:
            newuser = input('是否继续执行? (y/n): ')
            if newuser == "y":
                # 移动到指定关节位置 (拾取位置)
                if not robot_control.move_to_joint_goal((-3.1557, -1.0119, -2.1765, -1.5426, 1.5686, -3.1643), 10):
                    return 'aborted'
                # 关闭夹爪抓取手机
                if not robot_control.gripper_controller.send_gripper_command('close'):
                    return 'aborted'
                # 重新打开夹爪 (测试用)
                if not robot_control.gripper_controller.send_gripper_command('open'):
                    return 'aborted'
                    
                # 创建移动计划
                plan = []
                plan.append(robot_control.convert_to_pose(rb_arm_transition_over_m))
                if not robot_control.move_to_taget_plan(plan, 10):
                    return 'aborted'
                
                # 执行拾取操作
                if not robot_control.pick_up(rb_arm_on_m[self.counter]):
                    return 'aborted'
                
                # 移动到最终位置
                if not robot_control.move_to_joint_goal((-3.8423, -1.0118, -2.3565, -2.8601, -0.7018, -3.1867), 20):
                    return 'aborted'
                
                self.counter += 1  # 增加计数器
                rospy.loginfo(f"拾取手机 {self.counter}")
                return 'succeeded_with_HD'
                
            elif newuser == "n":
                rospy.loginfo('继续执行')
                return 'succeeded_with_HD'

class MHold(smach.State):
    """
    手机保持状态类
    维持手机的当前位置和状态
    """
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        
    def execute(self, userdata):
        rospy.loginfo(f"执行状态: {self.__class__.__name__}")
                
class MHoldHD(smach.State):
    """
    手机支架保持状态类
    在人机交互过程中保持手机在支架位置
    """
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'succeeded_end', 'aborted'])
        self.hm = get_Hum_mertics()  # 初始化人体测量对象
        
    def execute(self, userdata):
        rospy.loginfo(f"执行状态: {self.__class__.__name__}")

        newuser = input('是否执行交接操作? (y/n/a): ')
        if newuser == "y":
            # 执行向人类交接手机的操作
            if not robot_control.handover_to_hum(5):
                return 'aborted'
            
            # 启动人体角度监测线程
            self.hm.stop_event = threading.Event()
            thread = threading.Thread(target=self.hm.get_arm_angels)
            thread.start()
            
            # 等待用户输入停止监测
            input("按回车键停止角度监测...\n")
            self.hm.stop_event.set()  # 设置停止事件
            thread.join()  # 等待线程结束

            # 返回到预定义位置
            if not robot_control.move_to_joint_goal((-3.8423, -1.0118, -2.3565, -2.8601, -0.7018, -3.1867), 20):
                return 'aborted'
            return 'succeeded'
            
        elif newuser == "n":
            rospy.loginfo('跳过交接操作')
            return 'succeeded'
            
        elif newuser == "a":
            rospy.loginfo('中止操作')
            return 'aborted'

class MPositioning(smach.State):
    """
    手机定位状态类
    负责将手机定位到指定位置，为后续装配做准备
    """
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'succeeded_to_PCB', 'aborted'])
        self.robot_control = robot_control  # 机器人控制实例
        self.counter = 0  # 计数器
        
    def execute(self, userdata):
        rospy.loginfo(f"执行状态: {self.__class__.__name__}")

        self.counter += 1
        # 每4次操作后自动跳转，否则等待用户确认
        if not (self.counter % 4 == 0):
            newuser = input('是否执行定位操作? (y/n): ')
            if newuser == "y":
                # 移动到过渡位置1
                if not robot_control.move_to_target_carth(robot_control.convert_to_pose(rb_arm_transition_over_gb0_1), 10):
                    return 'aborted'
                    
                # 打开夹爪释放手机
                if not robot_control.gripper_controller.send_gripper_command('open'):
                    return 'aborted'
                    
                # 移动到过渡位置2
                if not robot_control.move_to_target_carth(robot_control.convert_to_pose(rb_arm_transition_over_gb0_2), 10):
                    return 'aborted'
                return 'succeeded_to_PCB'
                
            elif newuser == "n":
                rospy.loginfo('跳过定位操作')
                return 'succeeded_to_PCB'
        else:
            # 自动跳转到PCB状态
            return 'succeeded_to_PCB'

class PCB1PickUpAndPositioning(smach.State):
    """
    PCB1拾取和定位状态类
    负责拾取第一块PCB板并将其定位到指定位置
    """
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])
        self.counter = 0  # 计数器
        
    def execute(self, userdata):
        rospy.loginfo(f"执行状态: {self.__class__.__name__}")
        while True:
            newuser = input('是否执行PCB1拾取和定位? (y/n): ')
            if newuser == "y":
                # 移动到PCB1拾取预备位置
                if not robot_control.move_to_joint_goal((-3.1299, -2.1996, -0.6071, -1.8830, 1.5654, -3.1786), 10):
                    return 'aborted'
                
                # 拾取PCB1
                if not robot_control.pick_up(rb_arm_on_pcb1[self.counter]):
                    return 'aborted'
                
                # 创建移动到目标位置的路径规划
                plan = []
                plan.append(robot_control.convert_to_pose(rb_arm_transition_over_gb1_1))  # 过渡位置1
                plan.append(robot_control.convert_to_pose(rb_arm_transition_over_gb1_2))  # 过渡位置2
                plan.append(robot_control.convert_to_pose(rb_arm_transition_over_gb1_3))  # 最终放置位置
                
                if not robot_control.move_to_target_carth_plan(plan, 10):
                    return 'aborted'
                
                # 打开夹爪放置PCB1
                if not robot_control.gripper_controller.send_gripper_command('open'):
                    return 'aborted'
                
                # 创建返回路径规划
                plan = []
                plan.append(robot_control.convert_to_pose(rb_arm_transition_over_gb1_2))  # 返回过渡位置2
                plan.append(robot_control.convert_to_pose(rb_arm_transition_over_gb1_1))  # 返回过渡位置1
                
                if not robot_control.move_to_target_carth_plan(plan, 10):
                    return 'aborted'
                    
                self.counter += 1  # 增加计数器
                return 'succeeded'
                
            elif newuser == "n":
                print("weiter")
                return 'succeeded'        

class PCB2PickUpAndPositioning(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded','aborted'])
        self.counter = 0
    def execute(self, userdata):
        rospy.loginfo(f"Führe state: {self.__class__.__name__} aus.")
        while True:
            newuser = input('enter y/n: ')
            if newuser == "y":
                if not robot_control.move_to_joint_goal((-3.4437, -1.5349, -1.6576, -1.5354, 1.5145, -3.469),10):
                    return 'aborted'
                if not robot_control.pick_up(rb_arm_on_pcb2[self.counter]):
                    return 'aborted'
                plan = []
                plan.append(robot_control.convert_to_pose(rb_arm_transition_over_gb3_1))
                plan.append(robot_control.convert_to_pose(rb_arm_transition_over_gb3_2))
                plan.append(robot_control.convert_to_pose(rb_arm_transition_over_gb3_3))
                if not robot_control.move_to_target_carth_plan(plan,10):
                    return 'aborted' 
                if not robot_control.gripper_controller.send_gripper_command('open'):
                    return 'aborted'
                
                plan = []
                plan.append(robot_control.convert_to_pose(rb_arm_transition_over_gb3_2))
                plan.append(robot_control.convert_to_pose(rb_arm_transition_over_gb3_1))

                if not robot_control.move_to_target_carth_plan(plan,10):
                    return 'aborted'
                
                self.counter +=1
                return 'succeeded'
            elif newuser == "n":
                print("Exiting")
                return 'succeeded' 

class BatteryPickUpAndPositioning(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded','succeeded_end','aborted'])
        self.counter = 0
    def execute(self, userdata):
        rospy.loginfo(f"Führe state: {self.__class__.__name__} aus.")
        while True:
            newuser = input('enter y/n: ')
            if newuser == "y":
                if not robot_control.move_to_joint_goal((-2.8680, -1.9416, -1.1650, -1.6055, 1.5637, -1.3022),10):
                    return 'aborted' 
                if not robot_control.pick_up(rb_arm_on_battery[self.counter]):
                    return 'aborted' 
                plan = []
                plan.append(robot_control.convert_to_pose(rb_arm_transition_over_gb2_1))
                plan.append(robot_control.convert_to_pose(rb_arm_transition_over_gb2_2))
                #plan.append(robot_control.convert_to_pose(rb_arm_transition_over_gb2_3))
                if not robot_control.move_to_target_carth_plan(plan,10):
                    return 'aborted' 
                if not robot_control.gripper_controller.send_gripper_command('open'):
                    return 'aborted'
                # plan = []
                # plan.append(robot_control.convert_to_pose(rb_arm_transition_over_gb2_2))
                # plan.append(robot_control.convert_to_pose(rb_arm_transition_over_gb2_1))
                # if not robot_control.move_to_target_carth_plan(plan,10):
                #     return 'aborted' 
                # return 'succeeded'
            
            elif newuser == "n":
                print("Exiting")
                return 'succeeded' 

class Aborted(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded_end','succeeded'])
    def execute(self, userdata):
        rospy.loginfo(f"Führe state: {self.__class__.__name__} aus.")
        while True:
            newuser = input('neuer Versuch? y/n: ')
            if newuser == "y":
                return 'succeeded'
            elif newuser == "n":
                return 'succeeded_end'

class Test(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded_end','aborted'])
    def execute(self, userdata):
        rospy.loginfo(f"Führe state: {self.__class__.__name__} aus.")
        newuser = input('start? y/n: ')
        if newuser == "y":
            command_list = [
                {
                    "type": "joint",
                    "joints": (-3.1557, -1.0119, -2.1765, -1.5426, 1.5686, -3.1643)
                },
                {
                    "type": "gripper",
                    "action": 'close' 
                },{
                    "type": "gripper",
                    "action": 'open'  
                },
                {
                    "type": "cartesian",
                    "pose": robot_control.convert_to_pose(rb_arm_transition_over_m)
                },{
                    "type": "joint",
                    "joints": (-3.8423, -1.0118, -2.3565, -2.8601, -0.7018, -3.1867)
                }
                ]
            command_list[4:4] = robot_control.pick_up_plan(rb_arm_on_m[0])
            robot_control.planner(command_list)
            return 'succeeded_end'

            # newuser = input('neuer Versuch? y/n: ')
            # if newuser == "y":
            #     command_list = [
            #         {
            #             "type": "joint",
            #             "joints": (-3.1557, -1.0119, -2.1765, -1.5426, 1.5686, -3.1643)
            #         },{
            #             "type": "gripper",
            #             "action": 'close' 
            #         },{
            #             "type": "gripper",
            #             "action": 'open'  
            #         },{
            #             "type": "cartesian",
            #             "pose": robot_control.convert_to_pose(rb_arm_transition_over_m)
            #         },{
            #             "type": "joint",
            #             "joints": (-3.8423, -1.0118, -2.3565, -2.8601, -0.7018, -3.1867)
            #         }
            #     ]
            #     command_list[4:4] = robot_control.pick_up_plan(rb_arm_on_m[0])
            #     robot_control.planner(command_list)
            #     return 'succeeded_end'

if __name__ == "__main__":
    """
    主程序入口
    初始化ROS节点，设置机器人，创建并执行SMACH状态机
    """
    # 初始化ROS节点
    rospy.init_node('ur5_moveit_control', anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)

    # 初始化夹爪：激活并打开
    robot_control.gripper_controller.send_gripper_command('activate')
    robot_control.gripper_controller.send_gripper_command('open')

    sm = smach.StateMachine(outcomes=['finished', 'aborted', 'preempted'])
    
    with sm:
    # 入口：Start（可选；若你没有 Start，可把本段注释掉并在后面 set_initial_state 改为 'MPickUp'）
    add_gui_state(sm, 'Start', Start(), {
        'MPickUp': 'MPickUp',
        'MHoldHD': 'MHoldHD',
        'MPositioning': 'MPositioning',
        'PCB1PickUpAndPositioning': 'PCB1PickUpAndPositioning',
        'PCB2PickUpAndPositioning': 'PCB2PickUpAndPositioning',
        'BatteryPickUpAndPositioning': 'BatteryPickUpAndPositioning',
        'succeeded_end': 'finished',
        'test': 'finished',
        'aborted': 'aborted',
    })

    # 1) 电机拾取
    add_gui_state(sm, 'MPickUp', MPickUp(), {
        'succeeded': 'MHoldHD',
        'succeeded_with_HD': 'MHoldHD',
        'aborted': 'aborted',
    })

    # 2) 人机持握/交接
    add_gui_state(sm, 'MHoldHD', MHoldHD(), {
        'succeeded': 'MPositioning',
        'succeeded_end': 'MPositioning',
        'aborted': 'aborted',
    })

    # 3) 装配起点定位
    add_gui_state(sm, 'MPositioning', MPositioning(), {
        'succeeded': 'PCB1PickUpAndPositioning',
        'succeeded_to_PCB': 'PCB1PickUpAndPositioning',
        'aborted': 'aborted',
    })

    # 4) PCB1
    add_gui_state(sm, 'PCB1PickUpAndPositioning', PCB1PickUpAndPositioning(), {
        'succeeded': 'PCB2PickUpAndPositioning',
        'aborted': 'aborted',
    })

    # 5) PCB2
    add_gui_state(sm, 'PCB2PickUpAndPositioning', PCB2PickUpAndPositioning(), {
        'succeeded': 'BatteryPickUpAndPositioning',
        'aborted': 'aborted',
    })

    # 6) 电池
    add_gui_state(sm, 'BatteryPickUpAndPositioning', BatteryPickUpAndPositioning(), {
        'succeeded': 'finished',
        'succeeded_end': 'finished',
        'aborted': 'aborted',
    })

   """  # 创建SMACH状态机，定义最终结果
    sm = smach.StateMachine(outcomes=['finished'])
    
    with sm:
        # ===== SMACH状态定义和转换关系 =====
        
        # 起始状态 - 根据用户选择跳转到不同的装配状态
        smach.StateMachine.add('Start', Start(),
                               transitions={'MPickUp': 'MPickUp',                     # 手机拾取
                                          'MHoldHD': 'MHoldHD',                       # 手机支架保持
                                          'MPositioning': 'MPositioning',             # 手机定位
                                          'PCB1PickUpAndPositioning': 'PCB1PickUpAndPositioning',  # PCB1装配
                                          'PCB2PickUpAndPositioning': 'PCB2PickUpAndPositioning',  # PCB2装配
                                          'BatteryPickUpAndPositioning': 'BatteryPickUpAndPositioning',  # 电池装配
                                          'test': 'Test',                             # 测试模式
                                          'succeeded_end': 'finished'})               # 结束
        
        # 拾取状态
        smach.StateMachine.add('MPickUp', MPickUp(),
                               transitions={'succeeded': 'MHold',                     # 成功->保持状态
                                          'aborted': 'Aborted',                       # 失败->中止状态
                                          'succeeded_with_HD': 'MHoldHD'})            # 成功带支架->支架保持状态
        
        # 保持状态
        smach.StateMachine.add('MHold', MHold(),
                               transitions={'succeeded': 'MPositioning',             # 成功->定位状态
                                          'aborted': 'Aborted'})                     # 失败->中止状态
        
        # 手机支架保持状态
        smach.StateMachine.add('MHoldHD', MHoldHD(),
                               transitions={'succeeded': 'MPositioning',             # 成功->定位状态
                                          'succeeded_end': 'finished',
                                            'aborted':'Aborted'})
        smach.StateMachine.add('MPositioning', MPositioning(),
                               transitions={'succeeded':'MPickUp',
                                            'succeeded_to_PCB':'PCB1PickUpAndPositioning',
                                            'aborted':'Aborted'})
        smach.StateMachine.add('PCB1PickUpAndPositioning', PCB1PickUpAndPositioning(),
                               transitions={'succeeded':'PCB2PickUpAndPositioning',
                                            'aborted':'Aborted'})
        smach.StateMachine.add('PCB2PickUpAndPositioning', PCB2PickUpAndPositioning(),
                               transitions={'succeeded':'BatteryPickUpAndPositioning',
                                            'aborted':'Aborted'})
        smach.StateMachine.add('BatteryPickUpAndPositioning', BatteryPickUpAndPositioning(),
                               transitions={'succeeded_end':'finished',
                                            'aborted':'Aborted',
                                            'succeeded':'MPickUp'})
        smach.StateMachine.add('Test', Test(),
                               transitions={'succeeded_end':'finished',
                                            'aborted':'Aborted'})
        smach.StateMachine.add('Aborted', Aborted(),
                               transitions={'succeeded_end':'finished',
                                            'succeeded': 'MPickUp'}) """

    # ===== 初始化状态机监控服务器 =====
    try:
        # 创建SMACH内省服务器，用于可视化状态机执行过程
        sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
        sis.start()  # 启动服务器
        rospy.loginfo("SMACH内省服务器启动成功")
    except AttributeError as e:
        rospy.logwarn(f"内省服务器启动失败，回退到手动调试模式。错误: {e}")

    # ===== 执行状态机 =====
    rospy.loginfo("开始执行装配流程状态机")
    outcome = sm.execute()  # 执行状态机并获取最终结果
    rospy.loginfo(f"状态机执行完成，最终结果: {outcome}")
    
    # 保持节点运行，等待回调
    rospy.spin() 
