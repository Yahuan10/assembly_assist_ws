#!/usr/bin/env python

import numpy as np # type: ignore
import rospy # type: ignore
import moveit_commander # type: ignore
import sys
import smach # type: ignore
import smach_ros # type: ignore
import tf   # type: ignore
import math
import copy
import time
import csv
import threading
import random
#from schledluer.msg import robot_msgs # type: ignore
from tf.transformations import quaternion_from_euler  # type: ignore
from geometry_msgs.msg import Pose, PoseStamped , PointStamped # type: ignore
from moveit_msgs.msg import Grasp, PlaceLocation # type: ignore
from moveit_commander.move_group import MoveGroupCommander # type: ignore
from moveit_commander import PlanningSceneInterface # type: ignore
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg # type: ignore
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_input as inputMsg # type: ignore
from moveit_msgs.msg import RobotTrajectory # type: ignore
from trajectory_msgs.msg import JointTrajectoryPoint # type: ignore
from std_msgs.msg import String # type: ignore
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_input  as inputMsg # type: ignore
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output, Robotiq2FGripper_robot_input # type: ignore

#test
#======Konstanten====== 
#Konstanten für TCP-Ausrichtung
tcp_to_hum = [0.47471946904520335, 0.508672230393878, -0.5204373469522815, 0.4950140963981013]

#Konstanten für Roboterposen
rb_arm_home = np.array([-0.28531283917512756,  0.08176575019716574, 0.3565888897535509, 0.021838185570339213, -0.9997536365149914, 0.0006507883874787611, 0.003916171666392069])

rb_arm_on_m =  [np.array([0.2641105225136129,    0.11513901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),
                np.array([0.2641105225136129,    0.06813901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),
                np.array([0.2641105225136129,    0.02113901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),
                np.array([0.2641105225136129,    -0.02613901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),
                np.array([0.3441105225136129,    0.11513901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),
                np.array([0.3441105225136129,    0.06813901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),
                np.array([0.3441105225136129,    0.02113901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),
                np.array([0.3441105225136129,    -0.02613901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),
                np.array([0.4231105225136129,    0.11513901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),
                np.array([0.4231105225136129,    0.06813901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),
                np.array([0.4231105225136129,    0.02113901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),
                np.array([0.4231105225136129,    -0.02613901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),
                np.array([0.5051105225136129,    0.11513901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),
                np.array([0.5051105225136129,    0.06813901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),
                np.array([0.5051105225136129,    0.02113901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008]),
                np.array([0.5051105225136129,    -0.02613901314207496, 0.19474944789272417 ,0.018266303149021744, 0.9997308933491994, -0.010420321910118447, 0.009792851666864008])]

rb_arm_on_hum_static = np.array([0.01127138298740326, -0.40789791168606154, 0.4347020900402719,0.65967278113823, 0.13322073168864898, -0.04031615244060301, 0.7385517357139446])

rb_arm_transition =             np.array([0.22048980978459626, -0.11962800779329041, 0.22232535871506093 ,-0.00519597519482744, -0.7000337195214675, 0.7140685181262056, 0.005651972731604554])
rb_arm_transition_over_m =      np.array([0.32755193192480295, 0,                    0.3552028979677898 ,-0.002982105237080432, -0.9999915258909946, 0.00274986658972347, 0.0007024445132438654])
rb_arm_transition_over_pcb1 =   np.array([0.7371109279194257, -0.12405534656551466, 0.3564804416147143 ,0.701903624069778, 0.7119699919919052, 0.017634125210474614, 0.010911949817505392])
rb_arm_transition_over_pcb2 =   np.array([0.39299783753064255, -0.25037007326362604, 0.4098002793824048  ,0.9994996999243878, 0.018132610811126923, -0.01488422170868633, 0.021213632889197198])


rb_arm_transition_over_gb0_1 =  np.array([0.43920883565114404, -0.27118297223348553, 0.21533919567733978,-0.0017450344372635439, -0.6960370290652399, 0.7180016795804389, 0.0017312263041814983])
rb_arm_transition_over_gb0_2 =  np.array([0.4017174799606998, -0.2181725740604259, 0.2921709170604791, -0.0017450344372635439, -0.6960370290652399, 0.7180016795804389, 0.0017312263041814983])

rb_arm_place_m_on_gb =          np.array([0.39854600328887463, -0.27554380366129894, 0.2096079683752514, 0.013454136070054674, -0.6962561587863013, 0.7174988234549298, 0.01554946672864364])

rb_arm_transition_over_gb1_1 =  np.array([0.4907380230958256, -0.33196635637380156, 0.3911058561909323,0.003330260900274425, 0.9999775446059118, -0.0015901397197829994, 0.0055938450049654006])
rb_arm_transition_over_gb1_2 =  np.array([0.4907380230958256, -0.45983847995419647, 0.3910704893338057  ,0.003330260900274425, 0.9999775446059118, -0.0015901397197829994, 0.0055938450049654006])
rb_arm_transition_over_gb1_3 =  np.array([0.4907380230958256, -0.45983847995419647, 0.35395950043452323 ,0.003330260900274425, 0.9999775446059118, -0.0015901397197829994, 0.0055938450049654006])

rb_arm_transition_over_gb2_1 =  np.array([0.5486854170473805, -0.3885145028949433, 0.3814376455406984 ,-0.0020243784347149197, 0.9996642271967776, 0.02193613735553668, 0.013643336576571479])
rb_arm_transition_over_gb2_2 =  np.array([0.5541683902247382, -0.4608776309406269, 0.3516115882341859,-0.0019221405935429784, 0.9999968121680831, 0.0016100890850873695, 0.0002977286491069227])

rb_arm_transition_over_gb2_3 =  np.array([0.5486854170473805, -0.27057393609447655, 0.35141967557281056 ,-0.0020243784347149197, 0.9996642271967776, 0.02193613735553668, 0.013643336576571479])

rb_arm_transition_over_gb3_1 =  np.array([0.6402808547359244, -0.26790083190492066, 0.38354439061807685 ,0.019159378644141387, 0.999622466840548, -0.005839393784188026, 0.018808069486830555])
rb_arm_transition_over_gb3_2 =  np.array([0.6402808547359244, -0.46790083190492066, 0.38354439061807685 ,0.019159378644141387, 0.999622466840548, -0.005839393784188026, 0.018808069486830555])
rb_arm_transition_over_gb3_3 =  np.array([0.6395384342520363, -0.469121998164546, 0.34939538969817263, 0.010291666360366757, 0.9997544304898899, -0.005685602308402095, 0.01878388260594522])

rb_arm_transition_over_gb4_3 =  np.array([])

rb_arm_on_pcb1  =  [np.array([0.6316488317010515, -0.13953502575569454, 0.16890158973568933 ,-0.7289965096816865, -0.6845173343498824, 0.00032839232615476167, 0]),
                    np.array([0.6866488317010515, -0.13953502575569454, 0.16890158973568933  ,-0.7289965096816865, -0.6845173343498824, 0.00032839232615476167, 0]),
                    np.array([0.7416488317010515, -0.13953502575569454, 0.16890158973568933  ,-0.7289965096816865, -0.6845173343498824, 0.00032839232615476167, 0]),
                    np.array([0.7966488317010515, -0.13953502575569454, 0.16890158973568933 ,-0.7289965096816865, -0.6845173343498824, 0.00032839232615476167, 0])]

rb_arm_on_pcb2  =  [np.array([0.40558901752394194, -0.26155397106696143,  0.15524703844596804  ,0.7090144359375834, 0.7047068297621026, 0.006839549337615578, 0.02529889886172804]),
                    np.array([]),
                    np.array([]),
                    np.array([])]

rb_arm_on_pcb3  =  [np.array([]),
                    np.array([]),
                    np.array([]),
                    np.array([])]

rb_arm_on_battery =[np.array([0.6057899329831553, -0.019193581297668794, 0.15691817631772764   ,0.001276412480525014, -0.9999579745548456, 0.0048883027126173095, 0.007650123655222502]),
                    np.array([0.6057899329831553,  0.103193581297668794, 0.15691817631772764   ,0.001276412480525014, -0.9999579745548456, 0.0048883027126173095, 0.007650123655222502]),
                    np.array([0.7017839913625549, -0.019193581297668794, 0.15691817631772764   ,0.001276412480525014, -0.9999579745548456, 0.0048883027126173095, 0.007650123655222502]),
                    np.array([0.7017839913625549,  0.101393581297668794, 0.15691817631772764   ,0.001276412480525014, -0.9999579745548456, 0.0048883027126173095, 0.007650123655222502])]

rb_arm_to_hum_sta = np.array([-0.021777125261783766, -0.37363507849091354, 0.19917983570199801, 0.47471946904520335, 0.508672230393878, -0.5204373469522815, 0.4950140963981013])


#Konstanten für ergonomie Berechnungen
forearmlenghdin = 0.3325  # Aus DIN 33402-2 gemittelt aus Mann und Frau über alle Altersklassen
upperarmlenghtdin = 0.3425  # Aus DIN 33402-2 gemittelt aus Mann und Frau über alle Altersklassen

forearmlenghdin_max = 0.395 #
forearmlenghdin_min = 0.285 #

upperarmlenghtdin_max = 0.405 #
upperarmlenghtdin_min = 0.285 #

tcp_coversion = 0.15

savety_koord_1 = np.array([ 0.10,  0.3, 0.6])
savety_koord_2 = np.array([-0.24, -0.7, 0.08])



use_built_in_rb_control = True

def wait_for_moveit():
    rospy.loginfo("Warte auf MoveIt-Services...")
    rospy.wait_for_service('/get_planning_scene')
    rospy.loginfo("MoveIt bereit.")

#======Robot Control Class======
class RobotControl:
    
    def __init__(self, group_name):

        self.gripper_controller = GripperController()

    
        if use_built_in_rb_control:       

            #Initialisiert die MoveIt-Gruppe und die Greifer-Node
            self.group_name = group_name
            self.move_group = MoveGroupCommander(self.group_name)
            
            self.scene = PlanningSceneInterface()
            self.robot = moveit_commander.RobotCommander()
            self.planned_traj_pub = rospy.Publisher("/planned_trajectory", RobotTrajectory, queue_size=10)

            rospy.sleep(2)  

            

            # Tischfläche, Wände ung Grundplatten in MoveIt zur Kollisionserkennung hinzufügen
            planning_frame = self.move_group.get_planning_frame()
            rospy.loginfo("Planungsrahmen: %s", planning_frame)

            Tisch = PoseStamped()
            Tisch.header.frame_id = planning_frame
            Tisch.pose.position.x = 0.0
            Tisch.pose.position.y = 0.0
            Tisch.pose.position.z = -0.09 
            
            self.scene.add_box("Tisch", Tisch, size=(3, 2, 0.05))
            rospy.loginfo("Tisch wurde Planungszene hinzugefügt.")

            Wand_links = PoseStamped()
            Wand_links.header.frame_id = planning_frame 
            Wand_links.pose.position.x = -0.37
            Wand_links.pose.position.y = 0.00
            Wand_links.pose.position.z = 0.00 
            
            self.scene.add_box("Wand_links", Wand_links, size=(0.05, 3, 3))
            rospy.loginfo("Wand_links wurde Planungszene hinzugefügt")

            Wand_hinten = PoseStamped()
            Wand_hinten.header.frame_id = planning_frame 
            Wand_hinten.pose.position.x = 0.00
            Wand_hinten.pose.position.y = 0.34
            Wand_hinten.pose.position.z = 0.00 
            
            self.scene.add_box("Wand_hinten", Wand_hinten, size=(3, 0.05, 3))
            rospy.loginfo("Wand_hinten wurde Planungszene hinzugefügt")

            Decke = PoseStamped()
            Decke.header.frame_id = planning_frame  
            Decke.pose.position.x = 0.0
            Decke.pose.position.y = 0.0
            Decke.pose.position.z = 0.92 
            
            self.scene.add_box("Decke", Decke, size=(3, 2, 0.05))
            rospy.loginfo("Decke wurde Planungszene hinzugefügt.")

            Halter_Grundplatte = PoseStamped()
            Halter_Grundplatte.header.frame_id = planning_frame  
            Halter_Grundplatte.pose.position.x = 2*0.28
            Halter_Grundplatte.pose.position.y = -0.59
            Halter_Grundplatte.pose.position.z = -0.04
            
            self.scene.add_box("Halter_Grundplatte", Halter_Grundplatte, size=(0.60, 0.22, 0.22))
            rospy.loginfo("Halter_Grundplatte wurde Planungszene hinzugefügt.")

            eef_link = self.move_group.get_end_effector_link()
            rospy.loginfo("Endeffektor-Link: %s", eef_link)

            self.move_group.set_max_velocity_scaling_factor(0.1)
            self.move_group.set_max_acceleration_scaling_factor(0.1)

        if not use_built_in_rb_control:  

            self.command_pub = rospy.Publisher('/robot/command', robot_msgs, queue_size=10)
            self.status_sub = rospy.Subscriber('/robot/status', String, self.status_callback)
            self.status_event = threading.Event()
            self.completed_cmds = 0
            self.expected_cmds = 0
            self.cmd_nr = 0
            self.cmd_buffer = []



        while True:
            user_input = input('Gebe initialen ein: ')
            user = user_input
            if (user == ""):
                user = "test"
                print(f"user: {user}")

            break
        
    def convert_to_pose(self, koords):
        #Konvertiert ein 1x7-Array in eine Pose
        target_pose = Pose()
        target_pose.position.x = koords[0]
        target_pose.position.y = koords[1]
        target_pose.position.z = koords[2]
        target_pose.orientation.x = koords[3]
        target_pose.orientation.y = koords[4]
        target_pose.orientation.z = koords[5]
        target_pose.orientation.w = koords[6]
        return target_pose
    
    def convert_to_koords(self, pose = Pose()):
        #Konvertiert eine Pose in ein 1x7-Array
        koords = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        koords[0] = pose.position.x
        koords[1] = pose.position.y
        koords[2] = pose.position.z
        koords[3] = pose.orientation.x
        koords[4] = pose.orientation.y
        koords[5] = pose.orientation.z
        koords[6] = pose.orientation.w
        return koords

    def move_to_target(self, target_pose, speed):
        if not use_built_in_rb_control:
            command = [{'type':'p2p','pose':target_pose}] 
            if not self.publish_rb_cmds(command):
                return False
            return True
        
        #Bewegt den Roboter zu einer Pose
        self.move_group.set_max_velocity_scaling_factor(speed / 100.0)
        self.move_group.set_pose_target(target_pose)
        rospy.loginfo("Bewege Roboter zu: x={}, y={}, z={}".format(target_pose.position.x, target_pose.position.y, target_pose.position.z))
        # rospy.loginfo("0.2662104568594572, -0.35661957908057046, 0.24265798894634866 | Orientation: 0.0050765060764118896, -0.8027125907596652, 0.5948306511336113, 0.042464363811632704")
        success = self.move_group.go(wait=True)
        if success:
            rospy.loginfo("Bewegung erfolgreich!")
            return True
        else:
            rospy.logwarn("Bewegung fehlgeschlagen!")
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            return False

    def move_to_target_carth(self, target_pose, speed):
        #Bewegt den Roboter in einer Linie zur Zielpose
        if not use_built_in_rb_control:
            command = [{'type':'cartesian','pose':target_pose}]
            if not self.publish_rb_cmds(command):
                return False
            return True


        self.move_group.set_max_velocity_scaling_factor(speed / 100.0)
        waypoints = []
        waypoints.append(target_pose)
        self.move_group.set_planning_time(10.0) 
        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.05) 
        if fraction < 1.0:
            return False
        rospy.loginfo("Bewege Roboter in einer Linie zu: x={}, y={}, z={}".format(target_pose.position.x, target_pose.position.y, target_pose.position.z))

        success = self.move_group.execute(plan, wait=True)
        if success:
            rospy.loginfo("Bewegung erfolgreich!")
            return True
        else:
            rospy.logwarn("Bewegung fehlgeschlagen!")
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            return False
        
    def move_to_target_carth_plan(self, waypoints, speed):
        #Bewegt den Roboter in einer Linie zur Zielpose mit mehreren Waypoints
        if not use_built_in_rb_control:
            for pose in waypoints:
                command = [{'type':'cartesian','pose':pose}]
                if not self.publish_rb_cmds(command):
                    return False
                return True

        self.move_group.set_max_velocity_scaling_factor(speed / 100.0)

        self.move_group.set_planning_time(10.0) 
        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.05) 
        if fraction < 1.0:
            return False
        success = self.move_group.execute(plan, wait=True)
        if success:
            rospy.loginfo("Bewegung erfolgreich!")
            return True
        else:
            rospy.logwarn("Bewegung fehlgeschlagen!")
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            return False 

    def move_to_taget_plan(self, waypoints, speed):
        #Fahre mit dem Roboterarm eine Reihe von Waipoints an

        if not use_built_in_rb_control:
            for pose in waypoints:
                command = [{'type':'p2p','pose':pose}]
                if not self.publish_rb_cmds(command):
                    return False
                return True
            
        self.move_group.set_max_velocity_scaling_factor(speed / 100.0)
        for i, waypoint in enumerate(waypoints):
            self.move_group.set_pose_target(waypoint)
            plan = self.move_group.plan()  
            if plan[0]:  
                rospy.loginfo(f"Führe Waypoint {i+1} aus...")
                self.move_group.execute(plan[1], wait=True)
            else:
                rospy.logwarn(f"Konnte Waypoint {i+1} nicht erreichen!")
                return False
        return True

    def move_to_joint_goal(self, joint_goal, speed):
        #Bewegt den Roboter zu einem Gelenkwinkel

        if not use_built_in_rb_control:
            command = [{'type':'joint','joints':joint_goal}]
            rospy.loginfo(type(command))
            rospy.loginfo(command)
            if not self.publish_rb_cmds(command):
                return False
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
        ###wird nicht benutzt
        #Stoppt die Bewegung des Roboters
        self.move_group.stop()
        rospy.loginfo("Roboter gestoppt!")

    def reset_robot(self):
        ###wird nicht benutzt
        #Setzt den Roboter auf die Home-Position zurück
        self.move_group.set_named_target("home")
        self.move_group.go(wait=True)
        rospy.loginfo("Roboter auf 'Home' Position zurückgesetzt!")

    def handover_to_hum(self, speed):
        rospy.loginfo("Starte Übergabe an den Menschen...")

        speed += speed * (random.uniform(-50, 50) / 100)
        rospy.loginfo(f'geschwindigkeit: {speed}') 
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            rospy.loginfo(f"Versuch {attempt} von {max_retries}...")

            # Lokaler TF-Listener
            local_listener = tf.TransformListener()

            try:
                handover_pose_end = self.point_inside(self.calc_handover_position_schoulder(local_listener))
            except Exception as e:
                rospy.logwarn(f"Fehler bei der Berechnung der Übergabeposition: {e}")
                handover_pose_end = None

            if handover_pose_end is None:
                rospy.logwarn("Übergabeposition konnte nicht bestimmt werden.")
                time.sleep(2)
                continue

            handover_pose_start = copy.deepcopy(handover_pose_end)
            handover_pose_start.position.y += 0.1

            rospy.loginfo("Bewege Roboter zur Startposition der Übergabe...")
            if not self.move_to_target_carth(handover_pose_start, speed):
                rospy.logwarn("Bewegung zur Startposition fehlgeschlagen.")
                time.sleep(2)
                continue

            rospy.loginfo("Bewege Roboter zur Übergabeposition...")
            if not self.move_to_target_carth(handover_pose_end, speed):
                rospy.logwarn("Bewegung zur Übergabeposition fehlgeschlagen.")
                time.sleep(2)
                continue

            rospy.loginfo("Übergabe erfolgreich abgeschlossen.")
            return True

        rospy.logerr("Übergabe nach mehreren Versuchen fehlgeschlagen.")
        return False
        
        rospy.loginfo("Bewege Roboter zu: x={}, y={}, z={}".format(handover_pose_start.position.x, handover_pose_start.position.y, handover_pose_start.position.z))

        if not self.move_to_target_carth(handover_pose_start,speed):
            return False
        if not self.move_to_target_carth(handover_pose_end,speed):
            return False
        return True

    def calc_handover_position_schoulder(self,listener_lc):
        #Berechnet die ergonomischste Übergabeposition basierend auf Schulterkoordinaten aus Tracking
        
        for i in range(1):
            for sek in range(10):
                broadcaster = tf.TransformBroadcaster()
                #listener = tf.TransformListener()  
                hm = get_Hum_mertics()
                hm.camera_listener()
                hm.calc_arm_lenght()
                if not(all(x == 0 for x in hm.shoulderkoords)) and not(all(x == 0 for x in hm.elbowkoords)) and not(all(x == 0 for x in hm.handkoords)):
                #if not(all(x == 0 for x in hm.shoulderkoords)) and not(all(x == 0 for x in hm.elbowkoords)) and not(all(x == 0 for x in hm.handkoords)):    
                    rospy.loginfo("Schulter, Ellbogen und Hand erkannt")
                    rospy.loginfo("Unterarmlänge: %s", hm.forearmlenght)
                    rospy.loginfo("Oberarmlänge: %s", hm.uperarmlenght)
                    rospy.loginfo("Schulterkoords: %s", hm.shoulderkoords)

                    translation = [ 0, (hm.forearmlenght + tcp_coversion),- hm.uperarmlenght]
                    break

                elif not(all(x == 0 for x in hm.shoulderkoords)) and not(all(x == 0 for x in hm.elbowkoords)):
                    
                    rospy.loginfo("Schulter, Ellbogen erkannt")
                    rospy.loginfo("Unterarmlänge: %s", hm.forearmlenght)
                    rospy.loginfo("Oberarmlänge: %s", hm.uperarmlenght)
                    rospy.loginfo("Schulterkoords: %s", hm.shoulderkoords)

                    translation = [ 0,(forearmlenghdin + tcp_coversion),- hm.uperarmlenght]
                    break

                elif not(all(x == 0 for x in hm.shoulderkoords)):
                    rospy.loginfo("Schulter erkannt")
                    rospy.loginfo("Unterarmlänge: %s", hm.forearmlenght)
                    rospy.loginfo("Oberarmlänge: %s", hm.uperarmlenght)
                    rospy.loginfo("Schulterkoords: %s", hm.shoulderkoords)

                    translation = [ 0,(forearmlenghdin + tcp_coversion),- upperarmlenghtdin]
                    break

                else:
                    rospy.loginfo(f"Nichts erkannt Versuch: {sek}/10")
                    time.sleep(1)

            try:    
                    
                    handover_point = PointStamped()
                    handover_point.header.frame_id = "shoulder"
                    handover_point.header.stamp = rospy.Time.now() 
                    handover_point.point.x = translation[0]
                    handover_point.point.y = -translation[1]
                    handover_point.point.z = translation[2]

                    time.sleep(3)

                    listener_lc.waitForTransform("base","shoulder", rospy.Time(0), rospy.Duration(4.0))
                        
                    broadcaster.sendTransform(
                        (handover_point.point.x, handover_point.point.y,handover_point.point.z),
                        (0.0, 0.0, 0.0, 1.0),  
                        rospy.Time.now(),
                        "handover_position",
                        "shoulder"
                    )
                    time.sleep(3)
                    #atest_time = listener.getLatestCommonTime("base", "handover_position")
                    listener_lc.waitForTransform("base","handover_position", rospy.Time(0) , rospy.Duration(4.0))
                    hand_over_position_koords, _ = listener_lc.lookupTransform("base","handover_position",  rospy.Time(0))

                    
                    
                        #hand_over_position = self.convert_to_pose(np.array([0,-0.4,0.4,tcp_to_hum[0],tcp_to_hum[1],tcp_to_hum[2],tcp_to_hum[3]]))
                    hand_over_position = self.convert_to_pose(np.array([-hand_over_position_koords[0],-hand_over_position_koords[1],hand_over_position_koords[2],tcp_to_hum[0],tcp_to_hum[1],tcp_to_hum[2],tcp_to_hum[3]]))
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                    rospy.logwarn(f"Error transforming point: {e}")

            return hand_over_position
        
    def reset_robot(self):
        #### nicht benutzt
        self.move_group.set_named_target("home")
        self.move_group.go(wait=True)
        rospy.loginfo("Roboter auf 'Home' Position zurückgesetzt!")

    def point_inside(self, pose):
        #Überprüft, ob die Übergabeposition innerhalb eines Sicherheitsrechtecks liegt

        point = [pose.position.x, pose.position.y, pose.position.z]
        xmin, xmax = sorted([savety_koord_1[0], savety_koord_2[0]])
        ymin, ymax = sorted([savety_koord_1[1], savety_koord_2[1]])
        zmin, zmax = sorted([savety_koord_1[2], savety_koord_2[2]])
        
        if point[0] < xmin or point[0] > xmax or point[1] < ymin or point[1]  > ymax or point[2] < zmin or point[2] >zmax:
            rospy.logwarn(f"Punkt liegt außerhalb")

        pose.position.x = max(xmin, min(point[0], xmax))
        pose.position.y = max(ymin, min(point[1], ymax))
        pose.position.z = max(zmin, min(point[2], zmax))

        return pose

    def pick_up(self,target,speed=10):
        #Nehme Bautteile auf die bei Target liegen

        speed += speed * (random.uniform(-20, 20) / 100)
        rospy.loginfo(f'geschwindigkeit: {speed}') 
        over_target = target.copy()
        over_target[2] = over_target[2] + 0.1  
        
        if not use_built_in_rb_control:
            over_target_pose = self.convert_to_pose(over_target)
            target_pose = self.convert_to_pose(target)

            command1 =   [{'type':'p2p','pose':over_target_pose},
                         {'type':'cartesian','pose':target_pose},
                         {'type':'gripper','action':'close'},
                         {'type':'cartesian','pose':over_target_pose}]
            
            if not self.publish_rb_cmds(command1):
                return False

            return True

        if not self.move_to_target(self.convert_to_pose(over_target),speed):
            return False
        
        if not self.gripper_controller.send_gripper_command('open'):
           return False

        if not self.move_to_target_carth(self.convert_to_pose(target), speed * 0.5):
            return False
        
        if not self.gripper_controller.send_gripper_command('close'):
            return False

        if not self.move_to_target_carth(self.convert_to_pose(over_target), speed):
            return False
        
        return True
    
    def pick_up_plan(self,target):
        ###wird nicht benutzt

        over_target = target.copy()
        over_target[2] = over_target[2] + 0.1  
        return [
        {"type": "cartesian", "pose": self.convert_to_pose(over_target)},
        {"type": "gripper", "action": 'open'},
        {"type": "cartesian", "pose": self.convert_to_pose(target)},
        {"type": "gripper", "action": 'close'},
        {"type": "cartesian", "pose": self.convert_to_pose(over_target)}]
        
    def planner(self, command_list, speed):
        self.move_group.set_max_velocity_scaling_factor(speed / 100.0)
        waypoints = []
        ############# wird nicht verwendet ##############
        def extract_plan(plan_result):

            if isinstance(plan_result, tuple):
                for item in plan_result:
                    if hasattr(item, "joint_trajectory"):
                        return item
                return None  
            return plan_result

        def publish_plan(plan):
            try:
                if self.planned_traj_pub is not None and plan and hasattr(plan, "joint_trajectory"):
                    if plan.joint_trajectory.points:
                        self.planned_traj_pub.publish(plan)
            except Exception as exc:
                rospy.logwarn(f"Publizieren des geplanten Trajekts fehlgeschlagen: {exc}")

        for command in command_list:
            try:
                if command['type'] == 'cartesian':
                    waypoints.append(command['pose'])

                else:

                    if waypoints:
                        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
                        if fraction < 1.0:
                            rospy.logwarn("Kartesische Bewegung konnte nicht vollständig geplant werden.")
                            return False
                        publish_plan(plan)
                        if not self.move_group.execute(plan, wait=True):
                            return False
                        waypoints = []

                    if command['type'] == "joint":
                        self.move_group.set_joint_value_target(command["joints"])
                        plan_result = self.move_group.plan()
                        plan = extract_plan(plan_result)
                        if not plan or not plan.joint_trajectory.points:
                            rospy.logwarn("Gelenkbewegung konnte nicht geplant werden.")
                            return False
                        publish_plan(plan)
                        self.move_group.execute(plan, wait=True)

                    elif command['type'] == "p2p":
                        self.move_group.set_pose_target(command["pose"])
                        plan_result = self.move_group.plan()
                        plan = extract_plan(plan_result)
                        if not plan or not plan.joint_trajectory.points:
                            rospy.logwarn("P2P Bewegung konnte nicht geplant werden.")
                            return False
                        publish_plan(plan)
                        self.move_group.execute(plan, wait=True)

                    elif command['type'] == "gripper":
                        if not self.gripper_controller.send_gripper_command(command["action"]):
                            return False

            except Exception as e:
                rospy.logerr(f"Fehler beim Verarbeiten des Befehls: {e}")
                return False


        if waypoints:
            (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
            if fraction < 1.0:
                rospy.logwarn("Kartesische Bewegung konnte nicht vollständig geplant werden.")
                return False
            publish_plan(plan)
            if not self.move_group.execute(plan, wait=True):
                return False
        return True

    def status_callback(self, msg):
        #callback von bewegungssteuerung
        if msg.data == "done":
            self.completed_cmds += 1
            rospy.loginfo(f"Bewegung bestätigt: {self.completed_cmds}/{self.expected_cmds}")
            if self.completed_cmds >= self.expected_cmds:
                self.status_event.set()

    def wait_for_all_done(self, timeout=15.0):
        #warte auf Bestätigung der übermittelten bewegungen
        success = self.status_event.wait(timeout=self.cmd_nr*timeout)
        self.status_event.clear()
        return success

    def publish_rb_cmds(self, commands):
        #Publichen der Bewegungsbefehle als Batch

        for cmd in commands:
            if cmd["type"] == "stop":
                for cmd2 in self.cmd_buffer:
                    self.command_pub.publish(cmd2)
                    self.completed_cmds = 0
                    self.expected_cmds = 0
                    self.cmd_nr = 0
                    self.cmd_buffer = []
                continue
            elif cmd["type"] == "gripper":

                rospy.loginfo(f"Gesendet: {self.cmd_buffer}")
                for cmd2 in self.cmd_buffer:
                    self.command_pub.publish(cmd2)
                    #rospy.loginfo(f"Gesendet: {cmd2['type']} #{self.cmd_nr}")
                rospy.loginfo("Warte auf Abschluss aller bisherigen Bewegungen...")
                if not self.wait_for_all_done():
                    rospy.logwarn("Timeout beim Warten auf Bewegungsabschluss vor Greiferkommando.")
                    return False
                self.completed_cmds = 0
                self.expected_cmds = 0
                self.cmd_nr = 0
                self.cmd_buffer = []
                rospy.loginfo("Führe Greiferbefehl aus.")
                self.gripper_controller.send_gripper_command(cmd["action"])
                rospy.sleep(1)  
                continue


            msg = robot_msgs()
            msg.nr = self.cmd_nr
            msg.type = cmd["type"]

            if cmd["type"] == "joint":
                msg.joints = list(cmd["joints"])
                msg.pose = Pose()
            elif cmd["type"] in ["cartesian", "p2p"]:
                msg.pose = cmd["pose"]
                msg.joints = (0,0,0,0,0,0)
            
            self.cmd_buffer.append(msg)


            
            self.cmd_nr += 1
            self.expected_cmds += 1
            rospy.sleep(0.05)  
            
        return True


    def place_on_board(self,target,speed,distance=0.2,gripper_val = 'open'):
        #Bautteile auf grundplatte ablegen

        speed += speed * (random.uniform(-40, 40) / 100)
        rospy.loginfo(f'geschwindigkeit: {speed}')    
        next_board = target.copy()
        over_board = target.copy()
        next_board[1] = next_board[1] + distance

        next_board[2] = next_board[2] + 0.04
        over_board[2] = over_board[2] + 0.04
        
        if not use_built_in_rb_control:
            command =   [{'type':'cartesian','pose':self.convert_to_pose(next_board)},
                         {'type':'cartesian','pose':self.convert_to_pose(over_board)},
                         {'type':'cartesian','pose':self.convert_to_pose(target)},
                         {'type':'gripper','action':gripper_val},
                         {'type':'cartesian','pose':self.convert_to_pose(over_board)},
                         {'type':'cartesian','pose':self.convert_to_pose(next_board)}]

            if not self.publish_rb_cmds(command):
                return False
            return True



        plan1 = []
        plan1.append(self.convert_to_pose(next_board))
        plan1.append(self.convert_to_pose(over_board))
        plan1.append(self.convert_to_pose(target))
        rospy.loginfo(plan1)
        plan2 = []
        plan2.append(self.convert_to_pose(over_board))
        plan2.append(self.convert_to_pose(next_board))

        rospy.loginfo(plan2)
        if not self.move_to_target_carth_plan(plan1,speed):
            return False
        if not self.gripper_controller.send_gripper_command(gripper_val):
            return False
        if not self.move_to_target_carth_plan(plan2,speed):
            return False
        
        if not self.gripper_controller.send_gripper_command('open'):
            return False
        
        return True

#======Gripper Control======
class GripperController:
    def __init__(self):
        self.pub = rospy.Publisher('Robotiq2FGripperRobotOutput', Robotiq2FGripper_robot_output, queue_size=10)
        self.status = None
        rospy.Subscriber("Robotiq2FGripperRobotInput", Robotiq2FGripper_robot_input, self.status_callback)
        self.command = Robotiq2FGripper_robot_output()
    def status_callback(self, msg):
        self.status = msg

    def wait_for_gripper_stop(self, timeout=5.0):
        #warte auf Gripper ausführung
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)  

        while not rospy.is_shutdown():
            if self.status is None:
                rospy.logwarn("Kein Gripperstatus")
            else:
                if self.status.gFLT == 0x09:
                    rospy.logerr("Kommunikationsfehler: Chip nicht bereit.")
                    return False
                elif self.status.gSTA == 3:
                    rospy.sleep(0.5) 
                    return True

            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logwarn("Timeout beim Warten auf Gripper-Stillstand.")
                return False

            rate.sleep()


        return False

    def send_gripper_command(self, action_type):
        #senden der Gripper cmds an die Gripper Node -> Robotiq
        if action_type == 'open':
            self.command.rPR = 0
        elif action_type == 'close':
            self.command.rPR = 255
        elif action_type == 'activate':
            
            self.command.rACT = 1
            self.command.rGTO = 1
            self.command.rSP = 255
            self.command.rFR = 100
        elif action_type == 'reset':
            self.command.rACT = 0
        elif isinstance(action_type, int):
            self.command.rPR = max(0, min(255, int(action_type)))

        self.pub.publish(self.command)
        rospy.loginfo(f"Gripper-Befehl '{action_type}' gesendet – warte auf Abschluss...")
        return self.wait_for_gripper_stop()
        #return True

#======Get Hum Data======
class get_Hum_mertics:
    #innitiere tracking des Menschen
    def __init__(self):
        self.uperarmlenght = 0.0
        self.forearmlenght = 0.0
        self.shoulderkoords = [0.0, 0.0, 0.0]
        self.elbowkoords =    [0.0, 0.0, 0.0]
        self.handkoords =     [0.0, 0.0, 0.0]
        self.rightshoulderkoords = [0.0, 0.0, 0.0]
        self.rightelbowkoords =    [0.0, 0.0, 0.0]
        self.righthandkoords =     [0.0, 0.0, 0.0]
        self.leftshoulderkoords = [0.0, 0.0, 0.0]
        self.leftelbowkoords =    [0.0, 0.0, 0.0]
        self.lefthandkoords =     [0.0, 0.0, 0.0]
        self.inside_norm_upper = True
        self.inside_norm_fore  = True
        #self.calc_arm_lenght()
        self.stop_event = threading.Event()

    def camera_listener(self):
    #lese tf für Mittelpunkte Schulter Elebogen und Hand aus 
        try:

            time = rospy.Time(0)
            #listener = tf.TransformListener()

            listener.waitForTransform("base", "shoulder", time, rospy.Duration(4.0))
            listener.waitForTransform("base", "elbow",    time, rospy.Duration(4.0))
            listener.waitForTransform("base", "hand",     time, rospy.Duration(4.0))

            shoulder_trans, _ = listener.lookupTransform("base", "shoulder",  time)
            elbow_trans,    _ = listener.lookupTransform("base", "elbow",     time)
            hand_trans,     _ = listener.lookupTransform("base", "hand",      time)

            self.shoulderkoords =   [shoulder_trans[0], shoulder_trans[1], shoulder_trans[2]]
            self.elbowkoords =      [elbow_trans[0], elbow_trans[1], elbow_trans[2]]
            self.handkoords =       [hand_trans[0], hand_trans[1], hand_trans[2]]

        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF Error: {e}")
    
    def camera_listener_arms(self):
    #lese tf für Schulter Elebogen und Hand aus
        try:
            self.camera_listener()
            time = rospy.Time(0)
            #listener = tf.TransformListener()

            listener.waitForTransform("base", "right_shoulder", time, rospy.Duration(1.0))
            listener.waitForTransform("base", "right_elbow",    time, rospy.Duration(1.0))
            listener.waitForTransform("base", "right_hand",     time, rospy.Duration(1.0))

            right_shoulder_trans, _ = listener.lookupTransform("base", "right_shoulder",  time)
            right_elbow_trans,    _ = listener.lookupTransform("base", "right_elbow",     time)
            right_hand_trans,     _ = listener.lookupTransform("base", "right_hand",      time)

            self.rightshoulderkoords =   [right_shoulder_trans[0],  right_shoulder_trans[1], right_shoulder_trans[2]]
            self.rightelbowkoords =      [right_elbow_trans[0],     right_elbow_trans[1],    right_elbow_trans[2]]
            self.righthandkoords =       [right_hand_trans[0],      right_hand_trans[1],     right_hand_trans[2]]

            listener.waitForTransform("base", "left_shoulder", time, rospy.Duration(1.0))
            listener.waitForTransform("base", "left_elbow",    time, rospy.Duration(1.0))
            listener.waitForTransform("base", "left_hand",     time, rospy.Duration(1.0))

            left_shoulder_trans, _ = listener.lookupTransform("base", "left_shoulder",  time)
            left_elbow_trans,    _ = listener.lookupTransform("base", "left_elbow",     time)
            left_hand_trans,     _ = listener.lookupTransform("base", "left_hand",      time)

            self.leftshoulderkoords =   [left_shoulder_trans[0], left_shoulder_trans[1], left_shoulder_trans[2]]
            self.leftelbowkoords =      [left_elbow_trans[0],    left_elbow_trans[1],    left_elbow_trans[2]]
            self.lefthandkoords =       [left_hand_trans[0],     left_hand_trans[1],     left_hand_trans[2]]

        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF Error: {e}")

    def calc_arm_lenght(self):
    #bestimme ober und unterarm länge
        #self.camera_listener()
        self.camera_listener()
        self.uperarmlenght = self.calc_euclidean_distance(self.shoulderkoords,  self.elbowkoords)
        self.forearmlenght = self.calc_euclidean_distance(self.handkoords,      self.elbowkoords)
        rospy.loginfo(f"Oberarmlänge: {self.uperarmlenght}")
        rospy.loginfo(f"Unterarmlänge: {self.forearmlenght}")
        self.is_inside_norm()

        with open(f'armlaengen{user}.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f'{user}{rospy.Time.now()} oberarmlänge: {self.uperarmlenght}'])
            writer.writerow([f'{user}{rospy.Time.now()} unterarmlänge:{self.uperarmlenght}'])

    def calc_euclidean_distance(self, point1, point2):
    #bestimme den euclidischen Abstand zwischen zwei Punkten

        distance = 0.0
        for i in range(len(point1)):
            distance += (point2[i] - point1[i]) ** 2
        return math.sqrt(distance)
    
    def calc_angel(self, point1, point2, point3):
    #Berechne den Winkel zweier Graden
        vec1  = point1-point2
        vec2  = point2-point3

        angelrad = np.arccos(np.dot(vec1,vec2)/ (np.sqrt((vec1*vec1).sum())*np.sqrt((vec2*vec2).sum())))
        angle = angelrad * 360 / 2 / np.pi
        return angle

    def is_inside_norm(self):
    #Überprüft ob Ober und Unterarm innerhalb des 5. und 95 Perzentil sind

        if (self.uperarmlenght <= upperarmlenghtdin_max) and (self.uperarmlenght >= upperarmlenghtdin_min):
            self.inside_norm_upper = True
        else:
            rospy.logwarn(f"Oberarmmaße sind außerhalb 5. bis 95 Perzentil")
            self.inside_norm_upper = False
            self.uperarmlenght = upperarmlenghtdin

        if(self.forearmlenght <= forearmlenghdin_max) and (self.forearmlenght <= forearmlenghdin_min):
            self.inside_norm_fore = True
        else:
            rospy.logwarn(f"Unterarmmaße sind außerhalb 5. bis 95 Perzentil")
            self.inside_norm_fore = False
            self.forearmlenght = forearmlenghdin
    
    def get_arm_angels(self):
    #Lese Arm Koords aus und über gebe an Winkel berechnung + Logs
        while not self.stop_event.is_set():
            self.camera_listener_arms()

            right_shoulder = np.array([self.rightshoulderkoords[0],self.rightshoulderkoords[1],self.rightshoulderkoords[2]])
            right_elbow = np.array([self.rightelbowkoords[0],self.rightelbowkoords[1],self.rightelbowkoords[2]])
            right_hand = np.array([self.righthandkoords[0],self.righthandkoords[1],self.righthandkoords[2]])

            left_shoulder = np.array([self.leftshoulderkoords[0],self.leftshoulderkoords[1],self.leftshoulderkoords[2]])
            left_elbow = np.array([self.leftelbowkoords[0],self.leftelbowkoords[1],self.leftelbowkoords[2]])
            left_hand = np.array([self.lefthandkoords[0],self.lefthandkoords[1],self.lefthandkoords[2]])


            right_angle = self.calc_angel(right_shoulder,right_elbow,right_hand)
            left_angle = self.calc_angel(left_shoulder,left_elbow,left_hand)


            print(f'rechts: {right_angle} links: {left_angle}', end='\r') 
            with open(f'armlaengen{user}.csv','a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f'{user} {rospy.Time.now()} elbogenwinkel rechts: {right_angle}'])
                writer.writerow([f'{user} {rospy.Time.now()} elbogenwinkel links:  {left_angle}'])
            #return elbowangle


    
################################ Initialisiere Smachstates ################################
class Start(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['MPickUp','MHoldHD','MPositioning','PCB1PickUpAndPositioning','PCB2PickUpAndPositioning','BatteryPickUpAndPositioning','succeeded_end','test'])
    def execute(self, userdata):
        rospy.loginfo(f"Führe state: {self.__class__.__name__} aus.")
        #Hauptmenuü
        if( user ==  "test" ):

            print("\n--- Hauptmenü ---")
            print("1. MPickUp")
            print("2. MHoldHD")
            print("3. MPositioning")
            print("4. PCB1PickUpAndPositioning")
            print("5. PCB2PickUpAndPositioning")
            print("6. BatteryPickUpAndPositioning")
            print("7. Beenden")
            print("8. Test")

            while True:
                    start = input("Bitte wähle einen State aus (1–8): ")
                    if start == "1" or start == "":
                        print("\nDu hast MPickUp gewählt.")
                        return 'MPickUp'
                    elif start == "2":
                        print("\nDu hast MHoldHD gewählt.")
                        return 'MHoldHD'
                    elif start == "3":
                        print("\nDu hast MPositioning gewählt.")
                        return 'MPositioning'
                    elif start == "4":
                        print("\nDu hast PCB1PickUpAndPositioning gewählt.")
                        return 'PCB1PickUpAndPositioning'
                    elif start == "5":
                        print("\nDu hast PCB2PickUpAndPositioning gewählt.")
                        return 'PCB2PickUpAndPositioning'
                    elif start == "6":
                        print("\nDu hast BatteryPickUpAndPositioning gewählt.")
                        return 'BatteryPickUpAndPositioning'
                    elif start == "7":
                        print("\nDu hast Abort gewählt.")
                        return 'succeeded_end'
                    elif start == "8":
                        print("\nDu hast Test gewählt.")
                        return 'test'
                    else:
                        print(f"\n'{start}' gibts nich du Depp!")
        else:

            print("\n--- Hauptmenü ---")
            print("1. MPickUp")
            print("2. MHoldHD")
            print("3. MPositioning")
            print("4. PCB1PickUpAndPositioning")
            print("5. PCB2PickUpAndPositioning")
            print("6. BatteryPickUpAndPositioning")
            print("7. Beenden")

            while True:
                    start = input("Bitte wähle einen State aus (1–8): ")
                    
                    if start == "1" or start == "":
                        print("\nDu hast MPickUp gewählt.")
                        return 'MPickUp'
                    elif start == "2":
                        print("\nDu hast MHoldHD gewählt.")
                        return 'MHoldHD'
                    elif start == "3":
                        print("\nDu hast MPositioning gewählt.")
                        return 'MPositioning'
                    elif start == "4":
                        print("\nDu hast PCB1PickUpAndPositioning gewählt.")
                        return 'PCB1PickUpAndPositioning'
                    elif start == "5":
                        print("\nDu hast PCB2PickUpAndPositioning gewählt.")
                        return 'PCB2PickUpAndPositioning'
                    elif start == "6":
                        print("\nDu hast BatteryPickUpAndPositioning gewählt.")
                        return 'BatteryPickUpAndPositioning'
                    elif start == "7":
                        print("\nDu hast Abort gewählt.")
                        return 'succeeded_end'
                    else:
                        print(f"\n'{start}' gibt es nicht. Try again")

class MPickUp(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'succeeded_with_HD','aborted'])
        #self.robot_control = robot_control
        
        self.counter = 0

    def execute(self, userdata):
        #nehme Motor auf
        ### Kommentieren für testen
        # if not robot_control.gripper_controller.send_gripper_command('reset'):
        #    return 'aborted'
        # if not robot_control.gripper_controller.send_gripper_command('activate'):
        #    return 'aborted'
        #return 'succeeded_with_HD'

        rospy.loginfo(f"Führe state: {self.__class__.__name__} aus.")

        while True:
            newuser = input('enter y/n: ')
            if newuser == "y":
                if not robot_control.move_to_joint_goal( (-3.1557, -1.0119, -2.1765, -1.5426, 1.5686, -3.1643), 10):
                    return 'aborted'
                # if not robot_control.move_to_joint_goal( (-3.1557, -1.0119, -2.1765, -1.5426, 1.5686, -3.1643), 10):
                #     return 'aborted'
                # if not robot_control.gripper_controller.send_gripper_command('close'):
                #     return 'aborted'
                # if not robot_control.gripper_controller.send_gripper_command('open'):
                #     return 'aborted'
                plan = []
                plan.append(robot_control.convert_to_pose(rb_arm_transition_over_m))
                if not robot_control.move_to_taget_plan(plan,20):
                     return 'aborted'
                if not robot_control.pick_up(rb_arm_on_m[self.counter],20):
                    return 'aborted'
                if not robot_control.move_to_joint_goal( (-3.8497, -1.0055, -2.3556, -2.8687, -0.7227, -1.6213), 20):
                    return 'aborted'
                self.counter += 1
                rospy.loginfo(f"Nehme Motor {self.counter} auf")
                return 'succeeded_with_HD'
            elif newuser == "n":
                rospy.loginfo('weiter')
                return 'succeeded_with_HD'
             

class MHold(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded','aborted'])
    def execute(self, userdata):
        #unergonomische übergabe zu Probanten
        rospy.loginfo(f"Führe state: {self.__class__.__name__} aus.")
        newuser = input('enter y/n: ')
        while True:
            if newuser == "y":
                if not robot_control.move_to_target_cartesian(robot_control.convert_to_pose(rb_arm_to_hum_sta), 20):
                    return 'aborted'
                self.hm.stop_event = threading.Event()
                thread = threading.Thread(target=self.hm.get_arm_angels)
                thread.start()
                input("Drücke Enter, um zu stoppen...\n")
                self.hm.stop_event.set()
                thread.join(timeout=2.0)
                if not robot_control.move_to_joint_goal( (-3.8472, -1.0107, -2.3570, -2.8612, -0.7213, -1.6747), 20):
                        return 'aborted'
                return 'succeeded'

            elif newuser == "n":
                rospy.loginfo('weiter')
                return 'succeeded'
             
                
class MHoldHD(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded','succeeded_end','aborted'])
        self.hm = get_Hum_mertics()
    def execute(self, userdata):
        #ergonomische übergabe zu Probanten
        rospy.loginfo(f"Führe state: {self.__class__.__name__} aus.")
        newuser = input('enter y/n: ')
        while True:
            if newuser == "y":
                if not robot_control.handover_to_hum(5):
                    return 'aborted'
                self.hm.stop_event = threading.Event()
                thread = threading.Thread(target=self.hm.get_arm_angels)
                thread.start()
                input("Drücke Enter, um zu stoppen...\n")
                self.hm.stop_event.set()
                thread.join(timeout=2.0)

                if not robot_control.move_to_joint_goal( (-3.8472, -1.0107, -2.3570, -2.8612, -0.7213, -1.6747), 20):
                        return 'aborted'
                return 'succeeded'
            elif newuser == "n":
                rospy.loginfo('weiter')
                return 'succeeded'
            elif newuser == "a":
                rospy.loginfo('abort')
                return 'aborted'
            elif newuser == "t":
                rospy.loginfo('test_modus')
                while True:
                   
                        if not robot_control.handover_to_hum(5):
                            return 'aborted'

                        continue

             
            
class MPositioning(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded','succeeded_to_PCB','aborted'])
        self.robot_control = robot_control
        self.counter = 0

    def execute(self, userdata):
        #plaziere Motor auf Grundplatte
        rospy.loginfo(f"Führe state: {self.__class__.__name__} aus.")

        #self.counter += 1
        
        newuser = input('enter y/n: ')
        while True:
            if newuser == "y":
                if not robot_control.place_on_board(rb_arm_place_m_on_gb,20,0.02):
                    return 'aborted'
                self.counter += 1
                if not (self.counter % 4==0):
                    return 'succeeded'
                else:
                    return 'succeeded_to_PCB'
            elif newuser == "n":
                rospy.loginfo('weiter')
                return 'succeeded_to_PCB'
             

class PCB1PickUpAndPositioning(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded','aborted'])
        self.counter = 0
    def execute(self, userdata):
        rospy.loginfo(f"Führe state: {self.__class__.__name__} aus.")
        #plaziere PCB1 auf Grundplatte
        while True:
            newuser = input('enter y/n: ')
            if newuser == "y":
                if not robot_control.move_to_joint_goal((-3.1850, -1.9856, -1.4840, -1.2426, 1.5657, -3.1280),20):
                    return 'aborted'
                if not robot_control.pick_up(rb_arm_on_pcb1[self.counter]):
                    return 'aborted'
                if not robot_control.place_on_board(rb_arm_transition_over_gb1_3,10,0.2,130):
                    return 'aborted'
                self.counter +=1
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
        #plaziere PCB2 auf Grundplatte
        while True:
            newuser = input('enter y/n: ')
            if newuser == "y":
                if not robot_control.pick_up(rb_arm_on_pcb2[self.counter]):
                    return 'aborted'

                if not robot_control.place_on_board(rb_arm_transition_over_gb3_3,10):
                    return 'aborted'
            
                self.counter +=1
                return 'succeeded'
            elif newuser == "n":
                print("Exiting")
                return 'succeeded'
             

class PCB3PickUpAndPositioning(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded','aborted'])
        self.counter = 0
    def execute(self, userdata):
        rospy.loginfo(f"Führe state: {self.__class__.__name__} aus.")
        if random.random() <= 1/3:  # 1/3 Wahrscheinlichkeit
            while True:
                newuser = input('enter y/n: ')
                if newuser == "y":
                    if not robot_control.pick_up(rb_arm_on_pcb3[self.counter]):
                        return 'aborted'
                    if not robot_control.place_on_board(rb_arm_transition_over_gb4_3,10):
                        return 'aborted'
                    self.counter +=1
                    return 'succeeded'
                elif newuser == "n":
                    print("Exiting")
                    return 'succeeded'
                else:
                    print("Ungültige Eingabe. Bitte 'y' oder 'n' eingeben.")
        else:
            return 'succeeded'

class BatteryPickUpAndPositioning(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded','succeeded_end','aborted'])
        self.counter = 0
    def execute(self, userdata):
        rospy.loginfo(f"Führe state: {self.__class__.__name__} aus.")
        #plaziere Batterie auf Grundplatte
        while True:
            newuser = input('enter y/n: ')
            if newuser == "y":
                if not robot_control.pick_up(rb_arm_on_battery[self.counter]):
                    return 'aborted' 
                if not robot_control.place_on_board(rb_arm_transition_over_gb2_2 , 10,0.2,70):
                    return 'aborted'

                return 'succeeded'
            
            elif newuser == "n":
                print("Exiting")
                return 'succeeded'
             

class Aborted(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded_end','succeeded','start'])
    def execute(self, userdata):
        rospy.loginfo(f"Führe state: {self.__class__.__name__} aus.")
        #Ruckfallstate für Fehler
        while True:
            newuser = input('neuer Versuch? y/n: ')
            if newuser == "y":
                return 'start'
            elif newuser == "n":
                return 'succeeded_end'
             

class Test(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded_end','aborted'])
    def execute(self, userdata):
        rospy.loginfo(f"Führe state: {self.__class__.__name__} aus.")
        ####### DEBUG STATE NICHT VERWENDEN #######
        while True:
            newuser = input('neuer Versuch? y/n: ')
            if newuser == "y":


                command_list = [
                    {
                        "type": "joint",
                        "joints": (-3.1557, -1.0119, -2.1765, -1.5426, 1.5686, -3.1643)
                    },{
                        "type": "gripper",
                        "action": 'close' 
                    },{
                        "type": "gripper",
                        "action": 'open'  
                    },{
                        "type": "cartesian",
                        "pose": robot_control.convert_to_pose(rb_arm_transition_over_m)
                    },{
                        "type": "joint",
                        "joints": (-3.8423, -1.0118, -2.3565, -2.8601, -0.7018, -3.1867)
                    }
                ]
                command_list[4:4] = robot_control.pick_up_plan(rb_arm_on_m[0])
                robot_control.planner(command_list,10)
                return 'succeeded_end'

            elif newuser == "n":
                return 'aborted'
            elif newuser == "t":
                wahl = input("Welches Objekt soll der Roboter aufheben? ")
                # Zugriff über globals()
                if wahl in globals():
                    ausgewähltes_objekt = globals()[wahl]
                    wahl2 = input("Welche nr?")
                    robot_control.pick_up(ausgewähltes_objekt[wahl2])
                else:
                    print("Ungültige Eingabe oder Variable nicht definiert.")
                

if __name__ == "__main__":


    rospy.init_node('ur5_moveit_control', anonymous=True)
    wait_for_moveit()
    robot_control = RobotControl("manipulator")
    moveit_commander.roscpp_initialize(sys.argv)
    robot_control.gripper_controller.send_gripper_command('reset')
    robot_control.gripper_controller.send_gripper_command('activate')
    # robot_control.gripper_controller.send_gripper_command('open')
    listener = tf.TransformListener()
    user = ""
    sm = smach.StateMachine(outcomes=['finished'])
    with sm:
        # Smachstates und Smachverbindungen
        smach.StateMachine.add('Start', Start(),
                               transitions={'MPickUp':'MPickUp',
                                            'MHoldHD':'MHoldHD',
                                            'MPositioning':'MPositioning',
                                            'PCB1PickUpAndPositioning':'PCB1PickUpAndPositioning',
                                            'PCB2PickUpAndPositioning':'PCB2PickUpAndPositioning',
                                            'BatteryPickUpAndPositioning':'BatteryPickUpAndPositioning',
                                            'test':'Test',
                                            'succeeded_end':'finished'})
        smach.StateMachine.add('MPickUp', MPickUp(),
                               transitions={'succeeded':'MHold',
                                            'aborted':'Aborted',
                                            'succeeded_with_HD':'MHoldHD'})
        smach.StateMachine.add('MHold', MHold(),
                               transitions={'succeeded':'MPositioning',
                                            'aborted':'Aborted'})
        smach.StateMachine.add('MHoldHD', MHoldHD(),
                               transitions={'succeeded':'MPositioning',
                                            'succeeded_end':'finished',
                                            'aborted':'Aborted'})
        smach.StateMachine.add('MPositioning', MPositioning(),
                               transitions={'succeeded':'MPickUp',
                                            'succeeded_to_PCB':'PCB1PickUpAndPositioning',
                                            'aborted':'Aborted'})
        smach.StateMachine.add('PCB1PickUpAndPositioning', PCB1PickUpAndPositioning(),
                               transitions={'succeeded':'PCB2PickUpAndPositioning',
                                            'aborted':'Aborted'})
        smach.StateMachine.add('PCB2PickUpAndPositioning', PCB2PickUpAndPositioning(),
                               transitions={'succeeded':'PCB3PickUpAndPositioning',
                                            'aborted':'Aborted'})
        smach.StateMachine.add('PCB3PickUpAndPositioning', PCB3PickUpAndPositioning(),
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
                                            'succeeded':'MPickUp',
                                            'start':'Start'})

    # Iniizialisiere den introspection server
    try:
        sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
        sis.start()
    except AttributeError as e:
        rospy.logwarn(f"IntrospectionServer failed. Falling back to manual debugging. Error: {e}")

    # Führe die Statemachine aus
    outcome = sm.execute()
    rospy.spin() 
