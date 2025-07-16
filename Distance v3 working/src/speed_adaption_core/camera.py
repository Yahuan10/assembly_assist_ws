import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import copy
import mediapipe as mp
import math
import csv
from speed_adaption_core.robot_data import RobotInterface
from speed_adaption_core.data_buffer import store_single_value, store_short_term

# --- KONFIGURATION & INITIALISIERUNG ---
camera_serial = '317222071793'

# MediaPipe Initialisierung
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2  # Explizit auf 2 Hände setzen
)
landmarker = HandLandmarker.create_from_options(options)


# Hilfsfunktion für Abstandsberechnung
def distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


# Laden der Kalibrierungsdaten
current_dir = os.path.dirname(os.path.abspath(__file__))
calibration_data = np.load(os.path.join(current_dir, "../calibration_data.npz"))
T_world2cam = calibration_data["T_world2cam"]
camera_matrix = calibration_data["camera_matrix"]
dist_coeffs = calibration_data["dist_coeffs"]

# Initialisierung der Roboter-Schnittstelle
robot = RobotInterface()

# Konfiguration der Kamera-Pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_device(camera_serial)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
align_to = rs.stream.color
alignedFs = rs.align(align_to)


def hand_tracking():
    while True:
        timestamp = time.time()
        frames = pipeline.wait_for_frames()
        aligned_frames = alignedFs.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        store_single_value("color_image", (timestamp, color_image))
        store_single_value("depth_image", depth_image)

        tcp_pose = robot.get_tcp_position()
        store_single_value("robot_tcp", (timestamp, tcp_pose))

        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = landmarker.detect(mp_image)

        all_distances = {}
        all_min_points = {}

        if results.hand_landmarks:
            # --- DEBUG-AUSGABE 1 ---
            print("=================================================")
            print(f"DEBUG: Erkannte Hände in diesem Frame: {len(results.hand_landmarks)}")

            for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                hand_label = handedness[0].category_name.lower()

                # --- DEBUG-AUSGABE 2 ---
                print(f"DEBUG: Verarbeite Hand: '{hand_label}' mit Konfidenz: {handedness[0].score:.2f}")

                min_dist_for_this_hand = float('inf')
                min_point_for_this_hand = None

                for landmark in hand_landmarks:
                    x, y = int(landmark.x * 640), int(landmark.y * 480)

                    if 0 <= y < 480 and 0 <= x < 640:
                        depth = depth_image[y, x]
                        if depth > 0:
                            pixel_coords = np.array([x, y, 1], dtype=np.float32)
                            camera_coords = np.linalg.inv(camera_matrix) @ (pixel_coords * depth).reshape(3, 1)
                            world_coords = np.linalg.inv(T_world2cam[:3, :3]) @ (
                                        camera_coords - T_world2cam[:3, 3].reshape(3, 1))
                            real_x, real_y, real_z = world_coords.flatten()
                            dist = distance(tcp_pose[0], tcp_pose[1], tcp_pose[2], real_x, real_y, real_z)

                            if dist < min_dist_for_this_hand:
                                min_dist_for_this_hand = dist
                                min_point_for_this_hand = (real_x, real_y, real_z)

                if min_point_for_this_hand:
                    all_distances[hand_label] = min_dist_for_this_hand
                    all_min_points[hand_label] = min_point_for_this_hand

            # --- DEBUG-AUSGABE 3 ---
            print(f"DEBUG: Finale Distanzen vor dem Speichern: {all_distances}")
            print("=================================================\n")

        store_single_value("hand_distances", (timestamp, all_distances))