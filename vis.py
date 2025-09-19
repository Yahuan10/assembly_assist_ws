import cv2
import numpy as np
from rtde_receive import RTDEReceiveInterface
import time
import importlib.util
import os

# === Canvas and Window Settings ===
width, height = 1024, 768  # Canvas size
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
win_name = "TCP Trajectory"

# === Homography Matrix (pre-calibrated) ===
H = np.array([
    [7.35135136e+02, 0., 2.03448646e+02],
    [0., 1.02857148e+03, 1.71085702e+02],
    [0., 0., 1.00000000e+00]
])

# === Load rb_arm_on_m from scheduler.py ===
scheduler_path = "/Users/yujiangtao/my_project/PJ_Auto/src/schledluer/src/scheduler.py"

# Dynamically import scheduler.py
spec = importlib.util.spec_from_file_location("scheduler", scheduler_path)
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)

# Extract rb_arm_on_m
rb_arm_on_m = scheduler.rb_arm_on_m
rb_arm_on_m_xy = [pose[:2] for pose in rb_arm_on_m]  # Only x, y

# Convert world (m) to pixel coordinates
def world_to_pixel(x, y):
    point = np.array([x, y, 1.0])
    pixel = H @ point
    pixel /= pixel[2]
    return int(round(pixel[0])), int(round(pixel[1]))

rb_arm_on_m_pixels = [world_to_pixel(x, y) for x, y in rb_arm_on_m_xy]
xs, ys = zip(*rb_arm_on_m_pixels)
top_left = (min(xs), min(ys))
bottom_right = (max(xs), max(ys))

# === Robot Connection ===
robot_ip = "192.168.0.107"
rtde_receive = RTDEReceiveInterface(robot_ip)

# === Fullscreen Window ===
cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# === Trajectory Buffer ===
trajectory = []

# === Main Loop ===
while True:
    # Get TCP pose
    tcp_pose = rtde_receive.getActualTCPPose()
    x, y = tcp_pose[0], tcp_pose[1]

    # Convert to pixel
    px, py = world_to_pixel(x, y)
    print(f"[TCP] x={x:.3f} m, y={y:.3f} m → Pixel: ({px}, {py})")

    # Save if in canvas bounds
    if 0 <= px < width and 0 <= py < height:
        trajectory.append((px, py))
    else:
        print("⚠️ TCP position is outside the canvas!")

    # Redraw canvas
    canvas[:] = 255

    # Draw TCP trajectory
    for pt in trajectory:
        cv2.circle(canvas, pt, 3, (0, 0, 255), -1)

    # Draw current TCP text
    cv2.putText(canvas, f"x={x:.3f} y={y:.3f}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # Draw rb_arm_on_m rectangle
    cv2.rectangle(canvas, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(canvas, "rb_arm_on_m", (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)

    # Check if TCP is inside the region
    if top_left[0] <= px <= bottom_right[0] and top_left[1] <= py <= bottom_right[1]:
        cv2.putText(canvas, "✅ TCP in M board region", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 0), 2)
        print("✅ TCP 在 M 板区域内")
    else:
        print("🟡 TCP 不在 M 板区域")

    # Show canvas
    cv2.imshow(win_name, canvas)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

    # Wait for next update
    time.sleep(0.1)

# === Cleanup ===
cv2.destroyAllWindows()
