import cv2
import numpy as np
from rtde_receive import RTDEReceiveInterface
import time

# === Canvas and Window Settings ===
width, height = 1024, 768  # Canvas dimensions in pixels
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255  # White canvas
win_name = "TCP Trajectory"

# === Homography Matrix ===
# This matrix maps real-world (x, y) in meters to image pixel coordinates
H = np.array([
    [7.35135136e+02, 0., 2.03448646e+02],
    [0., 1.02857148e+03, 1.71085702e+02],
    [0., 0., 1.00000000e+00]
])

# === Robot Connection Setup ===
robot_ip = "192.168.0.107"
rtde_receive = RTDEReceiveInterface(robot_ip)

# === Convert world coordinates (x, y) to image pixel coordinates ===
def world_to_pixel(x, y):
    point = np.array([x, y, 1.0])
    pixel = H @ point
    pixel /= pixel[2]
    return int(round(pixel[0])), int(round(pixel[1]))

# === Define rb_arm_on_m positions (x, y) only ===
rb_arm_on_m = [
    np.array([0.2631105225136129, 0.11513901314207496]),
    np.array([0.2631105225136129, 0.06813901314207496]),
    np.array([0.2631105225136129, 0.02113901314207496]),
    np.array([0.2631105225136129, -0.02613901314207496]),
    np.array([0.3431105225136129, 0.11513901314207496]),
    np.array([0.3431105225136129, 0.06813901314207496]),
    np.array([0.3431105225136129, 0.02113901314207496]),
    np.array([0.3431105225136129, -0.02613901314207496]),
    np.array([0.4231105225136129, 0.11513901314207496]),
    np.array([0.4231105225136129, 0.06813901314207496]),
    np.array([0.4231105225136129, 0.02113901314207496]),
    np.array([0.4231105225136129, -0.02613901314207496]),
    np.array([0.5051105225136129, 0.11513901314207496]),
    np.array([0.5051105225136129, 0.06813901314207496]),
    np.array([0.5051105225136129, 0.02113901314207496]),
    np.array([0.5051105225136129, -0.02613901314207496])
]

# === Precompute pixel coordinates for rb_arm_on_m ===
rb_arm_on_m_pixels = [world_to_pixel(x, y) for x, y in rb_arm_on_m]
xs, ys = zip(*rb_arm_on_m_pixels)
top_left = (min(xs), min(ys))
bottom_right = (max(xs), max(ys))

# === List to store TCP trajectory points ===
trajectory = []

# === Initialize Fullscreen Window ===
cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# === Main Loop ===
while True:
    # Get current TCP pose: [x, y, z, Rx, Ry, Rz]
    tcp_pose = rtde_receive.getActualTCPPose()
    x, y = tcp_pose[0], tcp_pose[1]

    # Convert to pixel
    px, py = world_to_pixel(x, y)
    print(f"[TCP] x={x:.3f} m, y={y:.3f} m → Pixel: ({px}, {py})")

    # Append if within canvas
    if 0 <= px < width and 0 <= py < height:
        trajectory.append((px, py))
    else:
        print("⚠️ Point is outside canvas!")

    # Clear canvas
    canvas[:] = 255

    # Draw TCP trajectory
    for pt in trajectory:
        cv2.circle(canvas, pt, 3, (0, 0, 255), -1)  # Red

    # Draw current TCP coordinates
    cv2.putText(canvas, f"x={x:.3f} y={y:.3f}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # Draw rb_arm_on_m rectangular region
    cv2.rectangle(canvas, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(canvas, "rb_arm_on_m", (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)

    # Show canvas
    cv2.imshow(win_name, canvas)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

    # Sleep to limit update rate
    time.sleep(0.1)

# === Clean up ===
cv2.destroyAllWindows()
