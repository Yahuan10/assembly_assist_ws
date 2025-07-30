import cv2
import numpy as np
from rtde_receive import RTDEReceiveInterface
import time

# === Canvas and Window Settings ===
width, height = 1024, 768  # Canvas dimensions in pixels
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255  # White canvas (3 channels: RGB)
win_name = "TCP Trajectory"

# === Homography Matrix ===
# This matrix transforms real-world (x, y) coordinates (in meters) to pixel coordinates on the canvas.
# It must be precomputed using calibration
H = np.array([
    [7.35135136e+02, 0., 2.03448646e+02],
    [0., 1.02857148e+03, 1.71085702e+02],
    [0., 0., 1.00000000e+00]
])

# === Robot Connection Setup ===
robot_ip = "192.168.0.107"
rtde_receive = RTDEReceiveInterface(robot_ip)  # Create a connection interface to receive robot data

# List to store the TCP trajectory points (in pixel coordinates)
trajectory = []

def world_to_pixel(x, y):
    """
    Convert real-world coordinates (meters) to image pixel coordinates using the homography matrix.
    param x: Real-world x-coordinate (in meters)
    param y: Real-world y-coordinate (in meters)
    return: Corresponding pixel (px, py) as integers
    """
    point = np.array([x, y, 1.0])     # Convert to homogeneous coordinates
    pixel = H @ point                 # Apply homography transformation
    pixel /= pixel[2]                # Normalize to make the last coordinate 1
    return int(round(pixel[0])), int(round(pixel[1]))  # Return integer pixel positions

# === Initialize Fullscreen Window ===
cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# === Main Loop to Continuously Read and Visualize TCP Trajectory ===
while True:
    # Get the current TCP pose: [x, y, z, Rx, Ry, Rz] in meters and radians
    tcp_pose = rtde_receive.getActualTCPPose()
    x, y = tcp_pose[0], tcp_pose[1]  # Extract only the x and y coordinates

    # Convert the world coordinates (x, y) to pixel coordinates (px, py)
    px, py = world_to_pixel(x, y)

    # Print both world and pixel coordinates for debugging
    print(f"[TCP] x={x:.3f} m, y={y:.3f} m → Pixel: ({px}, {py})")

    # Only add the point to trajectory if it's within the canvas bounds
    if 0 <= px < width and 0 <= py < height:
        trajectory.append((px, py))  # Save point to draw later
    else:
        print("⚠️ Point is outside the visible canvas!")

    # Clear the canvas (make it white again)
    canvas[:] = 255

    # Draw all trajectory points on the canvas as red circles
    for pt in trajectory:
        cv2.circle(canvas, pt, 3, (0, 0, 255), -1)  # Red dot, radius=3

    # Display the current (x, y) coordinates on the canvas
    cv2.putText(canvas, f"x={x:.3f} y={y:.3f}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # Show the updated canvas in the window
    cv2.imshow(win_name, canvas)

    # Press ESC key to break the loop (ASCII 27)
    if cv2.waitKey(1) == 27:
        break

    # Wait for 0.1 seconds before next update (10 Hz refresh rate)
    time.sleep(0.1)

# === Clean up and close the window after exiting the loop ===
cv2.destroyAllWindows()
