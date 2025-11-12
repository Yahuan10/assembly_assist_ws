
import numpy as np
import cv2

# TCP coordinates (in meters)
real_points = np.array([
    [-0.140, 0.500],
    [-0.140, 0.200],
    [-0.873, 0.398],
    [-0.983, 0.200],
    [-0.985, -0.113],  
    [-0.345, 0.310],
    [-0.560, 0.200],


], dtype=np.float32)

# Corresponding pixel coordinates
pixel_points = np.array([
    [102.4, 672],  # P1
    [102.4, 384],  # P2
    [819.2, 576],  # P3
    [921.6, 384],  # P4
    [921.6, 96],   # P5
    [307.2, 288],   # P6
    [512, 384],  # P7
], dtype=np.float32)

# Compute the Homography matrix
H, status = cv2.findHomography(real_points, pixel_points)

print("Homography:")
print(H)
