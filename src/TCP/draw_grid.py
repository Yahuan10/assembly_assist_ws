import numpy as np
import cv2

# === Grid Settings ===
width, height = 1024, 768
cols, rows = 10, 8
cell_width = width // cols
cell_height = height // rows

# === Create a Black Image (Background) ===
# Create an empty (black) image of size (height x width) with 3 color channels (RGB)
grid = np.zeros((height, width, 3), dtype=np.uint8)

# === Draw Vertical Grid Lines ===
for i in range(1, cols):  # Skip the 0-th line (left border)
    x = i * cell_width
    # Draw vertical line from top to bottom
    cv2.line(grid, (x, 0), (x, height), (255, 255, 255), 1)  # White color, 1 pixel thickness

# === Draw Horizontal Grid Lines ===
for j in range(1, rows):  # Skip the 0-th line (bottom border)
    y = j * cell_height
    # Draw horizontal line from left to right
    cv2.line(grid, (0, y), (width, y), (255, 255, 255), 1)  # White color, 1 pixel thickness

# === Function to Convert Grid Coordinates to Pixel Coordinates ===
# Adjust Y-axis: convert from bottom-left origin (Cartesian style) to top-left origin (image coordinates)
def grid_intersection(col, row):
    x = col * cell_width                    # X position directly from column index
    y = height - (row * cell_height)        # Invert Y to match image coordinate system
    return x, y

# === Define Points in Grid Coordinates (col, row) ===
points = {
    'P1': (1, 1),
    'P2': (1, 4),
    'P3': (9, 1),
    'P4': (9, 4),
    'P5': (9, 7),
}

# === Draw Points on the Grid ===
for name, (col, row) in points.items():
    x, y = grid_intersection(col, row)      # Convert grid position to pixel position

    # Use red color for P1 and P2, green for others
    color = (0, 0, 255) if name in ['P1', 'P2'] else (0, 255, 0)

    # Draw a filled circle at the position
    cv2.circle(grid, (x, y), 8, color, -1)

    # Label the point with its name
    cv2.putText(grid, name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# === Show the Result in Fullscreen Window ===
cv2.namedWindow("Grid", cv2.WND_PROP_FULLSCREEN)  # Create a named window
cv2.setWindowProperty("Grid", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Set it to fullscreen
cv2.imshow("Grid", grid)  # Show the image in the window

# === Wait for Any Key Press and Close the Window ===
cv2.waitKey(0)            # Wait indefinitely for a key press
cv2.destroyAllWindows()   # Close all OpenCV windows
