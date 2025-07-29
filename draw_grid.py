import numpy as np
import cv2

# إعدادات الشبكة
width, height = 1024, 768
cols, rows = 10, 8
cell_width = width // cols
cell_height = height // rows

# إنشاء الصورة السوداء
grid = np.zeros((height, width, 3), dtype=np.uint8)

# رسم الشبكة
for i in range(1, cols):
    x = i * cell_width
    cv2.line(grid, (x, 0), (x, height), (255, 255, 255), 1)

for j in range(1, rows):
    y = j * cell_height
    cv2.line(grid, (0, y), (width, y), (255, 255, 255), 1)

# دالة لحساب نقطة التقاطع مع عكس Y
def grid_intersection(col, row):
    x = col * cell_width
    y = height - (row * cell_height)
    return x, y

# تعريف النقاط
points = {
    'P1': (1, 1),
    'P2': (1, 4),
    'P3': (9, 1),
    'P4': (9, 4),
    'P5': (9, 7),
}

# رسم النقاط المصححة
for name, (col, row) in points.items():
    x, y = grid_intersection(col, row)
    color = (0, 0, 255) if name in ['P1', 'P2'] else (0, 255, 0)
    cv2.circle(grid, (x, y), 8, color, -1)
    cv2.putText(grid, name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# عرض النافذة
cv2.namedWindow("Grid", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Grid", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Grid", grid)

cv2.waitKey(0)
cv2.destroyAllWindows()
