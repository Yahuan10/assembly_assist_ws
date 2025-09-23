import cv2
import numpy as np

# ===== 画布 =====
W, H = 1024, 768
img = np.ones((H, W, 3), np.uint8) * 255

# ===== 原点（画布中心）=====
ORIGIN = (W // 2, H // 2)

# 可视化坐标轴（可选）
cv2.line(img, (0, ORIGIN[1]), (W, ORIGIN[1]), (220,220,220), 1)  # X轴
cv2.line(img, (ORIGIN[0], 0), (ORIGIN[0], H), (220,220,220), 1)  # Y轴
cv2.circle(img, ORIGIN, 5, (0,0,255), -1)
cv2.putText(img, "(0,0)", (ORIGIN[0]+8, ORIGIN[1]-8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

# =========================================================
# 方式 A：直接用“像素偏移”固定与原点的相对位置（推荐，最直观）
BATTERY_OFFSET_PX = (200, -140)    # (dx, dy)，右为+，上为-（OpenCV y向下）
# ---------------------------------------------------------
# 方式 B：如果你有“世界坐标偏移”(例如来自Blender)，用统一比例映射到像素
USE_WORLD_COORDS = False
BATTERY_OFFSET_WORLD = (952.28, 4894.7)  # 例：世界坐标中的相对位移（单位自定）
POS_SCALE = 0.1                          # 世界坐标 → 像素的比例（自己定）
# =========================================================

# ===== battery 外观：固定长宽比 =====
ASPECT_WH = 2/3           # 宽:高 比例（例：2:3）
BATTERY_HEIGHT_PX = 200   # 固定高度（像素）
BATTERY_WIDTH_PX  = int(BATTERY_HEIGHT_PX * ASPECT_WH)

# ===== 计算 battery 中心像素坐标 =====
if USE_WORLD_COORDS:
    dx_px = int(BATTERY_OFFSET_WORLD[0] * POS_SCALE)
    dy_px = int(-BATTERY_OFFSET_WORLD[1] * POS_SCALE)  # 上为+
else:
    dx_px, dy_px = BATTERY_OFFSET_PX

cx = ORIGIN[0] + dx_px
cy = ORIGIN[1] + dy_px  # 注意：dy_px 已按“上为负/下为正”的屏幕坐标给出

# ===== 矩形顶点 =====
tl = (cx - BATTERY_WIDTH_PX // 2, cy - BATTERY_HEIGHT_PX // 2)
br = (cx + BATTERY_WIDTH_PX // 2, cy + BATTERY_HEIGHT_PX // 2)

# 画 battery
cv2.rectangle(img, tl, br, (0, 0, 255), 2)
cv2.putText(img, "battery", (tl[0], tl[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# 显示
cv2.imshow("Layout", img)
cv2.waitKey(0)
cv2.destroyAllWindows()