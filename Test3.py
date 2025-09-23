import cv2
import numpy as np
import time

# 屏幕分辨率
width, height = 1024, 768
center = (width // 2, height // 2)

# 全局缩放比例（由最大矩形决定）
rect1_w, rect1_h = 18200, 14000
scale = min(width / rect1_w, height / rect1_h)

# 矩形数据: (X, Y, W, H, 名称)
rectangles = [
    (0, 0, 18200, 14000, "Bound"),
    (3071.6, 1514.1, 4400, 600, "PCB1"),
    (-3126.8, 1463.4, 7100, 1200, "PCB2"),
    (-6619.7, -1977.6, 2900, 3600, "PCB3"),
    (952.28, 4894.7, 3200, 4200, "Motor"),
    (-4815.2, 4892.9, 5600, 4200, "Battery"),
]

# 倒计时参数
countdown_time = 60  # 秒
start_time = time.time()

def world_to_screen(wx, wy):
    """Blender 世界坐标 -> 屏幕坐标"""
    sx = int(center[0] + wx * scale)
    sy = int(center[1] - wy * scale)  # 注意Y轴反转
    return (sx, sy)

def draw_rect(img, cx, cy, w, h, color, name=""):
    """绘制矩形 + 名称"""
    pt1 = world_to_screen(cx - w/2, cy - h/2)
    pt2 = world_to_screen(cx + w/2, cy + h/2)
    cv2.rectangle(img, pt1, pt2, color, 2)
    if name:
        cv2.putText(img, name, (pt1[0], pt1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_timer(img, total_time, start_time):
    elapsed = int(time.time() - start_time)
    remaining = max(total_time - elapsed, 0)

    radius = 50
    thickness = 8
    pos = (width - 80, height - 80)

    angle = int(360 * remaining / total_time)

    # 超采样因子
    scale_up = 4
    big_size = (img.shape[1] * scale_up, img.shape[0] * scale_up)
    big_img = np.zeros((big_size[1], big_size[0], 3), dtype=np.uint8)

    big_pos = (pos[0] * scale_up, pos[1] * scale_up)
    cv2.ellipse(big_img, big_pos, (radius * scale_up, radius * scale_up),
                0, -90, -90 + angle, (0, 255, 0),
                thickness * scale_up, lineType=cv2.LINE_AA)

    # 缩小回原始尺寸
    small_img = cv2.resize(big_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    mask = small_img > 0
    img[mask] = small_img[mask]

    # 文字绘制
    text = str(remaining)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness_text = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness_text)
    text_x = pos[0] - text_w // 2
    text_y = pos[1] + text_h // 2
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness_text, lineType=cv2.LINE_AA)

    return remaining


def render(highlight_idx=None):
    """绘制所有矩形，highlight_idx 高亮编号"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # 中心点
    cv2.circle(img, center, 5, (0, 0, 255), -1)
    for i, (x, y, w, h, name) in enumerate(rectangles):
        color = (0, 255, 0)  # 默认绿色
        if highlight_idx == i:  # 高亮红色
            color = (0, 0, 255)
        draw_rect(img, x, y, w, h, color, name)
    return img

# 初始渲染
highlight = None
cv2.namedWindow("Blender Rectangles Projection", cv2.WINDOW_NORMAL)

while True:
    img = render(highlight)

    # 绘制倒计时器
    remaining = draw_timer(img, countdown_time, start_time)

    cv2.imshow("Blender Rectangles Projection", img)
    key = cv2.waitKey(100) & 0xFF

    # 窗口关闭检测
    if cv2.getWindowProperty("Blender Rectangles Projection", cv2.WND_PROP_VISIBLE) < 1:
        break

    if key == 27 or remaining == 0:  # ESC 或倒计时结束
        break
    elif key in [49, 50, 51, 52, 53]:  # 数字 1-5
        highlight = key - 49 + 1
    elif key == 48:  # 数字 0，取消高亮
        highlight = None

cv2.destroyAllWindows()
