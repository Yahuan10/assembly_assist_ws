import cv2
import numpy as np
import time

# 屏幕分辨率
width, height = 1024, 768
center = (width // 2, height // 2)

# 矩形框数据 (X, Y, W, H, 名称)
rectangles = [
    (0, 0, 18200, 14000, "Bound"),
    (3071.6, 1514.1, 4400, 600, "PCB1"),
    (-3126.8, 1463.4, 7100, 1200, "PCB2"),
    (-6619.7, -1977.6, 2900, 3600, "PCB3"),
    (952.28, 4894.7, 3200, 4200, "Motor"),
    (-4815.2, 4892.9, 5600, 4200, "Battery"),
    (-6619.7, -5577.6, 2900, 1800, "Handover")
]

# 矩形与图片绑定（修改为你的实际路径）
image_map = {
    "Bound": "bound.jpg",
    "PCB1": "pcb1.jpg",
    "PCB2": "pcb2.jpg",
    "PCB3": "pcb3.jpg",
    "Motor": "motor.jpg",
    "Battery": "battery.jpg"
}

# 全局缩放比例
rect1_w, rect1_h = 18200, 14000
scale = min(width / rect1_w, height / rect1_h)

# 倒计时参数
per_image_time = 6  # 每张图显示秒数
start_time = time.time()

# 当前和下一张
current_idx = 0
next_idx = 1

# ---------------- 工具函数 ----------------
def world_to_screen(wx, wy):
    sx = int(center[0] + wx * scale)
    sy = int(center[1] - wy * scale)
    return (sx, sy)

def draw_rect(img, cx, cy, w, h, color, name=""):
    pt1 = world_to_screen(cx - w/2, cy - h/2)
    pt2 = world_to_screen(cx + w/2, cy + h/2)
    cv2.rectangle(img, pt1, pt2, color, 2, lineType=cv2.LINE_AA)
    if name:
        cv2.putText(img, name, (pt1[0], pt1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, lineType=cv2.LINE_AA)

def load_and_resize(path, size):
    img = cv2.imread(path)
    if img is None:
        img = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
        cv2.putText(img, "No Img", (10, size[1]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

def place_image(canvas, img, top_left, size, label=""):
    h, w = size[1], size[0]
    x, y = top_left

    # 背景框（黑边 + 白底）
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 2)     # 黑色边框
    cv2.rectangle(canvas, (x+2, y+2), (x+w-2, y+h-2), (240, 240, 240), -1)  # 淡灰底

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)

    if img is not None:
        img_resized = cv2.resize(img, (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)
        canvas[y1:y2, x1:x2] = img_resized

    if label:
        cv2.putText(canvas, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def draw_info_window(img, current_path, next_text):
    """只显示大窗口和Next文字"""
    big_size = (200, 150)

    # 大窗口位置：倒计时器上方
    margin = 20
    big_pos = (width - big_size[0] - 20, height - 200 - big_size[1] - margin)

    # 加载图片
    big_img = load_and_resize(current_path, big_size)

    # 绘制大窗口
    place_image(img, big_img, big_pos, big_size, "Current")

    # --- Next文字 ---
    text = f"Next: {next_text}" if next_text else "Next: ---"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5       # ✅ 与 "Current" 保持一致
    thick = 1
    color = (0, 0, 0)

    cv2.putText(img, text, (big_pos[0], big_pos[1] + big_size[1] + 30),
                font, scale, color, thick, lineType=cv2.LINE_AA)

def draw_timer(img, total_time, start_time):
    elapsed = int(time.time() - start_time)
    remaining = max(total_time - elapsed, 0)

    # 参数
    radius = 50
    thickness = 10
    margin = 20  # 与底部间距
    pos = (width - 80, height - margin - radius)  # 圆心位置更贴近底部

    # 背景圆环（灰色槽）
    cv2.circle(img, pos, radius, (220, 220, 220), thickness, lineType=cv2.LINE_AA)

    # 剩余角度
    angle = int(360 * remaining / total_time)

    # 动态颜色
    if remaining / total_time > 0.5:
        color = (0, 200, 0)      # 绿色
    elif remaining / total_time > 0.2:
        color = (0, 200, 200)    # 黄色
    else:
        color = (0, 0, 255)      # 红色

    # 绘制进度圆环
    cv2.ellipse(img, pos, (radius, radius), 0, -90, -90 + angle,
                color, thickness, lineType=cv2.LINE_AA)

    # 在中心绘制数字
    text = str(remaining)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = pos[0] - tw // 2
    ty = pos[1] + th // 2
    cv2.putText(img, text, (tx, ty), font, scale, (50, 50, 50), thick, lineType=cv2.LINE_AA)

    return remaining

def draw_status_light(img, highlight_idx):
    """右上角 Ready 状态标签（自适应宽度）"""
    # 状态颜色
    if highlight_idx is None:
        color = (0, 200, 0)  # 绿色 = Ready
    else:
        color = (0, 0, 255)  # 红色 = Not Ready

    # 文字内容
    text = "Ready:"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

    # 内边距 & 圆点大小
    padding_x, padding_y = 15, 10
    dot_radius = 10
    dot_spacing = 20  # 文字与圆点间距

    # 动态计算框大小
    box_w = tw + dot_spacing + dot_radius * 2 + padding_x * 2
    box_h = th + padding_y * 2

    # 右上角位置
    x, y = width - box_w - 20, 20

    # 绘制背景框
    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (0, 0, 0), 2)
    cv2.rectangle(img, (x + 2, y + 2), (x + box_w - 2, y + box_h - 2), (255, 255, 255), -1)

    # 绘制文字
    text_x = x + padding_x
    text_y = y + box_h // 2 + th // 3
    cv2.putText(img, text, (text_x, text_y), font, scale, (0, 0, 0), thick, lineType=cv2.LINE_AA)

    # 圆点位置
    dot_x = text_x + tw + dot_spacing
    dot_y = y + box_h // 2
    cv2.circle(img, (dot_x, dot_y), dot_radius, color, -1, lineType=cv2.LINE_AA)

def render(highlight_idx=None):
    # 白色背景
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # 世界原点
    cv2.circle(img, center, 5, (0, 0, 255), -1)

    # 绘制矩形框
    for i, (x, y, w, h, name) in enumerate(rectangles):
        color = (0, 255, 0) if highlight_idx != i else (0, 0, 255)
        draw_rect(img, x, y, w, h, color, name)
    return img

# ---------------- 主循环 ----------------
# 定义菜单映射
menu_map = {
    1: "MPickUp",
    2: "MHoldHD",
    3: "MPositioning",
    4: "PCB1PickUpAndPositioning",
    5: "PCB2PickUpAndPositioning",
    6: "BatteryPickUpAndPositioning",
    7: "Beenden",
    8: "Test"
}

highlight = None
next_text = None
current_idx = 0
next_idx = 1
start_time = time.time()

cv2.namedWindow("Blender Rectangles Projection", cv2.WINDOW_NORMAL)

while True:
    # 背景 + 矩形
    canvas = render(highlight)

    # 当前与下一张图片名字
    current_name = rectangles[current_idx][4]

    # 倒计时
    remaining = draw_timer(canvas, per_image_time, start_time)

    # 大窗口 + Next 文字
    draw_info_window(canvas, image_map.get(current_name, ""), next_text)

    # 状态指示灯（Ready: ●）
    draw_status_light(canvas, highlight)

    # 显示画面
    cv2.imshow("Blender Rectangles Projection", canvas)
    key = cv2.waitKey(50) & 0xFF

    # 窗口关闭
    if cv2.getWindowProperty("Blender Rectangles Projection", cv2.WND_PROP_VISIBLE) < 1:
        break

    # ESC退出
    if key == 27:
        break

    # 数字键 1–8 控制
    elif key in [49, 50, 51, 52, 53, 54, 55, 56]:
        num = key - 48
        highlight = num if num < len(rectangles) else None
        current_idx = highlight if highlight is not None else 0
        next_idx = (current_idx + 1) % len(rectangles)
        start_time = time.time()

        # 切换 Next 显示内容
        next_text = menu_map.get(num, None)

    # 数字键 0 取消高亮
    elif key == 48:
        highlight = None
        next_text = None

    # 倒计时结束 → 自动切换到下一个
    if remaining == 0:
        current_idx = (current_idx + 1) % len(rectangles)
        next_idx = (current_idx + 1) % len(rectangles)
        start_time = time.time()

cv2.destroyAllWindows()

