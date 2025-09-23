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
    (-6619.7, -5577.6, 2900, 1800, "Handover"),  # 新增交接矩形
    (6619.7, -4577.6, 4500, 4000, "Information"),  # 新增信息矩形
]

# 倒计时参数
countdown_time = 60  # 秒
start_time = time.time()

# 文字显示内容列表 (对应矩形索引 0-5) - 支持多行文字
text_contents = [
    "Bound\nhallo\nHallo again",           # 对应 Bound (索引0) - 多行
    "PCB1 Pick\nnihao\nfuck",        # 对应 PCB1 (索引1) - 多行
    "PCB2 Pick\nshit\nhahaha",        # 对应 PCB2 (索引2) - 多行
    "PCB3 Pick\n扩展电路板\n状态: 空闲",      # 对应 PCB3 (索引3) - 多行
    "电机驱动模块\nMotor Control\n功率: 500W",  # 对应 Motor (索引4) - 多行
    "电池供电模块\nBattery Pack\n电压: 24V",   # 对应 Battery (索引5) - 多行
]

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


def draw_info_text(img, text_content, highlight_idx=None):
    """绘制信息文字显示 - 支持多行文字"""
    if text_content is None or highlight_idx is None:
        return
    
    # 文字显示参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_color = (255, 255, 255)  # 白色文字
    line_spacing = 10  # 行间距
    
    # 分割文本为多行（支持\n换行符）
    lines = text_content.split('\n')
    
    # 计算所有行的最大宽度和总高度
    max_width = 0
    line_heights = []
    for line in lines:
        (line_w, line_h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, line_w)
        line_heights.append(line_h)
    
    total_height = sum(line_heights) + line_spacing * (len(lines) - 1)
    
    # 文字位置：使用世界坐标
    world_x, world_y = 6619.7, -4577.6
    screen_pos = world_to_screen(world_x, world_y)
    
    # 计算起始位置（整体文本块居中）
    start_x = screen_pos[0] - max_width // 2
    start_y = screen_pos[1] - total_height // 2
    
    # 逐行绘制文字
    current_y = start_y
    for i, line in enumerate(lines):
        if line.strip():  # 跳过空行
            # 计算当前行的居中位置
            (line_w, line_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            line_x = screen_pos[0] - line_w // 2  # 每行都居中对齐
            line_y = current_y + line_h
            
            # 绘制当前行
            cv2.putText(img, line, (line_x, line_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # 移动到下一行位置
        current_y += line_heights[i] + line_spacing


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
    
    # 添加文字显示功能
    if highlight_idx is not None and 0 <= highlight_idx < len(text_contents):
        text_content = text_contents[highlight_idx]
        draw_info_text(img, text_content, highlight_idx)
    
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
