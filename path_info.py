import cv2
import numpy as np
import time
import math

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

# 矩形与文本绑定
text_map = {
    "Bound": "test\ntest\ntest",
    "PCB1": "PCB1 Pick up\nPlease install PCB1\nPrecision positioning operation",
    "PCB2": "PCB2 Pick up\nPlease install PCB2\nPrecision positioning operation",
    "PCB3": "PCB3 Pick up\nPlease install PCB3\nPrecision positioning operation",
    "Motor": "Motor Pick up\nPlease install Motor\nPrecision positioning operation",
    "Battery": "Battery Pick up\nPlease install Battery\nPrecision positioning operation",
    "Handover": "Handover\nMaterial handover point\nTransfer operation area"
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

# ============ 新增：机器臂路径相关参数 ============
# 路径动画参数
path_animation = {
    'active': False,           # 是否正在显示路径动画
    'from_rect': None,         # 起始矩形索引
    'to_rect': 6,              # 目标矩形索引(Handover)
    'progress': 0.0,           # 动画进度 0.0-1.0
    'start_time': 0,           # 路径动画开始时间
    'duration': 3.0,           # 路径动画持续时间(秒)
    'trail_points': [],        # 轨迹点历史记录
    'max_trail_length': 30     # 最大轨迹点数量
}

# 机器臂当前位置
robot_arm = {
    'x': 0,
    'y': 0,
    'target_x': 0,
    'target_y': 0
}

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

    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 2)
    cv2.rectangle(canvas, (x+2, y+2), (x+w-2, y+h-2), (240, 240, 240), -1)

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)

    if img is not None:
        img_resized = cv2.resize(img, (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)
        canvas[y1:y2, x1:x2] = img_resized

    if label:
        cv2.putText(canvas, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def draw_info_window(img, current_text, next_text):
    """显示文本信息窗口"""
    window_size = (250, 200)
    
    # 信息窗口位置：倒计时器上方
    margin = 20
    window_pos = (width - window_size[0] - 20, height - 200 - window_size[1] - margin)
    
    # 绘制窗口背景
    x, y = window_pos
    w, h = window_size
    
    # 背景框（黑边 + 白底）
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)     # 黑色边框
    cv2.rectangle(img, (x+2, y+2), (x+w-2, y+h-2), (250, 250, 250), -1)  # 白色底
    
    # 标题
    cv2.putText(img, "Current Info:", (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    
    # 显示当前文本内容（多行文本）
    if current_text:
        lines = current_text.split('\n')
        for i, line in enumerate(lines):
            y_offset = y + 50 + i * 25
            if y_offset < y + h - 20:  # 确保不超出窗口
                cv2.putText(img, line, (x + 10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, lineType=cv2.LINE_AA)
    else:
        cv2.putText(img, "No Information", (x + 10, y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, lineType=cv2.LINE_AA)
    
    # --- Next文字 ---
    next_display = f"Next: {next_text}" if next_text else "Next: ---"
    cv2.putText(img, next_display, (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

def draw_timer(img, total_time, start_time):
    elapsed = int(time.time() - start_time)
    remaining = max(total_time - elapsed, 0)

    radius = 50
    thickness = 10
    margin = 20
    pos = (width - 80, height - margin - radius)

    cv2.circle(img, pos, radius, (220, 220, 220), thickness, lineType=cv2.LINE_AA)

    angle = int(360 * remaining / total_time)

    if remaining / total_time > 0.5:
        color = (0, 200, 0)
    elif remaining / total_time > 0.2:
        color = (0, 200, 200)
    else:
        color = (0, 0, 255)

    cv2.ellipse(img, pos, (radius, radius), 0, -90, -90 + angle,
                color, thickness, lineType=cv2.LINE_AA)

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
    """右上角Ready状态标签"""
    if highlight_idx is None:
        color = (0, 200, 0)  # 绿色 = Ready
    else:
        color = (0, 0, 255)  # 红色 = Not Ready

    text = "Ready:"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

    padding_x, padding_y = 15, 10
    dot_radius = 10
    dot_spacing = 20

    box_w = tw + dot_spacing + dot_radius * 2 + padding_x * 2
    box_h = th + padding_y * 2

    x, y = width - box_w - 20, 20

    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (0, 0, 0), 2)
    cv2.rectangle(img, (x + 2, y + 2), (x + box_w - 2, y + box_h - 2), (255, 255, 255), -1)

    text_x = x + padding_x
    text_y = y + box_h // 2 + th // 3
    cv2.putText(img, text, (text_x, text_y), font, scale, (0, 0, 0), thick, lineType=cv2.LINE_AA)

    dot_x = text_x + tw + dot_spacing
    dot_y = y + box_h // 2
    cv2.circle(img, (dot_x, dot_y), dot_radius, color, -1, lineType=cv2.LINE_AA)

# ============ 新增：机器臂路径绘制相关函数 ============
def get_rect_center(rect_idx):
    """获取矩形中心点世界坐标"""
    if 0 <= rect_idx < len(rectangles):
        x, y, w, h, name = rectangles[rect_idx]
        return (x, y)
    return (0, 0)

def smooth_interpolation(start_pos, end_pos, progress):
    """平滑插值函数，创建更自然的机器臂运动路径"""
    # 使用三次贝塞尔曲线创建平滑路径
    # 添加中间控制点使路径更像机器臂运动
    
    start_x, start_y = start_pos
    end_x, end_y = end_pos
    
    # 计算中间控制点（创建弧形路径）
    mid_x = (start_x + end_x) / 2
    mid_y = (start_y + end_y) / 2
    
    # 添加高度偏移，模拟机器臂提升动作
    height_offset = abs(end_x - start_x) * 0.3 + abs(end_y - start_y) * 0.2
    control_y = mid_y - height_offset
    
    # 三次贝塞尔曲线插值
    t = progress
    t2 = t * t
    t3 = t2 * t
    
    # 贝塞尔曲线系数
    b0 = (1 - t) ** 3
    b1 = 3 * t * (1 - t) ** 2
    b2 = 3 * t2 * (1 - t)
    b3 = t3
    
    # 控制点
    p0 = start_pos
    p1 = (start_x, control_y)
    p2 = (end_x, control_y)
    p3 = end_pos
    
    # 计算当前位置
    current_x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
    current_y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
    
    return (current_x, current_y)

def start_path_animation(from_idx, to_idx=6):
    """启动路径动画"""
    global path_animation, robot_arm
    
    print(f"启动路径动画: from_idx={from_idx}, to_idx={to_idx}")  # 调试信息
    
    if from_idx is not None and 0 <= from_idx < len(rectangles) and from_idx != to_idx:
        path_animation['active'] = True
        path_animation['from_rect'] = from_idx
        path_animation['to_rect'] = to_idx
        path_animation['progress'] = 0.0
        path_animation['start_time'] = time.time()
        path_animation['trail_points'] = []
        
        # 设置机器臂起始位置
        start_pos = get_rect_center(from_idx)
        robot_arm['x'] = start_pos[0]
        robot_arm['y'] = start_pos[1]
        print(f"机器臂起始位置: {start_pos}")
    else:
        print(f"无效的路径动画参数: from_idx={from_idx}, to_idx={to_idx}")

def update_path_animation():
    """更新路径动画状态"""
    global path_animation, robot_arm
    
    if not path_animation['active']:
        return
    
    # 计算动画进度
    elapsed = time.time() - path_animation['start_time']
    progress = min(elapsed / path_animation['duration'], 1.0)
    path_animation['progress'] = progress
    
    # 获取起点和终点
    start_pos = get_rect_center(path_animation['from_rect'])
    end_pos = get_rect_center(path_animation['to_rect'])
    
    # 平滑插值计算当前位置
    current_pos = smooth_interpolation(start_pos, end_pos, progress)
    robot_arm['x'] = current_pos[0]
    robot_arm['y'] = current_pos[1]
    
    # 添加轨迹点
    screen_pos = world_to_screen(current_pos[0], current_pos[1])
    path_animation['trail_points'].append(screen_pos)
    
    # 限制轨迹点数量
    if len(path_animation['trail_points']) > path_animation['max_trail_length']:
        path_animation['trail_points'].pop(0)
    
    # 动画结束
    if progress >= 1.0:
        path_animation['active'] = False

def draw_robot_arm_path(img):
    """绘制机器臂路径和当前位置"""
    if not path_animation['active'] and len(path_animation['trail_points']) == 0:
        return
    
    # 绘制轨迹线
    if len(path_animation['trail_points']) > 1:
        for i in range(1, len(path_animation['trail_points'])):
            # 渐变透明度效果
            alpha = i / len(path_animation['trail_points'])
            color_intensity = int(255 * alpha)
            
            # 绘制轨迹线段
            cv2.line(img, 
                    path_animation['trail_points'][i-1], 
                    path_animation['trail_points'][i],
                    (0, color_intensity, 255),  # 蓝色到亮蓝色渐变
                    3, lineType=cv2.LINE_AA)
    
    # 绘制当前机器臂位置
    if path_animation['active']:
        current_screen_pos = world_to_screen(robot_arm['x'], robot_arm['y'])
        
        # 机器臂主体（圆形）
        cv2.circle(img, current_screen_pos, 15, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, current_screen_pos, 15, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        
        # 机器臂方向指示器
        if path_animation['progress'] < 1.0:
            # 计算指向目标的方向
            target_pos = get_rect_center(path_animation['to_rect'])
            target_screen = world_to_screen(target_pos[0], target_pos[1])
            
            # 绘制指向线 - 修复OpenCV兼容性问题
            try:
                cv2.arrowedLine(img, current_screen_pos, target_screen, 
                               (255, 100, 0), 2, tipLength=0.05)
            except:
                # 如果arrowedLine不支持，使用普通线条代替
                cv2.line(img, current_screen_pos, target_screen, 
                        (255, 100, 0), 2, lineType=cv2.LINE_AA)
                # 在目标点绘制一个小三角形作为箭头
                cv2.circle(img, target_screen, 5, (255, 100, 0), -1, lineType=cv2.LINE_AA)
    
    # 绘制目标位置标记
    if path_animation['from_rect'] is not None:
        handover_pos = get_rect_center(6)  # Handover位置
        handover_screen = world_to_screen(handover_pos[0], handover_pos[1])
        
        # 目标点闪烁效果
        flash_intensity = int(abs(math.sin(time.time() * 5)) * 255)
        cv2.circle(img, handover_screen, 20, (0, flash_intensity, 0), 3, lineType=cv2.LINE_AA)
        cv2.putText(img, "TARGET", (handover_screen[0]-30, handover_screen[1]-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1, lineType=cv2.LINE_AA)

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

# 主循环
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
    try:
        # 更新路径动画
        update_path_animation()
        
        # 背景 + 矩形
        canvas = render(highlight)
        
        # *** 新增：绘制机器臂路径 ***
        draw_robot_arm_path(canvas)

        # 当前图片名字
        current_name = rectangles[current_idx][4]

        # 倒计时
        remaining = draw_timer(canvas, per_image_time, start_time)

        # 大窗口 + Next 文字
        draw_info_window(canvas, text_map.get(current_name, ""), next_text)

        # 状态指示灯
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
            print(f"按下数字键: {num}")  # 调试信息
            
            # 确保索引在有效范围内
            if 1 <= num <= len(rectangles):
                highlight = num - 1  # 转换为0基索引
                current_idx = highlight
                next_idx = (current_idx + 1) % len(rectangles)
                start_time = time.time()

                # 切换 Next 显示内容
                next_text = menu_map.get(num, None)
                
                # *** 新增：启动路径动画 ***
                if highlight != 6:  # 不是Handover时才显示路径
                    start_path_animation(highlight, 6)  # 到Handover的路径
                    print(f"启动路径动画: 从 {rectangles[highlight][4]} 到 Handover")
            else:
                print(f"无效的数字键: {num}")

        # 数字键 0 取消高亮
        elif key == 48:
            highlight = None
            next_text = None
            path_animation['active'] = False  # 停止路径动画
            path_animation['trail_points'] = []
            print("取消高亮和路径动画")

        # 倒计时结束 → 自动切换到下一个
        if remaining == 0:
            current_idx = (current_idx + 1) % len(rectangles)
            next_idx = (current_idx + 1) % len(rectangles)
            start_time = time.time()

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        break

cv2.destroyAllWindows()