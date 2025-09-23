import numpy as np
import cv2

def draw_rectangle(canvas, x_real, y_real, width_real, height_real, 
                  color=(255, 255, 255), scale_inv_x=None, text=None):
    """
    在图像上绘制矩形，支持实际坐标到像素坐标的转换
    
    参数:
        canvas: 输入画布
        x_real, y_real: 矩形左上角的实际坐标
        width_real, height_real: 矩形的实际宽度和高度
        color: 矩形颜色 (B, G, R)，文字颜色将与矩形颜色相同
        scale_inv_x: 坐标转换矩阵
        text: 要显示的文字（可选）
    
    返回:
        绘制了矩形的图像
    """
    
    # 如果没有提供比例尺，则根据画布尺寸计算
    if scale_inv_x is None:
        w_virtual = canvas.shape[1]  # 获取画布宽度（像素）
        w_real = 100  # 假设实际宽度为100cm
        scale_inv_x = w_virtual / w_real
    
    # 将现实世界的厘米坐标转换为像素坐标
    x_pixel = int(x_real * scale_inv_x)
    y_pixel = int(y_real * scale_inv_x)
    
    # 将现实世界的尺寸转换为像素尺寸
    width_pixel = int(width_real * scale_inv_x)
    height_pixel = int(height_real * scale_inv_x)
    
    # 计算矩形结束点的像素坐标
    end_x_pixel = x_pixel + width_pixel
    end_y_pixel = y_pixel + height_pixel
    
    # 绘制矩形框，粗细固定为3
    cv2.rectangle(canvas, 
                  (x_pixel, y_pixel), 
                  (end_x_pixel, end_y_pixel), 
                  color, 
                  3)
    
    # 如果提供了文字，则在矩形下方绘制文字
    if text is not None:
        # 设置字体
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.5  # 固定文字大小
        text_offset = 5  # 固定文字偏移
        
        # 获取文字尺寸，用于居中对齐
        text_size = cv2.getTextSize(text, font, font_size, 1)[0]
        
        # 计算文字位置（在矩形底部居中）
        text_x = x_pixel + (width_pixel - text_size[0]) // 2
        text_y = end_y_pixel + text_offset + text_size[1]
        
        # 确保文字不会超出画布边界
        if text_x < 0:
            text_x = 0
        elif text_x + text_size[0] > canvas.shape[1]:
            text_x = canvas.shape[1] - text_size[0]
        
        if text_y > canvas.shape[0]:
            text_y = canvas.shape[0] - 5
        
        # 绘制文字，颜色与矩形颜色相同
        cv2.putText(canvas, text, (text_x, text_y), font, font_size, color, 1, cv2.LINE_AA)
    
    return canvas

# --- 步骤 1: 定义画布和比例尺 ---
# 虚拟画布（OpenCV）尺寸
h_virtual, w_virtual = 768, 1024

# 现实世界画布尺寸（单位：厘米）
h_real, w_real = 68.0, 103.5 

# 创建一个白色的OpenCV画布
# np.ones() 创建一个用1填充的数组，乘以255使其变为白色
canvas = np.ones((h_virtual, w_virtual, 3), dtype='uint8') * 255

# 计算X方向的反向比例尺（像素/厘米）
scale_inv_x = w_virtual / w_real

# --- 步骤 2: 使用封装好的函数绘制多个带文字的矩形 ---

# 绘制第一个绿色矩形，带有绿色标签
canvas = draw_rectangle(canvas, x_real=12.0, y_real=0.0, 
                       width_real=28.0, height_real=21.0, 
                       color=(0, 255, 0), scale_inv_x=scale_inv_x,
                       text="motor")

canvas = draw_rectangle(canvas, x_real=9.0, y_real=31.0, 
                       width_real=35.0, height_real=6.0, 
                       color=(0, 255, 0), scale_inv_x=scale_inv_x,
                       text="PCB2")



# --- 步骤 3: 显示结果 ---
cv2.imshow('Canvas with Multiple Rectangles', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()