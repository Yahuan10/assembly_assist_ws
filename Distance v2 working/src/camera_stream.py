import pyrealsense2 as rs
import numpy as np
import cv2
import time
import sys
import os
import copy

def save_data(save_path, color_img, depth_img, img_cnt, pose = None):
    img_cnt_str_len = len(str(img_cnt))
    cv2.imwrite(os.path.join(save_path, "{}.jpg".format("0" * (6 - img_cnt_str_len) + str(img_cnt))),color_img)
    np.save(os.path.join(    save_path, "{}.npy".format("0" * (6 - img_cnt_str_len) + str(img_cnt))),depth_img)
    print(f"count of images: {img_cnt}, in {save_path}" )
    if pose is None:
        pass
    else:
        np.save(os.path.join(save_path, "{}_pose.npy".format("0" * (6 - img_cnt_str_len) + str(img_cnt))),pose)
    pass

# Start streaming
camera_serial = '317222071793'

pipeline = rs.pipeline()
config = rs.config()
config.enable_device(camera_serial)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
# 设定需要对齐的方式（这里是深度对齐彩色，彩色图不变，深度图变换）
align_to = rs.stream.color
alignedFs = rs.align(align_to)

# for saving
log_save_path = r"../data/log03282025"
os.makedirs(log_save_path, exist_ok=True)
data_save_path = r"D:/Abschluss/data"
os.makedirs(data_save_path, exist_ok=True)
data_set_cnt = 1
saving = False
i_save = 0
frame_count = 0
save_interval = 1 / 24  # 每秒保存 10 帧（根据需要调整）
last_save_time = time.time()
img_cnt = 1


try:
    # 初始化跟踪器
    tracker = cv2.TrackerMIL().create()
    # tracker = cv2.TrackerNano().create()
    enable_tracker = False
    start_tick = cv2.getTickCount()
    last_frame_time = time.time()
    while True:        

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = alignedFs.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # 获取相机内参
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays 把图像转换为numpy data
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        ori_img = color_image
        ori_img_vergin = copy.deepcopy(ori_img)

        # add fps, show image
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - last_frame_time)
        last_frame_time = current_time

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first) 在深度图上用颜色渲染
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        # Add FPS text to frames
        cv2.putText(images, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Show images 展示一下图片
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        # stream save
        if saving and (time.time() - last_save_time) >= save_interval:
            save_data(data_set_save_path, ori_img_vergin, depth_image, frame_count)
            frame_count += 1
            last_save_time = time.time()
 
        key = cv2.waitKey(1)

        # exit video
        if key == ord(' '):
            break

        # Save Single picture
        if key == ord('s'):
            # save picture as log
            save_data(log_save_path, ori_img_vergin, depth_image, img_cnt)
            img_cnt = img_cnt + 1
            pass

        if key == ord('r'):
            if not saving:
                print("start streaming saving")
                data_set_save_path = os.path.join(data_save_path, f"idx_{i_save:04d}")
                os.makedirs(data_set_save_path, exist_ok=True)
                saving = True
                frame_count = 0
                last_save_time = time.time()

        if key == ord('e'):
            if saving:
                print("stop streaming saving")
                i_save += 1
                saving = False
finally:
    # Stop streaming
    pipeline.stop()
    pass