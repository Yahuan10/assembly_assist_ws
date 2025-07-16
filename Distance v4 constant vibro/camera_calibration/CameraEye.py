import numpy as np
import cv2
import os


def calibrate_camera_from_images(chessboard_size, square_size, image_count, cali_img_path, ref_point):
    """
    通过多张棋盘格图像进行相机标定，获取世界坐标系到相机坐标系的变换矩阵。

    参数：
    - chessboard_size: (cols, rows) 棋盘格内角点数量
    - square_size: 棋盘格每个方格的实际大小 (单位：mm)
    - image_count: 需要读取的图片数量
    - cali_img_path: 棋盘格图像存储路径
    - ref_point: 最后一张图片中右上角角点的位置 (x, y)

    返回：
    - T_world2cam: 世界坐标系 → 相机坐标系的变换矩阵
    - camera_matrix: 相机内参
    - dist_coeffs: 相机畸变系数
    """
    obj_points = []  # 世界坐标系下的角点
    img_points = []  # 相机像素坐标系下的角点

    # 生成棋盘格的世界坐标（原点设为测得的右上角位置）
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2) * square_size
    objp[:, 0] = ref_point[0] - square_size - objp[:, 0]  # X 轴从右向左
    objp[:, 1] = ref_point[1] + square_size + objp[:, 1]  # Y 轴从上到下
    objp[:, 2] = 0

    for i in range(1,image_count+1):
        image_file = os.path.join(cali_img_path, f"{i:06d}.jpg")
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            obj_points.append(objp)
            img_points.append(corners)
            # 绘制棋盘格角点并显示
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow('Chessboard Detection', img)
            cv2.waitKey(500)
        else:
            print(f"未检测到角点: {i}")

    if not obj_points:
        raise ValueError("未找到有效的棋盘格图像")

    # 进行相机标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    # 取最后一张图像的外参作为参考
    R, _ = cv2.Rodrigues(rvecs[-1])
    t = tvecs[-1].reshape(3, 1)

    # 计算变换矩阵 T_world2cam
    T_world2cam = np.eye(4)
    T_world2cam[:3, :3] = R
    T_world2cam[:3, 3] = t.flatten()

    return T_world2cam, camera_matrix, dist_coeffs


# 读取本地棋盘格图片
cali_img_path = r"../data/log03282025"
chessboard_size = (12, 8)  # 角点数量（列, 行）
square_size = 20  # mm
image_count = 9  # 读取的图片数量
ref_point = (-180, 190)  # 测得的左上角角点位置

# 计算 T_world2cam
T_world2cam, camera_matrix, dist_coeffs = calibrate_camera_from_images(
    chessboard_size, square_size, image_count, cali_img_path, ref_point
)

# 保存转换矩阵
np.savez("../scripts/calibration_data.npz",
         T_world2cam=T_world2cam,
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs)
np.savez("../data/calibration/calibration_data.npz",
         T_world2cam=T_world2cam,
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs)

# 打印最终结果
print("T_world2cam:")
print(T_world2cam)
print("Camera Matrix:")
print(camera_matrix)
print("Distortion Coefficients:")
print(dist_coeffs)

print("变换矩阵已保存至 calibration_data.npz")
