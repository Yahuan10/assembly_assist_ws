import cv2
import numpy as np
import json

# 设置棋盘格参数
chessboard_size = (12, 8)  # 内角点 (13x9 需减1)
square_size = 20  # 单位：毫米 (根据实际情况调整)

# 读取棋盘格图像
image_path = r"..\log\chessboard.jpg"
chessboard_image = cv2.imread(image_path)
gray_image = cv2.cvtColor(chessboard_image, cv2.COLOR_BGR2GRAY)

# 生成实际世界坐标
obj_points = np.zeros((np.prod(chessboard_size), 3), np.float32)
obj_points[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2) * square_size
obj_points[:, 1] *= -1

# 查找棋盘格角点
found, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)

if found:
    # 计算亚像素角点
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

    # 相机标定
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj_points], [corners_refined], gray_image.shape[::-1], None, None)

    # 计算旋转矩阵
    R, _ = cv2.Rodrigues(rvecs[0])

    # 保存标定数据
    calibration_data = {
        "K": K.tolist(),
        "dist": dist.tolist(),
        "R": R.tolist(),
        "t": tvecs[0].tolist()
    }

    with open("camera_calibration.json", "w") as f:
        json.dump(calibration_data, f)

    print("相机标定完成，数据已保存到 camera_calibration.json")

    # 标注角点
    cv2.drawChessboardCorners(chessboard_image, chessboard_size, corners_refined, found)
    cv2.imshow("Chessboard Corners", chessboard_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("未检测到棋盘格，请检查图像质量！")
