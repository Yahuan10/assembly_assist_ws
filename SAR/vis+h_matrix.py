import cv2
import numpy as np
from rtde_receive import RTDEReceiveInterface
import time

# إعدادات الشاشة
width, height = 1024, 768
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
win_name = "TCP Trajectory"

# مصفوفة Homography (حسب ما حسبناها أنت سابقًا)
#H = np.array([
#    [7.35135136e+02, 0., 2.03448646e+02],
#    [0., 1.02857148e+03, 1.71085702e+02],
#    [0., 0., 1.00000000e+00]
#])

H = np.array([
    [-944.336237, -87.4102614, -4.95230579],
    [-62.4513781, 874.191558,134.016380],
    [-0.00962922264, -0.142556057, 1.00000000e+00]
])

# إعداد الاتصال مع الروبوت
robot_ip = "192.168.0.107"
rtde_receive = RTDEReceiveInterface(robot_ip)

trajectory = []

def world_to_pixel(x, y):
    """تحويل إحداثيات العالم إلى بكسل باستخدام Homography"""
    point = np.array([x, y, 1.0])
    pixel = H @ point
    pixel /= pixel[2]
    return int(round(pixel[0])), int(round(pixel[1]))

# تهيئة نافذة العرض
cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # الحصول على الموضع الحالي من الروبوت
    tcp_pose = rtde_receive.getActualTCPPose()
    x, y = tcp_pose[0], tcp_pose[1]  # بالمتر
    px, py = world_to_pixel(x, y)

    print(f"[TCP] x={x:.3f} m, y={y:.3f} m → Pixel: ({px}, {py})")

    if 0 <= px < width and 0 <= py < height:
        trajectory.append((px, py))
    else:
        print("⚠️ النقطة خارج مجال العرض!")

    # إعادة ضبط الصورة البيضاء
    canvas[:] = 255

    # رسم المسار
    for pt in trajectory:
        cv2.circle(canvas, pt, 3, (0, 0, 255), -1)

    # عرض آخر نقطة بالإحداثيات
    cv2.putText(canvas, f"x={x:.3f} y={y:.3f}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # عرض الصورة
    cv2.imshow(win_name, canvas)
    if cv2.waitKey(1) == 27:
        break

    time.sleep(0.1)

cv2.destroyAllWindows()
