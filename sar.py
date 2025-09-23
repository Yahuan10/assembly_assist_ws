import cv2
import numpy as np

# إعدادات الشاشة
width, height = 1024, 768  # دقة البروجيكتور (4:3)

# إنشاء صورة بيضاء بالكامل
image = np.ones((height, width, 3), dtype=np.uint8) * 255

# إحداثيات البيكسل للنقاط الأربعة
px1, py1 = 102, 672
px2, py2 = 102, 384
px3, py3 = 918, 672
px4, py4 = 918, 384
px5, py5 = 918, 96
px6, py6 = 102, 96
px7, py7 = 510, 384

# رسم النقاط كدوائر حمراء
for (px, py) in [(px1, py1), (px2, py2), (px3, py3), (px4, py4), (px5, py5),(px6,py6),(px7,py7)]:
    cv2.circle(image, (px, py), radius=5, color=(0, 0, 255), thickness=-1)

# رسم المحور X (سهم)
start_x = (100, height - 50)
end_x = (300, height - 50)
cv2.arrowedLine(image, start_x, end_x, (0, 128, 0), 3, tipLength=0.05)
cv2.putText(image, 'X', (end_x[0] + 10, end_x[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)

# رسم المحور Y (سهم)
start_y = (100, height - 50)
end_y = (100, height - 250)
cv2.arrowedLine(image, start_y, end_y, (255, 0, 0), 3, tipLength=0.05)
cv2.putText(image, 'Y', (end_y[0] - 30, end_y[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255       , 0, 0), 2)

# عرض الصورة في نافذة ملء الشاشة
cv2.namedWindow("Point", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Point", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Point", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
