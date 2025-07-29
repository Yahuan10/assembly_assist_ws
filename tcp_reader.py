from rtde_receive import RTDEReceiveInterface
import time

# عنوان IP للروبوت
ROBOT_IP = "192.168.0.107"

def main():
    # إنشاء اتصال
    rtde_receive = RTDEReceiveInterface(ROBOT_IP)

    while True:
        # قراءة موضع TCP [x, y, z, Rx, Ry, Rz]
        tcp_pose = rtde_receive.getActualTCPPose()  # بالمتر والراديان

        x, y, z = tcp_pose[:3]
        print(f"[TCP] x={x:.3f} m, y={y:.3f} m, z={z:.3f} m")

        # sleep بين كل قراءة
        time.sleep(0.2)

if __name__ == "__main__":
    main()
