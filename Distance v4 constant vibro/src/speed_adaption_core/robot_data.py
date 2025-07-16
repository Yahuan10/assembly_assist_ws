from rtde_receive import RTDEReceiveInterface

ROBOT_IP = "192.168.0.107"  # 机器人 IP

class RobotInterface:
    def __init__(self, ip=ROBOT_IP):
        """初始化 UR5 机器人 RTDE 接口"""
        self.rtde_receive = RTDEReceiveInterface(ip)

    def get_tcp_position(self):
        """获取当前 TCP 位置，单位转换为 mm"""
        tcp_pose = self.rtde_receive.getActualTCPPose()
        return (tcp_pose[0] * 1000, tcp_pose[1] * 1000, tcp_pose[2] * 1000)

    def get_tcp_velocity(self):
        """获取当前 TCP 位置，单位转换为 mm"""
        tcp_velocity = self.rtde_receive.getActualTCPSpeed()
        return (tcp_velocity[0] * 1000, tcp_velocity[1] * 1000, tcp_velocity[2] * 1000)


