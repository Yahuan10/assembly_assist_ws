from rtde_receive import RTDEReceiveInterface

# IP address of the robot (must be reachable on the network)
ROBOT_IP = "192.168.0.107"

def main():
    # Create a connection to the robot's RTDE data receiver
    rtde_receive = RTDEReceiveInterface(ROBOT_IP)

    # Infinite loop to continuously read and display TCP position
    while True:
        # Get the current TCP pose from the robot
        tcp_pose = rtde_receive.getActualTCPPose()

        # Extract only the position (first three values)
        x, y, z = tcp_pose[:3]

        # Print the position of the robot’s tool center point (TCP)
        print(f"[TCP] x={x:.3f} m, y={y:.3f} m, z={z:.3f} m")

        # Wait for 0.2 seconds before reading the next data point
        time.sleep(0.2)

# Entry point of the script
# Ensures main() runs only when this script is executed directly (not imported)
if __name__ == "__main__":
    main()
