from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='camera_node',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[],
        ),
    ])
