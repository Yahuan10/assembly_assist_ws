from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='projection_node',
            executable='projection_node',
            name='projection_node',
            output='screen',
        ),
    ])
