from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='fake_camera_node',
            executable='fake_camera_node',
            name='fake_camera_node',
            output='screen',
        ),
        Node(
            package='gesture_processor_node',
            executable='gesture_processor_node',
            name='gesture_processor_node',
            output='screen',
        ),
        Node(
            package='state_manager_node',
            executable='state_manager_node',
            name='state_manager_node',
            output='screen',
        ),
        Node(
            package='projection_node',
            executable='projection_node',
            name='projection_node',
            output='screen',
        ),
    ])
