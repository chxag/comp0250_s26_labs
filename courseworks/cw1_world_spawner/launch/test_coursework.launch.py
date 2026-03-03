from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cw1_world_spawner',
            executable='test_coursework.py',
            name='cw1_world_spawner_test',
            output='screen',
        )
    ])
