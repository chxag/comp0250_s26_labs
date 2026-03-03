from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    camera_name = LaunchConfiguration('camera_name')
    rvizconfig = LaunchConfiguration('rvizconfig')

    camera_processing = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('rpl_panda_with_rs'),
                'launch',
                'camera_proc.launch.py',
            ])
        ]),
        launch_arguments={
            'camera_name': camera_name,
            'use_sim_time': 'false',
        }.items(),
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '0.031', '-0.0285', '0.07',
            '0.7264432', '-0.0260592', '0.6857034', '0.0375764',
            'panda_hand',
            [TextSubstitution(text=''), camera_name, TextSubstitution(text='/camera_link')],
        ],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rvizconfig],
    )

    return LaunchDescription([
        DeclareLaunchArgument('camera_name', default_value='d435i'),
        DeclareLaunchArgument(
            'rvizconfig',
            default_value=PathJoinSubstitution([
                FindPackageShare('rpl_panda_with_rs'),
                'rviz',
                'urdf.rviz',
            ]),
        ),
        camera_processing,
        static_tf,
        rviz,
    ])
