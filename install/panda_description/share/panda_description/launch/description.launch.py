from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import FindExecutable


def generate_launch_description():
    use_joint_state_publisher = LaunchConfiguration('use_joint_state_publisher')
    use_rviz = LaunchConfiguration('use_rviz')
    robot_urdf_xacro = LaunchConfiguration('robot_urdf_xacro')
    rviz_config = LaunchConfiguration('rviz_config')

    default_urdf = PathJoinSubstitution([
        FindPackageShare('panda_description'),
        'urdf',
        'panda_with_rs.urdf.xacro'
    ])

    default_rviz = PathJoinSubstitution([
        FindPackageShare('panda_description'),
        'rviz',
        'urdf.rviz'
    ])

    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', robot_urdf_xacro]),
        value_type=str
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_joint_state_publisher', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        DeclareLaunchArgument('robot_urdf_xacro', default_value=default_urdf),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),

        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            condition=IfCondition(use_joint_state_publisher),
            output='screen',
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            condition=IfCondition(use_rviz),
            output='screen',
            arguments=['-d', rviz_config],
        ),
    ])
