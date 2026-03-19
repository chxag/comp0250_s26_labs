from ament_index_python.packages import get_package_share_directory
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    launch_delay = LaunchConfiguration('launch_delay')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_gazebo_gui = LaunchConfiguration('use_gazebo_gui')
    use_rviz = LaunchConfiguration('use_rviz')
    enable_realsense = LaunchConfiguration('enable_realsense')
    enable_camera_processing = LaunchConfiguration('enable_camera_processing')
    control_mode = LaunchConfiguration('control_mode')
    physics_engine = LaunchConfiguration('physics_engine')
    use_software_rendering = LaunchConfiguration('use_software_rendering')
    enable_cloud_viewer = LaunchConfiguration('enable_cloud_viewer')
    move_home_on_start = LaunchConfiguration('move_home_on_start')
    use_cartesian_reach = LaunchConfiguration('use_cartesian_reach')
    allow_position_only_fallback = LaunchConfiguration('allow_position_only_fallback')
    cartesian_eef_step = LaunchConfiguration('cartesian_eef_step')
    cartesian_jump_threshold = LaunchConfiguration('cartesian_jump_threshold')
    cartesian_min_fraction = LaunchConfiguration('cartesian_min_fraction')
    enable_task1_snap = LaunchConfiguration('enable_task1_snap')
    return_home_between_pick_place = LaunchConfiguration('return_home_between_pick_place')
    return_home_after_pick_place = LaunchConfiguration('return_home_after_pick_place')
    pick_offset_z = LaunchConfiguration('pick_offset_z')
    task3_pick_offset_z = LaunchConfiguration('task3_pick_offset_z')
    place_offset_z = LaunchConfiguration('place_offset_z')
    grasp_approach_offset_z = LaunchConfiguration('grasp_approach_offset_z')
    post_grasp_lift_z = LaunchConfiguration('post_grasp_lift_z')
    gripper_grasp_width = LaunchConfiguration('gripper_grasp_width')
    task2_capture_enabled = LaunchConfiguration('task2_capture_enabled')
    task2_capture_dir = LaunchConfiguration('task2_capture_dir')
    panda_moveit_share = get_package_share_directory('panda_moveit_config')
    with open(f'{panda_moveit_share}/config/kinematics.yaml', 'r', encoding='utf-8') as stream:
        moveit_kinematics = {
            'robot_description_kinematics': yaml.safe_load(stream) or {}
        }

    return LaunchDescription([
        DeclareLaunchArgument('launch_delay', default_value='5.0'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('use_gazebo_gui', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('enable_realsense', default_value='true'),
        DeclareLaunchArgument('enable_camera_processing', default_value='true'),
        DeclareLaunchArgument('control_mode', default_value='effort'),
        DeclareLaunchArgument('physics_engine', default_value='ode'),
        DeclareLaunchArgument('use_software_rendering', default_value='false'),
        DeclareLaunchArgument('enable_cloud_viewer', default_value='false'),
        DeclareLaunchArgument('move_home_on_start', default_value='false'),
        DeclareLaunchArgument('use_cartesian_reach', default_value='true'),
        DeclareLaunchArgument('allow_position_only_fallback', default_value='false'),
        DeclareLaunchArgument('cartesian_eef_step', default_value='0.005'),
        DeclareLaunchArgument('cartesian_jump_threshold', default_value='0.0'),
        DeclareLaunchArgument('cartesian_min_fraction', default_value='0.98'),
        DeclareLaunchArgument('enable_task1_snap', default_value='false'),
        DeclareLaunchArgument('return_home_between_pick_place', default_value='false'),
        DeclareLaunchArgument('return_home_after_pick_place', default_value='false'),
        DeclareLaunchArgument('pick_offset_z', default_value='0.12'),
        DeclareLaunchArgument('task3_pick_offset_z', default_value='0.13'),
        DeclareLaunchArgument('place_offset_z', default_value='0.35'),
        DeclareLaunchArgument('grasp_approach_offset_z', default_value='0.015'),
        DeclareLaunchArgument('post_grasp_lift_z', default_value='0.05'),
        DeclareLaunchArgument('gripper_grasp_width', default_value='0.03'),
        DeclareLaunchArgument('task2_capture_enabled', default_value='false'),
        DeclareLaunchArgument('task2_capture_dir', default_value='/tmp/cw1_task2_capture'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare('rpl_panda_with_rs'),
                '/launch/display.launch.py'
            ]),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'use_gazebo_gui': use_gazebo_gui,
                'use_rviz': use_rviz,
                'enable_realsense': enable_realsense,
                'enable_camera_processing': enable_camera_processing,
                'control_mode': control_mode,
                'physics_engine': physics_engine,
                'use_software_rendering': use_software_rendering,
            }.items()
        ),

        TimerAction(
            period=launch_delay,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        FindPackageShare('cw1_world_spawner'),
                        '/launch/world_spawner.launch.py'
                    ])
                )
            ]
        ),

        Node(
            package='cw1_team_x',
            executable='cw1_solution_node',
            name='cw1_solution_node',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'enable_cloud_viewer': enable_cloud_viewer,
                'use_gazebo_gui': use_gazebo_gui,
                'move_home_on_start': move_home_on_start,
                'use_cartesian_reach': use_cartesian_reach,
                'allow_position_only_fallback': allow_position_only_fallback,
                'cartesian_eef_step': cartesian_eef_step,
                'cartesian_jump_threshold': cartesian_jump_threshold,
                'cartesian_min_fraction': cartesian_min_fraction,
                'enable_task1_snap': enable_task1_snap,
                'return_home_between_pick_place': return_home_between_pick_place,
                'return_home_after_pick_place': return_home_after_pick_place,
                'pick_offset_z': pick_offset_z,
                'task3_pick_offset_z': task3_pick_offset_z,
                'place_offset_z': place_offset_z,
                'grasp_approach_offset_z': grasp_approach_offset_z,
                'post_grasp_lift_z': post_grasp_lift_z,
                'gripper_grasp_width': gripper_grasp_width,
                'task2_capture_enabled': task2_capture_enabled,
                'task2_capture_dir': task2_capture_dir,
            },
            moveit_kinematics,
            ],
        ),
    ])
