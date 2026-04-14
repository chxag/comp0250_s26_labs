import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression, TextSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_realsense = LaunchConfiguration('enable_realsense')
    enable_camera_processing = LaunchConfiguration('enable_camera_processing')
    camera_name = LaunchConfiguration('camera_name')
    load_gripper = LaunchConfiguration('load_gripper')
    robot_urdf_xacro = LaunchConfiguration('robot_urdf_xacro')
    robot_srdf_xacro = LaunchConfiguration('robot_srdf_xacro')
    robot_srdf_xacro_no_gripper = LaunchConfiguration('robot_srdf_xacro_no_gripper')
    rvizconfig = LaunchConfiguration('rvizconfig')
    world = LaunchConfiguration('world')
    physics_engine = LaunchConfiguration('physics_engine')
    spawn_delay = LaunchConfiguration('spawn_delay')
    controller_delay = LaunchConfiguration('controller_delay')
    load_controllers = LaunchConfiguration('load_controllers')
    controller_manager = LaunchConfiguration('controller_manager')
    control_mode = LaunchConfiguration('control_mode')
    use_rviz = LaunchConfiguration('use_rviz')
    use_gazebo_gui = LaunchConfiguration('use_gazebo_gui')
    use_software_rendering = LaunchConfiguration('use_software_rendering')
    extra_gazebo_model_path = LaunchConfiguration('extra_gazebo_model_path')
    ros2_control_params = LaunchConfiguration('ros2_control_params')

    panda_share = get_package_share_directory('panda_description')
    rpl_share = get_package_share_directory('rpl_panda_with_rs')
    realsense_share = get_package_share_directory('realsense_gazebo_plugin')
    gazebo_ros2_share = get_package_share_directory('gazebo_ros2_control')
    panda_share_parent = os.path.dirname(panda_share)
    realsense_share_parent = os.path.dirname(realsense_share)
    realsense_prefix = os.path.dirname(realsense_share_parent)
    gazebo_ros2_share_parent = os.path.dirname(gazebo_ros2_share)
    gazebo_ros2_prefix = os.path.dirname(gazebo_ros2_share_parent)
    ros2_controllers_position_path = os.path.join(rpl_share, 'config', 'ros2_controllers.yaml')
    ros2_controllers_effort_path = os.path.join(rpl_share, 'config', 'ros2_controllers_effort.yaml')
    ros2_controllers_hybrid_path = os.path.join(rpl_share, 'config', 'ros2_controllers_hybrid.yaml')
    default_ros2_control_params = PythonExpression([
        "'",
        ros2_controllers_effort_path,
        "' if '",
        control_mode,
        "' == 'effort' else ('",
        ros2_controllers_hybrid_path,
        "' if '",
        control_mode,
        "' == 'hybrid' else '",
        ros2_controllers_position_path,
        "')",
    ])
    gazebo_resource_path = (
        f"{panda_share_parent}:{realsense_share_parent}:/usr/share/gazebo-11"
    )
    gazebo_model_path = (
        f"{panda_share_parent}:{realsense_share_parent}:/usr/share/gazebo-11/models"
    )
    gazebo_plugin_path = (
        f"{os.path.join(realsense_prefix, 'lib')}"
        f":{os.path.join(gazebo_ros2_prefix, 'lib')}"
        f":/opt/ros/humble/lib:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins"
    )

    robot_description_command = Command([
        FindExecutable(name='xacro'),
        ' ',
        robot_urdf_xacro,
        ' ',
        TextSubstitution(text='use_gazebo:='),
        enable_realsense,
        ' ',
        TextSubstitution(text='sensor_prefix:='),
        camera_name,
        ' ',
        TextSubstitution(text='ros2_control_params:='),
        ros2_control_params,
    ])
    # gazebo_ros2_control in some Humble builds is fragile with multi-line XML
    # passed via parameter override; flatten newlines to a single-line string.
    robot_description = {
        'robot_description': ParameterValue(
            PythonExpression([
                "'''",
                robot_description_command,
                "'''.replace('\\n', ' ')",
            ]),
            value_type=str,
        )
    }
    moveit_config = (
        MoveItConfigsBuilder('panda', package_name='panda_moveit_config')
        .robot_description(
            file_path=os.path.join(rpl_share, 'urdf', 'panda_with_rs.xacro'),
            mappings={
                'use_gazebo': enable_realsense,
                'sensor_prefix': camera_name,
                'ros2_control_params': ros2_control_params,
            },
        )
        .robot_description_semantic(
            file_path=os.path.join(rpl_share, 'srdf', 'panda_with_rs.srdf.xacro'),
        )
        .robot_description_kinematics()
        .trajectory_execution(
            file_path=os.path.join(rpl_share, 'config', 'moveit_controllers.yaml')
        )
        .joint_limits()
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .planning_pipelines(
            pipelines=['ompl'],
            default_planning_pipeline='ompl',
            load_all=False,
        )
        .to_moveit_configs()
    )
    moveit_config_dict = moveit_config.to_dict()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}],
    )

    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gzserver.launch.py',
            ])
        ]),
        launch_arguments={
            'world': world,
            'physics': physics_engine,
        }.items(),
    )

    # Launch gzclient directly to avoid gazebo_ros EOL GUI plugin crashes under WSL/X forwarding.
    gazebo_client = ExecuteProcess(
        cmd=['gzclient'],
        output='screen',
        condition=IfCondition(use_gazebo_gui),
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=['-topic', '/robot_description', '-entity', 'panda'],
    )
    delayed_spawn = TimerAction(period=spawn_delay, actions=[spawn_entity])

    controller_spawner_js = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager',
            controller_manager,
        ],
        condition=IfCondition(load_controllers),
    )
    controller_spawner_arm = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=[
            'panda_arm_controller',
            '--controller-manager',
            controller_manager,
        ],
        condition=IfCondition(load_controllers),
    )
    controller_spawner_hand = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=[
            'panda_hand_controller',
            '--controller-manager',
            controller_manager,
        ],
        condition=IfCondition(load_controllers),
    )
    delayed_controllers = TimerAction(
        period=controller_delay,
        actions=[controller_spawner_js, controller_spawner_arm, controller_spawner_hand],
    )

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
            'use_sim_time': use_sim_time,
        }.items(),
        condition=IfCondition(enable_camera_processing),
    )

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            moveit_config_dict,
            {
                'use_sim_time': use_sim_time,
                'jiggle_fraction': 0.05,
            },
        ],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rvizconfig],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {'use_sim_time': use_sim_time},
        ],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('enable_realsense', default_value='true'),
        DeclareLaunchArgument('enable_camera_processing', default_value='true'),
        DeclareLaunchArgument('camera_name', default_value='r200'),
        DeclareLaunchArgument('load_gripper', default_value='true'),
        DeclareLaunchArgument(
            'robot_urdf_xacro',
            default_value=PathJoinSubstitution([
                FindPackageShare('rpl_panda_with_rs'),
                'urdf',
                'panda_with_rs.xacro',
            ]),
        ),
        DeclareLaunchArgument(
            'robot_srdf_xacro',
            default_value=PathJoinSubstitution([
                FindPackageShare('rpl_panda_with_rs'),
                'srdf',
                'panda_with_rs.srdf.xacro',
            ]),
        ),
        DeclareLaunchArgument(
            'robot_srdf_xacro_no_gripper',
            default_value=PathJoinSubstitution([
                FindPackageShare('panda_moveit_config'),
                'config',
                'panda_arm.srdf.xacro',
            ]),
        ),
        DeclareLaunchArgument('control_mode', default_value='effort'),
        DeclareLaunchArgument(
            'ros2_control_params',
            default_value=default_ros2_control_params,
        ),
        DeclareLaunchArgument(
            'rvizconfig',
            default_value=PathJoinSubstitution([
                FindPackageShare('rpl_panda_with_rs'),
                'rviz',
                'urdf.rviz',
            ]),
        ),
        DeclareLaunchArgument(
            'world',
            default_value=PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'worlds',
                'empty.world',
            ]),
        ),
        DeclareLaunchArgument('physics_engine', default_value='ode'),
        DeclareLaunchArgument('spawn_delay', default_value='3.0'),
        DeclareLaunchArgument('controller_delay', default_value='0.0'),
        DeclareLaunchArgument('load_controllers', default_value='true'),
        DeclareLaunchArgument('controller_manager', default_value='/controller_manager'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_gazebo_gui', default_value='true'),
        DeclareLaunchArgument('use_software_rendering', default_value='false'),
        DeclareLaunchArgument('extra_gazebo_model_path', default_value=''),
        SetEnvironmentVariable('GAZEBO_RESOURCE_PATH', gazebo_resource_path),
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', [gazebo_model_path, ':', extra_gazebo_model_path]),
        SetEnvironmentVariable('GAZEBO_PLUGIN_PATH', gazebo_plugin_path),
        SetEnvironmentVariable('GAZEBO_MASTER_URI', 'http://localhost:11345'),
        SetEnvironmentVariable('OGRE_RESOURCE_PATH', '/usr/lib/x86_64-linux-gnu/OGRE-1.9.0'),
        SetEnvironmentVariable('GAZEBO_RENDER_ENGINE', 'ogre'),
        SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '1',
                               condition=IfCondition(use_software_rendering)),
        gazebo_server,
        gazebo_client,
        robot_state_publisher,
        delayed_spawn,
        delayed_controllers,
        move_group,
        camera_processing,
        rviz,
    ])
