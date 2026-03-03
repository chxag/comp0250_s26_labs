from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def _camera_topic(camera_name, suffix):
    return [
        TextSubstitution(text='/'),
        camera_name,
        TextSubstitution(text=f'/camera/{suffix}'),
    ]


def generate_launch_description():
    camera_name = LaunchConfiguration('camera_name')
    use_sim_time = LaunchConfiguration('use_sim_time')

    rgb_camera_info = _camera_topic(camera_name, 'color/camera_info')
    depth_camera_info = _camera_topic(camera_name, 'depth/camera_info')
    rgb_img_raw = _camera_topic(camera_name, 'color/image_raw')
    depth_img_raw = _camera_topic(camera_name, 'depth/image_raw')

    rgb_img_rect = _camera_topic(camera_name, 'color/image_rect')
    depth_img_rect = _camera_topic(camera_name, 'depth/image_rect')
    depth_registered_imgrect = _camera_topic(camera_name, 'depth_registered/image_rect')
    depth_registered_camera_info = _camera_topic(camera_name, 'depth_registered/camera_info')
    out_cloud = _camera_topic(camera_name, 'depth_registered/points')

    rectify_color = ComposableNode(
        package='image_proc',
        plugin='image_proc::RectifyNode',
        name='rectify_color',
        remappings=[
            ('image', rgb_img_raw),
            ('camera_info', rgb_camera_info),
            ('image_rect', rgb_img_rect),
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    convert_metric = ComposableNode(
        package='depth_image_proc',
        plugin='depth_image_proc::ConvertMetricNode',
        name='convert_metric',
        remappings=[
            ('image_raw', depth_img_raw),
            ('image', depth_img_rect),
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    register_depth = ComposableNode(
        package='depth_image_proc',
        plugin='depth_image_proc::RegisterNode',
        name='register',
        remappings=[
            ('rgb/camera_info', rgb_camera_info),
            ('depth/camera_info', depth_camera_info),
            ('depth/image_rect', depth_img_rect),
            ('depth_registered/image_rect', depth_registered_imgrect),
            ('depth_registered/camera_info', depth_registered_camera_info),
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    point_cloud = ComposableNode(
        package='depth_image_proc',
        plugin='depth_image_proc::PointCloudXyzrgbNode',
        name='points_xyzrgb',
        remappings=[
            ('rgb/camera_info', rgb_camera_info),
            ('rgb/image_rect_color', rgb_img_rect),
            ('depth_registered/image_rect', depth_registered_imgrect),
            # ROS2 PointCloudXyzrgbNode publishes `points`; remap it to the ROS1-compatible topic.
            ('points', out_cloud),
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    container = ComposableNodeContainer(
        name='realsense_processing_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            rectify_color,
            convert_metric,
            register_depth,
            point_cloud,
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('camera_name', default_value='d435i'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        container,
    ])
