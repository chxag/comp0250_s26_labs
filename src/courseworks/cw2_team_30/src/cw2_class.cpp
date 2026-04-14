/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#include <cw2_class.h>

#include <utility>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>  // for tf2::toMsg
#include <tf2/LinearMath/Quaternion.h>

cw2::cw2(const rclcpp::Node::SharedPtr &node)
: node_(node),
  tf_buffer_(node->get_clock()),
  tf_listener_(tf_buffer_),
  g_cloud_ptr(new PointC)
{
  t1_service_ = node_->create_service<cw2_world_spawner::srv::Task1Service>(
    "/task1_start",
    std::bind(&cw2::t1_callback, this, std::placeholders::_1, std::placeholders::_2));
  t2_service_ = node_->create_service<cw2_world_spawner::srv::Task2Service>(
    "/task2_start",
    std::bind(&cw2::t2_callback, this, std::placeholders::_1, std::placeholders::_2));
  t3_service_ = node_->create_service<cw2_world_spawner::srv::Task3Service>(
    "/task3_start",
    std::bind(&cw2::t3_callback, this, std::placeholders::_1, std::placeholders::_2));

  pointcloud_topic_ = node_->declare_parameter<std::string>(
    "pointcloud_topic", "/r200/camera/depth_registered/points");
  pointcloud_qos_reliable_ =
    node_->declare_parameter<bool>("pointcloud_qos_reliable", true);

  pointcloud_callback_group_ =
    node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions pointcloud_sub_options;
  pointcloud_sub_options.callback_group = pointcloud_callback_group_;

  rclcpp::QoS pointcloud_qos = rclcpp::SensorDataQoS();
  if (pointcloud_qos_reliable_) {
    pointcloud_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
  }

  color_cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
    pointcloud_topic_,
    pointcloud_qos,
    std::bind(&cw2::cloud_callback, this, std::placeholders::_1),
    pointcloud_sub_options);

  arm_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "panda_arm");
  hand_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "hand");

  arm_group_->setPlanningTime(30.0);
  arm_group_->setNumPlanningAttempts(30);
  arm_group_->setMaxVelocityScalingFactor(0.6);
  arm_group_->setMaxAccelerationScalingFactor(0.6);
  arm_group_->clearPathConstraints();

  RCLCPP_INFO(
    node_->get_logger(),
    "cw2_team_30 template initialised with pointcloud topic '%s' (%s QoS)",
    pointcloud_topic_.c_str(),
    pointcloud_qos_reliable_ ? "reliable" : "sensor-data");
}

void cw2::cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  pcl::PCLPointCloud2 pcl_cloud;
  pcl_conversions::toPCL(*msg, pcl_cloud);

  PointCPtr latest_cloud(new PointC);
  pcl::fromPCLPointCloud2(pcl_cloud, *latest_cloud);

  std::lock_guard<std::mutex> lock(cloud_mutex_);
  g_input_pc_frame_id_ = msg->header.frame_id;
  g_cloud_ptr = std::move(latest_cloud);
  ++g_cloud_sequence_;
}


void cw2::moveToPreGraspPosition(const geometry_msgs::msg::Point &object_point, const std::string &shape_type, const geometry_msgs::msg::Point &goal_point)
{
  arm_group_->setNamedTarget("ready");
  arm_group_->move();

  tf2::Quaternion orientation;
  orientation.setRPY(M_PI, 0, -M_PI / 4); // Change when T1_ANY_ORIENTATION = True

  // Step 1: Move to a pre-grasp position above the object
  geometry_msgs::msg::Pose approach_pose;
  if (shape_type == "nought") {
    approach_pose.position.x = object_point.x;
    approach_pose.position.y = object_point.y + 0.08;
  }
  else {
    approach_pose.position.x = object_point.x + 0.06;
    approach_pose.position.y = object_point.y;
  }

  approach_pose.position.z = object_point.z + 0.5;
  approach_pose.orientation = tf2::toMsg(orientation);
  arm_group_->setPoseTarget(approach_pose);
  arm_group_->move();

  // Step 2: Open the gripper
  hand_group_->setNamedTarget("open");
  hand_group_->move();

  // Step 3: Move down to grasp the object
  geometry_msgs::msg::Pose grasp_pose = approach_pose;
  grasp_pose.position.z = object_point.z + 0.15;

  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(grasp_pose);

  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = arm_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
  RCLCPP_INFO(node_->get_logger(), "Cartesian path computed with %.2f%% success", fraction * 100.0);

  if (fraction > 0.9) {
    arm_group_->execute(trajectory);
  }

  // Step 4: Close the gripper to grasp the object
  hand_group_->setNamedTarget("close");
  hand_group_->move();

  // Step 5: Lift the object
  geometry_msgs::msg::Pose lift_pose = grasp_pose;
  lift_pose.position.z = object_point.z + 0.5;
  
  waypoints.clear();
  waypoints.push_back(lift_pose);
  fraction = arm_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
  RCLCPP_INFO(node_->get_logger(), "Cartesian path computed with %.2f%% success", fraction * 100.0);

  if (fraction > 0.9) {
    arm_group_->execute(trajectory);
  }

  // Step 6: Move back to basket position
  geometry_msgs::msg::Pose basket_pose;

  if (shape_type == "nought") {
    basket_pose.position.x = goal_point.x;
    basket_pose.position.y = goal_point.y + 0.08;
  }
  else {
    basket_pose.position.x = goal_point.x;
    basket_pose.position.y = goal_point.y;
  }
  basket_pose.position.z = goal_point.z + 0.5;
  basket_pose.orientation = tf2::toMsg(orientation);
  arm_group_->setPoseTarget(basket_pose);
  arm_group_->move();

  // Step 7: Lower the object into the basket
  geometry_msgs::msg::Pose lower_pose = basket_pose;
  lower_pose.position.z = goal_point.z + 0.15;
  waypoints.clear();
  waypoints.push_back(lower_pose);
  fraction = arm_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
  RCLCPP_INFO(node_->get_logger(), "Cartesian path computed with %.2f%% success", fraction * 100.0);
  if (fraction > 0.9) {
    arm_group_->execute(trajectory);
  }

  // Step 8: Open the gripper to release the object
  hand_group_->setNamedTarget("open");
  hand_group_->move();

  // Step 9: Lift back up after releasing the object
  geometry_msgs::msg::Pose lift_pose_after = lower_pose;
  lift_pose_after.position.z = goal_point.z + 0.5;
  
  waypoints.clear();
  waypoints.push_back(lift_pose_after);
  fraction = arm_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
  RCLCPP_INFO(node_->get_logger(), "Cartesian path computed with %.2f%% success", fraction * 100.0);

  if (fraction > 0.9) {
    arm_group_->execute(trajectory);
  }

  // Step 10: Move back to the ready position
  arm_group_->setNamedTarget("ready");
  arm_group_->move();

}


void cw2::t1_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response)
{
  (void)response;

  // Step 1: Move to a pre-grasp position above the object
  moveToPreGraspPosition(request->object_point.point, request->shape_type, request->goal_point.point);


}

void cw2::t2_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task2Service::Response> response)
{
  (void)request;
  response->mystery_object_num = -1;

  std::string frame_id;
  std::size_t point_count = 0;
  std::uint64_t sequence = 0;
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    frame_id = g_input_pc_frame_id_;
    point_count = g_cloud_ptr ? g_cloud_ptr->size() : 0;
    sequence = g_cloud_sequence_;
  }

  RCLCPP_WARN(
    node_->get_logger(),
    "Task 2 is not implemented in cw2_team_30. Latest cloud: seq=%llu frame='%s' points=%zu",
    static_cast<unsigned long long>(sequence),
    frame_id.c_str(),
    point_count);
}

void cw2::t3_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  response->total_num_shapes = 0;
  response->num_most_common_shape = 0;
  response->most_common_shape_vector.clear();

  std::string frame_id;
  std::size_t point_count = 0;
  std::uint64_t sequence = 0;
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    frame_id = g_input_pc_frame_id_;
    point_count = g_cloud_ptr ? g_cloud_ptr->size() : 0;
    sequence = g_cloud_sequence_;
  }

  RCLCPP_WARN(
    node_->get_logger(),
    "Task 3 is not implemented in cw2_team_30. Latest cloud: seq=%llu frame='%s' points=%zu",
    static_cast<unsigned long long>(sequence),
    frame_id.c_str(),
    point_count);
}
