/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire 
solution is contained within the cw1_team_<your_team_number> package */

#include <cw1_class.h>

#include <functional>
#include <memory>
#include <utility>
#include <cmath>
#include <chrono>
#include <thread>
#include <queue>
#include <algorithm>

#include <rmw/qos_profiles.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

///////////////////////////////////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////////////////////////////////

cw1::cw1(const rclcpp::Node::SharedPtr &node)
{
  node_ = node;
  service_cb_group_ = node_->create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);
  sensor_cb_group_ = node_->create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);

  t1_service_ = node_->create_service<cw1_world_spawner::srv::Task1Service>(
    "/task1_start",
    std::bind(&cw1::t1_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, service_cb_group_);
  t2_service_ = node_->create_service<cw1_world_spawner::srv::Task2Service>(
    "/task2_start",
    std::bind(&cw1::t2_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, service_cb_group_);
  t3_service_ = node_->create_service<cw1_world_spawner::srv::Task3Service>(
    "/task3_start",
    std::bind(&cw1::t3_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, service_cb_group_);

  rclcpp::SubscriptionOptions joint_state_sub_options;
  joint_state_sub_options.callback_group = sensor_cb_group_;
  auto joint_state_qos = rclcpp::QoS(rclcpp::KeepLast(50));
  joint_state_qos.reliable();
  joint_state_qos.durability_volatile();
  joint_state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", joint_state_qos,
    [this](const sensor_msgs::msg::JointState::ConstSharedPtr msg) {
      const int64_t stamp_ns =
        static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
        static_cast<int64_t>(msg->header.stamp.nanosec);
      latest_joint_state_stamp_ns_.store(stamp_ns, std::memory_order_relaxed);
      joint_state_msg_count_.fetch_add(1, std::memory_order_relaxed);
    },
    joint_state_sub_options);

  rclcpp::SubscriptionOptions cloud_sub_options;
  cloud_sub_options.callback_group = sensor_cb_group_;
  auto cloud_qos = rclcpp::QoS(rclcpp::KeepLast(10));
  cloud_qos.reliable();
  cloud_qos.durability_volatile();
  cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/r200/camera/depth_registered/points", cloud_qos,
    [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
      const int64_t stamp_ns =
        static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
        static_cast<int64_t>(msg->header.stamp.nanosec);
      latest_cloud_stamp_ns_.store(stamp_ns, std::memory_order_relaxed);
      cloud_msg_count_.fetch_add(1, std::memory_order_relaxed);
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      latest_cloud_ = msg;
    },
    cloud_sub_options);

  const bool use_gazebo_gui = node_->declare_parameter<bool>("use_gazebo_gui", true);
  (void)use_gazebo_gui;
  enable_cloud_viewer_ = node_->declare_parameter<bool>("enable_cloud_viewer", false);
  move_home_on_start_ = node_->declare_parameter<bool>("move_home_on_start", false);
  use_path_constraints_ = node_->declare_parameter<bool>("use_path_constraints", false);
  use_cartesian_reach_ = node_->declare_parameter<bool>("use_cartesian_reach", false);
  allow_position_only_fallback_ = node_->declare_parameter<bool>(
    "allow_position_only_fallback", allow_position_only_fallback_);
  cartesian_eef_step_ = node_->declare_parameter<double>("cartesian_eef_step", cartesian_eef_step_);
  cartesian_jump_threshold_ = node_->declare_parameter<double>("cartesian_jump_threshold", cartesian_jump_threshold_);
  cartesian_min_fraction_ = node_->declare_parameter<double>("cartesian_min_fraction", cartesian_min_fraction_);
  publish_programmatic_debug_ = node_->declare_parameter<bool>("publish_programmatic_debug", publish_programmatic_debug_);
  enable_task1_snap_ = node_->declare_parameter<bool>("enable_task1_snap", false);
  return_home_between_pick_place_ = node_->declare_parameter<bool>("return_home_between_pick_place", return_home_between_pick_place_);
  return_home_after_pick_place_ = node_->declare_parameter<bool>("return_home_after_pick_place", return_home_after_pick_place_);
  pick_offset_z_ = node_->declare_parameter<double>("pick_offset_z", pick_offset_z_);
  task3_pick_offset_z_ = node_->declare_parameter<double>("task3_pick_offset_z", task3_pick_offset_z_);
  task2_capture_enabled_ = node_->declare_parameter<bool>("task2_capture_enabled", task2_capture_enabled_);
  task2_capture_dir_ = node_->declare_parameter<std::string>("task2_capture_dir", task2_capture_dir_);
  place_offset_z_ = node_->declare_parameter<double>("place_offset_z", place_offset_z_);
  grasp_approach_offset_z_ = node_->declare_parameter<double>("grasp_approach_offset_z", grasp_approach_offset_z_);
  post_grasp_lift_z_ = node_->declare_parameter<double>("post_grasp_lift_z", post_grasp_lift_z_);
  gripper_grasp_width_ = node_->declare_parameter<double>("gripper_grasp_width", gripper_grasp_width_);
  joint_state_wait_timeout_sec_ = node_->declare_parameter<double>("joint_state_wait_timeout_sec", joint_state_wait_timeout_sec_);

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  RCLCPP_INFO(node_->get_logger(), "cw1 template class initialised with compatibility scaffold");

  arm_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "panda_arm");
  hand_group = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "hand");
  arm_group->setPlanningTime(10.0);
  arm_group->setMaxVelocityScalingFactor(0.3);
}

///////////////////////////////////////////////////////////////////////////////
// Helper functions for motion + Task 1
///////////////////////////////////////////////////////////////////////////////

static bool execJoints(
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> & arm_group,
  const std::vector<double> & joints)
{
  arm_group->setStartStateToCurrentState();
  arm_group->setJointValueTarget(joints);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool ok = (arm_group->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (ok) ok = (arm_group->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  return ok;
}

bool cw1::moveToLiftXY(double x, double y)
{
  const double r  = std::sqrt(x*x + y*y);
  const double j1 = std::atan2(y, x);
  const double j2 = 3.267*r - 1.784;
  const double j4 = 3.286*r - 3.457;
  return execJoints(arm_group, {j1, j2, 0.0, j4, 0.0, 1.571, 0.785});
}

bool cw1::moveToGraspZ(double x, double y, double z)
{
  arm_group->setStartStateToCurrentState();

  const double r      = std::sqrt(x*x + y*y);
  const double offset = 0.03;
  const double gx     = x + offset * (x / r);
  const double gy     = y + offset * (y / r);

  geometry_msgs::msg::PoseStamped current_stamped = arm_group->getCurrentPose();
  geometry_msgs::msg::Pose target;
  target.position.x  = gx;
  target.position.y  = gy;
  target.position.z  = z;
  target.orientation = current_stamped.pose.orientation;

  std::vector<geometry_msgs::msg::Pose> waypoints = {target};
  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = arm_group->computeCartesianPath(waypoints, 0.005, 0.0, trajectory);

  if (fraction < 0.9) {
    RCLCPP_WARN(node_->get_logger(),
      "moveToGraspZ: Cartesian path %.0f%% — retrying with pose target", fraction * 100.0);
    arm_group->setGoalOrientationTolerance(0.05);
    arm_group->setGoalPositionTolerance(0.01);
    arm_group->setPoseTarget(target);
    moveit::planning_interface::MoveGroupInterface::Plan p;
    bool ok2 = (arm_group->plan(p) == moveit::core::MoveItErrorCode::SUCCESS);
    if (ok2) ok2 = (arm_group->execute(p) == moveit::core::MoveItErrorCode::SUCCESS);
    return ok2;
  }

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  plan.trajectory_ = trajectory;
  return (arm_group->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
}

bool cw1::moveToPose(const geometry_msgs::msg::Pose target_pose)
{
  arm_group->setStartStateToCurrentState();
  arm_group->setGoalOrientationTolerance(0.01);
  arm_group->setGoalPositionTolerance(0.01);
  arm_group->setPoseTarget(target_pose);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool success = (arm_group->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (success) success = (arm_group->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  return success;
}

bool cw1::setGripper(double width)
{
  hand_group->setJointValueTarget("panda_finger_joint1", width);
  hand_group->setJointValueTarget("panda_finger_joint2", width);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool success = (hand_group->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (success) success = (hand_group->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  return success;
}

///////////////////////////////////////////////////////////////////////////////
// Task 1
///////////////////////////////////////////////////////////////////////////////

void cw1::t1_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task1Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task1Service::Response> response)
{
  (void)response;

  const double obj_x  = request->object_loc.pose.position.x;
  const double obj_y  = request->object_loc.pose.position.y;
  const double goal_x = request->goal_loc.point.x;
  const double goal_y = request->goal_loc.point.y;

  RCLCPP_INFO(node_->get_logger(),
    "Task 1: cube(%.3f,%.3f) basket(%.3f,%.3f)", obj_x, obj_y, goal_x, goal_y);

  // 1. Open gripper
  setGripper(0.04);

  // 2. Lift above cube
  moveToLiftXY(obj_x, obj_y);

  // 3. Descend to grasp height
  moveToGraspZ(obj_x, obj_y, 0.1434);

  // 4. Close gripper
  setGripper(0.010);

  // 5. Lift back up
  moveToLiftXY(obj_x, obj_y);

  // 6. Move above basket (push 3cm outward to clear near rim)
  {
    const double gr = std::sqrt(goal_x * goal_x + goal_y * goal_y);
    const double drop_outward = 0.02;
    moveToLiftXY(goal_x + drop_outward * (goal_x / gr),
                 goal_y + drop_outward * (goal_y / gr));
  }

  // 7. Release
  setGripper(0.04);

  RCLCPP_INFO(node_->get_logger(), "Task 1 done.");
}

///////////////////////////////////////////////////////////////////////////////
// Task 2 helpers  (KEPT EXACTLY FROM DOC 10)
///////////////////////////////////////////////////////////////////////////////

sensor_msgs::msg::PointCloud2::ConstSharedPtr cw1::waitForCloud(double timeout_sec)
{
  const auto start_time = node_->now();
  const uint64_t start_count = cloud_msg_count_.load(std::memory_order_relaxed);

  while (rclcpp::ok()) {
    if (cloud_msg_count_.load(std::memory_order_relaxed) > start_count) {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      return latest_cloud_;
    }
    if ((node_->now() - start_time).seconds() > timeout_sec) {
      return nullptr;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return nullptr;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cw1::cropAroundBasket(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
  const geometry_msgs::msg::PointStamped & basket_world_loc)
{
  auto cropped_cloud_ptr = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  if (!cloud) return cropped_cloud_ptr;
  auto stamped_basket_loc = basket_world_loc;
  stamped_basket_loc.header.stamp = cloud->header.stamp;

  geometry_msgs::msg::PointStamped basket_in_cam = transformToCameraFrame(stamped_basket_loc, cloud->header.frame_id);
  if (basket_in_cam.header.frame_id.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "Failed to transform basket location to camera frame");
    return cropped_cloud_ptr;
  }

  const float basket_x = static_cast<float>(basket_in_cam.point.x);
  const float basket_y = static_cast<float>(basket_in_cam.point.y);
  const float basket_z = static_cast<float>(basket_in_cam.point.z);

  RCLCPP_INFO(node_->get_logger(),
    "Cloud frame=%s basket_cam=(%.3f, %.3f, %.3f)",
    cloud->header.frame_id.c_str(), basket_x, basket_y, basket_z);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::fromROSMsg(*cloud, *pcl_cloud);

  constexpr float crop_radius = 0.08f;
  constexpr float crop_radius_sq = crop_radius * crop_radius;
  const float min_z = basket_z - 0.15f;
  const float max_z = basket_z + 0.06f;

  for (const auto & pt : pcl_cloud->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
    if (pt.z < min_z || pt.z > max_z) continue;
    float dx = pt.x - basket_x;
    float dy = pt.y - basket_y;
    if (dx*dx + dy*dy > crop_radius_sq) continue;
    cropped_cloud_ptr->points.push_back(pt);
  }

  cropped_cloud_ptr->width  = static_cast<uint32_t>(cropped_cloud_ptr->points.size());
  cropped_cloud_ptr->height = 1;
  cropped_cloud_ptr->is_dense = false;
  return cropped_cloud_ptr;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cw1::removeNoiseAndFloor(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud)
{
  auto cleaned_cloud_ptr = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  if (!cloud || cloud->points.empty()) return cleaned_cloud_ptr;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sor_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(20);
  sor.setStddevMulThresh(1.0);
  sor.filter(*sor_cloud);

  if (sor_cloud->points.empty()) return cleaned_cloud_ptr;

  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100);
  seg.setDistanceThreshold(0.01);
  seg.setInputCloud(sor_cloud);
  seg.segment(*inliers, *coefficients);

  if (inliers->indices.empty()) {
    RCLCPP_WARN(node_->get_logger(), "No plane found in point cloud");
    return sor_cloud;
  }

  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
  extract.setInputCloud(sor_cloud);
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*cleaned_cloud_ptr);
  return cleaned_cloud_ptr;
}

geometry_msgs::msg::PointStamped cw1::transformToCameraFrame(
  const geometry_msgs::msg::PointStamped & point_in_world,
  const std::string & target_frame)
{
  geometry_msgs::msg::PointStamped point_in_cam = point_in_world;
  try {
    point_in_cam = tf_buffer_->transform(point_in_world, target_frame, tf2::durationFromSec(0.5));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(node_->get_logger(), "TF transform failed: %s", ex.what());
    point_in_cam.header.frame_id = "";
  }
  return point_in_cam;
}

std::string cw1::detectBasketColour(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
  const geometry_msgs::msg::PointStamped & basket_world_loc)
{
  if (!cloud) return "none";
  auto cropped_cloud = cropAroundBasket(cloud, basket_world_loc);

  if (cropped_cloud->points.empty()) {
    RCLCPP_WARN(node_->get_logger(), "No points found around basket location");
    return "none";
  }
  auto basket_cloud = removeNoiseAndFloor(cropped_cloud);

  if (basket_cloud->points.empty()) {
    RCLCPP_WARN(node_->get_logger(), "No valid points left after noise/floor removal");
    return "none";
  }

  double red_sum = 0.0, green_sum = 0.0, blue_sum = 0.0;
  int count = 0;

  for (const auto & pt : basket_cloud->points) {
    const double red   = static_cast<double>(pt.r) / 255.0;
    const double green = static_cast<double>(pt.g) / 255.0;
    const double blue  = static_cast<double>(pt.b) / 255.0;
    if (std::max({red, green, blue}) < 0.25) continue;
    red_sum   += red;
    green_sum += green;
    blue_sum  += blue;
    count++;
  }

  if (count < 20) {
    RCLCPP_WARN(node_->get_logger(), "Not enough valid points to determine colour (only %d)", count);
    return "none";
  }

  const double red_avg   = red_sum   / count;
  const double green_avg = green_sum / count;
  const double blue_avg  = blue_sum  / count;

  struct KnownColour { const char * name; double r, g, b; };
  const KnownColour known_colours[] = {
    {"red",    0.8, 0.1, 0.1},
    {"blue",   0.1, 0.1, 0.8},
    {"purple", 0.8, 0.1, 0.8},
  };

  double best_dist = 1e9;
  std::string best_colour = "none";
  for (const auto & kc : known_colours) {
    double dr = red_avg - kc.r, dg = green_avg - kc.g, db = blue_avg - kc.b;
    double dist = std::sqrt(dr*dr + dg*dg + db*db);
    if (dist < best_dist) { best_dist = dist; best_colour = kc.name; }
  }

  if (best_dist > 0.4) {
    RCLCPP_WARN(node_->get_logger(),
      "Detected colour is far from known colours (distance %.3f) — returning 'none'", best_dist);
    return "none";
  }
  return best_colour;
}

///////////////////////////////////////////////////////////////////////////////
// Task 2 callback  (KEPT EXACTLY FROM DOC 10)
///////////////////////////////////////////////////////////////////////////////

void cw1::t2_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task2Service::Response> response)
{
  RCLCPP_INFO(node_->get_logger(), "Task 2 started: scanning baskets");

  const auto & basket_locs = request->basket_locs;
  std::vector<std::string> basket_colours;

  for (size_t i = 0; i < basket_locs.size(); ++i) {
    auto basket_loc = basket_locs[i];
    if (basket_loc.header.frame_id.empty()) {
      basket_loc.header.frame_id = "world";
    }

    RCLCPP_INFO(node_->get_logger(),
      "Scanning basket %zu at (%.3f, %.3f)", i+1, basket_loc.point.x, basket_loc.point.y);

    if (!moveToLiftXY(basket_loc.point.x, basket_loc.point.y)) {
      RCLCPP_ERROR(node_->get_logger(),
        "Failed to move above basket %zu — skipping colour detection", i+1);
      basket_colours.push_back("none");
      continue;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(400));

    auto fresh_cloud = waitForCloud(5.0);
    if (!fresh_cloud) {
      RCLCPP_ERROR(node_->get_logger(),
        "No fresh cloud received after moving above basket %zu — skipping colour detection", i+1);
      basket_colours.push_back("none");
      continue;
    }

    std::string colour = detectBasketColour(fresh_cloud, basket_loc);
    RCLCPP_INFO(node_->get_logger(), "Detected colour for basket %zu: %s", i+1, colour.c_str());
    basket_colours.push_back(colour);
  }

  response->basket_colours = basket_colours;
  RCLCPP_INFO(node_->get_logger(), "Task 2 done.");
}

///////////////////////////////////////////////////////////////////////////////
// Task 3 — scan scene, detect objects by colour, pick and place
///////////////////////////////////////////////////////////////////////////////

struct DetectedObject {
  std::string colour;
  std::string type;
  double x, y, z;
};

static std::vector<pcl::PointIndices> pclEuclideanClusters(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
  float tolerance, int min_pts, int max_pts)
{
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(
    new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud(cloud);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance(tolerance);
  ec.setMinClusterSize(min_pts);
  ec.setMaxClusterSize(max_pts);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);
  return cluster_indices;
}

void cw1::t3_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  (void)response;

  RCLCPP_INFO(node_->get_logger(), "Task 3 started: scanning scene");

  // ── Step 1: Multi-position scan ──────────────────────────────────────────
  geometry_msgs::msg::Pose scan_pose;
  scan_pose.orientation.x = 1.0;
  scan_pose.orientation.y = 0.0;
  scan_pose.orientation.z = 0.0;
  scan_pose.orientation.w = 0.0;

  const std::vector<std::pair<double,double>> scan_positions = {
    {0.40,  0.30},
    {0.40, -0.30},
    {0.55,  0.35},
    {0.55, -0.35},
    {0.55,  0.00},
  };

  pcl::PointCloud<pcl::PointXYZRGB> combined_colour_cloud;

  struct ColRef { float r, g, b; };
  const ColRef col_refs[] = {
    {0.8f, 0.1f, 0.1f},
    {0.1f, 0.1f, 0.8f},
    {0.8f, 0.1f, 0.8f},
  };
  const float max_col_dist_sq = 0.25f * 0.25f * 3.0f;

  for (const auto & pos : scan_positions) {
    scan_pose.position.x = pos.first;
    scan_pose.position.y = pos.second;
    scan_pose.position.z = 0.60;
    moveToPose(scan_pose);

    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    cloud_msg_count_.store(0, std::memory_order_relaxed);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto cm = waitForCloud(5.0);
    if (!cm) continue;

    pcl::PointCloud<pcl::PointXYZRGB> raw;
    pcl::fromROSMsg(*cm, raw);

    for (const auto & pt : raw.points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
      if (pt.z < 0.30f || pt.z > 0.58f) continue;
      float r = pt.r/255.f, g = pt.g/255.f, b = pt.b/255.f;
      if (std::max({r,g,b}) < 0.20f) continue;
      bool match = false;
      for (const auto & ref : col_refs) {
        float dr=r-ref.r, dg=g-ref.g, db=b-ref.b;
        if (dr*dr+dg*dg+db*db < max_col_dist_sq) { match=true; break; }
      }
      if (!match) continue;

      geometry_msgs::msg::PointStamped cam_pt, world_pt;
      cam_pt.header.frame_id = cm->header.frame_id;
      cam_pt.header.stamp    = cm->header.stamp;
      cam_pt.point.x = pt.x; cam_pt.point.y = pt.y; cam_pt.point.z = pt.z;
      try {
        world_pt = tf_buffer_->transform(cam_pt, "world", tf2::durationFromSec(0.5));
      } catch (...) { continue; }

      pcl::PointXYZRGB wpt;
      wpt.x = static_cast<float>(world_pt.point.x);
      wpt.y = static_cast<float>(world_pt.point.y);
      wpt.z = static_cast<float>(world_pt.point.z);
      wpt.r = pt.r; wpt.g = pt.g; wpt.b = pt.b;
      if (wpt.z < 0.03f) continue;
      combined_colour_cloud.push_back(wpt);
    }
    RCLCPP_INFO(node_->get_logger(),
      "Task 3: scan at (%.2f,%.2f): %zu total coloured pts so far",
      pos.first, pos.second, combined_colour_cloud.size());
  }

  if (combined_colour_cloud.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "Task 3: no coloured points found");
    return;
  }

  RCLCPP_INFO(node_->get_logger(),
    "Task 3: %zu total coloured points accumulated", combined_colour_cloud.size());

  // ── Step 2: Cluster ───────────────────────────────────────────────────────
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colour_cloud_ptr(
    new pcl::PointCloud<pcl::PointXYZRGB>(combined_colour_cloud));
  auto clusters = pclEuclideanClusters(colour_cloud_ptr, 0.02f, 200, 100000);

  RCLCPP_INFO(node_->get_logger(), "Task 3: found %zu clusters", clusters.size());

  // ── Step 3: Classify clusters ─────────────────────────────────────────────
  struct RefC { const char* name; float r,g,b; };
  const RefC refs[] = {
    {"blue",   0.1f, 0.1f, 0.8f},
    {"red",    0.8f, 0.1f, 0.1f},
    {"purple", 0.8f, 0.1f, 0.8f},
  };

  struct Candidate { std::string colour; double x, y, z; int pts; };
  std::vector<Candidate> candidates;

  for (const auto & cluster : clusters) {
    double sr=0,sg=0,sb=0, cx=0,cy=0,cz=0;
    for (int idx : cluster.indices) {
      const auto & pt = colour_cloud_ptr->points[idx];
      sr += pt.r/255.0; sg += pt.g/255.0; sb += pt.b/255.0;
      cx += pt.x; cy += pt.y; cz += pt.z;
    }
    int n = static_cast<int>(cluster.indices.size());
    double mr = sr/n, mg = sg/n, mb = sb/n;
    cx /= n; cy /= n; cz /= n;

    double best_dist = 1e9;
    const char * colour = nullptr;
    for (const auto & ref : refs) {
      double d = (mr-ref.r)*(mr-ref.r)+(mg-ref.g)*(mg-ref.g)+(mb-ref.b)*(mb-ref.b);
      if (d < best_dist) { best_dist = d; colour = ref.name; }
    }
    if (best_dist > 0.4 || !colour) continue;

    candidates.push_back({std::string(colour), cx, cy, cz, n});
    RCLCPP_INFO(node_->get_logger(),
      "  cluster: %s at world (%.3f, %.3f, %.3f) [%d pts, dist=%.3f]",
      colour, cx, cy, cz, n, best_dist);
  }

  // ── Step 4: Separate baskets from cubes ───────────────────────────────────
  std::vector<DetectedObject> cubes, baskets;
  const std::string colours[] = {"red", "blue", "purple"};
  const int BASKET_PT_THRESHOLD = 8000;

  for (const auto & col : colours) {
    std::vector<Candidate*> same_colour;
    for (auto & c : candidates) {
      if (c.colour == col) same_colour.push_back(&c);
    }
    if (same_colour.empty()) continue;

    for (const auto * c : same_colour) {
      DetectedObject obj;
      obj.colour = col;
      obj.x = c->x; obj.y = c->y; obj.z = c->z;
      if (c->pts >= BASKET_PT_THRESHOLD) {
        obj.type = "basket";
        baskets.push_back(obj);
      } else {
        obj.type = "cube";
        cubes.push_back(obj);
      }
      RCLCPP_INFO(node_->get_logger(),
        "  detected %s %s at world (%.3f, %.3f, %.3f) [%d pts]",
        obj.colour.c_str(), obj.type.c_str(), obj.x, obj.y, obj.z, c->pts);
    }
  }

  RCLCPP_INFO(node_->get_logger(),
    "Task 3: %zu cubes, %zu baskets detected", cubes.size(), baskets.size());

  // ── Step 5: Pick and place ────────────────────────────────────────────────
  int placed = 0;
  for (const auto & cube : cubes) {
    const DetectedObject * target = nullptr;
    for (const auto & basket : baskets) {
      if (basket.colour == cube.colour) { target = &basket; break; }
    }
    if (!target) {
      RCLCPP_WARN(node_->get_logger(),
        "Task 3: no %s basket found for %s cube — skipping",
        cube.colour.c_str(), cube.colour.c_str());
      continue;
    }

    RCLCPP_INFO(node_->get_logger(),
      "=== T3 PICK: %s cube at (%.3f, %.3f) -> basket (%.3f, %.3f) ===",
      cube.colour.c_str(), cube.x, cube.y, target->x, target->y);

    RCLCPP_INFO(node_->get_logger(), "  [1] Open gripper");
    setGripper(0.04);

    RCLCPP_INFO(node_->get_logger(), "  [2] Lift above cube (%.4f, %.4f)", cube.x, cube.y);
    if (!moveToLiftXY(cube.x, cube.y)) {
      RCLCPP_WARN(node_->get_logger(), "  [2] FAILED - skipping");
      continue;
    }

    RCLCPP_INFO(node_->get_logger(), "  [3] Descend to grasp");
    moveToGraspZ(cube.x, cube.y, 0.1434);

    RCLCPP_INFO(node_->get_logger(), "  [4] Close gripper, wait 500ms");
    setGripper(0.010);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    RCLCPP_INFO(node_->get_logger(), "  [5] Lift up");
    moveToLiftXY(cube.x, cube.y);

    // Push 3cm outward to land in basket centre
    {
      const double br = std::sqrt(target->x * target->x + target->y * target->y);
      const double drop_outward = 0.02;
      const double bx = target->x + drop_outward * (target->x / br);
      const double by = target->y + drop_outward * (target->y / br);
      RCLCPP_INFO(node_->get_logger(), "  [6] Move above basket (%.4f, %.4f)", bx, by);
      moveToLiftXY(bx, by);
    }

    RCLCPP_INFO(node_->get_logger(), "  [7] Release above basket");
    setGripper(0.04);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    ++placed;
    RCLCPP_INFO(node_->get_logger(),
      "=== T3 PLACED %s cube (%d/%zu) ===", cube.colour.c_str(), placed, cubes.size());
  }

  RCLCPP_INFO(node_->get_logger(),
    "Task 3 done: placed %d/%zu cubes", placed, cubes.size());

  moveToLiftXY(0.45, 0.0);
}