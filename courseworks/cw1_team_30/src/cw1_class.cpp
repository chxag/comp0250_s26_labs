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
// Motion helpers (shared by all tasks)
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

// Move to ~0.56m lift height above XY, gripper pointing straight down.
bool cw1::moveToLiftXY(double x, double y)
{
  const double r  = std::sqrt(x*x + y*y);
  const double j1 = std::atan2(y, x);
  const double j2 = 3.267*r - 1.784;
  const double j4 = 3.286*r - 3.457;
  return execJoints(arm_group, {j1, j2, 0.0, j4, 0.0, 1.571, 0.785});
}

// Descend in a straight vertical line to grasp height.
// Applies a 3cm forward offset so fingers centre on the cube face.
bool cw1::moveToGraspZ(double x, double y, double z)
{
  arm_group->setStartStateToCurrentState();
  const double r  = std::sqrt(x*x + y*y);
  const double gx = x + 0.03 * (x / r);
  const double gy = y + 0.03 * (y / r);

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

// Pose-target move — used for scan positions where exact orientation matters.
bool cw1::moveToPose(const geometry_msgs::msg::Pose target_pose)
{
  arm_group->setStartStateToCurrentState();
  arm_group->setGoalOrientationTolerance(0.01);
  arm_group->setGoalPositionTolerance(0.01);
  arm_group->setPoseTarget(target_pose);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool ok = (arm_group->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (ok) ok = (arm_group->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  return ok;
}

bool cw1::setGripper(double width)
{
  hand_group->setJointValueTarget("panda_finger_joint1", width);
  hand_group->setJointValueTarget("panda_finger_joint2", width);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool ok = (hand_group->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (ok) ok = (hand_group->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  return ok;
}

// Move above a basket drop position, applying a small outward radial offset
// to compensate for PCL centroid bias toward the near rim.
// Used by both Task 1 and Task 3 — single definition, no duplication.
bool cw1::moveAboveBasketDrop(double basket_x, double basket_y)
{
  const double r = std::sqrt(basket_x * basket_x + basket_y * basket_y);
  const double drop_outward = 0.02;  // push 2cm outward from centroid into basket centre
  return moveToLiftXY(basket_x + drop_outward * (basket_x / r),
                      basket_y + drop_outward * (basket_y / r));
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

  setGripper(0.04);                         // 1. open
  moveToLiftXY(obj_x, obj_y);              // 2. lift above cube
  moveToGraspZ(obj_x, obj_y, 0.1434);     // 3. descend
  setGripper(0.010);                        // 4. grasp
  moveToLiftXY(obj_x, obj_y);              // 5. lift back up
  moveAboveBasketDrop(goal_x, goal_y);     // 6. move above basket (shared helper)
  setGripper(0.04);                         // 7. release

  RCLCPP_INFO(node_->get_logger(), "Task 1 done.");
}

///////////////////////////////////////////////////////////////////////////////
// Task 2 helpers
///////////////////////////////////////////////////////////////////////////////

// Waits for a new cloud message after the arm has moved.
sensor_msgs::msg::PointCloud2::ConstSharedPtr cw1::waitForCloud(double timeout_sec)
{
  const auto start_time  = node_->now();
  const uint64_t start_count = cloud_msg_count_.load(std::memory_order_relaxed);
  while (rclcpp::ok()) {
    if (cloud_msg_count_.load(std::memory_order_relaxed) > start_count) {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      return latest_cloud_;
    }
    if ((node_->now() - start_time).seconds() > timeout_sec) return nullptr;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return nullptr;
}

// Transform a world-frame point into the camera frame using TF2.
geometry_msgs::msg::PointStamped cw1::transformToCameraFrame(
  const geometry_msgs::msg::PointStamped & point_in_world,
  const std::string & target_frame)
{
  geometry_msgs::msg::PointStamped result = point_in_world;
  try {
    result = tf_buffer_->transform(point_in_world, target_frame, tf2::durationFromSec(0.5));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(node_->get_logger(), "TF transform failed: %s", ex.what());
    result.header.frame_id = "";  // mark invalid
  }
  return result;
}

// Crop the cloud to an 8cm radius around the basket in camera XY,
// with a depth band [bz-0.15, bz+0.06] to exclude the table.
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cw1::cropAroundBasket(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
  const geometry_msgs::msg::PointStamped & basket_world_loc)
{
  auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  if (!cloud) return out;

  auto stamped = basket_world_loc;
  stamped.header.stamp = cloud->header.stamp;
  auto basket_cam = transformToCameraFrame(stamped, cloud->header.frame_id);
  if (basket_cam.header.frame_id.empty()) return out;

  const float bx = static_cast<float>(basket_cam.point.x);
  const float by = static_cast<float>(basket_cam.point.y);
  const float bz = static_cast<float>(basket_cam.point.z);

  RCLCPP_INFO(node_->get_logger(),
    "cropAroundBasket: basket_cam=(%.3f, %.3f, %.3f)", bx, by, bz);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::fromROSMsg(*cloud, *pcl_cloud);

  constexpr float R2 = 0.08f * 0.08f;
  const float zmin = bz - 0.15f, zmax = bz + 0.06f;

  for (const auto & pt : pcl_cloud->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
    if (pt.z < zmin || pt.z > zmax) continue;
    float dx = pt.x - bx, dy = pt.y - by;
    if (dx*dx + dy*dy > R2) continue;
    out->points.push_back(pt);
  }
  out->width = static_cast<uint32_t>(out->points.size());
  out->height = 1;
  out->is_dense = false;
  return out;
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr cw1::removeNoiseAndFloor(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud)
{
  auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  if (!cloud || cloud->points.empty()) return out;

  // Statistical outlier removal
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sor_out(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(20);
  sor.setStddevMulThresh(1.0);
  sor.filter(*sor_out);
  if (sor_out->points.empty()) return out;

  // RANSAC plane segmentation to remove table
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients());
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100);
  seg.setDistanceThreshold(0.01);
  seg.setInputCloud(sor_out);
  seg.segment(*inliers, *coeff);

  if (inliers->indices.empty()) {
    RCLCPP_WARN(node_->get_logger(), "removeNoiseAndFloor: no plane found — returning SOR cloud");
    return sor_out;
  }

  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
  extract.setInputCloud(sor_out);
  extract.setIndices(inliers);
  extract.setNegative(true);  // keep non-plane points
  extract.filter(*out);
  return out;
}

// Classify basket colour: crop → denoise → mean RGB → nearest known colour.
std::string cw1::detectBasketColour(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
  const geometry_msgs::msg::PointStamped & basket_world_loc)
{
  if (!cloud) return "none";

  auto cropped = cropAroundBasket(cloud, basket_world_loc);
  if (cropped->points.empty()) {
    RCLCPP_WARN(node_->get_logger(), "detectBasketColour: no points after crop");
    return "none";
  }

  auto cleaned = removeNoiseAndFloor(cropped);
  if (cleaned->points.empty()) {
    RCLCPP_WARN(node_->get_logger(), "detectBasketColour: no points after denoising");
    return "none";
  }

  double rs = 0, gs = 0, bs = 0;
  int count = 0;
  for (const auto & pt : cleaned->points) {
    double r = pt.r / 255.0, g = pt.g / 255.0, b = pt.b / 255.0;
    if (std::max({r, g, b}) < 0.25) continue;
    rs += r; gs += g; bs += b; ++count;
  }
  if (count < 20) {
    RCLCPP_WARN(node_->get_logger(), "detectBasketColour: only %d valid points", count);
    return "none";
  }

  const double mr = rs/count, mg = gs/count, mb = bs/count;
  RCLCPP_INFO(node_->get_logger(),
    "detectBasketColour: mean RGB=(%.3f, %.3f, %.3f) from %d pts", mr, mg, mb, count);

  struct KnownColour { const char* name; double r, g, b; };
  const KnownColour palette[] = {
    {"red",    0.8, 0.1, 0.1},
    {"blue",   0.1, 0.1, 0.8},
    {"purple", 0.8, 0.1, 0.8},
  };

  double best = 1e9;
  std::string result = "none";
  for (const auto & kc : palette) {
    double d = std::sqrt((mr-kc.r)*(mr-kc.r) + (mg-kc.g)*(mg-kc.g) + (mb-kc.b)*(mb-kc.b));
    if (d < best) { best = d; result = kc.name; }
  }

  if (best > 0.4) {
    RCLCPP_WARN(node_->get_logger(),
      "detectBasketColour: dist %.3f > 0.4 — returning 'none'", best);
    return "none";
  }
  RCLCPP_INFO(node_->get_logger(), "detectBasketColour: -> %s (dist=%.3f)", result.c_str(), best);
  return result;
}

///////////////////////////////////////////////////////////////////////////////
// Task 2 callback
///////////////////////////////////////////////////////////////////////////////

void cw1::t2_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task2Service::Response> response)
{
  RCLCPP_INFO(node_->get_logger(), "Task 2 started: %zu baskets", request->basket_locs.size());

  std::vector<std::string> basket_colours;

  for (size_t i = 0; i < request->basket_locs.size(); ++i) {
    auto loc = request->basket_locs[i];
    if (loc.header.frame_id.empty()) loc.header.frame_id = "world";

    RCLCPP_INFO(node_->get_logger(),
      "  basket %zu at (%.3f, %.3f)", i+1, loc.point.x, loc.point.y);

    if (!moveToLiftXY(loc.point.x, loc.point.y)) {
      RCLCPP_ERROR(node_->get_logger(), "  basket %zu: move failed — returning 'none'", i+1);
      basket_colours.push_back("none");
      continue;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    auto cloud = waitForCloud(5.0);
    if (!cloud) {
      RCLCPP_ERROR(node_->get_logger(), "  basket %zu: no cloud — returning 'none'", i+1);
      basket_colours.push_back("none");
      continue;
    }

    std::string colour = detectBasketColour(cloud, loc);
    RCLCPP_INFO(node_->get_logger(), "  basket %zu: %s", i+1, colour.c_str());
    basket_colours.push_back(colour);
  }

  response->basket_colours = basket_colours;
  RCLCPP_INFO(node_->get_logger(), "Task 2 done.");
}

///////////////////////////////////////////////////////////////////////////////
// Task 3 — scan, cluster, classify, pick and place
///////////////////////////////////////////////////////////////////////////////

struct DetectedObject {
  std::string colour, type;
  double x, y, z;
};

static std::vector<pcl::PointIndices> pclEuclideanClusters(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
  float tolerance, int min_pts, int max_pts)
{
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud(cloud);
  std::vector<pcl::PointIndices> indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance(tolerance);
  ec.setMinClusterSize(min_pts);
  ec.setMaxClusterSize(max_pts);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(indices);
  return indices;
}

void cw1::t3_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task3Service::Response> response)
{
  (void)request; (void)response;
  RCLCPP_INFO(node_->get_logger(), "Task 3 started: scanning scene");

  // ── Step 1: Multi-position scan ──────────────────────────────────────────
  geometry_msgs::msg::Pose scan_pose;
  scan_pose.orientation.x = 1.0;
  scan_pose.orientation.y = 0.0;
  scan_pose.orientation.z = 0.0;
  scan_pose.orientation.w = 0.0;

  const std::vector<std::pair<double,double>> scan_positions = {
    {0.40,  0.30}, {0.40, -0.30},
    {0.55,  0.35}, {0.55, -0.35}, {0.55,  0.00},
  };

  struct ColRef { float r, g, b; };
  const ColRef col_refs[] = {
    {0.8f, 0.1f, 0.1f}, {0.1f, 0.1f, 0.8f}, {0.8f, 0.1f, 0.8f},
  };
  const float max_col_dist_sq = 0.25f * 0.25f * 3.0f;

  pcl::PointCloud<pcl::PointXYZRGB> combined;

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
      combined.push_back(wpt);
    }
    RCLCPP_INFO(node_->get_logger(),
      "Task 3: scan (%.2f,%.2f): %zu pts so far", pos.first, pos.second, combined.size());
  }

  if (combined.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "Task 3: no coloured points found"); return;
  }
  RCLCPP_INFO(node_->get_logger(), "Task 3: %zu total pts", combined.size());

  // ── Step 2: Cluster ───────────────────────────────────────────────────────
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr(
    new pcl::PointCloud<pcl::PointXYZRGB>(combined));
  auto clusters = pclEuclideanClusters(cloud_ptr, 0.02f, 200, 100000);
  RCLCPP_INFO(node_->get_logger(), "Task 3: %zu clusters", clusters.size());

  // ── Step 3: Classify colour + basket/cube ─────────────────────────────────
  struct RefC { const char* name; float r,g,b; };
  const RefC refs[] = {
    {"blue",0.1f,0.1f,0.8f}, {"red",0.8f,0.1f,0.1f}, {"purple",0.8f,0.1f,0.8f},
  };

  struct Candidate { std::string colour; double x,y,z; int pts; };
  std::vector<Candidate> candidates;

  for (const auto & cl : clusters) {
    double sr=0,sg=0,sb=0, cx=0,cy=0,cz=0;
    for (int idx : cl.indices) {
      const auto & pt = cloud_ptr->points[idx];
      sr+=pt.r/255.0; sg+=pt.g/255.0; sb+=pt.b/255.0;
      cx+=pt.x; cy+=pt.y; cz+=pt.z;
    }
    int n = static_cast<int>(cl.indices.size());
    double mr=sr/n, mg=sg/n, mb=sb/n;
    cx/=n; cy/=n; cz/=n;

    double best=1e9; const char* col=nullptr;
    for (const auto & ref : refs) {
      double d=(mr-ref.r)*(mr-ref.r)+(mg-ref.g)*(mg-ref.g)+(mb-ref.b)*(mb-ref.b);
      if (d<best) { best=d; col=ref.name; }
    }
    if (best>0.4 || !col) continue;
    candidates.push_back({std::string(col), cx, cy, cz, n});
    RCLCPP_INFO(node_->get_logger(),
      "  cluster: %s at (%.3f,%.3f,%.3f) [%d pts]", col, cx, cy, cz, n);
  }

  std::vector<DetectedObject> cubes, baskets;
  const std::string colours[] = {"red","blue","purple"};
  const int BASKET_PT_THRESHOLD = 8000;

  for (const auto & col : colours) {
    for (auto & c : candidates) {
      if (c.colour != col) continue;
      DetectedObject obj;
      obj.colour=col; obj.x=c.x; obj.y=c.y; obj.z=c.z;
      if (c.pts >= BASKET_PT_THRESHOLD) { obj.type="basket"; baskets.push_back(obj); }
      else                              { obj.type="cube";   cubes.push_back(obj);   }
      RCLCPP_INFO(node_->get_logger(),
        "  detected %s %s at (%.3f,%.3f,%.3f) [%d pts]",
        obj.colour.c_str(), obj.type.c_str(), obj.x, obj.y, obj.z, c.pts);
    }
  }
  RCLCPP_INFO(node_->get_logger(),
    "Task 3: %zu cubes, %zu baskets", cubes.size(), baskets.size());

  // ── Step 4: Pick and place ────────────────────────────────────────────────
  int placed = 0;
  for (const auto & cube : cubes) {
    const DetectedObject * target = nullptr;
    for (const auto & basket : baskets)
      if (basket.colour == cube.colour) { target = &basket; break; }

    if (!target) {
      RCLCPP_WARN(node_->get_logger(), "Task 3: no %s basket — skipping", cube.colour.c_str());
      continue;
    }
    RCLCPP_INFO(node_->get_logger(),
      "=== PICK %s (%.3f,%.3f) -> (%.3f,%.3f) ===",
      cube.colour.c_str(), cube.x, cube.y, target->x, target->y);

    setGripper(0.04);                               // [1] open
    if (!moveToLiftXY(cube.x, cube.y)) continue;   // [2] lift above cube
    moveToGraspZ(cube.x, cube.y, 0.1434);           // [3] descend
    setGripper(0.010);                              // [4] grasp
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    moveToLiftXY(cube.x, cube.y);                  // [5] lift up
    moveAboveBasketDrop(target->x, target->y);     // [6] move above basket (shared helper)
    setGripper(0.04);                               // [7] release
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    RCLCPP_INFO(node_->get_logger(),
      "=== PLACED %s (%d/%zu) ===", cube.colour.c_str(), ++placed, cubes.size());
  }

  RCLCPP_INFO(node_->get_logger(), "Task 3 done: %d/%zu cubes placed", placed, cubes.size());
  moveToLiftXY(0.45, 0.0);
}