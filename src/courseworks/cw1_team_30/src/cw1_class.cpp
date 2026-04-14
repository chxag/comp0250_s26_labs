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

/* ------------------------ Motion helpers (shared by all tasks) ------------------------ */

// Helper to execute a joint-space plan to a given joint target.
static bool execJoints(
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> & arm_group,
  const std::vector<double> & joints)
{
  arm_group->setStartStateToCurrentState(); // Set current state as start to avoid issues with non-determinism of MoveIt!'s internal state
  arm_group->setJointValueTarget(joints); // Set the joint target
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool ok = (arm_group->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS); // Plan a trajectory to the target
  if (ok) ok = (arm_group->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS); // Execute the planned trajectory
  return ok; // Return whether the execution was successful
}

/** 
* Helper to move arm to a "lifted" pose above a given (x,y) location.
* Rather than using MoveIt! IK (which was causing the robot to "dance"), joint angles are computed analytically.
* 
*   j1 = atan2(y, x)      — rotate base to face the target (exact trig)
*   j2 = 3.267·r − 1.784  — extend shoulder outward; fitted linear in r
*   j4 = 3.286·r − 3.457  — bend elbow to match; fitted linear in r

* j2 and j4 are empirically fitted: the arm was posed at several radial distances and the joint angles recorded. 
* Both scale almost identically with r because shoulder and elbow must extend together to hold the wrist at a fixed height.
* 
* j3 and j5 are locked at 0, j6 = pi/2 points the tool down, j7 = pi/4 sets a 45 degree gripper approach angle which helps with grasping.
*
* The result is always the same arm shape, just reaching further or closer depending on r.
*
*/ 
bool cw1::moveToLiftXY(double x, double y)
{
  const double r  = std::sqrt(x*x + y*y);
  const double j1 = std::atan2(y, x);
  const double j2 = 3.267*r - 1.784;
  const double j4 = 3.286*r - 3.457;
  return execJoints(arm_group, {j1, j2, 0.0, j4, 0.0, 1.571, 0.785});
}

// Helper to move arm to grasp height z above a given (x,y).
bool cw1::moveToGraspZ(double x, double y, double z)
{
  arm_group->setStartStateToCurrentState();

  // Adds a small outward offset (3cm along the radial direction) to prevent the gripper from colliding with the object when descending straight down.
  // Helps grab it at object centre.
  const double r  = std::sqrt(x*x + y*y);
  const double gx = x + 0.03 * (x / r);
  const double gy = y + 0.03 * (y / r);

  // Keep current end-effector orientation and just move the position to the target. 
  geometry_msgs::msg::PoseStamped current_stamped = arm_group->getCurrentPose();
  geometry_msgs::msg::Pose target;
  target.position.x  = gx;
  target.position.y  = gy;
  target.position.z  = z;
  target.orientation = current_stamped.pose.orientation; // keep current orientation

  std::vector<geometry_msgs::msg::Pose> waypoints = {target};
  moveit_msgs::msg::RobotTrajectory trajectory;
  // Uses Cartesian path planning (straight-line interpolation with 5 mm resolution) to ensure a clean vertical descent.
  double fraction = arm_group->computeCartesianPath(waypoints, 0.005, 0.0, trajectory);

  // Falls back to a standard pose-goal plan if the Cartesian path covers less than 90% of the 
  // requested trajectory (e.g. if a singularity or joint limit is nearby).
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

// Move to an arbitrary 6-DOF target pose using IK-based planning.
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

// Helper to set the gripper width for grasping or releasing objects.
bool cw1::setGripper(double width)
{
  hand_group->setJointValueTarget("panda_finger_joint1", width);
  hand_group->setJointValueTarget("panda_finger_joint2", width);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool ok = (hand_group->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (ok) ok = (hand_group->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  return ok;
}

// Helper to position arm above centre of basket. 
bool cw1::moveAboveBasketDrop(double basket_x, double basket_y)
{
  const double r = std::sqrt(basket_x * basket_x + basket_y * basket_y);
  const double drop_outward = 0.02;  // push 2cm outward from centroid into basket centre to avoid collisions with basket rim and improve grasp success.
  return moveToLiftXY(basket_x + drop_outward * (basket_x / r),
                      basket_y + drop_outward * (basket_y / r));
}


/* ------------------------ Task 1 ------------------------ */

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

  setGripper(0.04); // 1. open gripper
  moveToLiftXY(obj_x, obj_y); // 2. lift above cube
  moveToGraspZ(obj_x, obj_y, 0.1434); // 3. descend
  setGripper(0.010); // 4. grasp
  moveToLiftXY(obj_x, obj_y); // 5. lift back up
  moveAboveBasketDrop(goal_x, goal_y); // 6. move above basket
  setGripper(0.04); // 7. release

  RCLCPP_INFO(node_->get_logger(), "Task 1 done.");
}

/* ------------------------ Task 2 helpers ------------------------ */

// Blocks until a new point cloud arrives (after moving robot) or timeout occurs.
sensor_msgs::msg::PointCloud2::ConstSharedPtr cw1::waitForCloud(double timeout_sec)
{
  const auto start_time  = node_->now();
  const uint64_t start_count = cloud_msg_count_.load(std::memory_order_relaxed); // how many cloud have been received so far, to detect new ones
  while (rclcpp::ok()) { // check for new cloud or timeout
    if (cloud_msg_count_.load(std::memory_order_relaxed) > start_count) {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      return latest_cloud_; // if a new cloud has arrived, return it
    }
    if ((node_->now() - start_time).seconds() > timeout_sec) return nullptr; // timeout
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return nullptr; // node shutdown
}

// Transforms a point from world frame to the target frame (e.g. camera frame) using TF2. 
// Used to transform the basket location into the camera frame for cropping and colour detection.
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

// Crop cloud to a ~16cm radius around the basket ---> reduces noise and speeds up processing for colour detection.
// Also applies rudementary depth filtering to remove points that are too far below or above the basket (based on expected cube size and table height).
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cw1::cropAroundBasket(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
  const geometry_msgs::msg::PointStamped & basket_world_loc)
{
  auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  if (!cloud) return out;

  auto stamped = basket_world_loc;
  stamped.header.stamp = cloud->header.stamp;

  // Transform the basket location from world frame to camera frame so we can crop the cloud around it. 
  // This is necessary because the cloud points are in the camera frame.
  auto basket_cam = transformToCameraFrame(stamped, cloud->header.frame_id);
  if (basket_cam.header.frame_id.empty()) return out;

  // Extract the basket location in camera frame as floats for easier processing.
  const float bx = static_cast<float>(basket_cam.point.x);
  const float by = static_cast<float>(basket_cam.point.y);
  const float bz = static_cast<float>(basket_cam.point.z); // Distance from camera to basket centre (if robot above basket) — used for rudimentary depth filtering

  RCLCPP_INFO(node_->get_logger(),
    "cropAroundBasket: basket_cam=(%.3f, %.3f, %.3f)", bx, by, bz);

  // Convert ROS PointCloud2 to PCL format for easier processing.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::fromROSMsg(*cloud, *pcl_cloud);

  constexpr float R2 = 0.08f * 0.08f; // Crop radius of 8cm around the basket (16cm diameter)

  /*
  ChatGPT was used to help implement the below depth filtering mechansim as just cropping and plane segmentation was not sufficient to remove all the noise from the table and surrounding area, 
  which was causing significant issues for colour detection accuracy. 
  The depth filtering is a simple way to remove points that are too far below or above the basket based on the expected cube size and table height.
  This significantly improved the colour detection accuracy by only keeping points that are likely to be on the basket and cube.
  */

  // Depth filtering: only consider points within a certain vertical range around the basket. 
  // Keeps points that are up to 15cm closer than the basket (to include cube points below basket centre, and account for some depth noise)
  // Keeps points that are up to 6cm further than the basket (to include points on the basket walls above the cube, but not points from the table)
  const float zmin = bz - 0.15f, zmax = bz + 0.06f; 

  // For every point in the cloud...
  for (const auto & pt : pcl_cloud->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue; // skip invalid points (sensor noise, missing depth, etc)
    if (pt.z < zmin || pt.z > zmax) continue; // keep points within depth range around the basket
    // calculate distance from basket in camera XY plane 
    float dx = pt.x - bx, dy = pt.y - by;
    if (dx*dx + dy*dy > R2) continue; // keep points within crop radius of basket centre in XY plane
    out->points.push_back(pt);
  }
  out->width = static_cast<uint32_t>(out->points.size());
  out->height = 1;
  out->is_dense = false;
  return out;
}

// Removes noise and floor points from the cropped cloud using a statistical outlier removal filter followed by RANSAC plane segmentation. 
// This helps to improve colour detection accuracy by only keeping points on the basket and cube, and removing points from the table and any remaining sensor noise.
// This follows the rudimentary filtering in cropAroundBasket with a more robust statistical outlier removal and explicit plane segmentation to remove the floor.
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cw1::removeNoiseAndFloor(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud)
{
  auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  if (!cloud || cloud->points.empty()) return out;

  // Follows Lab 5 filtering steps for plane segmentation, but applied to the cropped cloud around the basket rather than the whole scene.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sor_out(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(20);
  sor.setStddevMulThresh(1.0);
  sor.filter(*sor_out);
  if (sor_out->points.empty()) return out;


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

  // If no plane was found...
  if (inliers->indices.empty()) {
    RCLCPP_WARN(node_->get_logger(), "removeNoiseAndFloor: no plane found — returning SOR cloud");
    return sor_out; // return noise-filtered cloud without floor removal
  }

  // Extract points that are NOT part of the floor plane (inliers) to get the cleaned cloud with noise and floor removed.
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
  extract.setInputCloud(sor_out);
  extract.setIndices(inliers);
  extract.setNegative(true); // remove floor points
  extract.filter(*out); 
  return out;
}

// Detects the colour of the basket by analysing the average RGB values of the points in the cropped and cleaned cloud around the basket.
std::string cw1::detectBasketColour(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
  const geometry_msgs::msg::PointStamped & basket_world_loc)
{
  if (!cloud) return "none";

  auto cropped = cropAroundBasket(cloud, basket_world_loc); // crop to points around basket

  // If no points are found around the basket, return "none" as the colour. 
  // This can happen if the basket location is inaccurate, if the basket is not visible in the camera, or if there is too much noise in the cloud.
  if (cropped->points.empty()) {
    RCLCPP_WARN(node_->get_logger(), "detectBasketColour: no points after crop");
    return "none";
  }

  auto cleaned = removeNoiseAndFloor(cropped); // remove noise and floor points to improve colour detection accuracy
  // If no points are left after noise and floor removal, return "none" as the colour.
  if (cleaned->points.empty()) {
    RCLCPP_WARN(node_->get_logger(), "detectBasketColour: no points after denoising");
    return "none";
  }

  // Initialise sums for RGB values and a count of valid points. We will calculate the average colour of the points in the basket cloud to determine the basket colour.
  double rs = 0, gs = 0, bs = 0;
  int count = 0; // count of valid points considered for colour detection (used to calculate average and filter out very dark points)

  // For each point in the basket cloud...
  for (const auto & pt : cleaned->points) {
    // Get RGB values as doubles in the range [0,1] by normalising the uint8 RGB values from the point cloud.
    double r = pt.r / 255.0, g = pt.g / 255.0, b = pt.b / 255.0;
    if (std::max({r, g, b}) < 0.25) continue;
    rs += r; gs += g; bs += b; ++count; // Accumulate RGB values for valid points to calculate average colour later. 
  }
  // If we have too few valid points (e.g. due to noise, inaccurate basket location, or occlusion), return "none" as the colour.
  if (count < 20) {
    RCLCPP_WARN(node_->get_logger(), "detectBasketColour: only %d valid points", count);
    return "none";
  }

  // Find average RGB values of the points in the basket cloud to determine the overall colour of the basket. 
  const double mr = rs/count, mg = gs/count, mb = bs/count;
  RCLCPP_INFO(node_->get_logger(),
    "detectBasketColour: mean RGB=(%.3f, %.3f, %.3f) from %d pts", mr, mg, mb, count);

  // Structure to hold the known colours and their RBG values (as defined in the spec)
  struct KnownColour { const char* name; double r, g, b; };
  const KnownColour palette[] = {
    {"red",    0.8, 0.1, 0.1},
    {"blue",   0.1, 0.1, 0.8},
    {"purple", 0.8, 0.1, 0.8},
  };

  double best = 1e9;
  std::string result = "none";
  // For each known colour...
  for (const auto & kc : palette) {
    // Calculate the Euclidean distance in RGB space between the average colour of the basket cloud and the known colour.
    double d = std::sqrt((mr-kc.r)*(mr-kc.r) + (mg-kc.g)*(mg-kc.g) + (mb-kc.b)*(mb-kc.b));
    if (d < best) { best = d; result = kc.name; } // Keep track of the known colour with the smallest distance to the average colour of the basket cloud. This will be our detected colour for the basket.
  }

  // If the best distance to a known colour is above a certain threshold, we consider the detected colour to be unreliable and return "none". 
  // This can happen if the basket has a lot of noise in the cloud.
  if (best > 0.4) {
    RCLCPP_WARN(node_->get_logger(),
      "detectBasketColour: dist %.3f > 0.4 — returning 'none'", best);
    return "none";
  }
  RCLCPP_INFO(node_->get_logger(), "detectBasketColour: -> %s (dist=%.3f)", result.c_str(), best);
  return result;
}

/* ------------------------ Task 2 ------------------------ */

void cw1::t2_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task2Service::Response> response)
{
  RCLCPP_INFO(node_->get_logger(), "Task 2 started: %zu baskets", request->basket_locs.size());

  std::vector<std::string> basket_colours;

  // For each basket location provided in the request...
  for (size_t i = 0; i < request->basket_locs.size(); ++i) {
    auto loc = request->basket_locs[i];
    if (loc.header.frame_id.empty()) loc.header.frame_id = "world"; // assume world frame if not specified

    RCLCPP_INFO(node_->get_logger(),
      "  basket %zu at (%.3f, %.3f)", i+1, loc.point.x, loc.point.y);

    // Move the robot above the basket location to get a clear view for colour detection. 
    // If the move fails (e.g. due to unreachable position), skip colour detection for this basket and return "none" as the colour.
    if (!moveToLiftXY(loc.point.x, loc.point.y)) {
      RCLCPP_ERROR(node_->get_logger(), "  basket %zu: move failed — returning 'none'", i+1);
      basket_colours.push_back("none");
      continue;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(400)); // settle time

    // Wait for a fresh cloud after moving above the basket to ensure we are processing the most up-to-date view of the basket for colour detection.
    auto cloud = waitForCloud(5.0);
    if (!cloud) {
      RCLCPP_ERROR(node_->get_logger(), "  basket %zu: no cloud — returning 'none'", i+1);
      basket_colours.push_back("none");
      continue;
    }

    // Detect the colour of the basket using the fresh cloud and the basket location. 
    // This involves cropping the cloud around the basket, removing noise and floor points, and analysing the average RGB values to determine the colour.
    std::string colour = detectBasketColour(cloud, loc);
    RCLCPP_INFO(node_->get_logger(), "  basket %zu: %s", i+1, colour.c_str());
    basket_colours.push_back(colour);
  }

  response->basket_colours = basket_colours; // return the detected colours for all baskets in the response in the same order as the request.
  RCLCPP_INFO(node_->get_logger(), "Task 2 done.");
}

/* ------------------------ Task 3 ------------------------ */

// Structure to hold detected object information (colour, type, and position) for Task 3.
struct DetectedObject {
  std::string colour; // "red", "blue", "purple", or "none" 
  std::string type;  // "cube" or "basket"
  double x, y, z; // world frame coordinates of the detected object (e.g. cube centre or basket centre)
};

/** 
 * ChatGPT was used to help point towards the below Euclidean Cluster Extraction method in order to segment individual objects from the cleaned cloud.
 * The following tutorial was then used as reference for implementation: https://pcl.readthedocs.io/projects/tutorials/en/master/cluster_extraction.html
*/

// Segments the cleaned cloud into individual clusters (objects) using Euclidean Cluster Extraction based on the distance between points.
// Rejects clusters that are too small (e.g. noise) or too large (e.g. table) based on the specified min and max cluster size.
static std::vector<pcl::PointIndices> pclEuclideanClusters(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
  float tolerance, int min_pts, int max_pts)
{
  // Builds a KD-tree for efficient nearest neighbour search to find clusters of points that are close to each other in space.
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

  // Orientation for scanning; camera / gripper faces downwards.
  geometry_msgs::msg::Pose scan_pose;
  scan_pose.orientation.x = 1.0;
  scan_pose.orientation.y = 0.0;
  scan_pose.orientation.z = 0.0;
  scan_pose.orientation.w = 0.0;

  // Predefined scan positions (x,y) in front of the robot to move to and capture point clouds.
  // Gives overlapping coverage from the whole workspace.
  const std::vector<std::pair<double,double>> scan_positions = {
    {0.40,  0.30}, {0.40, -0.30},
    {0.55,  0.35}, {0.55, -0.35}, {0.55,  0.00},
  };

  
  struct KnownColours { float r, g, b; };
  const KnownColours palette[] = {
    {0.8f, 0.1f, 0.1f}, {0.1f, 0.1f, 0.8f}, {0.8f, 0.1f, 0.8f}, // red, blue, purple
  };
  // Maximum squared distance in RGB space to consider a point as matching one of the known colours.
  const float max_col_dist_sq = 0.25f * 0.25f * 3.0f;

  pcl::PointCloud<pcl::PointXYZRGB> combined;

  // ------------------ 1. Scan the scene ------------------
  for (const auto & pos : scan_positions) {
    scan_pose.position.x = pos.first;
    scan_pose.position.y = pos.second;
    scan_pose.position.z = 0.60;
    moveToPose(scan_pose); // Move arm to each scan position at height 0.6 with camera facing down.

    std::this_thread::sleep_for(std::chrono::milliseconds(600)); // Wait for robot to settle and motion blur to clear before capturing cloud.
    cloud_msg_count_.store(0, std::memory_order_relaxed);
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Delay to ensure next cloud captures is a fresh one after motion (not mid-motion).

    // Blocks until new cloud arrives 
    auto cloud = waitForCloud(5.0);
    if (!cloud) continue;

    // Convert ROS PointCloud2 message to PCL format for easier processing.
    pcl::PointCloud<pcl::PointXYZRGB> raw;
    pcl::fromROSMsg(*cloud, raw);

    // ------------------2. Filter and combine clouds from all scan positions ------------------

    // For each point in the raw point cloud (before filtering)...
    for (const auto & pt : raw.points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue; // skip invalid points (sensor noise, missing depth, etc)
      if (pt.z < 0.30f || pt.z > 0.58f) continue; // keep points within expected height range of objects on the table and drop table and anything too high

      float r = pt.r/255.f, g = pt.g/255.f, b = pt.b/255.f;
      if (std::max({r,g,b}) < 0.20f) continue; // skip very dark points

      // Check if point's colour is close enough to any of the three known colours
      // If not, discard it.
      bool match = false;
      for (const auto & kc: palette) { 
        float dr=r-kc.r, dg=g-kc.g, db=b-kc.b;
        if (dr*dr+dg*dg+db*db < max_col_dist_sq) { match=true; break; }
      }
      if (!match) continue;

      // Transform the point from camera frame to world frame using TF2 so we can work with consistent world coordinates for clustering and object detection.
      geometry_msgs::msg::PointStamped cam_pt, world_pt;
      cam_pt.header.frame_id = cloud->header.frame_id;
      cam_pt.header.stamp    = cloud->header.stamp;
      cam_pt.point.x = pt.x; cam_pt.point.y = pt.y; cam_pt.point.z = pt.z;
      try {
        world_pt = tf_buffer_->transform(cam_pt, "world", tf2::durationFromSec(0.5));
      } catch (...) { continue; }

      pcl::PointXYZRGB pcl_world_pt;
      pcl_world_pt.x = static_cast<float>(world_pt.point.x);
      pcl_world_pt.y = static_cast<float>(world_pt.point.y);
      pcl_world_pt.z = static_cast<float>(world_pt.point.z);
      pcl_world_pt.r = pt.r; pcl_world_pt.g = pt.g; pcl_world_pt.b = pt.b;
      if (pcl_world_pt.z < 0.03f) continue; // drop anything at the floor level
      combined.push_back(pcl_world_pt);
    }
    RCLCPP_INFO(node_->get_logger(),
      "Task 3: scan (%.2f,%.2f): %zu pts so far", pos.first, pos.second, combined.size());
  }

  if (combined.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "Task 3: no coloured points found"); return;
  }
  RCLCPP_INFO(node_->get_logger(), "Task 3: %zu total pts", combined.size());

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>(combined));
  
  // ------------------ 3. Cluster and detect objects ------------------

  // Group combined cloud into clusters (objects).
  auto clusters = pclEuclideanClusters(cloud_ptr, 0.02f, 200, 100000);
  RCLCPP_INFO(node_->get_logger(), "Task 3: %zu clusters", clusters.size());

  struct KnownColoursDict { const char* name; float r,g,b; };
  const KnownColoursDict dict[] = {
    {"blue",0.1f,0.1f,0.8f}, {"red",0.8f,0.1f,0.1f}, {"purple",0.8f,0.1f,0.8f},
  };

  struct Candidate { std::string colour; double x,y,z; int pts; };
  std::vector<Candidate> candidates;

  // For each cluster...
  for (const auto & cluster : clusters) {
    // Calculate the average RGB colour and centroid (x,y,z) of the points in the cluster to determine the cluster's overall colour and position.
    double rs=0,gs=0,bs=0, cluster_x=0,cluster_y=0,cluster_z=0;
    for (int idx : cluster.indices) {
      const auto & pt = cloud_ptr->points[idx];
      rs+=pt.r/255.0; gs+=pt.g/255.0; bs+=pt.b/255.0;
      cluster_x+=pt.x; cluster_y+=pt.y; cluster_z+=pt.z;
    }
    int n = static_cast<int>(cluster.indices.size());
    double mr=rs/n, mg=gs/n, mb=bs/n;
    cluster_x/=n; cluster_y/=n; cluster_z/=n;

    // Find the closest known colour in RGB space to the average colour of the cluster to determine the cluster's colour.
    double best=1e9; const char* colour=nullptr;
    for (const auto & kc : dict) {
      double d=(mr-kc.r)*(mr-kc.r)+(mg-kc.g)*(mg-kc.g)+(mb-kc.b)*(mb-kc.b);
      if (d<best) { best=d; colour=kc.name; }
    }
    if (best>0.4 || !colour) continue;
    candidates.push_back({std::string(colour), cluster_x, cluster_y, cluster_z, n});
    RCLCPP_INFO(node_->get_logger(),
      "  cluster: %s at (%.3f,%.3f,%.3f) [%d pts]", colour, cluster_x, cluster_y, cluster_z, n);
  }

  std::vector<DetectedObject> cubes, baskets;
  const std::string colours[] = {"red","blue","purple"};

  // ------------------- 4. Classify clusters as cubes or baskets ------------------

  // Threshold on the number of points in the cluster to classify it as a cube or a basket.
  // This is a simple heuristic based on the expected size of the objects and the density of the point cloud.
  // E.g. Baskets are larger and should have more points than cubes, so we can use a threshold to separate them.
  const int BASKET_PT_THRESHOLD = 8000; 

  // For each detected colour...
  for (const auto & col : colours) {
    // For each candidate cluster of that colour...
    for (auto & c : candidates) {
      if (c.colour != col) continue; // only consider candidates of the current colour
      DetectedObject obj;
      obj.colour=col; obj.x=c.x; obj.y=c.y; obj.z=c.z; // Object colour and position are taken from the candidate cluster information.

      // Classify the object as a basket or cube based on the number of points in the cluster (c.pts) compared to the threshold.
      if (c.pts >= BASKET_PT_THRESHOLD) { obj.type="basket"; baskets.push_back(obj); }
      else                              { obj.type="cube";   cubes.push_back(obj);   }
      RCLCPP_INFO(node_->get_logger(),
        "  detected %s %s at (%.3f,%.3f,%.3f) [%d pts]",
        obj.colour.c_str(), obj.type.c_str(), obj.x, obj.y, obj.z, c.pts);
    }
  }
  RCLCPP_INFO(node_->get_logger(),
    "Task 3: %zu cubes, %zu baskets", cubes.size(), baskets.size());

  // ------------------- 5. Matching cubes to baskets -------------------
  int placed = 0;
  // For each detected cube...
  for (const auto & cube : cubes) {
    const DetectedObject * target = nullptr;
    // For each detected basket..
    for (const auto & basket : baskets)
      if (basket.colour == cube.colour) { target = &basket; break; } // find the basket of the same colour as the cube to place it in

    if (!target) {
      RCLCPP_WARN(node_->get_logger(), "Task 3: no %s basket — skipping", cube.colour.c_str());
      continue;
    }
    RCLCPP_INFO(node_->get_logger(),
      "=== PICK %s (%.3f,%.3f) -> (%.3f,%.3f) ===",
      cube.colour.c_str(), cube.x, cube.y, target->x, target->y);
    
    // ------------------- 6. Pick and place the cube into the matched basket -------------------

    setGripper(0.04); // 1. open gripper 
    if (!moveToLiftXY(cube.x, cube.y)) continue;// 2. lift above cube
    moveToGraspZ(cube.x, cube.y, 0.1434); // 3. descend
    setGripper(0.010); // 4. grasp
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    moveToLiftXY(cube.x, cube.y); // 5. lift up
    moveAboveBasketDrop(target->x, target->y);// 6.move above basket
    setGripper(0.04); // 7. release
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    RCLCPP_INFO(node_->get_logger(),
      "=== PLACED %s (%d/%zu) ===", cube.colour.c_str(), ++placed, cubes.size());
  }

  RCLCPP_INFO(node_->get_logger(), "Task 3 done: %d/%zu cubes placed", placed, cubes.size());
  moveToLiftXY(0.45, 0.0);
}