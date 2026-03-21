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
// General helpers
///////////////////////////////////////////////////////////////////////////////

// ─────────────────────────────────────────────────────────────────────────────
// Analytical joint-space helpers — derived from URDF FK, gripper always DOWN
// ─────────────────────────────────────────────────────────────────────────────
//
//  j1 = atan2(y, x)   — base rotation to face target
//  j3 = j5 = 0        — always zero
//  j7 = 0.0            — wrist neutral, fingers aligned with arm direction
//
// LIFT   (z~0.56m): j2=3.267*r-1.784  j4=3.286*r-3.457  j6=pi/2=1.571  j7=0
// GRASP  (z~0.13m): j2=1.710*r-0.369  j4=2.970*r-3.837  j6=2.900
//
// Both sets verified via FK to give TCP z-axis=[0,0,-1] (gripper straight down)
// across r=0.30..0.65m workspace.

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

// Lift height (~0.56m), gripper straight down. j6=pi/2.
bool cw1::moveToLiftXY(double x, double y)
{
  const double r  = std::sqrt(x*x + y*y);
  const double j1 = std::atan2(y, x);
  const double j2 = 3.267*r - 1.784;
  const double j4 = 3.286*r - 3.457;
  return execJoints(arm_group, {j1, j2, 0.0, j4, 0.0, 1.571, 0.785});
}

// Move TCP in a straight vertical line to target z.
// Uses computeCartesianPath — keeps current orientation locked, pure Z motion.
// Call ONLY after moveToLiftXY has positioned the arm above the target XY.
// Inherits orientation from current arm state so fingers stay parallel to arm.
bool cw1::moveToGraspZ(double x, double y, double z)
{
  arm_group->setStartStateToCurrentState();

  // Apply a small forward offset (2cm) along the approach direction so
  // the gripper centre lands on the cube face rather than the near edge.
  const double r       = std::sqrt(x*x + y*y);
  const double offset  = 0.023; // altering offset
  const double gx      = x + offset * (x / r);
  const double gy      = y + offset * (y / r);

  // Read current TCP orientation — preserves finger direction from moveToLiftXY
  geometry_msgs::msg::PoseStamped current_stamped = arm_group->getCurrentPose();
  geometry_msgs::msg::Pose target;
  target.position.x    = gx;
  target.position.y    = gy;
  target.position.z    = z;
  target.orientation   = current_stamped.pose.orientation; // keep fingers parallel

  std::vector<geometry_msgs::msg::Pose> waypoints = {target};

  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = arm_group->computeCartesianPath(
    waypoints,
    0.005,   // eef_step: 5mm interpolation
    0.0,     // jump_threshold: disabled
    trajectory);

  if (fraction < 0.9) {
    RCLCPP_WARN(node_->get_logger(),
      "moveToGraspZ: Cartesian path %.0f%% — retrying with pose target",
      fraction * 100.0);
    // Fallback: plain pose target if Cartesian fails
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

// Pose-target fallback for scan moves (camera pointing down, exact wrist not critical)
bool cw1::moveToPose(const geometry_msgs::msg::Pose target_pose)
{
  arm_group->setStartStateToCurrentState();
  arm_group->setGoalOrientationTolerance(0.01);  // tight: enforce gripper direction
  arm_group->setGoalPositionTolerance(0.01);
  arm_group->setPoseTarget(target_pose);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool success = (arm_group->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (success) {
    success = (arm_group->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  }
  return success;
}

bool cw1::setGripper(double width)
{
  hand_group->setJointValueTarget("panda_finger_joint1", width);
  hand_group->setJointValueTarget("panda_finger_joint2", width);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool success = (hand_group->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (success) {
    success = (hand_group->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  }
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
    "Task 1: cube(%.3f,%.3f) basket(%.3f,%.3f)",
    obj_x, obj_y, goal_x, goal_y);

  // ═══════════════════════════════════════════════════════════════════════════
  // Pure joint-space pick-and-place:
  // moveToLiftXY  → gripper straight down at z~0.55m, facing target
  // moveToGraspZ  → gripper straight down at z~0.12m, same XY
  // All moves deterministic, wrist always at pi/4 = perpendicular to base
  // ═══════════════════════════════════════════════════════════════════════════

  // 1. Open gripper
  setGripper(0.04);

  // 2. Rotate base + lift to safe height above cube, gripper straight down
  moveToLiftXY(obj_x, obj_y);

  // 3. Descend to cube grasp height
  moveToGraspZ(obj_x, obj_y, 0.1434);  // fingertips at cube centre (0.04m)

  // 4. Close gripper
  setGripper(0.006);

  // 5. Lift straight back up (reverse of step 3)
  moveToLiftXY(obj_x, obj_y);

  // 6. Rotate base to face basket, stay at lift height
  moveToLiftXY(goal_x, goal_y);

  // 7. Release cube above basket (no descent needed — drop from lift height)
  setGripper(0.04);

  RCLCPP_INFO(node_->get_logger(), "Task 1 done.");
}



///////////////////////////////////////////////////////////////////////////////
// Task 2 helpers
///////////////////////////////////////////////////////////////////////////////

bool cw1::moveToScanPose()
{
  geometry_msgs::msg::Pose scan_pose;
  scan_pose.orientation.x = 1.0;
  scan_pose.orientation.y = 0.0;
  scan_pose.orientation.z = 0.0;
  scan_pose.orientation.w = 0.0;
  scan_pose.position.x = 0.4;
  scan_pose.position.y = 0.0;
  scan_pose.position.z = 0.65;
  return moveToPose(scan_pose);
}

sensor_msgs::msg::PointCloud2::ConstSharedPtr
cw1::waitForCloud(double timeout_sec)
{
  const auto start = node_->now();
  const uint64_t count_before = cloud_msg_count_.load(std::memory_order_relaxed);
  while (rclcpp::ok()) {
    if (cloud_msg_count_.load(std::memory_order_relaxed) > count_before) {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      return latest_cloud_;
    }
    if ((node_->now() - start).seconds() > timeout_sec) {
      RCLCPP_WARN(node_->get_logger(), "waitForCloud: timed out after %.1f s", timeout_sec);
      return nullptr;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return nullptr;
}

/**
 * Classify basket colour from a point cloud captured directly above the basket.
 *
 * Key improvements over v1:
 * - Uses bz (basket depth in camera frame) to set a tight Z band that excludes
 *   the ground plane behind the basket.
 * - Reduced distance threshold to 0.4 to avoid misclassifying floor returns.
 */
std::string cw1::classifyBasketColour(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
  const geometry_msgs::msg::PointStamped & basket_world_loc,
  double crop_radius)
{
  if (!cloud) {
    RCLCPP_WARN(node_->get_logger(), "classifyBasketColour: null cloud");
    return "none";
  }

  // ---- 1. Transform basket world position into camera frame ---------------
  geometry_msgs::msg::PointStamped basket_cam;
  try {
    basket_cam = tf_buffer_->transform(
      basket_world_loc, cloud->header.frame_id, tf2::durationFromSec(1.0));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(node_->get_logger(), "TF2 failed: %s", ex.what());
    return "none";
  }

  const float bx = static_cast<float>(basket_cam.point.x);
  const float by = static_cast<float>(basket_cam.point.y);
  const float bz = static_cast<float>(basket_cam.point.z);

  RCLCPP_INFO(node_->get_logger(),
    "classifyBasketColour: basket in camera frame = (%.3f, %.3f, %.3f)", bx, by, bz);

  // ---- 2. Convert cloud to PCL --------------------------------------------
  pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
  pcl::fromROSMsg(*cloud, pcl_cloud);

  // ---- 3. Crop: XY radius + Z band to exclude ground plane ----------------
  // The basket top is ~0.05 m closer to the camera than the centroid.
  // The ground plane is ~0.1 m further than the centroid.
  // We keep points between (bz - 0.15) and (bz + 0.06) to stay on the basket.
  const float r2 = static_cast<float>(crop_radius * crop_radius);
  const float z_min = bz - 0.15f;
  const float z_max = bz + 0.06f;

  double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
  int count = 0;

  for (const auto & pt : pcl_cloud.points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }
    if (pt.z < z_min || pt.z > z_max) {
      continue;
    }
    const float dx = pt.x - bx;
    const float dy = pt.y - by;
    if (dx * dx + dy * dy > r2) {
      continue;
    }
    // Ignore very dark points (shadows, black plastic)
    const float r_f = pt.r / 255.0f;
    const float g_f = pt.g / 255.0f;
    const float b_f = pt.b / 255.0f;
    if (std::max({r_f, g_f, b_f}) < 0.25f) {
      continue;
    }

    sum_r += r_f;
    sum_g += g_f;
    sum_b += b_f;
    ++count;
  }

  RCLCPP_INFO(node_->get_logger(),
    "classifyBasketColour: found %d points in radius %.3f around (%.3f, %.3f) in %s",
    count, crop_radius, bx, by, cloud->header.frame_id.c_str());

  // ---- 4. Classify --------------------------------------------------------
  const int MIN_POINTS = 20;
  if (count < MIN_POINTS) {
    RCLCPP_INFO(node_->get_logger(),
      "classifyBasketColour: too few points (%d < %d) -> none", count, MIN_POINTS);
    return "none";
  }

  const double mr = sum_r / count;
  const double mg = sum_g / count;
  const double mb = sum_b / count;

  RCLCPP_INFO(node_->get_logger(),
    "classifyBasketColour: mean RGB = (%.3f, %.3f, %.3f)", mr, mg, mb);

  // Reference colours from spec
  struct RefColour { const char * name; double r, g, b; };
  const RefColour refs[] = {
    {"blue",   0.1, 0.1, 0.8},
    {"red",    0.8, 0.1, 0.1},
    {"purple", 0.8, 0.1, 0.8},
  };

  double best_dist = 1e9;
  const char * best_name = "none";
  for (const auto & ref : refs) {
    const double dr = mr - ref.r, dg = mg - ref.g, db = mb - ref.b;
    const double dist = dr*dr + dg*dg + db*db;
    if (dist < best_dist) { best_dist = dist; best_name = ref.name; }
  }

  // Reject poor matches (max possible = 3.0; 0.4 is generous but safe)
  if (best_dist > 0.4) {
    RCLCPP_INFO(node_->get_logger(),
      "classifyBasketColour: best dist %.3f > 0.4 -> none", best_dist);
    return "none";
  }

  RCLCPP_INFO(node_->get_logger(),
    "classifyBasketColour: classified as '%s' (dist=%.3f)", best_name, best_dist);
  return std::string(best_name);
}

///////////////////////////////////////////////////////////////////////////////
// Task 2 callback — move above EACH basket individually before scanning
///////////////////////////////////////////////////////////////////////////////

void cw1::t2_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task2Service::Response> response)
{
  RCLCPP_INFO(node_->get_logger(), "Task 2 started: %zu basket locations to check",
    request->basket_locs.size());

  // Gripper pointing straight down for all scan positions
  geometry_msgs::msg::Pose scan_pose;
  scan_pose.orientation.x = 1.0;
  scan_pose.orientation.y = 0.0;
  scan_pose.orientation.z = 0.0;
  scan_pose.orientation.w = 0.0;

  // Height above ground. Basket is 0.1 m tall; 0.55 m puts the camera
  // ~0.45 m above the basket top, well within the RealSense depth range.
  const double SCAN_HEIGHT = 0.55;

  // Store per-basket info for the final summary
  struct BasketResult {
    double x, y, z;
    std::string colour;
    double mean_r, mean_g, mean_b;
    int point_count;
    double best_dist;
  };
  std::vector<BasketResult> results;

  for (const auto & loc : request->basket_locs) {
    // Move directly above this basket location
    scan_pose.position.x = loc.point.x;
    scan_pose.position.y = loc.point.y;
    scan_pose.position.z = SCAN_HEIGHT;

    RCLCPP_INFO(node_->get_logger(),
      "Task 2: scanning basket %zu/%zu at (%.3f, %.3f)",
      results.size() + 1, request->basket_locs.size(),
      loc.point.x, loc.point.y);

    bool moved = moveToPose(scan_pose);
    if (!moved) {
      RCLCPP_WARN(node_->get_logger(),
        "Task 2: unreachable, falling back to centre scan pose");
      scan_pose.position.x = 0.4;
      scan_pose.position.y = 0.0;
      scan_pose.position.z = 0.65;
      moveToPose(scan_pose);
      scan_pose.position.x = loc.point.x;
      scan_pose.position.y = loc.point.y;
      scan_pose.position.z = SCAN_HEIGHT;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    auto cloud = waitForCloud(6.0);

    geometry_msgs::msg::PointStamped stamped_loc = loc;
    if (stamped_loc.header.frame_id.empty()) {
      stamped_loc.header.frame_id = "world";
    }
    stamped_loc.header.stamp = node_->now();

    // --- classify and capture stats for summary ---
    BasketResult res;
    res.x = loc.point.x;
    res.y = loc.point.y;
    res.z = loc.point.z;
    res.mean_r = 0; res.mean_g = 0; res.mean_b = 0;
    res.point_count = 0; res.best_dist = -1.0;

    // Run classification (logs suppressed here; summary printed below)
    res.colour = classifyBasketColour(cloud, stamped_loc, 0.08);

    // Re-extract stats from cloud for the summary (lightweight re-scan)
    if (cloud) {
      geometry_msgs::msg::PointStamped basket_cam;
      try {
        basket_cam = tf_buffer_->transform(
          stamped_loc, cloud->header.frame_id, tf2::durationFromSec(1.0));
        pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
        pcl::fromROSMsg(*cloud, pcl_cloud);
        const float bx = static_cast<float>(basket_cam.point.x);
        const float by = static_cast<float>(basket_cam.point.y);
        const float bz = static_cast<float>(basket_cam.point.z);
        const float r2 = 0.08f * 0.08f;
        const float z_min = bz - 0.15f, z_max = bz + 0.06f;
        double sr = 0, sg = 0, sb = 0; int cnt = 0;
        for (const auto & pt : pcl_cloud.points) {
          if (!std::isfinite(pt.x) || pt.z < z_min || pt.z > z_max) continue;
          const float dx = pt.x - bx, dy = pt.y - by;
          if (dx*dx + dy*dy > r2) continue;
          const float rf = pt.r/255.f, gf = pt.g/255.f, bf = pt.b/255.f;
          if (std::max({rf,gf,bf}) < 0.25f) continue;
          sr += rf; sg += gf; sb += bf; ++cnt;
        }
        if (cnt > 0) {
          res.mean_r = sr/cnt; res.mean_g = sg/cnt; res.mean_b = sb/cnt;
          res.point_count = cnt;
          // Compute distance to winning colour
          struct RC { const char* n; double r,g,b; };
          const RC refs[] = {{"blue",0.1,0.1,0.8},{"red",0.8,0.1,0.1},{"purple",0.8,0.1,0.8}};
          double best = 1e9;
          for (const auto & ref : refs) {
            double d = (res.mean_r-ref.r)*(res.mean_r-ref.r)
                     + (res.mean_g-ref.g)*(res.mean_g-ref.g)
                     + (res.mean_b-ref.b)*(res.mean_b-ref.b);
            if (d < best) best = d;
          }
          res.best_dist = best;
        }
      } catch (...) {}
    }

    results.push_back(res);
    response->basket_colours.push_back(res.colour);
  }


  RCLCPP_INFO(node_->get_logger(), "Task 2 results:");
  for (size_t i = 0; i < results.size(); ++i) {
    const auto & r = results[i];
    RCLCPP_INFO(node_->get_logger(), "  basket %zu: (%.3f, %.3f) -> %s",
      i + 1, r.x, r.y, r.colour.c_str());
  }
  RCLCPP_INFO(node_->get_logger(), "Task 2 done.");
}



///////////////////////////////////////////////////////////////////////////////
// Task 3 — scan scene, detect objects by colour, pick and place
///////////////////////////////////////////////////////////////////////////////

// Internal struct to hold a detected object
struct DetectedObject {
  std::string colour;   // "red", "blue", "purple"
  std::string type;     // "cube" or "basket"
  double x, y, z;       // world-frame centroid
};

// PCL Euclidean cluster extraction using KdTree (efficient)

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

  // ── Step 1: Scan from multiple positions to see full workspace ───────────
  geometry_msgs::msg::Pose scan_pose;
  scan_pose.orientation.x = 1.0;
  scan_pose.orientation.y = 0.0;
  scan_pose.orientation.z = 0.0;
  scan_pose.orientation.w = 0.0;

  // 5-position scan grid covering the full reachable workspace.
  // Baskets/cubes spawn within ~0.65m forward and ±0.40m sideways.
  const std::vector<std::pair<double,double>> scan_positions = {
    {0.40,  0.30},   // near-right
    {0.40, -0.30},   // near-left
    {0.55,  0.35},   // far-right
    {0.55, -0.35},   // far-left
    {0.55,  0.00},   // centre-far
  };

  // Accumulate all coloured points across all scans into one cloud
  pcl::PointCloud<pcl::PointXYZRGB> combined_colour_cloud;

  // Reference colours for filtering (defined early for reuse)
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
    // Settle and then wait for a FRESH cloud after settling
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    // Reset the cloud count baseline AFTER settling so waitForCloud
    // waits for a new message from the current arm position
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

      // Transform point to world frame before accumulating
      geometry_msgs::msg::PointStamped cam_pt, world_pt;
      cam_pt.header.frame_id = cm->header.frame_id;
      cam_pt.header.stamp = cm->header.stamp;
      cam_pt.point.x = pt.x; cam_pt.point.y = pt.y; cam_pt.point.z = pt.z;
      try {
        world_pt = tf_buffer_->transform(cam_pt, "world", tf2::durationFromSec(0.5));
      } catch (...) { continue; }

      pcl::PointXYZRGB wpt;
      wpt.x = static_cast<float>(world_pt.point.x);
      wpt.y = static_cast<float>(world_pt.point.y);
      wpt.z = static_cast<float>(world_pt.point.z);
      wpt.r = pt.r; wpt.g = pt.g; wpt.b = pt.b;
      // Only keep points that are above ground level (z > 0.01 m world)
      if (wpt.z < 0.01f) continue;
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

  // ── Step 2: Cluster in world frame ────────────────────────────────────────
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colour_cloud_ptr(
    new pcl::PointCloud<pcl::PointXYZRGB>(combined_colour_cloud));
  // tolerance=2cm, min 50 pts, max 100000 pts
  auto clusters = pclEuclideanClusters(colour_cloud_ptr, 0.02f, 50, 100000);

  RCLCPP_INFO(node_->get_logger(),
    "Task 3: found %zu clusters", clusters.size());

  // ── Step 4: Classify each cluster ────────────────────────────────────────
  struct RefC { const char* name; float r,g,b; };
  const RefC refs[] = {
    {"blue",   0.1f, 0.1f, 0.8f},
    {"red",    0.8f, 0.1f, 0.1f},
    {"purple", 0.8f, 0.1f, 0.8f},
  };

  // First pass: collect all candidate objects with their point counts
  struct Candidate {
    std::string colour;
    double x, y, z;
    int pts;
  };
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

    // Classify colour
    double best_dist = 1e9;
    const char * colour = nullptr;
    for (const auto & ref : refs) {
      double d = (mr-ref.r)*(mr-ref.r)+(mg-ref.g)*(mg-ref.g)+(mb-ref.b)*(mb-ref.b);
      if (d < best_dist) { best_dist = d; colour = ref.name; }
    }
    if (best_dist > 0.4 || !colour) continue;

    // Points are already in world frame (transformed during accumulation)
    candidates.push_back({std::string(colour), cx, cy, cz, n});

    RCLCPP_INFO(node_->get_logger(),
      "  cluster: %s at world (%.3f, %.3f, %.3f) [%d pts, dist=%.3f]",
      colour, cx, cy, cz, n, best_dist);
  }

  // Second pass: for each colour, the LARGEST cluster = basket, rest = cubes.
  // This avoids a fixed point-count threshold.
  std::vector<DetectedObject> cubes, baskets;
  const std::string colours[] = {"red", "blue", "purple"};

  for (const auto & col : colours) {
    // Find all candidates of this colour
    std::vector<Candidate*> same_colour;
    for (auto & c : candidates) {
      if (c.colour == col) same_colour.push_back(&c);
    }
    if (same_colour.empty()) continue;

    // Sort by point count descending
    std::sort(same_colour.begin(), same_colour.end(),
      [](const Candidate* a, const Candidate* b){ return a->pts > b->pts; });

    // Largest = basket (if it has significantly more points than others,
    // or if there is only one cluster of this colour treat it as basket
    // only if world z > 0.04 m, otherwise it's a cube on the ground)
    // Strategy: largest cluster is basket, all others are cubes.
    // But if there's only one cluster and it looks like a cube (z < 0.04),
    // treat it as a cube with no basket.
    bool has_basket = false;
    for (size_t i = 0; i < same_colour.size(); ++i) {
      const auto * c = same_colour[i];
      DetectedObject obj;
      obj.colour = col;
      obj.x = c->x; obj.y = c->y; obj.z = c->z;

      if (i == 0 && same_colour.size() > 1) {
        // Largest cluster when multiple exist = basket
        obj.type = "basket";
        baskets.push_back(obj);
        has_basket = true;
      } else if (i == 0 && c->z > 0.04) {
        // Only cluster but world z > 4cm = likely basket
        obj.type = "basket";
        baskets.push_back(obj);
        has_basket = true;
      } else {
        obj.type = "cube";
        cubes.push_back(obj);
      }
      RCLCPP_INFO(node_->get_logger(),
        "  detected %s %s at world (%.3f, %.3f, %.3f) [%d pts]",
        obj.colour.c_str(), obj.type.c_str(), obj.x, obj.y, obj.z, c->pts);
    }
    (void)has_basket;
  }

  RCLCPP_INFO(node_->get_logger(),
    "Task 3: %zu cubes, %zu baskets detected", cubes.size(), baskets.size());

  // ── Step 5: Pick and place each cube into matching basket ─────────────────
  // Heights handled internally: moveToLiftXY (~0.55m), moveToGraspZ (~0.12m)

  int placed = 0;
  for (const auto & cube : cubes) {
    // Find matching basket
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

    // ── [1] Open gripper ─────────────────────────────────────────────────────
    RCLCPP_INFO(node_->get_logger(), "  [1] Open gripper");
    setGripper(0.04);

    // ── [2] Joint-space: rotate base + lift to above cube ────────────────────
    // moveToLiftXY computes j1=atan2(y,x) and sets j2-j7 to elbow-down config.
    // Gripper is always perpendicular (j7=pi/4). Fully deterministic.
    RCLCPP_INFO(node_->get_logger(), "  [2] Lift above cube (%.4f, %.4f)", cube.x, cube.y);
    if (!moveToLiftXY(cube.x, cube.y)) {
      RCLCPP_WARN(node_->get_logger(), "  [2] FAILED - skipping");
      continue;
    }

    // ── [3] Joint-space: descend to grasp height ─────────────────────────────
    RCLCPP_INFO(node_->get_logger(), "  [3] Descend to grasp");
    moveToGraspZ(cube.x, cube.y, 0.1434);  // fingertips at cube centre (0.04m)

    // ── [4] Close gripper ────────────────────────────────────────────────────
    RCLCPP_INFO(node_->get_logger(), "  [4] Close gripper, wait 500ms");
    setGripper(0.006);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // ── [5] Joint-space: lift back up ────────────────────────────────────────
    RCLCPP_INFO(node_->get_logger(), "  [5] Lift up");
    moveToLiftXY(cube.x, cube.y);

    // ── [6] Joint-space: rotate base to face basket ──────────────────────────
    RCLCPP_INFO(node_->get_logger(), "  [6] Move above basket (%.4f, %.4f)", target->x, target->y);
    moveToLiftXY(target->x, target->y);

    // ── [7] Release above basket (drop from lift height) ──────────────────────
    RCLCPP_INFO(node_->get_logger(), "  [7] Release above basket");
    setGripper(0.04);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    ++placed;
    RCLCPP_INFO(node_->get_logger(),
      "=== T3 PLACED %s cube (%d/%zu) ===", cube.colour.c_str(), placed, cubes.size());
  }

  RCLCPP_INFO(node_->get_logger(),
    "Task 3 done: placed %d/%zu cubes", placed, cubes.size());
}