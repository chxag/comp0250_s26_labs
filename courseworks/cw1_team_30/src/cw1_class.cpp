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

#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/* --------------------------------------------
Task 2 helper functions:
----------------------------------------------*/

// Blocks until a new point cloud arrives (after moving robot) or timeout occurs.
sensor_msgs::msg::PointCloud2::ConstSharedPtr cw1::waitForCloud(double timeout_sec)
{
  const auto start_time = node_->now(); 
  const uint64_t start_count = cloud_msg_count_.load(std::memory_order_relaxed); // how many cloud have been received so far, to detect new ones

  while (rclcpp::ok()) { // check for new cloud or timeout
    if (cloud_msg_count_.load(std::memory_order_relaxed) > start_count) {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      return latest_cloud_; // if a new cloud has arrived, return it
    }
    if ((node_->now() - start_time).seconds() > timeout_sec) {
      return nullptr;  // timeout
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return nullptr;  // node shutdown
}

// Crop cloud to a ~16cm radius around the basket ---> reduces noise and speeds up processing for colour detection.
// Also applies rudementary depth filtering to remove points that are too far below or above the basket (based on expected cube size and table height).
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cw1::cropAroundBasket(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud, 
  const geometry_msgs::msg::PointStamped & basket_world_loc)
{

  auto cropped_cloud_ptr = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(); 
  if (!cloud) return cropped_cloud_ptr;
  auto stamped_basket_loc = basket_world_loc;
  stamped_basket_loc.header.stamp = cloud->header.stamp;

  // Transform the basket location from world frame to camera frame so we can crop the cloud around it. 
  // This is necessary because the cloud points are in the camera frame.
  geometry_msgs::msg::PointStamped basket_in_cam = transformToCameraFrame(stamped_basket_loc, cloud->header.frame_id);
  if(basket_in_cam.header.frame_id.empty()){
    RCLCPP_ERROR(node_->get_logger(), "Failed to transform basket location to camera frame");
    return cropped_cloud_ptr;
  }

  // Extract the basket location in camera frame as floats for easier processing.
  const float basket_x = static_cast<float>(basket_in_cam.point.x);
  const float basket_y = static_cast<float>(basket_in_cam.point.y);
  const float basket_z = static_cast<float>(basket_in_cam.point.z); // Distance from camera to basket centre (if robot above basket) — used for rudimentary depth filtering

  RCLCPP_INFO(node_->get_logger(),
  "Cloud frame=%s basket_cam=(%.3f, %.3f, %.3f)",
  cloud->header.frame_id.c_str(), basket_x, basket_y, basket_z);

  // Convert ROS PointCloud2 to PCL format for easier processing.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::fromROSMsg(*cloud, *pcl_cloud);

  constexpr float crop_radius = 0.08f; // Crop radius of 8cm around the basket (16cm diameter)
  constexpr float crop_radius_sq = crop_radius * crop_radius;


  /*
  ChatGPT was used to help implement the below depth filtering mechansim as just cropping and plane segmentation was not sufficient to remove all the noise from the table and surrounding area, 
  which was causing significant issues for colour detection accuracy. 
  The depth filtering is a simple way to remove points that are too far below or above the basket based on the expected cube size and table height.
  This significantly improved the colour detection accuracy by only keeping points that are likely to be on the basket and cube.
  */

  // Depth filtering: only consider points within a certain vertical range around the basket. 
  const float min_z = basket_z - 0.15f; // Keeps points that are up to 15cm closer than the basket (to include cube points below basket centre, and account for some depth noise)
  const float max_z = basket_z + 0.06f; // Keeps points that are up to 6cm further than the basket (to include points on the basket walls above the cube, but not points from the table)

  // For every point in the cloud...
  for (const auto & pt : pcl_cloud->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue; // skip invalid points (sensor noise, missing depth, etc)
    if (pt.z < min_z || pt.z > max_z) continue; // keep points within depth range around the basket
    // calculate distance from basket in camera XY plane 
    float dx = pt.x - basket_x; 
    float dy = pt.y - basket_y; 
    if (dx*dx + dy*dy > crop_radius_sq) continue; // keep points within crop radius of basket centre in XY plane
    cropped_cloud_ptr->points.push_back(pt);
  }

  cropped_cloud_ptr->width = static_cast<uint32_t>(cropped_cloud_ptr->points.size());
  cropped_cloud_ptr->height = 1;
  cropped_cloud_ptr->is_dense = false;
  return cropped_cloud_ptr;

}

// Removes noise and floor points from the cropped cloud using a statistical outlier removal filter followed by RANSAC plane segmentation. 
// This helps to improve colour detection accuracy by only keeping points on the basket and cube, and removing points from the table and any remaining sensor noise.
// This follows the rudimentary filtering in cropAroundBasket with a more robust statistical outlier removal and explicit plane segmentation to remove the floor.
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cw1::removeNoiseAndFloor(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud)
{

  auto cleaned_cloud_ptr = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  if (!cloud || cloud->points.empty()) return cleaned_cloud_ptr;


  // Follows Lab 5 filtering steps for plane segmentation, but applied to the cropped cloud around the basket rather than the whole scene.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sor_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(20);
  sor.setStddevMulThresh(1.0);
  sor.filter(*sor_cloud); 
  
  if(sor_cloud->points.empty()) return cleaned_cloud_ptr;

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

  // If no plane was found...
  if(inliers->indices.empty()) { 
    RCLCPP_WARN(node_->get_logger(), "No plane found in point cloud");
    return sor_cloud;  // return noise-filtered cloud without floor removal
  }

  // Extract points that are NOT part of the floor plane (inliers) to get the cleaned cloud with noise and floor removed.
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
  extract.setInputCloud(sor_cloud);
  extract.setIndices(inliers);
  extract.setNegative(true);  // remove floor points
  extract.filter(*cleaned_cloud_ptr);

  return cleaned_cloud_ptr;

}

// Transforms a point from world frame to the target frame (e.g. camera frame) using TF2. 
// Used to transform the basket location into the camera frame for cropping and colour detection.
geometry_msgs::msg::PointStamped cw1::transformToCameraFrame(const geometry_msgs::msg::PointStamped & point_in_world, const std::string & target_frame)
{
  geometry_msgs::msg::PointStamped point_in_cam = point_in_world;
  try{
    point_in_cam = tf_buffer_->transform(point_in_world, target_frame, tf2::durationFromSec(0.5));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(node_->get_logger(), "TF transform failed: %s", ex.what());
    point_in_cam.header.frame_id = "";  // mark as invalid
  }
  return point_in_cam;
}

// Detects the colour of the basket by analysing the average RGB values of the points in the cropped and cleaned cloud around the basket.
std::string cw1::detectBasketColour(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
  const geometry_msgs::msg::PointStamped & basket_world_loc)
{
  if (!cloud) return "none";
  auto cropped_cloud = cropAroundBasket(cloud, basket_world_loc); // crop to points around basket

  // If no points are found around the basket, return "none" as the colour. 
  // This can happen if the basket location is inaccurate, if the basket is not visible in the camera, or if there is too much noise in the cloud.
  if(cropped_cloud->points.empty() || cropped_cloud->empty()) {
    RCLCPP_WARN(node_->get_logger(), "No points found around basket location");
    return "none";
  }
  auto basket_cloud = removeNoiseAndFloor(cropped_cloud); // remove noise and floor points to improve colour detection accuracy

  // If no points are left after noise and floor removal, return "none" as the colour.
  if (basket_cloud->points.empty() || basket_cloud->empty()) {
    RCLCPP_WARN(node_->get_logger(), "No valid points left after noise/floor removal");
    return "none";
  }

  // Initialise sums for RGB values and a count of valid points. We will calculate the average colour of the points in the basket cloud to determine the basket colour.
  double red_sum = 0.0, green_sum = 0.0, blue_sum = 0.0;
  int count = 0; // count of valid points considered for colour detection (used to calculate average and filter out very dark points)
  
  // For each point in the basket cloud...
  for (const auto & pt : basket_cloud->points) {

    // Get RGB values as doubles in the range [0,1] by normalising the uint8 RGB values from the point cloud.
    const double red = static_cast<double>(pt.r) / 255.0; 
    const double green = static_cast<double>(pt.g) / 255.0;
    const double blue = static_cast<double>(pt.b) / 255.0;

    if (std::max({red, green, blue}) < 0.25) continue;  // skip very dark points

    // Accumulate RGB values for valid points to calculate average colour later. 
    red_sum += red;
    green_sum += green;
    blue_sum += blue;
    count++;
  }

  // If we have too few valid points (e.g. due to noise, inaccurate basket location, or occlusion), return "none" as the colour.
  if (count < 20) {
    RCLCPP_WARN(node_->get_logger(), "Not enough valid points to determine colour (only %d)", count);
    return "none";
  }

  // Find average RGB values of the points in the basket cloud to determine the overall colour of the basket. 
  const double red_avg = red_sum / count;
  const double green_avg = green_sum / count;
  const double blue_avg = blue_sum / count;

  // Structure to hold the known colours and their RBG values (as defined in the spec)
  struct KnownColour {  
    const char * name;
    double r, g, b;
  };

  const KnownColour known_colours[] = {
    {"red", 0.8, 0.1, 0.1},
    {"blue", 0.1, 0.1, 0.8},
    {"purple", 0.8, 0.1, 0.8},
  };

  double best_distance_to_colour = 1e9;
  std::string best_colour = "none";

  // For each known colour...
  for (const auto & known_colour : known_colours) {
    // Calculate the Euclidean distance in RGB space between the average colour of the basket cloud and the known colour.
    double dr = red_avg - known_colour.r;
    double dg = green_avg - known_colour.g;
    double db = blue_avg - known_colour.b;
    double distance = std::sqrt(dr*dr + dg*dg + db*db);

    // Keep track of the known colour with the smallest distance to the average colour of the basket cloud. This will be our detected colour for the basket.
    if (distance < best_distance_to_colour) {
      best_distance_to_colour = distance;
      best_colour = known_colour.name;
    }
  } 

  // If the best distance to a known colour is above a certain threshold, we consider the detected colour to be unreliable and return "none". 
  // This can happen if the basket has a lot of noise in the cloud.
  if (best_distance_to_colour > 0.4){
    RCLCPP_WARN(node_->get_logger(), "Detected colour is far from known colours (distance %.3f) — returning 'none'", best_distance_to_colour);
    return "none";
  }

  return best_colour;

}

// Main callback for Task 2 service 
void cw1::t2_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task2Service::Response> response)
{
  RCLCPP_INFO(node_->get_logger(), "Task 2 started: scanning baskets");

  const auto & basket_locs = request->basket_locs;
  std::vector<std::string> basket_colours;

  // For each basket location provided in the request...
  for (size_t i = 0; i < basket_locs.size(); ++i) {
    auto basket_loc = basket_locs[i];
    if (basket_loc.header.frame_id.empty()) {
      basket_loc.header.frame_id = "world";  // assume world frame if not specified
    }
    
    RCLCPP_INFO(node_->get_logger(), "Scanning basket %zu at (%.3f, %.3f)", i+1, basket_loc.point.x, basket_loc.point.y);

    // Move the robot above the basket location to get a clear view for colour detection. 
    // If the move fails (e.g. due to unreachable position), skip colour detection for this basket and return "none" as the colour.
    if (!moveToLiftXY(basket_loc.point.x, basket_loc.point.y)) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to move above basket %zu — skipping colour detection", i+1);
      basket_colours.push_back("none");
      continue;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(400));  // settle time

    // Wait for a fresh cloud after moving above the basket to ensure we are processing the most up-to-date view of the basket for colour detection.
    auto fresh_cloud = waitForCloud(5.0);
    if (!fresh_cloud) {
      RCLCPP_ERROR(node_->get_logger(), "No fresh cloud received after moving above basket %zu — skipping colour detection", i+1);
      basket_colours.push_back("none");
      continue;
    }

    // Detect the colour of the basket using the fresh cloud and the basket location. 
    // This involves cropping the cloud around the basket, removing noise and floor points, and analysing the average RGB values to determine the colour.
    std::string colour = detectBasketColour(fresh_cloud, basket_loc);
    RCLCPP_INFO(node_->get_logger(), "Detected colour for basket %zu: %s ", i+1, colour.c_str());
    basket_colours.push_back(colour);

  }
  response->basket_colours = basket_colours; // return the detected colours for all baskets in the response in the same order as the request. 

  RCLCPP_INFO(node_->get_logger(), "Task 2 done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* --------------------------------------------
Task 3 helper functions:
----------------------------------------------*/

// Applies a voxel grid filter as seen in LAB 5 to downsample the point cloud and reduce noise, which can help improve the accuracy of subsequent processing
// steps like plane segmentation and clustering.
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cw1::applyVoxelFiltering(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) 
{

  auto voxel_filtered_cloud_ptr = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  if (!cloud) return voxel_filtered_cloud_ptr;

  pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
  voxel_filter.setInputCloud(cloud);
  voxel_filter.setLeafSize(0.003f, 0.003f, 0.003f);
  voxel_filter.filter(*voxel_filtered_cloud_ptr);
  return voxel_filtered_cloud_ptr;
}

// Similar to the Task 2 noise and floor removal.
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cw1::applyStatisticalOutlierRemoval(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) 
{
  auto sor_cloud_ptr = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  if (!cloud) return sor_cloud_ptr;

  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(20);
  sor.setStddevMulThresh(1.0);
  sor.filter(*sor_cloud_ptr);
  return sor_cloud_ptr;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cw1::removeFloorWithRANSAC(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &sor_cloud) 
{
  auto cleaned_cloud_ptr = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  if (!sor_cloud) return cleaned_cloud_ptr;

  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100);
  seg.setDistanceThreshold(0.006);
  seg.setInputCloud(sor_cloud);
  seg.segment(*inliers, *coefficients);

  // If no plane was found...
  if(inliers->indices.empty()) { 
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

/*
ChatGPT was used to help point towards the below Euclidean Cluster Extraction method in order to segment individual objects from the cleaned cloud.
The following tutorial was then used as reference for implementation: https://pcl.readthedocs.io/projects/tutorials/en/master/cluster_extraction.html
*/
// Segments the cleaned cloud into individual clusters (objects) using Euclidean Cluster Extraction based on the distance between points.
std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cw1::clusterWithEuclidean(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) 
{
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clusters;
  if (!cloud || cloud->empty()) return clusters;
  // Create a KD-Tree for the cloud to speed up the nearest neighbour search required for clustering.
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
  tree->setInputCloud(cloud);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance(0.02); // 2cm
  ec.setMinClusterSize(30);
  ec.setMaxClusterSize(50000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  for (const auto & indices : cluster_indices) {
    auto cluster = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    for (int idx : indices.indices) {
      cluster->points.push_back(cloud->points[idx]);
    }
    cluster->width = static_cast<uint32_t>(cluster->points.size());
    cluster->height = 1;
    cluster->is_dense = false;
    clusters.push_back(cluster);
  }
  return clusters;
}

geometry_msgs::msg::PointStamped cw1::calculateCentroidInWorldFrame(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cluster, const 
  std::string & cloud_frame, const rclcpp::Time & cloud_timestamp) 
{
  geometry_msgs::msg::PointStamped centroid_world;
  if (!cluster || cluster->empty()) return centroid_world;
  centroid_world.header.frame_id = "";

  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cluster, centroid);

  geometry_msgs::msg::PointStamped centroid_cam;
  centroid_cam.header.frame_id = cloud_frame;
  centroid_cam.header.stamp = cloud_timestamp;
  centroid_cam.point.x = centroid[0];
  centroid_cam.point.y = centroid[1];
  centroid_cam.point.z = centroid[2];

  try{
    centroid_world = tf_buffer_->transform(centroid_cam, "world", tf2::durationFromSec(0.5));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(node_->get_logger(), "TF transform failed: %s", ex.what());
  }
  return centroid_world;
}

std::string cw1::classifyColour(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cluster) 
{
  if (!cluster || cluster->empty()) return "none";

  double red_sum = 0.0, green_sum = 0.0, blue_sum = 0.0;
  int count = 0;

  for (const auto & pt: cluster->points) {
    const double red = static_cast<double>(pt.r) / 255.0;
    const double green = static_cast<double>(pt.g) / 255.0;
    const double blue = static_cast<double>(pt.b) / 255.0;

    if (std::max({red, green, blue}) < 0.15) continue; // skip very dark points

    red_sum += red;
    green_sum += green;
    blue_sum += blue;
    count++;
  }

  if (count < 20) return "none"; // not enough valid points to determine colour

  const double red_avg = red_sum / count;
  const double green_avg = green_sum / count;
  const double blue_avg = blue_sum / count;

  struct KnownColour {
    const char * name;
    double r, g, b;
  };

  const KnownColour known_colours[] = {
    {"red", 0.8, 0.1, 0.1},
    {"blue", 0.1, 0.1, 0.8},
    {"purple", 0.8, 0.1, 0.8},
  };

  double best_distance_to_colour = 1e9;
  std::string best_colour = "none";

  for (const auto & known_colour : known_colours) {
    double dr = red_avg - known_colour.r;
    double dg = green_avg - known_colour.g;
    double db = blue_avg - known_colour.b;
    double distance = std::sqrt(dr*dr + dg*dg + db*db);
    if (distance < best_distance_to_colour) {
      best_distance_to_colour = distance;
      best_colour = known_colour.name;
    }
  }

  if (best_distance_to_colour > 0.4){
  RCLCPP_WARN(node_->get_logger(), "Detected colour is far from known colours (distance %.3f) — returning 'none'", best_distance_to_colour);
  return "none";
  }

  return best_colour;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cw1::accumulateColouredPointClouds(const std::vector<std::pair<double,double>> & scan_positions) 
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr accumulated_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  struct ColRef { float r, g, b; };
  const ColRef col_refs[] = {
    {0.8f, 0.1f, 0.1f},
    {0.1f, 0.1f, 0.8f},
    {0.8f, 0.1f, 0.8f},
  };
  const float thresh_sq = 0.25f * 0.25f * 3.0f;

  for (const auto & pos: scan_positions){
    geometry_msgs::msg::Pose scan_pose;
    scan_pose.orientation.x = 1.0;
    scan_pose.orientation.y = 0.0;
    scan_pose.orientation.z = 0.0;
    scan_pose.orientation.w = 0.0;
    scan_pose.position.x = pos.first;
    scan_pose.position.y = pos.second;
    scan_pose.position.z = 0.60;
    moveToPose(scan_pose);
    std::this_thread::sleep_for(std::chrono::milliseconds(800));

    auto cm = waitForCloud(5.0);
    if (!cm) { RCLCPP_WARN(node_->get_logger(), "No cloud at (%.2f,%.2f)", pos.first, pos.second); continue; }

    pcl::PointCloud<pcl::PointXYZRGB> raw;
    pcl::fromROSMsg(*cm, raw);

    for (const auto & pt : raw.points){
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
      if (pt.z < 0.30f || pt.z > 0.58f) continue;

      float red = pt.r/255.f;
      float green = pt.g/255.f;
      float blue = pt.b/255.f;
      
      if (std::max({red, green, blue}) < 0.20f) continue;

      bool match = false;
      for (const auto & ref : col_refs) {
        float dr=red-ref.r;
        float dg=green-ref.g;
        float db=blue-ref.b;
        if (dr*dr+dg*dg+db*db < thresh_sq) { match=true; break; }
      }
      if (!match) continue;

      geometry_msgs::msg::PointStamped cam_pt, world_pt;
      cam_pt.header.frame_id = cm->header.frame_id;
      cam_pt.header.stamp    = cm->header.stamp;
      cam_pt.point.x = pt.x; cam_pt.point.y = pt.y; cam_pt.point.z = pt.z;
      try {
        world_pt = tf_buffer_->transform(cam_pt, "world", tf2::durationFromSec(0.5));
      } catch (...) { continue; }
      if (world_pt.point.z < 0.03) continue;

      pcl::PointXYZRGB coloured_pt;
      coloured_pt.x = static_cast<float>(world_pt.point.x);
      coloured_pt.y = static_cast<float>(world_pt.point.y);
      coloured_pt.z = static_cast<float>(world_pt.point.z);

      coloured_pt.r = pt.r;
      coloured_pt.g = pt.g;
      coloured_pt.b = pt.b;

      accumulated_cloud->points.push_back(coloured_pt);
    }
    RCLCPP_INFO(node_->get_logger(), "Scan at (%.2f,%.2f): %zu coloured pts accumulated",
      pos.first, pos.second, accumulated_cloud->size());
  }
  return accumulated_cloud;
}

std::string cw1::classifyCubeOrBasket(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cluster, float & size_x, float & size_y, float & size_z) 
{
  size_x = size_y = size_z = 0.0f;
  if (!cluster || cluster->empty()) return "none";

  pcl::PointXYZRGB min_pt, max_pt;
  pcl::getMinMax3D(*cluster, min_pt, max_pt);

  size_x = max_pt.x - min_pt.x;
  size_y = max_pt.y - min_pt.y;
  size_z = max_pt.z - min_pt.z;

  const float max_xy = std::max(size_x, size_y);
  const float min_xy = std::min(size_x, size_y);

  if (max_xy > 0.02 && max_xy < 0.065f && min_xy > 0.015f && size_z < 0.07f) return "cube";
  if (max_xy > 0.07f && max_xy < 0.16f && min_xy > 0.03f) return "basket";
  return "none";

std::vector<cw1::AllObjectsInScene> cw1::detectObjectsInCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & input_cloud)
{
  std::vector<AllObjectsInScene> objects_in_scene;
  if (!input_cloud || input_cloud->empty()) return objects_in_scene;

  auto voxel_filtered = applyVoxelFiltering(input_cloud);
  auto cleaned_cloud = applyStatisticalOutlierRemoval(voxel_filtered);
  auto removed_floor = removeFloorWithRANSAC(cleaned_cloud);
  auto clusters = clusterWithEuclidean(removed_floor);
  for (const auto & cluster : clusters) {
    AllObjectsInScene obj;

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cluster, centroid);
    obj.location_world.header.frame_id = "world";
    obj.location_world.point.x = centroid[0];
    obj.location_world.point.y = centroid[1];
    obj.location_world.point.z = centroid[2];

    obj.colour = classifyColour(cluster);
    obj.typeOfObject = classifyCubeOrBasket(cluster, obj.size_of_object_x, obj.size_of_object_y, obj.size_of_object_z);
    obj.pointCloud = cluster;
    if (obj.typeOfObject == "none" || obj.colour == "none" || obj.location_world.header.frame_id.empty()) continue;
    objects_in_scene.push_back(obj);
    RCLCPP_INFO(node_->get_logger(), "Detected %s of colour %s at (%.3f, %.3f, %.3f)", obj.typeOfObject.c_str(), obj.colour.c_str(), obj.location_world.point.x, obj.location_world.point.y, obj.location_world.point.z);
  }
  return objects_in_scene;
}

void cw1::t3_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  (void)response;

  RCLCPP_INFO(node_->get_logger(), "Task 3 started: scanning scene");

  const std::vector<std::pair<double,double>> scan_positions = {
    {0.35,  0.00},   // centre-near
    {0.35,  0.25},   // near-right
    {0.35, -0.25},   // near-left
    {0.45,  0.35},   // mid-right  (covers right corner)
    {0.45, -0.35},   // mid-left   (covers left corner)
    {0.55,  0.00},   // centre-far
    {0.55,  0.30},   // far-right
    {0.55, -0.30},   // far-left
  };

  auto complete_cloud = accumulateColouredPointClouds(scan_positions);
  auto objects_in_scene = detectObjectsInCloud(complete_cloud);

  RCLCPP_INFO(node_->get_logger(), "Detected %zu unique objects in scene", objects_in_scene.size());

  std::vector<std::pair<cw1::AllObjectsInScene, cw1::AllObjectsInScene>> cube_basket_pairs; // pairs of cubes and their corresponding baskets based on colour
  for (const auto & cube : objects_in_scene) {
    if (cube.typeOfObject != "cube") continue; // only consider cubes for pairing
    cw1::AllObjectsInScene * best_basket = nullptr;
    double best_distance = 1e9;
    for (const auto & basket : objects_in_scene) {
      if (basket.typeOfObject != "basket") continue; // only consider baskets for pairing
      if (basket.colour != cube.colour) continue; // only consider baskets of the same colour as the cube for pairing
      double dx = cube.location_world.point.x - basket.location_world.point.x;
      double dy = cube.location_world.point.y - basket.location_world.point.y;
      double distance = std::sqrt(dx*dx + dy*dy);
      if (distance < best_distance) {
        best_distance = distance;
        best_basket = const_cast<cw1::AllObjectsInScene*>(&basket); // find the closest basket of the same colour to the cube for pairing
      }
    }
    if (best_basket) {
      cube_basket_pairs.push_back(std::make_pair(cube, *best_basket));
      RCLCPP_INFO(node_->get_logger(), "Paired cube at (%.3f, %.3f) with basket at (%.3f, %.3f)", cube.location_world.point.x, cube.location_world.point.y, best_basket->location_world.point.x, best_basket->location_world.point.y);
    } else {
      RCLCPP_WARN(node_->get_logger(), "No matching basket found for cube at (%.3f, %.3f)", cube.location_world.point.x, cube.location_world.point.y);
    }
  }

  // 9. Use Task 1 for pick and place 
  RCLCPP_INFO(node_->get_logger(), "Placing cubes in corresponding baskets");
  int placed = 0;
  for (const auto & pair : cube_basket_pairs) {
    const auto & cube = pair.first;
    const auto & basket = pair.second;
    RCLCPP_INFO(node_->get_logger(), "Placing cube at (%.3f, %.3f) into basket at (%.3f, %.3f)", cube.location_world.point.x, cube.location_world.point.y, basket.location_world.point.x, basket.location_world.point.y);
    
    // Open gripper
    RCLCPP_INFO(node_->get_logger(), " Open gripper");
    setGripper(0.04);

    // Joint-space: rotate base + lift to above cube
    // moveToLiftXY computes j1=atan2(y,x) and sets j2-j7 to elbow-down config.
    // Gripper is always perpendicular (j7=pi/4). Fully deterministic.
    RCLCPP_INFO(node_->get_logger(), " Lift above cube (%.4f, %.4f)", cube.location_world.point.x, cube.location_world.point.y);
    if (!moveToLiftXY(cube.location_world.point.x, cube.location_world.point.y)) {
      RCLCPP_WARN(node_->get_logger(), " FAILED - skipping");
      continue;
    }

    // Descend to grasp height.
    {
      const double cx = cube.location_world.point.x;
      const double cy = cube.location_world.point.y;
      const double cr = std::sqrt(cx * cx + cy * cy);
      const double inward = 0.03;  // counteract moveToGraspZ's outward push
      const double pick_x = cx - inward * (cx / cr);
      const double pick_y = cy - inward * (cy / cr);
      RCLCPP_INFO(node_->get_logger(), " Descend to grasp (adjusted %.4f, %.4f)", pick_x, pick_y);
      moveToGraspZ(pick_x, pick_y, 0.1434);
    }

    // Let arm settle, then close gripper firmly
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    RCLCPP_INFO(node_->get_logger(), " Close gripper, wait 500ms");
    setGripper(0.005);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Joint-space: lift back up
    RCLCPP_INFO(node_->get_logger(), " Lift up");
    moveToLiftXY(cube.location_world.point.x, cube.location_world.point.y);

    // Joint-space: rotate base to face basket
    // Push drop point slightly outward (away from robot base) to compensate
    // for point-cloud centroid bias toward the camera/near side of basket.
    {
      const double bx = basket.location_world.point.x;
      const double by = basket.location_world.point.y;
      const double br = std::sqrt(bx * bx + by * by);
      const double drop_outward = 0.02;
      const double drop_x = bx + drop_outward * (bx / br);
      const double drop_y = by + drop_outward * (by / br);
      RCLCPP_INFO(node_->get_logger(), " Move above basket (%.4f, %.4f) -> drop at (%.4f, %.4f)", bx, by, drop_x, drop_y);
      moveToLiftXY(drop_x, drop_y);
    }

    // Release above basket (drop from lift height)
    RCLCPP_INFO(node_->get_logger(), " Release above basket");
    setGripper(0.04);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    ++placed;
    RCLCPP_INFO(node_->get_logger(),
      "=== T3 PLACED %s cube (%d/%zu) ===", cube.colour.c_str(), placed, cube_basket_pairs.size());
  }

  RCLCPP_INFO(node_->get_logger(),
    "Task 3 done: placed %d/%zu cubes", placed, cube_basket_pairs.size());

  // Return arm to a neutral forward-facing pose so it doesn't freeze in an
  // awkward configuration after the last cube is placed.
  moveToLiftXY(0.45, 0.0);
}