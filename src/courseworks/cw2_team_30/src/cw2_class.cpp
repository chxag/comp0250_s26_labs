/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirement is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#include <cw2_class.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <thread>
#include <atomic>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <unordered_set>
#include <array>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <octomap/OcTree.h>
#include <octomap/OcTreeKey.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <pcl/features/moment_of_inertia_estimation.h>

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
    // Declare parameters and set up point cloud subscription
  pointcloud_topic_ = node_->declare_parameter<std::string>(
    "pointcloud_topic", "/r200/camera/depth_registered/points");
  pointcloud_qos_reliable_ =
    node_->declare_parameter<bool>("pointcloud_qos_reliable", true);

    // point cloud subscription with callback group for accumulation
  pointcloud_callback_group_ =
    node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions pointcloud_sub_options;
  pointcloud_sub_options.callback_group = pointcloud_callback_group_;

  // use reliable QoS if requested, otherwise use sensor-data QoS
  rclcpp::QoS pointcloud_qos = rclcpp::SensorDataQoS();
  if (pointcloud_qos_reliable_) {
    pointcloud_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
  }

  // Subscribe to the point cloud topic
  color_cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
    pointcloud_topic_,
    pointcloud_qos,
    std::bind(&cw2::cloud_callback, this, std::placeholders::_1),
    pointcloud_sub_options);

  // Initialize MoveIt interfaces
  arm_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "panda_arm");
  hand_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "hand");

  arm_group_->setPlanningTime(30.0);
  arm_group_->setNumPlanningAttempts(30);
  arm_group_->setMaxVelocityScalingFactor(0.6);
  arm_group_->setMaxAccelerationScalingFactor(0.6);
  arm_group_->clearPathConstraints();

  // accumulated cloud for Task 3 
  accumulated_cloud_.reset(new PointC);
  is_scanning_ = false;

  RCLCPP_INFO(
    node_->get_logger(),
    "Initialised with pointcloud topic '%s' (%s QoS)",
    pointcloud_topic_.c_str(),
    pointcloud_qos_reliable_ ? "reliable" : "sensor-data");
}

// point cloud callback for accumulating data for Task 3 and providing latest cloud for Task 1/2
void cw2::cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  pcl::PCLPointCloud2 pcl_cloud;
  pcl_conversions::toPCL(*msg, pcl_cloud);

  PointCPtr latest_cloud(new PointC);
  pcl::fromPCLPointCloud2(pcl_cloud, *latest_cloud);

  // Store latest cloud for Task 1 and 2
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    g_input_pc_frame_id_ = msg->header.frame_id;
    g_cloud_ptr = latest_cloud;
    ++g_cloud_sequence_;
  }

  // only keep points on objects and ignores floor (for grasping and collision avoidance)
  if (is_scanning_ && latest_cloud && !latest_cloud->empty()) {
    geometry_msgs::msg::TransformStamped tf_msg;
    try {
      tf_msg = tf_buffer_.lookupTransform(
        "panda_link0",
        msg->header.frame_id,
        tf2::TimePointZero,
        tf2::durationFromSec(0.1));
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
        "TF error during scan accumulation: %s", ex.what());
      return;
    }

    tf2::Transform tf;
    tf2::fromMsg(tf_msg.transform, tf);

    std::lock_guard<std::mutex> lock(accumulated_cloud_mutex_);
    if (!accumulated_cloud_) {
      accumulated_cloud_ = std::make_shared<PointC>();
    }

    // Transform points to arm base frame and accumulate, while filtering out floor points
    for (const auto &pt : latest_cloud->points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
        continue;

      tf2::Vector3 v = tf * tf2::Vector3(pt.x, pt.y, pt.z);

      // filtering out points that are on the floor or too high, or outside the workspace bounds
      if (v.z() < 0.01 || v.z() > 0.5) continue;
      if (v.x() < -0.6 || v.x() > 0.7) continue;
      if (v.y() < -0.6 || v.y() > 0.6) continue;

      PointT new_pt = pt; // copy intensity and other fields if present
      new_pt.x = v.x();
      new_pt.y = v.y();
      new_pt.z = v.z();
      accumulated_cloud_->push_back(new_pt);
    }
  }
}

// Task 1 Helpers 

// Move the arm to a named pose defined in MoveIts configuration
void cw2::moveToNamedPose(const std::string &pose_name)
{
  arm_group_->clearPathConstraints();
  arm_group_->setNamedTarget(pose_name);
  arm_group_->move();
}

// Open the gripper by setting it to the "open" named target
void cw2::openGripper()
{
  hand_group_->setNamedTarget("open");
  hand_group_->move();
}

// Close the gripper by setting it to the "close" named target
void cw2::closeGripper()
{
  hand_group_->setNamedTarget("close");
  hand_group_->move();
}
// Move the arm to a specific pose using MoveIts pose target interface
void cw2::moveToPose(const geometry_msgs::msg::Pose &target_pose)
{
  arm_group_->setStartStateToCurrentState();
  arm_group_->setPoseTarget(target_pose);
  arm_group_->move();
}

// Compute a Cartesian path to the target pose and execute it if the path is mostly valid (fraction > 0.9). 
// This is used for straight-line motions during grasping and placing, where we want to avoid collisions with the table and objects.
bool cw2::computeAndExecuteCartesianPath(const geometry_msgs::msg::Pose &target)
{
  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(target);

  arm_group_->setStartStateToCurrentState();

  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = arm_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
  RCLCPP_INFO(node_->get_logger(), "Cartesian path computed with %.2f%% success", fraction * 100.0);

  if (fraction > 0.9) {
    arm_group_->execute(trajectory);
    return true;
  }
  RCLCPP_WARN(node_->get_logger(), "Cartesian path planning failed with only %.2f%% success", fraction * 100.0);
  return false;
}

// Helper to compute a grasp pose offset from the centroid of the object.
// There are slight adjustments based on shape type and orientation found through testing.
geometry_msgs::msg::Pose cw2::makeAGraspOffset(
  const geometry_msgs::msg::Point &point,
  const std::string &shape_type,
  double z_offset,
  const tf2::Quaternion &orientation,
  double shape_yaw)
{

  // Local frame offset. Either to the nought's wall or one of the cross's arm.
  geometry_msgs::msg::Pose pose;
  double dx = 0.0, dy = 0.0;
  if (shape_type == "nought") {
    dy = 0.08;
  } else if (shape_type == "cross") {
    dx = 0.06;
  }
  
  // Rotate the local offset by the shape's yaw (calculated later) to align the grasp point 
  // with the wall/arm direction of the shape.
  const double c = std::cos(shape_yaw);
  const double s = std::sin(shape_yaw);
  pose.position.x = point.x + dx * c - dy * s;
  pose.position.y = point.y + dx * s + dy * c;
  pose.position.z = point.z + z_offset;
  pose.orientation = tf2::toMsg(orientation);
  return pose;
}

// Estimates the yaw of a shape at a given point by extracting a local point cloud around the point, filtering to the top layer 
// and using orientatied bounding box (OBB) to estimate the object's alignment in the 3D space.
// (References https://pcl.readthedocs.io/projects/tutorials/en/latest/moment_of_inertia.html and https://stackoverflow.com/questions/61589904/not-correct-orientation-of-the-obb-box).
// Returns the yaw in [-pi / 4, pi / 4] since both shapes have 90 degree symmetry. 
double cw2::computeShapeOrientation(const geometry_msgs::msg::PointStamped &query_point)
{
  // Snapshot of latest cloud 
  PointCPtr cloud_snapshot;
  std::string cloud_frame;
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    if (!g_cloud_ptr || g_cloud_ptr->empty() || g_input_pc_frame_id_.empty()) {
      return 0.0;
    }
    cloud_snapshot = g_cloud_ptr;
    cloud_frame    = g_input_pc_frame_id_;
  }
  
  // Transform from the cloud frame to the query point frame (arm base).
  geometry_msgs::msg::TransformStamped tf_msg;
  try {
    tf_msg = tf_buffer_.lookupTransform(
      query_point.header.frame_id,
      cloud_frame,
      tf2::TimePointZero,
      tf2::durationFromSec(0.15));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN(node_->get_logger(), "TF failed: %s", ex.what());
    return 0.0;
  }

  tf2::Transform tf_cloud_to_target;
  tf2::fromMsg(tf_msg.transform, tf_cloud_to_target);

  // Crop the cloud around the query point. 
  std::vector<geometry_msgs::msg::Point> local_points;
  local_points.reserve(4096);

  const double xy_radius    = 0.12; // radius in xy plane to consider points around the query point
  const double xy_radius_sq = xy_radius * xy_radius; // filtering points to be within this radius in the xy plane
  const double z_min        = query_point.point.z - 0.03; // only consider points that are slightly below 
  const double z_max        = query_point.point.z + 0.12; // to moderately above the query point, to focus on the object and ignore table and floating noise 
  double observed_z_max     = -std::numeric_limits<double>::max(); // track highest point observed, to help filter to the top layer later 

  for (const auto &pt : cloud_snapshot->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
    const tf2::Vector3 p_t = tf_cloud_to_target * tf2::Vector3(pt.x, pt.y, pt.z);
    const double dx = p_t.x() - query_point.point.x;
    const double dy = p_t.y() - query_point.point.y;
    if ((dx * dx + dy * dy) > xy_radius_sq) continue;
    if (p_t.z() < z_min || p_t.z() > z_max) continue;
    geometry_msgs::msg::Point p;
    p.x = p_t.x(); p.y = p_t.y(); p.z = p_t.z();
    local_points.push_back(p);
    observed_z_max = std::max(observed_z_max, p.z);
  }

  if (local_points.size() < 80) return 0.0;

  // Keep only approx. top 3.5 cm of points to focus on the top surface of the shape 
  // and remove the sides which can skew OBB.
  const double top_z_min = observed_z_max - 0.035;
  pcl::PointCloud<pcl::PointXYZ>::Ptr top_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  top_cloud->reserve(local_points.size());
  for (const auto &p : local_points) {
    if (p.z >= top_z_min){
      pcl::PointXYZ pt;
      pt.x = p.x; pt.y = p.y; pt.z = p.z;
      top_cloud->push_back(pt);
    }
  }
  if (top_cloud->size() < 30) return 0.0; 

  /* 

  Claude was used to implement the following edge-based filterting to improve OBB orientation estimation for symmetric shapes.

  For the nought, the horizontal covariance has two near equal eigenvalues, so OBB's major axis can be noise-dominated, causing
  the yaw to land anywhere between [0, 90] degrees. By filtering to just the edge points (low density), we leave out the interior
  and get a cloud more biased towards the wall directions. This results in OBB aligning more reliable. This is especially important 
  for the nought, where the grasp orientation is aligned with the wall direction, so a wrong OBB orientation can cause the grasp to 
  be misaligned and fail.

  */
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(top_cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  const double search_radius = 0.008;
  const int density_threshold = 15;

  std::vector<int> point_indices;
  std::vector<float> point_distances;
  for (const auto &pt : top_cloud->points) {
    kdtree.radiusSearch(pt, search_radius, point_indices, point_distances);
    if (point_indices.size() < density_threshold) {
      edge_cloud->push_back(pt);
    }
  }

  // Fir an oriented bounding box to the edge cloud. Its major axis gives 
  // the dominant in-plane direction of the shape. 
  pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
  feature_extractor.setInputCloud(edge_cloud);
  feature_extractor.compute();

  pcl::PointXYZ min_OBB, max_OBB, position_OBB;
  Eigen::Matrix3f rotational_matrix_OBB;
  feature_extractor.getOBB(min_OBB, max_OBB, position_OBB, rotational_matrix_OBB);

  // Take the yaw of the major axis and wrap it to [-pi/4, pi/4] since both shapes have 90 degree symmetry.
  Eigen::Vector3f major = rotational_matrix_OBB.col(0);
  double yaw = std::atan2(major.y(), major.x());
  while (yaw > M_PI / 4.0) yaw += -M_PI / 2.0;
  while (yaw <= -M_PI / 4.0) yaw += M_PI / 2.0;
  return yaw;
}

// Ensures we have received fresh point cloud data after moving the arm, to improve reliability of perception before grasping or placing.
void cw2::waitForFreshCloud(int frames_to_wait, double timeout_sec)
{
  uint64_t start_seq;
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    start_seq = g_cloud_sequence_;
  }

  const auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::duration<double>(timeout_sec);

  while (std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      if (g_cloud_sequence_ >= start_seq + frames_to_wait) return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  RCLCPP_WARN(node_->get_logger(),
              "waitForFreshCloud timed out waiting for %d new frames",
              frames_to_wait);
}

// Main callback function for Task 1.
void cw2::t1_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response)
{
  (void)response;

  const auto &object     = request->object_point.point; // centroid of the object
  const auto &basket     = request->goal_point.point; // centroid of the basket
  const auto &shape_type = request->shape_type; // shape of the object ("nought" or "cross")

  // Wrap the oject point with its frame header so that computeShapeOrientation can transform 
  // it to the cloud frame for orientation estimation.
  geometry_msgs::msg::PointStamped object_stamped;
  object_stamped.header = request->object_point.header;
  object_stamped.point  = object;

  // Gripper orientation (pointing straight down) to observe the shape orientation clearly, or in other words, get a good point cloud.
  tf2::Quaternion observe_orientation;
  observe_orientation.setRPY(M_PI, 0, -M_PI / 4);

  // 1. Move to ready position 
  arm_group_->setNamedTarget("ready");
  
  // 2. Move / position camera directly above object to have a clean top-down view before measuring orientation.
  //    Passing " " as shape type keeps the arm centred on the centroid of the object without any grasping/placing offset.
  moveToPose(makeAGraspOffset(object, " ", 0.5, observe_orientation, 0.0));

  // 3. Wait for a fresh cloud after moving, to ensure we have an up-to-date view of the object before measuring its orientation.
  waitForFreshCloud();

  // 4. Measure the shape orientation from the point cloud. 
  const double shape_yaw = computeShapeOrientation(object_stamped);

  // 5. Build a grasp orientation by combining measured shape orientation with the default gripper yaw. 
  //    This aligns the gripper's fingers with the shape's rotation.
  tf2::Quaternion orientation;
  orientation.setRPY(M_PI, 0, -M_PI / 4 + shape_yaw);

  // 6. Move to a pre-grasp pose above object, with the correct orientation of the gripper to execute a reliable grasp.
  moveToPose(makeAGraspOffset(object, shape_type, 0.5, orientation, shape_yaw));

  // 7. Open gripper.
  openGripper();

  // 8. Move stright down to grasp pose.
  computeAndExecuteCartesianPath(makeAGraspOffset(object, shape_type, 0.15, orientation, shape_yaw));

  // 9. Close gripper to grasp the object.
  closeGripper();

  // 10. Move straight up with the object. 
  computeAndExecuteCartesianPath(makeAGraspOffset(object, shape_type, 0.5, orientation, shape_yaw));

  // 11. Placement orientation goes back to teh the default gripper yaw. The basket is axis aligned so no shape yaw is needed.
  tf2::Quaternion basket_orientation;
  basket_orientation.setRPY(M_PI, 0, -M_PI / 4);

  if (shape_type == "nought") { // if it's a nought...
    //12a. Move to a pose offset towards the wall of the basket, to avoid collisions with the basket edges.
    moveToPose(makeAGraspOffset(basket, "nought", 0.5, basket_orientation, 0.0));
    // 13a. Move straight down to place the nought with a slight offset from basket centroid to avoid collisions.
    computeAndExecuteCartesianPath(makeAGraspOffset(basket, "nought", 0.17, basket_orientation, 0.0));
  } else { // if it's a cross...
    // 12b. Move to a pose offset towards the center of the basket, since the cross is smaller and less likely to collide with the edges.
    moveToPose(makeAGraspOffset(basket, " ", 0.5, basket_orientation, 0.0));
    // 13b. Move straight down to place the cross at the basket centroid.
    computeAndExecuteCartesianPath(makeAGraspOffset(basket, " ", 0.17, basket_orientation, 0.0));
  }

  // 14. Open gripper to release the object.
  openGripper();
  
  // 15. Move straight up after placing.
  computeAndExecuteCartesianPath(makeAGraspOffset(basket, " ", 0.5, basket_orientation, 0.0));

  // 16. Move back to ready position.
  moveToNamedPose("ready");
}

void cw2::t2_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task2Service::Response> response)
{
  response->mystery_object_num = 1;

  tf2::Quaternion view_orientation;
  view_orientation.setRPY(M_PI, 0.0, -M_PI / 4.0);

  auto classify_with_viewpoint =
    [&](const geometry_msgs::msg::PointStamped &point_stamped) -> std::string {
      geometry_msgs::msg::Pose view_pose;
      view_pose.position.x  = point_stamped.point.x;
      view_pose.position.y  = point_stamped.point.y;
      view_pose.position.z  = point_stamped.point.z + 0.50;
      view_pose.orientation = tf2::toMsg(view_orientation);
      moveToPose(view_pose);

      std::uint64_t before_seq = 0;
      {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        before_seq = g_cloud_sequence_;
      }

      const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(900);
      while (std::chrono::steady_clock::now() < deadline) {
        {
          std::lock_guard<std::mutex> lock(cloud_mutex_);
          if (g_cloud_sequence_ > before_seq) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
      }

      return classifyShapeAtPoint(point_stamped);
    };

  std::vector<std::string> ref_shapes;
  ref_shapes.reserve(request->ref_object_points.size());
  for (std::size_t i = 0; i < request->ref_object_points.size(); ++i) {
    const auto shape = classify_with_viewpoint(request->ref_object_points[i]);
    ref_shapes.push_back(shape);
    RCLCPP_INFO(node_->get_logger(),
      "Task2 reference #%zu classified as: %s", i + 1, shape.c_str());
  }

  const auto mystery_shape = classify_with_viewpoint(request->mystery_object_point);
  RCLCPP_INFO(node_->get_logger(), "Task2 mystery classified as: %s", mystery_shape.c_str());

  int64_t match_index = 1;
  if (ref_shapes.size() >= 2) {
    const bool match_ref1 = (mystery_shape != "unknown" && mystery_shape == ref_shapes[0]);
    const bool match_ref2 = (mystery_shape != "unknown" && mystery_shape == ref_shapes[1]);
    if      (match_ref1 && !match_ref2)                          match_index = 1;
    else if (match_ref2 && !match_ref1)                          match_index = 2;
    else if (ref_shapes[0] == "unknown" && ref_shapes[1] != "unknown") match_index = 2;
    else                                                         match_index = 1;
  }
  response->mystery_object_num = match_index;

  RCLCPP_INFO(node_->get_logger(),
    "Task2 final output: ref1=%s ref2=%s mystery=%s -> mystery_object_num=%lld",
    ref_shapes.size() > 0 ? ref_shapes[0].c_str() : "n/a",
    ref_shapes.size() > 1 ? ref_shapes[1].c_str() : "n/a",
    mystery_shape.c_str(),
    static_cast<long long>(response->mystery_object_num));

  moveToNamedPose("ready");
}

// Helper function to classify shape at a given point using the latest point cloud
std::string cw2::classifyShapeAtPoint(const geometry_msgs::msg::PointStamped &query_point)
{
  PointCPtr cloud_snapshot;
  std::string cloud_frame;
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    if (!g_cloud_ptr || g_cloud_ptr->empty() || g_input_pc_frame_id_.empty()) {
      return "unknown";
    }
    cloud_snapshot = g_cloud_ptr;
    cloud_frame    = g_input_pc_frame_id_;
  }
  // Transform the query point to the cloud frame
  geometry_msgs::msg::TransformStamped tf_msg;
  try {
    tf_msg = tf_buffer_.lookupTransform(
      query_point.header.frame_id,
      cloud_frame,
      tf2::TimePointZero,
      tf2::durationFromSec(0.15));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN(node_->get_logger(), "classifyShapeAtPoint TF failed: %s", ex.what());
    return "unknown";
  }

  tf2::Transform tf_cloud_to_target;
  tf2::fromMsg(tf_msg.transform, tf_cloud_to_target);

  std::vector<geometry_msgs::msg::Point> local_points;
  local_points.reserve(4096);

  const double xy_radius    = 0.12; // radius in xy plane to consider points around the query point
  const double xy_radius_sq = xy_radius * xy_radius; // filtering points to be within this radius in the xy plane
  const double z_min        = query_point.point.z - 0.03;
  const double z_max        = query_point.point.z + 0.12;
  double observed_z_max     = -std::numeric_limits<double>::max();

  
  for (const auto &pt : cloud_snapshot->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }
    const tf2::Vector3 p_t = tf_cloud_to_target * tf2::Vector3(pt.x, pt.y, pt.z);
    const double dx = p_t.x() - query_point.point.x;
    const double dy = p_t.y() - query_point.point.y;
    if ((dx * dx + dy * dy) > xy_radius_sq) continue;
    if (p_t.z() < z_min || p_t.z() > z_max)  continue;
    geometry_msgs::msg::Point p;
    p.x = p_t.x(); p.y = p_t.y(); p.z = p_t.z();
    local_points.push_back(p);
    observed_z_max = std::max(observed_z_max, p.z);
  }

  if (local_points.size() < 80) return "unknown";

  const double top_z_min = observed_z_max - 0.035;
  std::vector<geometry_msgs::msg::Point> top_points;
  top_points.reserve(local_points.size());
  for (const auto &p : local_points)
    if (p.z >= top_z_min) top_points.push_back(p);
  if (top_points.size() < 30) return "unknown";

  double cx = 0.0, cy = 0.0;
  for (const auto &p : top_points) { cx += p.x; cy += p.y; }
  cx /= static_cast<double>(top_points.size());
  cy /= static_cast<double>(top_points.size());

  const double center_radius    = 0.005;
  const double center_radius_sq = center_radius * center_radius;
  int center_count = 0;
  for (const auto &p : top_points) {
    const double dx = p.x - cx, dy = p.y - cy;
    if (dx * dx + dy * dy <= center_radius_sq) ++center_count;
  }

  return (center_count < 20) ? "nought" : "cross";
}


// Task 3 

  // struct to hold detected object information
  struct T3Object {
    std::string category;   // "object" or "basket" 
    std::string shape;      // "nought" or "cross"
    geometry_msgs::msg::Point centroid;
    double min_x, max_x, min_y, max_y, min_z, max_z;
    float mean_r, mean_g, mean_b;
    PointCPtr cloud;
  };

void cw2::t3_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  response->total_num_shapes = 0;
  response->num_most_common_shape = 0;

  RCLCPP_INFO(node_->get_logger(), "Task 3 started: scanning environment and building OctoMap");

  // Clearing previous data and resetting state for Task 3
  {
    std::lock_guard<std::mutex> lk(accumulated_cloud_mutex_);
    if (!accumulated_cloud_) accumulated_cloud_.reset(new PointC);
    accumulated_cloud_->clear();
  }
  if (latest_octree_) latest_octree_->clear();
  is_scanning_ = false;
  arm_group_->clearPathConstraints();

  // downward orientation for scanning poses
  tf2::Quaternion down_q;
  down_q.setRPY(M_PI, 0.0, -M_PI / 4.0); // 45 degree tilt to get better view of object tops

  // create a pose from x,y,z with the fixed downward orientation
  auto makePose = [&](double x, double y, double z) -> geometry_msgs::msg::Pose {
    geometry_msgs::msg::Pose p;
    p.position.x = x; p.position.y = y; p.position.z = z;
    p.orientation = tf2::toMsg(down_q);
    return p;
  };

  // scan sub area defined by 4 corner points, returns false if path planning failed
  auto scanSubArea = [&](const std::vector<std::array<double,3>> &corners) -> bool {
    if (corners.size() != 4) return false;
    moveToPose(makePose(corners[0][0], corners[0][1], corners[0][2]));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // generate waypoints in a boustrophedon pattern between the corners
    std::vector<geometry_msgs::msg::Pose> waypoints;
    for (int i = 1; i < 4; ++i)
      waypoints.push_back(makePose(corners[i][0], corners[i][1], corners[i][2]));

    moveit_msgs::msg::RobotTrajectory trajectory;
    arm_group_->setStartStateToCurrentState();
    double fraction = arm_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);
    if (fraction < 0.90) {
      RCLCPP_WARN(node_->get_logger(), "Scan Cartesian path only %.1f%%", fraction * 100.0);
      return false;
    }

    // execute the planned trajectory while allowing the point cloud callback to accumulate data
    is_scanning_ = true;
    arm_group_->execute(trajectory);
    is_scanning_ = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return true;
  };

  openGripper();
  moveToNamedPose("ready");

  // save initial joint state to return to between scans
  std::vector<double> initial_joints;
  try { initial_joints = arm_group_->getCurrentJointValues(); } catch (...) {}
  auto goToInitial = [&]() {
    arm_group_->clearPathConstraints();
    if (!initial_joints.empty()) {
      arm_group_->setJointValueTarget(initial_joints);
      arm_group_->move();
    } else {
      moveToNamedPose("ready");
    }
  };

  // Scan the area in 4 passes from different angles to get better coverage, starting with a front view
  moveToPose(makePose(0.5, 0.0, 0.5));
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  RCLCPP_INFO(node_->get_logger(), "T3: scanning front sub-area");
  scanSubArea({{{0.5,0.45,0.5},{0.5,-0.45,0.5},{0.3,-0.45,0.5},{0.3,0.45,0.5}}});
  goToInitial();

  RCLCPP_INFO(node_->get_logger(), "T3: scanning left sub-area");
  scanSubArea({{{0.10,0.45,0.5},{0.10,0.40,0.5},{-0.10,0.40,0.5},{-0.10,0.45,0.5}}});
  goToInitial();

  RCLCPP_INFO(node_->get_logger(), "T3: scanning right sub-area");
  scanSubArea({{{0.10,-0.45,0.5},{0.10,-0.40,0.5},{-0.10,-0.40,0.5},{-0.10,-0.45,0.5}}});
  goToInitial();

  RCLCPP_INFO(node_->get_logger(), "T3: scanning back sub-area");
  scanSubArea({{{-0.5,-0.45,0.5},{-0.5,0.45,0.5},{-0.40,0.45,0.5},{-0.40,-0.45,0.5}}});

  // Downsample accumulated cloud to speed up OctoMap insertion and clustering, using a voxel grid filter with 3mm leaf size
  PointCPtr acc_ds(new PointC);
  {
    std::lock_guard<std::mutex> lk(accumulated_cloud_mutex_);
    if (!accumulated_cloud_ || accumulated_cloud_->empty()) {
      RCLCPP_ERROR(node_->get_logger(), "T3: accumulated cloud empty – aborting");
      goToInitial(); return;
    }
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(accumulated_cloud_);
    vg.setLeafSize(0.003f, 0.003f, 0.003f); // 3mm voxel grid
    vg.filter(*acc_ds);
  }

  // ===============================================================
  // Preprocess cloud before clustering:
  //   1. z-filter to drop tile top
  //   2. split by colour into 3 streams: shapes (saturated), basket
  //      (low-saturation dark red), obstacles (near-black).
  //      Green (tile) points are dropped entirely.
  //   3. cluster each stream separately → no cross-class bridging.
  // ===============================================================

  pcl::PassThrough<PointT> pt_filter;
  pt_filter.setInputCloud(acc_ds);
  pt_filter.setFilterFieldName("z");
  pt_filter.setFilterLimits(0.035f, 0.50f);
  PointCPtr acc_ds_f(new PointC);
  pt_filter.filter(*acc_ds_f);
  acc_ds = acc_ds_f;

  PointCPtr cloud_shapes_red(new PointC);
  PointCPtr cloud_shapes_blue(new PointC);
  PointCPtr cloud_shapes_purple(new PointC);
  PointCPtr cloud_basket(new PointC);
  PointCPtr cloud_obstacles(new PointC);

  for (const auto &pt : acc_ds->points) {
    const uint8_t maxc = std::max({pt.r, pt.g, pt.b});
    const uint8_t minc = std::min({pt.r, pt.g, pt.b});
    const bool green_dominant = (pt.g > pt.r + 15 && pt.g > pt.b + 15);

    if (green_dominant) continue;                 // tile — drop

    if (maxc < 30) {
      cloud_obstacles->push_back(pt);             // black pillar
    } else {
      const float ratio = (float)maxc / std::max<uint8_t>(minc, 1);
      if (ratio > 3.0f && pt.z < 0.070f) {
        // Split shapes by hue so different-colour shapes can't bridge in clustering
        const bool r_high = (pt.r > pt.g && pt.r > pt.b);
        const bool b_high = (pt.b > pt.g && pt.b > pt.r);
        const bool rb_both = (pt.r > pt.g && pt.b > pt.g &&
                              std::abs((int)pt.r - (int)pt.b) < 40);
        if (rb_both)      cloud_shapes_purple->push_back(pt);
        else if (r_high)  cloud_shapes_red->push_back(pt);
        else if (b_high)  cloud_shapes_blue->push_back(pt);
        else              cloud_shapes_red->push_back(pt);  // rare fallback
      } else if (maxc > 35 && pt.r > pt.g + 10 && pt.r > pt.b + 10) {
        cloud_basket->push_back(pt);              // dark red = basket
      }
    }
  }

  RCLCPP_INFO(node_->get_logger(),
    "T3: colour split — red=%zu blue=%zu purple=%zu basket=%zu obstacles=%zu (total=%zu)",
    cloud_shapes_red->size(), cloud_shapes_blue->size(), cloud_shapes_purple->size(),
    cloud_basket->size(), cloud_obstacles->size(), acc_ds->size());
  
  std::vector<T3Object> detected;

  auto makeDetected =
    [&](const PointCPtr &src, double tol, const std::string &cat) {
      if (!src || src->size() < 80) return;

      pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
      tree->setInputCloud(src);
      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<PointT> ec;
      ec.setClusterTolerance(tol);
      ec.setMinClusterSize(80);
      ec.setMaxClusterSize(500000);
      ec.setSearchMethod(tree);
      ec.setInputCloud(src);
      ec.extract(cluster_indices);

      RCLCPP_INFO(node_->get_logger(), "T3[%s]: %zu clusters",
                  cat.c_str(), cluster_indices.size());

      for (const auto &indices : cluster_indices) {
        if (indices.indices.size() < 80) continue;

        T3Object d;
        d.category = cat;
        d.shape = "N/A";
        d.cloud.reset(new PointC);
        d.cloud->reserve(indices.indices.size());
        d.centroid.x = d.centroid.y = d.centroid.z = 0.0;

        double min_x=1e9, max_x=-1e9, min_y=1e9, max_y=-1e9, min_z=1e9, max_z=-1e9;
        double sum_r=0, sum_g=0, sum_b=0;
        for (int idx : indices.indices) {
          const PointT &pt = src->points[idx];
          d.cloud->push_back(pt);
          min_x = std::min(min_x, (double)pt.x); max_x = std::max(max_x, (double)pt.x);
          min_y = std::min(min_y, (double)pt.y); max_y = std::max(max_y, (double)pt.y);
          min_z = std::min(min_z, (double)pt.z); max_z = std::max(max_z, (double)pt.z);
          sum_r += pt.r; sum_g += pt.g; sum_b += pt.b;
        }
        const double n = (double)indices.indices.size();
        d.min_x=min_x; d.max_x=max_x;
        d.min_y=min_y; d.max_y=max_y;
        d.min_z=min_z; d.max_z=max_z;
        d.mean_r = (float)(sum_r / n);
        d.mean_g = (float)(sum_g / n);
        d.mean_b = (float)(sum_b / n);

        // Sanity: reject bridged clusters. Shapes are strict (40 mm tall max);
        // basket can be up to 50 mm; obstacles up to 120 mm.
        const double extent = std::max(max_x - min_x, max_y - min_y);
        const double cluster_h = max_z - min_z;
        if (cat == "object" && (extent > 0.37 || cluster_h > 0.055)) {
          RCLCPP_WARN(node_->get_logger(),
            "T3[object]: rejecting (bridged?) ext=%.2f h=%.3f",
            extent, cluster_h);
          continue;
        }
        if (cat == "basket" && (extent > 0.45 || cluster_h > 0.07)) {
          RCLCPP_WARN(node_->get_logger(),
            "T3[basket]: rejecting ext=%.2f h=%.3f", extent, cluster_h);
          continue;
        }
        if (cat == "obstacle" && (extent > 0.35 || cluster_h > 0.15)) {
          RCLCPP_WARN(node_->get_logger(),
            "T3[obstacle]: rejecting ext=%.2f h=%.3f", extent, cluster_h);
          continue;
        }

        // Shape-specific: centroid + nought/cross classification
        if (cat == "object") {
          // Use bounding-box centre as the shape's geometric centre.
          const double cx = 0.5 * (min_x + max_x);
          const double cy = 0.5 * (min_y + max_y);

          d.centroid.x = cx; d.centroid.y = cy; d.centroid.z = 0.0;

          // Snap measured extent to nominal shape size.
          double nominal_bbox;
          if      (extent < 0.125) nominal_bbox = 0.100;
          else if (extent < 0.175) nominal_bbox = 0.150;
          else                     nominal_bbox = 0.200;
          const double x_size = nominal_bbox / 5.0;
          const double r = x_size / 2.5;
          const double r2 = r * r;

          // Count cluster points within radius r of the bbox centre, using the
          // ENTIRE cluster (not just the top band). For a nought the central
          // region is empty at every height (hole goes all the way through).
          // For a cross the central region is filled at every height (solid
          // central column). Works even with incomplete top-surface scans.
          int centre_count = 0;
          for (const auto &pt : d.cloud->points) {
            const double ddx = pt.x - cx, ddy = pt.y - cy;
            if (ddx*ddx + ddy*ddy <= r2) ++centre_count;
          }

          // Scale threshold by cluster size — more points means a higher count is needed
          // to avoid noise dominating. About 1% of total points is a good heuristic.
          const int threshold = std::max(15, (int)(d.cloud->size() * 0.01));
          d.shape = (centre_count >= threshold) ? "cross" : "nought";

          RCLCPP_DEBUG(node_->get_logger(),
            "T3[object]: centre_count=%d threshold=%d total_pts=%zu → %s",
            centre_count, threshold, d.cloud->size(), d.shape.c_str());
        }

        detected.push_back(d);
      }
    };

  // Tolerances:
  //  - shapes: 6 mm (compact, don't bridge at this size)
  //  - basket: 10 mm (contiguous single object, looser ok)
  //  - obstacles: 12 mm (consolidate any fragmentation of black pillars)
  makeDetected(cloud_shapes_red,    0.006, "object");
  makeDetected(cloud_shapes_blue,   0.006, "object");
  makeDetected(cloud_shapes_purple, 0.006, "object");
  makeDetected(cloud_basket,        0.006, "basket");
  makeDetected(cloud_obstacles,     0.012, "obstacle");

  for (const auto &d : detected) {
    RCLCPP_INFO(node_->get_logger(),
      "T3 cluster: cat=%-8s shape=%-6s ext=(%.2f,%.2f) h=%.3f @(%.2f,%.2f) rgb=(%.0f,%.0f,%.0f)",
      d.category.c_str(), d.shape.c_str(),
      d.max_x - d.min_x, d.max_y - d.min_y, d.max_z - d.min_z,
      (d.min_x + d.max_x) * 0.5, (d.min_y + d.max_y) * 0.5,
      d.mean_r, d.mean_g, d.mean_b);
  }

  // Count detected objects and determine target shape based on majority
  int n_noughts = 0, n_crosses = 0;
  for (const auto &d : detected)
    if (d.category == "object") {
      if (d.shape == "nought") ++n_noughts;
      else if (d.shape == "cross") ++n_crosses;
    }

   // If no shapes detected then log an error and return to initial pose 
  if (n_noughts + n_crosses == 0) {
    RCLCPP_ERROR(node_->get_logger(), "T3: no shapes detected");
    goToInitial(); return;
  }

  // Determine target shape based on majority or randomly if same
  std::string target_shape = (n_noughts > n_crosses) ? "nought" :
                             (n_crosses > n_noughts) ? "cross" :
                             (std::rand() % 2 ? "nought" : "cross");

  response->total_num_shapes = static_cast<int64_t>(n_noughts + n_crosses);
  response->num_most_common_shape = (n_noughts == n_crosses) ?
    static_cast<int64_t>(n_noughts) : static_cast<int64_t>(std::max(n_noughts, n_crosses));

  // Finding basket and target object based on detected clusters
  T3Object basket, target;
  bool basket_ok = false, target_ok = false;

  // Find biggest basket cluster (real basket is ~0.35 m, noise is smaller)
  double best_basket_extent = 0.0;
  for (auto &d : detected) {
    if (d.category == "basket") {
      const double ext_b = std::max(d.max_x - d.min_x, d.max_y - d.min_y);
      if (ext_b > best_basket_extent) {
        best_basket_extent = ext_b;
        basket = d;
        basket_ok = (ext_b > 0.15);  // accept only if sensibly sized
      }
    }
    if (!target_ok && d.category == "object" && d.shape == target_shape) {
      target = d; target_ok = true;
    }
  }

  // Basket can only be at (-0.41, ±0.36) per spawner. Snap to closer one.
  if (basket_ok) {
    const std::array<std::pair<double, double>, 2> sites = {{
      {-0.41, -0.36}, {-0.41, 0.36}
    }};
    double best_d2 = 1e9;
    std::pair<double,double> snap = sites[0];
    for (const auto &s : sites) {
      const double ddx = basket.centroid.x - s.first;
      const double ddy = basket.centroid.y - s.second;
      const double d2 = ddx*ddx + ddy*ddy;
      if (d2 < best_d2) { best_d2 = d2; snap = s; }
    }
    basket.centroid.x = snap.first;
    basket.centroid.y = snap.second;
    basket.centroid.z = 0.04;
    RCLCPP_INFO(node_->get_logger(),
      "T3: basket snapped to known location (%.2f, %.2f)",
      snap.first, snap.second);
  }

  
  if (!target_ok) {
    RCLCPP_ERROR(node_->get_logger(), "T3: target object missing");
    goToInitial(); return;
  }
  if (!basket_ok) {
    RCLCPP_WARN(node_->get_logger(), "T3: basket not detected – using fallback");
    basket.centroid.x = -0.41;
    basket.centroid.y = (target.centroid.y < 0) ? -0.36 : 0.36;
    basket.centroid.z = 0.04;
  }

  // Add detected obstacles for collision avoidance during pick-and-place
  std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
  int obs_idx = 0; // unique ID for each obstacle
  for (const auto& d : detected) {
    if (d.category == "obstacle") {
      moveit_msgs::msg::CollisionObject co;
      co.id = "obs_" + std::to_string(obs_idx++);
      co.header.frame_id = arm_group_->getPlanningFrame();
      co.primitives.resize(1);
      co.primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
      co.primitives[0].dimensions.resize(3);          // <-- fixed for Humble
      co.primitives[0].dimensions[0] = d.max_x - d.min_x + 0.02;
      co.primitives[0].dimensions[1] = d.max_y - d.min_y + 0.02;
      co.primitives[0].dimensions[2] = d.max_z - d.min_z + 0.02;

      // place the box primitive at the center of the detected obstacle cluster
      co.primitive_poses.resize(1);
      co.primitive_poses[0].position.x = (d.min_x + d.max_x) * 0.5;
      co.primitive_poses[0].position.y = (d.min_y + d.max_y) * 0.5;
      co.primitive_poses[0].position.z = (d.min_z + d.max_z) * 0.5;
      co.primitive_poses[0].orientation.w = 1.0;
      co.operation = moveit_msgs::msg::CollisionObject::ADD;
      collision_objects.push_back(co);
    }
  }
  if (!collision_objects.empty()) {
    planning_scene_interface_.addCollisionObjects(collision_objects);
    RCLCPP_INFO(node_->get_logger(), "T3: added %zu obstacles to planning scene", collision_objects.size());
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // Compute grasp and place poses based on detected target object and basket
  goToInitial();

  geometry_msgs::msg::PointStamped target_stamped;
  target_stamped.header.frame_id = arm_group_->getPlanningFrame();
  target_stamped.header.stamp = node_->get_clock()->now();
  target_stamped.point = target.centroid;

  geometry_msgs::msg::Point target_point = target.centroid;
  geometry_msgs::msg::Point basket_point = basket.centroid;

  tf2::Quaternion observe_orientation;
  observe_orientation.setRPY(M_PI, 0, -M_PI / 4);

  moveToPose(makeAGraspOffset(target_point, " ", 0.5, observe_orientation, 0.0));

  waitForFreshCloud();

  const double shape_yaw = computeShapeOrientation(target_stamped);
  RCLCPP_INFO(node_->get_logger(), "T3: target object classified as %s with yaw %.2f degrees", target_shape.c_str(), shape_yaw * 180.0 / M_PI);

  tf2::Quaternion grasp_orientation;
  grasp_orientation.setRPY(M_PI, 0, -M_PI / 4 + shape_yaw);

  moveToPose(makeAGraspOffset(target_point, target_shape, 0.5, grasp_orientation, shape_yaw));

  openGripper();
  computeAndExecuteCartesianPath(makeAGraspOffset(target_point, target_shape, 0.15, grasp_orientation, shape_yaw));

  closeGripper();
  computeAndExecuteCartesianPath(makeAGraspOffset(target_point, target_shape, 0.5, grasp_orientation, shape_yaw));

  tf2::Quaternion basket_orientation;
  basket_orientation.setRPY(M_PI, 0, -M_PI / 4);

  if (target_shape == "nought") {
    moveToPose(makeAGraspOffset(basket_point, "nought", 0.5, basket_orientation, 0.0));
    computeAndExecuteCartesianPath(makeAGraspOffset(basket_point, "nought", 0.17, basket_orientation, 0.0));
  } else {
    moveToPose(makeAGraspOffset(basket_point, " ", 0.5, basket_orientation, 0.0));
    computeAndExecuteCartesianPath(makeAGraspOffset(basket_point, " ", 0.17, basket_orientation, 0.0));
  }

  openGripper();
  computeAndExecuteCartesianPath(makeAGraspOffset(basket_point, " ", 0.5, basket_orientation, 0.0));

  moveToNamedPose("ready");

  RCLCPP_INFO(node_->get_logger(),
    "Task 3 complete: picked a %s total=%lld most_common=%lld ",
    target_shape.c_str(),
    static_cast<long long>(response->total_num_shapes),
    static_cast<long long>(response->num_most_common_shape));
}