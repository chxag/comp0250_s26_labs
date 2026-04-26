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
#include <pcl/io/pcd_io.h>
#include <pcl/filters/radius_outlier_removal.h>

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
      const auto stamp = tf2_ros::fromMsg(msg->header.stamp);
      tf_msg = tf_buffer_.lookupTransform(
        "panda_link0",
        msg->header.frame_id,
        stamp,
        tf2::durationFromSec(0.25));
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
      if (v.z() < 0.03|| v.z() > 0.5) continue;
      if (v.x() < -0.6 || v.x() > 0.7) continue;
      if (v.y() < -0.6 || v.y() > 0.6) continue;

      PointT new_pt = pt; // copy intensity and other fields if present
      new_pt.x = v.x();
      new_pt.y = v.y();
      new_pt.z = v.z();
      const double arm_base_xy_radius_sq = 0.15 * 0.15;
      if ((v.x() * v.x() + v.y() * v.y()) < arm_base_xy_radius_sq) continue; 
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


std::vector<DetectedObj> cw2::classifyAccumulatedCloud()
{
  std::vector<DetectedObj> detected;

  // Snapshot accumulated cloud
  PointCPtr snapshot(new PointC);
  {
    std::lock_guard<std::mutex> lock(accumulated_cloud_mutex_);
    if (!accumulated_cloud_ || accumulated_cloud_->empty()) return detected;
    *snapshot = *accumulated_cloud_;
  }

  // ---- HSV helpers ----
  // RGB→HSV per-point. H in [0, 360), S and V in [0, 1].
  auto rgbToHSV = [](float r, float g, float b, float &H, float &S, float &V) {
    r /= 255.0f; g /= 255.0f; b /= 255.0f;
    const float mx = std::max({r, g, b});
    const float mn = std::min({r, g, b});
    const float d  = mx - mn;
    V = mx;
    S = (mx > 1e-6f) ? (d / mx) : 0.0f;
    if (d < 1e-6f) { H = 0.0f; return; }
    if      (mx == r) H = 60.0f * std::fmod((g - b) / d, 6.0f);
    else if (mx == g) H = 60.0f * ((b - r) / d + 2.0f);
    else              H = 60.0f * ((r - g) / d + 4.0f);
    if (H < 0) H += 360.0f;
  };

  // Median-over-cluster HSV. Robust to ~30% edge contamination, unlike mean RGB.
  auto medianHSV = [&](const PointCPtr &cluster, float &H, float &S, float &V) {
    std::vector<float> hs, ss, vs;
    hs.reserve(cluster->size());
    ss.reserve(cluster->size());
    vs.reserve(cluster->size());
    for (const auto &p : cluster->points) {
      float h, s, v;
      rgbToHSV(p.r, p.g, p.b, h, s, v);
      hs.push_back(h); ss.push_back(s); vs.push_back(v);
    }
    auto med = [](std::vector<float> &x) {
      std::nth_element(x.begin(), x.begin() + x.size()/2, x.end());
      return x[x.size()/2];
    };
    // Hue is circular (0 ≈ 360), so naive median fails for reds spanning the wrap.
    // Project to unit circle, take mean direction, convert back.
    double sx = 0.0, sy = 0.0;
    for (float h : hs) {
      const double rad = h * M_PI / 180.0;
      sx += std::cos(rad); sy += std::sin(rad);
    }
    double h_mean = std::atan2(sy, sx) * 180.0 / M_PI;
    if (h_mean < 0) h_mean += 360.0;
    H = static_cast<float>(h_mean);
    S = med(ss);
    V = med(vs);
  };

  // ---- Filter: drop floor (HSV-green), arm base, height bounds ----
  PointCPtr filtered(new PointC);
  filtered->reserve(snapshot->size());
  for (const auto &pt : snapshot->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
    if (pt.z < 0.035 || pt.z > 0.25) continue;
    if ((pt.x * pt.x + pt.y * pt.y) < (0.15 * 0.15)) continue;
    
    // HSV-based green floor rejection: catches the table without rejecting
    // dark or saturated colors that happen to have high green channel.
    float h, s, v;
    rgbToHSV(pt.r, pt.g, pt.b, h, s, v);
    if (h > 80.0f && h < 160.0f && s > 0.25f && v > 0.20f) continue;  // green

    filtered->push_back(pt);
  }
  if (filtered->empty()) return detected;

  PointCPtr voxelised(new PointC);
  pcl::VoxelGrid<PointT> voxel;
  voxel.setInputCloud(filtered);
  voxel.setLeafSize(0.005f, 0.005f, 0.005f);
  voxel.filter(*voxelised);

  PointCPtr cleaned(new PointC);
  pcl::RadiusOutlierRemoval<PointT> ror;
  ror.setInputCloud(voxelised);
  ror.setRadiusSearch(0.02);
  ror.setMinNeighborsInRadius(4);
  ror.filter(*cleaned);

  // Cluster
  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud(cleaned);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(0.012);
  ec.setMinClusterSize(200);
  ec.setMaxClusterSize(100000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cleaned);
  ec.extract(cluster_indices);

  RCLCPP_INFO(node_->get_logger(), "Detected %zu clusters", cluster_indices.size());

  static constexpr std::array<std::pair<double, double>, 2> BASKET_SPAWN_LOCS = {{
    {-0.41, -0.36}, {-0.41,  0.36}
  }};

  std::vector<PointCPtr> clusters_to_process;
  for (size_t i = 0; i < cluster_indices.size(); ++i) {
    PointCPtr cluster(new PointC);
    cluster->reserve(cluster_indices[i].indices.size());
    for (int idx : cluster_indices[i].indices) cluster->push_back(cleaned->points[idx]);

    // Quick footprint check
    double mnx=1e9, mxx=-1e9, mny=1e9, mxy=-1e9;
    for (const auto &p : cluster->points) {
      mnx = std::min(mnx, (double)p.x); mxx = std::max(mxx, (double)p.x);
      mny = std::min(mny, (double)p.y); mxy = std::max(mxy, (double)p.y);
    }
    const double max_dim = std::max(mxx - mnx, mxy - mny);

    if (max_dim < 0.40) {
      clusters_to_process.push_back(cluster);
    } else {
      RCLCPP_INFO(node_->get_logger(), "Cluster %zu (max_dim=%.2f) re-splitting", i, max_dim);
      pcl::search::KdTree<PointT>::Ptr sub_tree(new pcl::search::KdTree<PointT>);
      sub_tree->setInputCloud(cluster);
      std::vector<pcl::PointIndices> sub_indices;
      pcl::EuclideanClusterExtraction<PointT> sub_ec;
      sub_ec.setClusterTolerance(0.008);
      sub_ec.setMinClusterSize(150);
      sub_ec.setMaxClusterSize(100000);
      sub_ec.setSearchMethod(sub_tree);
      sub_ec.setInputCloud(cluster);
      sub_ec.extract(sub_indices);

      for (const auto &sub : sub_indices) {
        PointCPtr sub_c(new PointC);
        for (int idx : sub.indices) sub_c->push_back(cluster->points[idx]);
        clusters_to_process.push_back(sub_c);
      }
    }
  }

  for (size_t i = 0; i < clusters_to_process.size(); ++i) {
    PointCPtr cluster = clusters_to_process[i];

    // Centroid + AABB
    Eigen::Vector4f c4;
    pcl::compute3DCentroid(*cluster, c4);
    double mnx=1e9, mny=1e9, mnz=1e9, mxx=-1e9, mxy=-1e9, mxz=-1e9;
    for (const auto &p : cluster->points) {
      mnx = std::min(mnx, (double)p.x); mxx = std::max(mxx, (double)p.x);
      mny = std::min(mny, (double)p.y); mxy = std::max(mxy, (double)p.y);
      mnz = std::min(mnz, (double)p.z); mxz = std::max(mxz, (double)p.z);
    }

    DetectedObj obj;
    obj.centroid.x = 0.5 * (mnx + mxx);
    obj.centroid.y = 0.5 * (mny + mxy);
    obj.centroid.z = c4[2];
    obj.min_x = mnx; obj.max_x = mxx;
    obj.min_y = mny; obj.max_y = mxy;
    obj.min_z = mnz; obj.max_z = mxz;

    const double fx = mxx - mnx, fy = mxy - mny;
    const double max_dim = std::max(fx, fy);
    const size_t pts = cluster->size();

    // ---- Sanity floor: reject phantoms ----
    constexpr size_t kMinValidPts       = 600;
    constexpr double kMinValidFootprint = 0.07;
    constexpr double kMaxValidFootprint = 0.45;
    if (pts < kMinValidPts ||
        max_dim < kMinValidFootprint ||
        max_dim > kMaxValidFootprint)
    {
      RCLCPP_INFO(node_->get_logger(),
        "Cluster %zu REJECTED (pts=%zu, footprint=%.2fx%.2f) — likely noise",
        i, pts, fx, fy);
      continue;
    }

    // ---- HSV classification ----
    float H, S, V;
    medianHSV(cluster, H, S, V);

    auto near_basket = [&]() {
      for (const auto &b : BASKET_SPAWN_LOCS) {
        const double dx = obj.centroid.x - b.first;
        const double dy = obj.centroid.y - b.second;
        if (dx*dx + dy*dy < 0.15 * 0.15) return true;
      }
      return false;
    };

    const bool is_large    = (fx > 0.3 || fy > 0.3);
    const bool is_dark     = (V < 0.15f);                            // black obstacle
    const bool is_gray     = (S < 0.20f && V >= 0.15f);              // washed-out — phantom
    // Reddish hue: near 0 or 360 in HSV. Basket also tends to be lower V than fresh red shape.
    const bool is_reddish  = ((H < 20.0f || H > 340.0f) && S > 0.30f);

    // Reject mid-saturation gray clusters — these are the phantom self-points
    // that mean-RGB couldn't catch. A real shape always has S > ~0.4.
    if (is_gray) {
      RCLCPP_INFO(node_->get_logger(),
        "Cluster %zu REJECTED (gray: H=%.0f S=%.2f V=%.2f) — phantom",
        i, H, S, V);
      continue;
    }

    if (is_large && is_reddish && near_basket()) {
      obj.category = "basket";
    } else if (is_dark) {
      obj.category = "obstacle";
    } else {
      obj.category = "object";
    }

    RCLCPP_INFO(node_->get_logger(),
      "Cluster %zu: cat=%s pts=%zu centroid=(%.2f, %.2f, %.2f) footprint=%.2fx%.2f HSV=(%.0f, %.2f, %.2f)",
      i, obj.category.c_str(), pts,
      obj.centroid.x, obj.centroid.y, obj.centroid.z, fx, fy, H, S, V);

    detected.push_back(obj);
  }
  return detected;
}


// Task 3 
void cw2::t3_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  response->total_num_shapes = 0;
  response->num_most_common_shape = 0;

  {
    std::lock_guard<std::mutex> lock(accumulated_cloud_mutex_);
    if (!accumulated_cloud_) accumulated_cloud_.reset(new PointC);
    accumulated_cloud_->clear();
  }
  is_scanning_ = false;
  arm_group_->clearPathConstraints();

  tf2::Quaternion down_q;
  down_q.setRPY(M_PI, 0, -M_PI / 4);

  auto makePose = [&](double x, double y, double z) -> geometry_msgs::msg::Pose {
    geometry_msgs::msg::Pose pose;
    pose.position.x = x;
    pose.position.y = y;
    pose.position.z = z;
    pose.orientation = tf2::toMsg(down_q);
    return pose;
  };

  auto observeAt = [&](double x, double y, double z, const std::string &label){
    is_scanning_ = false;
    moveToPose(makePose(x, y, z));
    std::this_thread::sleep_for(std::chrono::milliseconds(700));
    is_scanning_ = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    is_scanning_ = false;
    RCLCPP_INFO(node_->get_logger(), "T3 observed at %s (%.2f, %.2f, %.2f)",
                label.c_str(), x, y, z);
  };
  openGripper();
  moveToNamedPose("ready");

  std::vector<double> initial_joints;
  try { initial_joints = arm_group_->getCurrentJointValues(); } catch (...) {}
  auto goToInitial = [&](){
    arm_group_->clearPathConstraints();
    if (!initial_joints.empty()) {
      arm_group_->setJointValueTarget(initial_joints);
      arm_group_->move();
    } else {
      moveToNamedPose("ready");
    }
  };

  const double scan_z = 0.6;

  // Front row
  observeAt( 0.50,  0.40, scan_z, "front_left");
  observeAt( 0.50,  0.00, scan_z, "front_centre");
  observeAt( 0.50, -0.40, scan_z, "front_right");
  goToInitial();

  // Middle row  
  observeAt( 0.10,  0.40, scan_z, "mid_left");
  observeAt( 0.10, -0.40, scan_z, "mid_right");
  goToInitial();

  observeAt(-0.30,  0.40, scan_z, "back_left");
  observeAt(-0.30,  0.00, scan_z, "back_centre");
  observeAt(-0.30, -0.40, scan_z, "back_right");
  goToInitial();

  observeAt(-0.50,  0.00, scan_z, "deep_back");
  goToInitial();

  if (accumulated_cloud_ && !accumulated_cloud_->empty()) {
    pcl::io::savePCDFileBinary("/home/charvi/t3_scan.pcd", *accumulated_cloud_);
    RCLCPP_INFO(node_->get_logger(), "Saved scan to /home/charvi/t3_scan.pcd");
  }
  
  auto detected = classifyAccumulatedCloud();
  RCLCPP_INFO(node_->get_logger(), "T3 Phase 2: located %zu objects", detected.size());

  RCLCPP_INFO(node_->get_logger(), "T3 Phase 3: inspecting each candidate");

  for (auto &obj : detected) {
    if (obj.category != "object") continue;

    geometry_msgs::msg::Pose inspect = makePose(obj.centroid.x, obj.centroid.y, 0.5);
    moveToPose(inspect);
    waitForFreshCloud(3, 2.0);

    // Use Task 2 classifier
    geometry_msgs::msg::PointStamped query;
    query.header.frame_id = "panda_link0";
    query.point.x = obj.centroid.x;
    query.point.y = obj.centroid.y;
    query.point.z = obj.centroid.z;
    obj.shape = classifyShapeAtPoint(query);

    RCLCPP_INFO(node_->get_logger(),
      "Inspected (%.2f, %.2f) -> %s",
      obj.centroid.x, obj.centroid.y, obj.shape.c_str());
  }

  goToInitial();

  int nought_count = 0, cross_count = 0;
  for (const auto &obj : detected) {
    if (obj.category == "object") {
      if (obj.shape == "nought") ++nought_count;
      else if (obj.shape == "cross") ++cross_count;
    }
  }

  const int total_shapes = nought_count + cross_count;
  const int most_common = std::max(nought_count, cross_count);
  response->total_num_shapes = total_shapes;
  response->num_most_common_shape = most_common;

  RCLCPP_INFO(node_->get_logger(), "T3 Pick and Place");

  if (total_shapes == 0) {
    RCLCPP_INFO(node_->get_logger(), "No shapes detected.");
    return;
  }

  const std::string target_shape = (most_common == nought_count) ? "nought" : "cross";

  const DetectedObj* basket_ptr = nullptr;
  for (const auto &obj : detected) {
    if (obj.category == "basket") { basket_ptr = &obj; break; }
  }
  if (!basket_ptr) {
    RCLCPP_WARN(node_->get_logger(), "No basket detected, cannot proceed with Task 3 pick and place.");
    return;
  }

  std::vector<const DetectedObj*> candidates;
  for (const auto &obj : detected) {
    if (obj.category == "object" && obj.shape == target_shape) {
      candidates.push_back(&obj);
    }
  }
  if (candidates.empty()) {
    RCLCPP_WARN(node_->get_logger(), "No objects of the most common shape (%s) detected, cannot proceed with Task 3 pick and place.", target_shape.c_str());
    return;
  }

  std::sort(candidates.begin(), candidates.end(),
    [](const DetectedObj* a, const DetectedObj* b) {
      const double da = std::hypot(a->centroid.x, a->centroid.y);
      const double db = std::hypot(b->centroid.x, b->centroid.y);
      return da < db;
    });

    auto planning_scene_interface = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();
    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    int obstacle_id = 0;
    for (const auto &obj : detected) {
      if (obj.category == "obstacle") {
        moveit_msgs::msg::CollisionObject co;
        co.header.frame_id = arm_group_->getPlanningFrame();
        co.id = "obstacle_" + std::to_string(obstacle_id++);

        shape_msgs::msg::SolidPrimitive box;
        box.type = shape_msgs::msg::SolidPrimitive::BOX;
        box.dimensions.resize(3);

        box.dimensions[0] = obj.max_x - obj.min_x + 0.04;
        box.dimensions[1] = obj.max_y - obj.min_y + 0.04;
        box.dimensions[2] = obj.max_z - obj.min_z + 0.04;
        
        geometry_msgs::msg::Pose box_pose;
        box_pose.orientation.w = 1.0;
        box_pose.position.x = obj.centroid.x;
        box_pose.position.y = obj.centroid.y;
        box_pose.position.z = 0.5 * (obj.min_z + obj.max_z);

        co.primitives.push_back(box);
        co.primitive_poses.push_back(box_pose);
        co.operation = moveit_msgs::msg::CollisionObject::ADD;
        collision_objects.push_back(co);
      }
    }
    planning_scene_interface->applyCollisionObjects(collision_objects);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    bool placed = false;
    for (const DetectedObj *target : candidates) {
      geometry_msgs::msg::Point target_point = target->centroid;
      target_point.z = 0.0;

      geometry_msgs::msg::Point basket_point = basket_ptr->centroid;
      basket_point.z = 0.0;

      goToInitial();
      tf2::Quaternion observe_orienation;
      observe_orienation.setRPY(M_PI, 0, -M_PI / 4);
      moveToPose(makeAGraspOffset(target_point, target_shape, 0.5, observe_orienation, 0.0));
      waitForFreshCloud();

      geometry_msgs::msg::PointStamped target_stamped;
      target_stamped.header.frame_id = arm_group_->getPlanningFrame();
      target_stamped.header.stamp = node_->get_clock()->now();
      target_stamped.point = target_point;
      const double shape_yaw = computeShapeOrientation(target_stamped);

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

      placed = true;
      RCLCPP_INFO(node_->get_logger(), "Successfully placed a %s in the basket!", target_shape.c_str());
      break;
    }

    std::vector<std::string> remove_objects;
    for (const auto &co : collision_objects) {
      remove_objects.push_back(co.id);
    }
    planning_scene_interface->removeCollisionObjects(remove_objects);

    if (!placed) {
      RCLCPP_WARN(node_->get_logger(), "Failed to place any %s in the basket.", target_shape.c_str());
    }

    moveToNamedPose("ready");

  RCLCPP_INFO(node_->get_logger(),
    "T3 counting result: noughts=%d crosses=%d total=%d most_common=%d",
    nought_count, cross_count, total_shapes, most_common);

}
