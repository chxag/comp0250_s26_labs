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

  arm_group_->setPlanningTime(50.0);
  arm_group_->setNumPlanningAttempts(50);
  arm_group_->setMaxVelocityScalingFactor(0.5);
  arm_group_->setMaxAccelerationScalingFactor(0.5);
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
      // filter out points that are too close to the arm base to avoid accumulating noise from the arm itself.
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
  double shape_yaw, 
  double size)
{

  // Local frame offset. Either to the nought's wall or one of the cross's arm.
  geometry_msgs::msg::Pose pose;
  double dx = 0.0, dy = 0.0;
  if (shape_type == "nought") {
    dy = 0.4 * size;
  } else if (shape_type == "cross") {
    dx = 0.3 * size;
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
  moveToPose(makeAGraspOffset(object, " ", 0.5, observe_orientation, 0.0, 0.2));

  // 3. Wait for a fresh cloud after moving, to ensure we have an up-to-date view of the object before measuring its orientation.
  waitForFreshCloud();

  // 4. Measure the shape orientation from the point cloud. 
  const double shape_yaw = computeShapeOrientation(object_stamped);

  // 5. Build a grasp orientation by combining measured shape orientation with the default gripper yaw. 
  //    This aligns the gripper's fingers with the shape's rotation.
  tf2::Quaternion orientation;
  orientation.setRPY(M_PI, 0, -M_PI / 4 + shape_yaw);

  // 6. Move to a pre-grasp pose above object, with the correct orientation of the gripper to execute a reliable grasp.
  moveToPose(makeAGraspOffset(object, shape_type, 0.5, orientation, shape_yaw, 0.2));

  // 7. Open gripper.
  openGripper();

  // 8. Move stright down to grasp pose.
  computeAndExecuteCartesianPath(makeAGraspOffset(object, shape_type, 0.15, orientation, shape_yaw, 0.2));

  // 9. Close gripper to grasp the object.
  closeGripper();

  // 10. Move straight up with the object. 
  computeAndExecuteCartesianPath(makeAGraspOffset(object, shape_type, 0.5, orientation, shape_yaw, 0.2));

  // 11. Placement orientation goes back to teh the default gripper yaw. The basket is axis aligned so no shape yaw is needed.
  tf2::Quaternion basket_orientation;
  basket_orientation.setRPY(M_PI, 0, -M_PI / 4);

  if (shape_type == "nought") { // if it's a nought...
    //12a. Move to a pose offset towards the wall of the basket, to avoid collisions with the basket edges.
    moveToPose(makeAGraspOffset(basket, "nought", 0.5, basket_orientation, 0.0, 0.2));
    // 13a. Move straight down to place the nought with a slight offset from basket centroid to avoid collisions.
    computeAndExecuteCartesianPath(makeAGraspOffset(basket, "nought", 0.17, basket_orientation, 0.0, 0.2));
  } else { // if it's a cross...
    // 12b. Move to a pose offset towards the center of the basket, since the cross is smaller and less likely to collide with the edges.
    moveToPose(makeAGraspOffset(basket, " ", 0.5, basket_orientation, 0.0, 0.2));
    // 13b. Move straight down to place the cross at the basket centroid.
    computeAndExecuteCartesianPath(makeAGraspOffset(basket, " ", 0.17, basket_orientation, 0.0, 0.2));
  }

  // 14. Open gripper to release the object.
  openGripper();
  
  // 15. Move straight up after placing.
  computeAndExecuteCartesianPath(makeAGraspOffset(basket, " ", 0.5, basket_orientation, 0.0, 0.2));

  // 16. Move back to ready position.
  moveToNamedPose("ready");
}

// Task 2
void cw2::t2_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task2Service::Response> response)
{
  response->mystery_object_num = 1;
  // The same viewpoint is used for classifying all objects to ensure a fair comparison between the mystery object and the reference objects, 
  // and to simplify the implementation by not having to compute separate orientations for each object. 
  tf2::Quaternion view_orientation;
  view_orientation.setRPY(M_PI, 0.0, -M_PI / 4.0);

  // Helper function to move to a viewpoint above the given point, wait for a fresh cloud, and classify the shape at that point.
  auto classify_with_viewpoint =
    [&](const geometry_msgs::msg::PointStamped &point_stamped) -> std::string {
      geometry_msgs::msg::Pose view_pose;
      view_pose.position.x  = point_stamped.point.x;
      view_pose.position.y  = point_stamped.point.y;
      view_pose.position.z  = point_stamped.point.z + 0.50;
      view_pose.orientation = tf2::toMsg(view_orientation);
      moveToPose(view_pose); // move to the viewpoint above the object

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

      // Classification
      return classifyShapeAtPoint(point_stamped);
    };

  std::vector<std::string> ref_shapes;
  ref_shapes.reserve(request->ref_object_points.size());

  // Classify each reference object and log the results. 
  // The classifications will be used to determine which reference object the mystery object matches with.
  for (std::size_t i = 0; i < request->ref_object_points.size(); ++i) {
    const auto shape = classify_with_viewpoint(request->ref_object_points[i]);
    ref_shapes.push_back(shape);
    RCLCPP_INFO(node_->get_logger(),
      "Task2 reference #%zu classified as: %s", i + 1, shape.c_str());
  }

  const auto mystery_shape = classify_with_viewpoint(request->mystery_object_point);
  RCLCPP_INFO(node_->get_logger(), "Task2 mystery classified as: %s", mystery_shape.c_str());

  // Determine which reference object the mystery object matches with, based on the classifications.
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

  moveToNamedPose("ready");

  RCLCPP_INFO(node_->get_logger(),
    "Task2 final output: ref1=%s ref2=%s mystery=%s -> mystery_object_num=%lld",
    ref_shapes.size() > 0 ? ref_shapes[0].c_str() : "n/a",
    ref_shapes.size() > 1 ? ref_shapes[1].c_str() : "n/a",
    mystery_shape.c_str(),
    static_cast<long long>(response->mystery_object_num));
}

// Classifies the shape ("nought" or "cross") at the given 3D query point using the latest point cloud snapshot.
// The classification is based on whether the top surface of the shape has occupied points at its geometric centre:
//   nought (O-shape): hollow centre -> no points near centroid -> center_count low -> "nought"
//   cross  (+ shape): solid centre  -> points cluster at centroid -> center_count high -> "cross"
// Returns "unknown" if the point cloud is unavailable or too sparse to make a reliable decision.
std::string cw2::classifyShapeAtPoint(const geometry_msgs::msg::PointStamped &query_point)
{
  // --- Step 1: Take a thread-safe snapshot of the latest point cloud ---
  // We copy the shared pointer under the mutex so that the rest of the function can work on a consistent cloud without holding the lock.
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

  // --- Step 2: Look up the transform from the sensor frame to the arm base frame ---
  geometry_msgs::msg::TransformStamped tf_msg;
  try {
    tf_msg = tf_buffer_.lookupTransform(
      query_point.header.frame_id,  // target frame: arm base (panda_link0)
      cloud_frame,                  // source frame: camera / sensor frame
      tf2::TimePointZero,           // use the latest available transform
      tf2::durationFromSec(0.15));  // wait up to 150ms for the transform to become available
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN(node_->get_logger(), "classifyShapeAtPoint TF failed: %s", ex.what());
    return "unknown";
  }

  tf2::Transform tf_cloud_to_target;
  tf2::fromMsg(tf_msg.transform, tf_cloud_to_target);

  // --- Step 3: Crop the cloud to a local cylinder around the query point ---
  // We only keep points that are:
  //   (a) within xy_radius (12 cm) of the query point in the horizontal plane.
  //   (b) within a vertical band [z_min, z_max] around the known object height to discard table surface noise below and floating sensor artefacts above.
  // We also track observed_z_max to identify the top surface in the next step.
  std::vector<geometry_msgs::msg::Point> local_points;
  local_points.reserve(4096);

  const double xy_radius    = 0.12; // horizontal crop radius [m]
  const double xy_radius_sq = xy_radius * xy_radius;
  const double z_min        = query_point.point.z - 0.03;  // 3 cm below centroid (table tolerance)
  const double z_max        = query_point.point.z + 0.12;  // 12 cm above centroid (object height + margin)
  double observed_z_max     = -std::numeric_limits<double>::max(); // running max z in the cropped region

  for (const auto &pt : cloud_snapshot->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue; // skip NaN / Inf points produced by the depth sensor at range boundaries
    }
    // Transform the point from the camera frame into the arm base frame
    const tf2::Vector3 p_t = tf_cloud_to_target * tf2::Vector3(pt.x, pt.y, pt.z);
    const double dx = p_t.x() - query_point.point.x;
    const double dy = p_t.y() - query_point.point.y;
    if ((dx * dx + dy * dy) > xy_radius_sq) continue; // outside horizontal cylinder
    if (p_t.z() < z_min || p_t.z() > z_max)  continue; // outside vertical band
    geometry_msgs::msg::Point p;
    p.x = p_t.x(); p.y = p_t.y(); p.z = p_t.z();
    local_points.push_back(p);
    observed_z_max = std::max(observed_z_max, p.z);
  }

  if (local_points.size() < 80) return "unknown";

  // --- Step 4: Keep only the top 3.5 cm layer (the visible top surface) ---
  // Side-wall points would shift the centroid and pollute the centre-density test,
  // so we discard everything below the topmost observed point minus a small margin.
  const double top_z_min = observed_z_max - 0.035;
  std::vector<geometry_msgs::msg::Point> top_points;
  top_points.reserve(local_points.size());
  for (const auto &p : local_points)
    if (p.z >= top_z_min) top_points.push_back(p);
  if (top_points.size() < 30) return "unknown"; // too few top-surface points

  // --- Step 5: Compute the 2D centroid of the top-surface point cloud ---
  // For both nought and cross the centroid lies at the geometric centre of the shape,
  // because both shapes are rotationally symmetric around their centre.
  double cx = 0.0, cy = 0.0;
  for (const auto &p : top_points) { cx += p.x; cy += p.y; }
  cx /= static_cast<double>(top_points.size());
  cy /= static_cast<double>(top_points.size());

  // --- Step 6: Count points within a 5 mm radius of the centroid ---
  // classification criterion:
  //   nought: the centre is a hollow hole → almost no points near (cx, cy) -> center_count ≈ 0
  //   cross:  the centre is solid material → many points near (cx, cy)     -> center_count >> 0
  const double center_radius    = 0.005; // 5 mm decision radius
  const double center_radius_sq = center_radius * center_radius;
  int center_count = 0;
  for (const auto &p : top_points) {
    const double dx = p.x - cx, dy = p.y - cy;
    if (dx * dx + dy * dy <= center_radius_sq) ++center_count;
  }

  // Empirically setting a threshold of 20 points.
  return (center_count < 20) ? "nought" : "cross";
}

// Task 3 Helper

// Clusters accumulated point cloud into individual objects and classifies them.
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

  // Drop points that are likely on the green floor or are too high, and points that are likely noise (e.g. halo pixels at object boundaries).
  PointCPtr filtered(new PointC);
  filtered->reserve(snapshot->size());
  for (const auto &pt : snapshot->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue; // invalid points
    if (pt.z > 0.25) continue; // points that are too high 
    if (pt.g > 1.3f * pt.r && pt.g > 1.3f * pt.b && pt.g > 60.0f) continue; //  points that are likely on the green floor
    filtered->push_back(pt);
  }
  if (filtered->empty()) return detected;

  // Voxel grid filter to downsample and unify point density. 
  PointCPtr voxelised(new PointC);
  pcl::VoxelGrid<PointT> voxel;
  voxel.setInputCloud(filtered);
  voxel.setLeafSize(0.005f, 0.005f, 0.005f);
  voxel.filter(*voxelised);

  // Radius outlier removel to drop isolated points that are likely noise, e.g. residual arm points, sensor artefacts etc.
  // Added as a result of testing to improve robustness of clustering and classification.
  PointCPtr cleaned(new PointC);
  pcl::RadiusOutlierRemoval<PointT> ror;
  ror.setInputCloud(voxelised);
  ror.setRadiusSearch(0.02);
  ror.setMinNeighborsInRadius(4);
  ror.filter(*cleaned);

  // Euclidean clustering to find individual object clusters.
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

  // Known basket spawn locations used to help identify clusters that are likely the baskets based on proximity, 
  // instead of relying on just colour.
  static constexpr std::array<std::pair<double, double>, 2> BASKET_SPAWN_LOCATIONS = {{
    {-0.41, -0.36}, {-0.41, 0.36}
  }};

  // Categorise each cluster
  for (size_t i = 0; i < cluster_indices.size(); ++i) {
    PointCPtr cluster(new PointC); // extract the cluster point cloud
    cluster->reserve(cluster_indices[i].indices.size());
    for (int index : cluster_indices[i].indices) {
      cluster->push_back(cleaned->points[index]);
    }

    // Compute the centroid and axis-aligned bounding box of the cluster. 
    // The centroid is used for computing object positions, and the bounding box is used for size estimation (obstacles)
    // and filtering out noise clusters and large clusters that are likely the table or floor.
    Eigen::Vector4f c4;
    pcl::compute3DCentroid(*cluster, c4);
    double min_x = 1e9, min_y = 1e9, min_z = 1e9, max_x = -1e9, max_y = -1e9, max_z = -1e9;
    for (const auto &pt : cluster->points) {
      min_x = std::min(min_x, (double)pt.x); max_x = std::max(max_x, (double)pt.x);
      min_y = std::min(min_y, (double)pt.y); max_y = std::max(max_y, (double)pt.y);
      min_z = std::min(min_z, (double)pt.z); max_z = std::max(max_z, (double)pt.z);
    }

    DetectedObj obj;
    // The centroid is the midpoint of the bounding box in x and y since it's more robust to partial views.
    // PCL centroid fine for z since all visible points sit on shape's top surface.
    obj.centroid.x = 0.5 * (min_x + max_x);
    obj.centroid.y = 0.5 * (min_y + max_y);
    obj.centroid.z = c4[2];
    obj.min_x = min_x; obj.max_x = max_x;
    obj.min_y = min_y; obj.max_y = max_y;
    obj.min_z = min_z; obj.max_z = max_z;

    const double footprint_x = max_x - min_x;
    const double footprint_y = max_y - min_y;
    const double max_dim = std::max({footprint_x, footprint_y, max_z - min_z});
    const size_t points = cluster->size();

    // Reject phantom and degenerate clusters that are too small (likely noise), or too large (likely the table or floor).
    if (points < 600 || max_dim < 0.07 || max_dim > 0.45){
      continue;
    }

    // Mean RBG normalised to [0,1] to get a simple colour descriptor for classification.
    double mean_r = 0, mean_g = 0, mean_b = 0;
    for (const auto &pt : cluster->points) {
      mean_r += pt.r;
      mean_g += pt.g;
      mean_b += pt.b;
    }
    const double n = static_cast<double>(cluster->size());
    mean_r /= (n * 255.0);
    mean_g /= (n * 255.0);
    mean_b /= (n * 255.0);

    // If the cluster's centroid is near a known basket location, and the cluster is large and reddish, we classify it as a basket.
    auto near_basket = [&](){
      for (const auto &b : BASKET_SPAWN_LOCATIONS) {
        const double dx = obj.centroid.x - b.first;
        const double dy = obj.centroid.y - b.second;
        if ((dx * dx + dy * dy) < 0.15 * 0.15) return true;
      }
      return false;
    };

    // Three simple rules based on colour and size to classify the cluster:
    //   - If it's very dark, it's likely an obstacle (e.g. the black box).
    //   - If it's large, reddish, and near a known basket location, it's likely a basket.
    //   - Otherwise, it's classified as a generic object.
    const bool is_dark = (mean_r < 0.10 && mean_g < 0.10 && mean_b < 0.10);
    const bool is_large = (footprint_x > 0.30 || footprint_y > 0.30);
    const bool is_reddish = (mean_r > 1.5 * mean_g && mean_r > 1.5 * mean_b && mean_r > 0.15);

    if (is_dark) {
      obj.category = "obstacle";
    } else if (is_large && is_reddish && near_basket()) {
      obj.category = "basket";
    } else {
      obj.category = "object";
    }

    detected.push_back(obj);
  }

  return detected;
}

// Main callback function for Task 3.
void cw2::t3_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  response->total_num_shapes = 0;
  response->num_most_common_shape = 0;

  // Reset accumulated cloud and state for a fresh scan.
  {
    std::lock_guard<std::mutex> lock(accumulated_cloud_mutex_);
    if (!accumulated_cloud_) accumulated_cloud_.reset(new PointC);
    accumulated_cloud_->clear();
  }
  is_scanning_ = false;
  arm_group_->clearPathConstraints();

  // Standard end-effector orientation, pointing straight down. 
  tf2::Quaternion down_q;
  down_q.setRPY(M_PI, 0, -M_PI / 4);

  // Helper function to create a pose with the standard down-facing orientation and given position.
  auto makePose = [&](double x, double y, double z) -> geometry_msgs::msg::Pose {
    geometry_msgs::msg::Pose pose;
    pose.position.x = x;
    pose.position.y = y;
    pose.position.z = z;
    pose.orientation = tf2::toMsg(down_q);
    return pose;
  };

  // Helper function to move robot to a viewpoint with scanning off, settle so the arm has fully stopped, 
  // then accumulate cloud with scanning on for a short fixed duration, then stop scanning and log the observation.
  // No motion blur.
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

  // Cache the joint configuration of the ready pose to return to a known-good state between scan rows.
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

  // Phase 1: Discovering the scene with a coarse scan to find the rought locations of obstacles.
  // Without, the robot was crashing into obstacles during the finer scan in phase 2. 
  const double preliminary_scan_z = 0.65; // high z to avoid obstacles while discovering.
  observeAt( 0.30,  0.30, preliminary_scan_z, "discovery_front_left");
  observeAt( 0.30, -0.30, preliminary_scan_z, "discovery_front_right");
  observeAt(-0.30,  0.30, preliminary_scan_z, "discovery_back_left");
  observeAt(-0.30, -0.30, preliminary_scan_z, "discovery_back_right");
  goToInitial();

  auto initial_detection = classifyAccumulatedCloud();

  // Phase 2: Register the detected obstacles in the planning scene.
  // Tells MoveIt about obstacles before fine scan to avoid collision whilst scanning.
  auto planning_scene_interface =
    std::make_shared<moveit::planning_interface::PlanningSceneInterface>();
  std::vector<moveit_msgs::msg::CollisionObject> collision_objects;

  int obstacle_id = 0; // used to give unique IDs to each obstacle, which is required by MoveIt to manage them in the planning scene.
  for (const auto &obj : initial_detection) {
    if (obj.category != "obstacle") continue;
    
    moveit_msgs::msg::CollisionObject co;
    co.header.frame_id = arm_group_->getPlanningFrame();
    co.id = "obstacle_" + std::to_string(obstacle_id++);

    // Create a box primitive for the obstacle based on the bounding box of the cluster. 
    shape_msgs::msg::SolidPrimitive box;
    box.type = shape_msgs::msg::SolidPrimitive::BOX;
    box.dimensions.resize(3);
    // Add a small margin to the bounding box dimensions to account for sensor noise and ensure the box fully encompasses the obstacle.
    box.dimensions[0] = (obj.max_x - obj.min_x) + 0.06;
    box.dimensions[1] = (obj.max_y - obj.min_y) + 0.06;
    box.dimensions[2] = std::max(0.25, (obj.max_z - obj.min_z) + 0.06); // ensure a minimum height to account for any underestimation of the obstacle height due to partial views.

    geometry_msgs::msg::Pose box_pose;
    box_pose.orientation.w = 1.0;
    box_pose.position.x = obj.centroid.x;
    box_pose.position.y = obj.centroid.y;
    // Set the box pose z to be half of the box height so that the box sits on the table surface. 
    box_pose.position.z = box.dimensions[2] / 2.0;

    co.primitives.push_back(box);
    co.primitive_poses.push_back(box_pose);
    co.operation = moveit_msgs::msg::CollisionObject::ADD;
    collision_objects.push_back(co);

  }
  planning_scene_interface->applyCollisionObjects(collision_objects);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Phase 3: Systematic fine scan to classify objects. The robot moves in a grid pattern with 8 rows of 5 viewpoints each, 
  // covering the whole workspace with significant overlap between viewpoints to ensure good coverage and classification accuracy.

  // Clear the accumulated cloud before starting the fine scan to ensure we only have points from the fine scan, which has better coverage. 
  {
    std::lock_guard<std::mutex> lock(accumulated_cloud_mutex_);
    accumulated_cloud_->clear();
  }

  // 36 viewpoints in a 8x5 grid pattern covering the workspace with goToInitial() called between rows to return to a known-good state 
  // to avoid awkward joint configurations between viewpoints. 

  const double scan_z = 0.55;

  // Front far row
  observeAt( 0.55,  0.45, scan_z, "front_far_far_left");
  observeAt( 0.55,  0.20, scan_z, "front_far_left");
  observeAt( 0.55,  0.00, scan_z, "front_far_middle");
  observeAt( 0.55, -0.20, scan_z, "front_far_right");
  observeAt( 0.55, -0.45, scan_z, "front_far_far_right");
  goToInitial(); 

  // Front row 
  observeAt( 0.30,  0.45, scan_z, "front_far_left");
  observeAt( 0.30,  0.20, scan_z, "front_left");
  observeAt( 0.30,  0.00, scan_z, "front_middle");
  observeAt( 0.30, -0.20, scan_z, "front_right");
  observeAt( 0.30, -0.45, scan_z, "front_far_right");
  goToInitial();

  // Midfront row
  observeAt( 0.05,  0.45, scan_z, "midfront_far_left");
  observeAt( 0.05,  0.20, scan_z, "midfront_left");
  observeAt( 0.05, -0.20, scan_z, "midfront_right");
  observeAt( 0.05, -0.45, scan_z, "midfront_far_right");
  goToInitial();

  // Left edge traversal: front-to-back
  observeAt( 0.40,  0.45, scan_z, "left_edge_front");
  observeAt( 0.10,  0.45, scan_z, "left_edge_midfront");
  observeAt(-0.10,  0.45, scan_z, "left_edge_midback");
  observeAt(-0.30,  0.45, scan_z, "left_edge_back");
  goToInitial();

  // Right edge traversal: front-to-back
  observeAt( 0.40, -0.45, scan_z, "right_edge_front");
  observeAt( 0.10, -0.45, scan_z, "right_edge_midfront");
  observeAt(-0.10, -0.45, scan_z, "right_edge_midback");
  observeAt(-0.30, -0.45, scan_z, "right_edge_back");
  goToInitial();

  // Midback row
  observeAt(-0.20,  0.45, scan_z, "midback_far_left");
  observeAt(-0.20,  0.20, scan_z, "midback_left");
  observeAt(-0.20, -0.20, scan_z, "midback_right");
  observeAt(-0.20, -0.45, scan_z, "midback_far_right");
  goToInitial();

  // Back row
  observeAt(-0.40,  0.45, scan_z, "back_far_left");
  observeAt(-0.40,  0.20, scan_z, "back_left");
  observeAt(-0.40,  0.00, scan_z, "back_middle");
  observeAt(-0.40, -0.20, scan_z, "back_right");
  observeAt(-0.40, -0.45, scan_z, "back_far_right");
  goToInitial();

  // Back far row
  observeAt(-0.55,  0.45, scan_z, "back_far_far_left");
  observeAt(-0.55,  0.20, scan_z, "back_far_left");
  observeAt(-0.55,  0.00, scan_z, "back_far_middle");
  observeAt(-0.55, -0.20, scan_z, "back_far_right");
  observeAt(-0.55, -0.45, scan_z, "back_far_far_right");
  goToInitial();

  // Phase 4: Cluster and categorise accumulated cloud.

  // Return each cluster as a DetectedObj tagged with its category ("object", "basket", "obstacle"), centroid, bounding box, and point count.
  auto detected = classifyAccumulatedCloud();


  // Phase 5: Inspect each candidate object overhead to classify its shape using Task 2.
  for (auto &obj : detected) {
    if (obj.category != "object") continue;
    
    // Move to a viewpoint above the object's centroid
    geometry_msgs::msg::Pose inspect = makePose(obj.centroid.x, obj.centroid.y, 0.5);
    moveToPose(inspect);
    // Wait for a fresh cloud to ensure the classification is based on the latest observation from the new viewpoint, which has better coverage of the object. 
    waitForFreshCloud(3, 2.0);

    // Classify the shape at the object's centroid using the same method as Task 2. 
    geometry_msgs::msg::PointStamped query;
    query.header.frame_id = "panda_link0";
    query.point.x = obj.centroid.x;
    query.point.y = obj.centroid.y;
    query.point.z = obj.centroid.z;
    obj.shape = classifyShapeAtPoint(query);
  }

  goToInitial();

  // Count how many noughts and crosses were classified for the output. 
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

  // Phase 6: Pick and place the most common shape into the basket.

  // If no shapes just return. 
  if (total_shapes == 0) {
    RCLCPP_INFO(node_->get_logger(), "No shapes detected.");
    return;
  }

  // Tie-break by picking nought. 
  const std::string target_shape = (most_common == nought_count) ? "nought" : "cross";

  // Find the basket from the finer scan. If none was detected we can't place and break.
  const DetectedObj* basket_ptr = nullptr;
  for (const auto &obj : detected) {
    if (obj.category == "basket") { basket_ptr = &obj; break; }
  }
  if (!basket_ptr) {
    RCLCPP_WARN(node_->get_logger(), "No basket detected, cannot proceed with Task 3 pick and place.");
    return;
  }

  // Collect all candidate objects of the target shape (either nought or cross).
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

  // Sort candidates by distance from the arm base. The closes is most likely to be reachable.
  std::sort(candidates.begin(), candidates.end(),
  [](const DetectedObj* a, const DetectedObj* b) {
    const double da = std::hypot(a->centroid.x, a->centroid.y);
    const double db = std::hypot(b->centroid.x, b->centroid.y);
    return da < db;
  });

  // Refine the obstacle list using the finer scan. Any obstacle the discovery scan missed gets added now.
  for (const auto &obj : detected) {
    if (obj.category == "obstacle") {
      bool already_added = false;
      for (const auto &existing : collision_objects) {
        const double dx = obj.centroid.x - existing.primitive_poses[0].position.x;
        const double dy = obj.centroid.y - existing.primitive_poses[0].position.y;
        if (std::hypot(dx, dy) < 0.1) {already_added = true; break;} // if there's already an obstacle within 10 cm, we consider it the same obstacle and skip adding a new one.
      }
      if (already_added) continue;

      // Add new obstacle to the planning scene.
      moveit_msgs::msg::CollisionObject co;
      co.header.frame_id = arm_group_->getPlanningFrame();
      co.id = "obstacle_" + std::to_string(obstacle_id++);
      shape_msgs::msg::SolidPrimitive box;
      box.type = shape_msgs::msg::SolidPrimitive::BOX;
      box.dimensions.resize(3);
      box.dimensions[0] = (obj.max_x - obj.min_x) + 0.06;
      box.dimensions[1] = (obj.max_y - obj.min_y) + 0.06;
      box.dimensions[2] = std::max(0.25, (obj.max_z - obj.min_z) + 0.06);

      geometry_msgs::msg::Pose box_pose;
      box_pose.orientation.w = 1.0;
      box_pose.position.x = obj.centroid.x;
      box_pose.position.y = obj.centroid.y;
      box_pose.position.z = box.dimensions[2] / 2.0;

      co.primitives.push_back(box);
      co.primitive_poses.push_back(box_pose);
      co.operation = moveit_msgs::msg::CollisionObject::ADD;
      collision_objects.push_back(co);
    }
  }
  planning_scene_interface->applyCollisionObjects(collision_objects);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Attempt to pick and place each candidate until one succeeds.
  bool placed = false;
  for (const DetectedObj *target : candidates) {
    geometry_msgs::msg::Point target_point = target->centroid;
    target_point.z = 0.0; // Override z to be level with the ground, the cluster centroid sits on top of the shape so we need to make it 0 for grasping. 

    // Measure the shape's size (20, 30, 40 mm) to scale the grasp offset so that the gripper lands on the right part of the shape for grasping.
    const double target_size = std::max(target->max_x - target->min_x, target->max_y - target->min_y);

    geometry_msgs::msg::Point basket_point = basket_ptr->centroid;
    basket_point.z = 0.0; // Override z for placing in the same way as target.

    // Same sequence as Task 1 for pick and place 
    goToInitial();

    // 1. Move directly above object to have a clean point cloud for orientation estimation.
    tf2::Quaternion observe_orienation;
    observe_orienation.setRPY(M_PI, 0, -M_PI / 4);
    moveToPose(makeAGraspOffset(target_point, target_shape, 0.5, observe_orienation, 0.0, target_size));
    waitForFreshCloud();
    
    // 2. Compute orientation for grasping.
    geometry_msgs::msg::PointStamped target_stamped;
    target_stamped.header.frame_id = arm_group_->getPlanningFrame();
    target_stamped.header.stamp = node_->get_clock()->now();
    target_stamped.point = target_point;
    const double shape_yaw = computeShapeOrientation(target_stamped);

    // 3. Move to the pre-grasp pose with the computed orientation.
    tf2::Quaternion grasp_orientation;
    grasp_orientation.setRPY(M_PI, 0, -M_PI / 4 + shape_yaw);
    moveToPose(makeAGraspOffset(target_point, target_shape, 0.5, grasp_orientation, shape_yaw, target_size));
    openGripper();
    
    // 4. Move down to grasp the object, close the gripper.
    computeAndExecuteCartesianPath(makeAGraspOffset(target_point, target_shape, 0.15, grasp_orientation, shape_yaw, target_size));
    closeGripper();
    
    // 5. Lift the object up.
    computeAndExecuteCartesianPath(makeAGraspOffset(target_point, target_shape, 0.6, grasp_orientation, shape_yaw, target_size));

    tf2::Quaternion basket_orientation;
    basket_orientation.setRPY(M_PI, 0, -M_PI / 4);

    // 6. Move above the basket, then down to place the object.
    if (target_shape == "nought") {
      moveToPose(makeAGraspOffset(basket_point, "nought", 0.6, basket_orientation, 0.0, target_size));
      computeAndExecuteCartesianPath(makeAGraspOffset(basket_point, "nought", 0.17, basket_orientation, 0.0, target_size));
    } else {
      moveToPose(makeAGraspOffset(basket_point, " ", 0.6, basket_orientation, 0.0, target_size));
      computeAndExecuteCartesianPath(makeAGraspOffset(basket_point, " ", 0.17, basket_orientation, 0.0, target_size));
    }
    
    // 7. Release the object.
    openGripper();

    // 8. Move up to clear the basket.
    computeAndExecuteCartesianPath(makeAGraspOffset(basket_point, " ", 0.5, basket_orientation, 0.0, target_size));

    // If we reached this point without exceptions, we consider the place successful and break the loop.
    placed = true;
    break;
  }

  // Clean up the collision objects from the planning scene after we're done.
  std::vector<std::string> remove_objects;
  for (const auto &co : collision_objects) {
    remove_objects.push_back(co.id);
  }
  planning_scene_interface->removeCollisionObjects(remove_objects);

  if (!placed) {
    RCLCPP_WARN(node_->get_logger(), "Failed to place any %s in the basket.", target_shape.c_str());
  }

  moveToNamedPose("ready");

  // Log the counting result for Task 3.
  RCLCPP_INFO(node_->get_logger(),
    "T3 counting result: noughts=%d crosses=%d total=%d most_common=%d",
    nought_count, cross_count, total_shapes, most_common);
}
