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

// Extra for Task 3 
#include <unordered_set>
#include <array>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <octomap/OcTree.h>
#include <octomap/OcTreeKey.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

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

  // Initialize Task 3 members
  accumulated_cloud_.reset(new PointC);
  is_scanning_ = false;

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

  // Store latest cloud for Task 1 & 2
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    g_input_pc_frame_id_ = msg->header.frame_id;
    g_cloud_ptr = latest_cloud;
    ++g_cloud_sequence_;
  }

  // Task 3 accumulation while scanning
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

    for (const auto &pt : latest_cloud->points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
        continue;

      tf2::Vector3 v = tf * tf2::Vector3(pt.x, pt.y, pt.z);

      // Keep points on the table and objects (z between 0.01 and 0.5)
      if (v.z() < 0.01 || v.z() > 0.5) continue;
      if (v.x() < -0.8 || v.x() > 0.8) continue;
      if (v.y() < -0.7 || v.y() > 0.7) continue;

      PointT new_pt = pt;
      new_pt.x = v.x();
      new_pt.y = v.y();
      new_pt.z = v.z();
      accumulated_cloud_->push_back(new_pt);
    }
  }
}

void cw2::moveToNamedPose(const std::string &pose_name)
{
  arm_group_->clearPathConstraints();
  arm_group_->setNamedTarget(pose_name);
  arm_group_->move();
}

void cw2::openGripper()
{
  hand_group_->setNamedTarget("open");
  hand_group_->move();
}

void cw2::closeGripper()
{
  hand_group_->setNamedTarget("close");
  hand_group_->move();
}

void cw2::moveToPose(const geometry_msgs::msg::Pose &target_pose)
{
  arm_group_->setStartStateToCurrentState();
  arm_group_->setPoseTarget(target_pose);
  arm_group_->move();
}

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

geometry_msgs::msg::Pose cw2::makeAGraspOffset(
  const geometry_msgs::msg::Point &point,
  const std::string &shape_type,
  double z_offset,
  const tf2::Quaternion &orientation)
{
  geometry_msgs::msg::Pose pose;
  if (shape_type == "nought") {
    pose.position.x = point.x;
    pose.position.y = point.y + 0.08;
  } else if (shape_type == "cross") {
    pose.position.x = point.x + 0.06;
    pose.position.y = point.y;
  } else {
    pose.position.x = point.x;
    pose.position.y = point.y;
  }
  pose.position.z = point.z + z_offset;
  pose.orientation = tf2::toMsg(orientation);
  return pose;
}

// ---------- Task 1 (unchanged) ----------
void cw2::t1_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response)
{
  (void)response;

  const auto &object     = request->object_point.point;
  const auto &basket     = request->goal_point.point;
  const auto &shape_type = request->shape_type;

  tf2::Quaternion orientation;
  orientation.setRPY(M_PI, 0, -M_PI / 4);

  arm_group_->setNamedTarget("ready");
  moveToPose(makeAGraspOffset(object, shape_type, 0.5, orientation));
  openGripper();
  computeAndExecuteCartesianPath(makeAGraspOffset(object, shape_type, 0.15, orientation));
  closeGripper();
  computeAndExecuteCartesianPath(makeAGraspOffset(object, shape_type, 0.5, orientation));

  if (shape_type == "nought") {
    moveToPose(makeAGraspOffset(basket, "nought", 0.5, orientation));
    computeAndExecuteCartesianPath(makeAGraspOffset(basket, "nought", 0.17, orientation));
  } else {
    moveToPose(makeAGraspOffset(basket, " ", 0.5, orientation));
    computeAndExecuteCartesianPath(makeAGraspOffset(basket, " ", 0.17, orientation));
  }

  openGripper();
  computeAndExecuteCartesianPath(makeAGraspOffset(basket, " ", 0.5, orientation));
  moveToNamedPose("ready");
}

// ---------- Task 2 (unchanged) ----------
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

// ---------- Helper for shape classification (exactly as used in Task 2) ----------
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

  const double xy_radius    = 0.12;
  const double xy_radius_sq = xy_radius * xy_radius;
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

// These mirror the helpers from cw2_pcl_helper.cpp and cw2_octomap_helper.cpp.
namespace {
  // KeyHash for octomap flood-fill
  struct T3KeyHash {
    std::size_t operator()(const octomap::OcTreeKey& k) const {
      return ((std::size_t(k.k[0]) * 73856093) ^
              (std::size_t(k.k[1]) * 19349663) ^
              (std::size_t(k.k[2]) * 83492791));
    }
  };

  // One detected object from the flood-fill
  struct T3Object {
    std::string category;   // "object" | "basket" | "obstacle"
    std::string shape;      // "nought" | "cross" | "N/A"
    geometry_msgs::msg::Point centroid;
    std::unordered_set<octomap::OcTreeKey, T3KeyHash> voxel_keys;
    double min_x, max_x, min_y, max_y, min_z, max_z;
  };
}

void cw2::t3_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  response->total_num_shapes = 0;
  response->num_most_common_shape = 0;

  RCLCPP_INFO(node_->get_logger(), "=== Task 3 start (fixed grasp + collision avoidance) ===");

  // Clear previous data
  {
    std::lock_guard<std::mutex> lk(accumulated_cloud_mutex_);
    if (!accumulated_cloud_) accumulated_cloud_.reset(new PointC);
    accumulated_cloud_->clear();
  }
  if (latest_octree_) latest_octree_->clear();
  is_scanning_ = false;
  arm_group_->clearPathConstraints();

  // Fixed downward orientation (same as Task 1/2 – this fixes the grasp hitting)
  tf2::Quaternion down_q;
  down_q.setRPY(M_PI, 0.0, -M_PI / 4.0);

  auto makePose = [&](double x, double y, double z) -> geometry_msgs::msg::Pose {
    geometry_msgs::msg::Pose p;
    p.position.x = x; p.position.y = y; p.position.z = z;
    p.orientation = tf2::toMsg(down_q);
    return p;
  };


  auto scanSubArea = [&](const std::vector<std::array<double,3>> &corners) -> bool {
    if (corners.size() != 4) return false;
    moveToPose(makePose(corners[0][0], corners[0][1], corners[0][2]));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

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

    is_scanning_ = true;
    arm_group_->execute(trajectory);
    is_scanning_ = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return true;
  };

  openGripper();
  moveToNamedPose("ready");

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

  // ── SCAN 4 SUB-AREAS ─────────────────────────────────────────────────────
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

  // Downsample accumulated cloud
  PointCPtr acc_ds(new PointC);
  {
    std::lock_guard<std::mutex> lk(accumulated_cloud_mutex_);
    if (!accumulated_cloud_ || accumulated_cloud_->empty()) {
      RCLCPP_ERROR(node_->get_logger(), "T3: accumulated cloud empty – aborting");
      goToInitial(); return;
    }
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(accumulated_cloud_);
    vg.setLeafSize(0.003f, 0.003f, 0.003f);
    vg.filter(*acc_ds);
  }

  // Build OctoMap
  const double OCT_RES = 0.005;
  latest_octree_ = std::make_shared<octomap::OcTree>(OCT_RES);
  for (const auto &pt : acc_ds->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z) || pt.z < 0.04) continue;
    latest_octree_->updateNode(octomap::point3d(pt.x, pt.y, pt.z), true);
  }
  latest_octree_->updateInnerOccupancy();

  // ── FLOOD-FILL CLUSTERING (exactly as in your original code) ─────────────
  using Key = octomap::OcTreeKey;
  const int deltas[26][3] = {
    {-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1},
    {-1,-1,0},{-1,1,0},{1,-1,0},{1,1,0},
    {-1,0,-1},{-1,0,1},{1,0,-1},{1,0,1},
    {0,-1,-1},{0,-1,1},{0,1,-1},{0,1,1},
    {-1,-1,-1},{-1,-1,1},{-1,1,-1},{-1,1,1},
    {1,-1,-1},{1,-1,1},{1,1,-1},{1,1,1}
  };

  std::unordered_set<Key, T3KeyHash> occ;
  for (auto it = latest_octree_->begin_leafs(); it != latest_octree_->end_leafs(); ++it) {
    if (latest_octree_->isNodeOccupied(*it)) occ.insert(it.getKey());
  }

  const int MIN_VOXELS = 150;
  std::unordered_set<Key, T3KeyHash> vis;
  std::vector<Key> stack;
  std::vector<T3Object> detected;

  for (const Key &seed : occ) {
    if (vis.count(seed)) continue;

    double min_x=1e9, max_x=-1e9, min_y=1e9, max_y=-1e9, min_z=1e9, max_z=-1e9;
    std::unordered_set<Key, T3KeyHash> cluster;
    stack.clear();
    stack.push_back(seed);
    vis.insert(seed);

    while (!stack.empty()) {
      Key cur = stack.back(); stack.pop_back();
      cluster.insert(cur);
      octomap::point3d p = latest_octree_->keyToCoord(cur);
      min_x = std::min(min_x, (double)p.x()); max_x = std::max(max_x, (double)p.x());
      min_y = std::min(min_y, (double)p.y()); max_y = std::max(max_y, (double)p.y());
      min_z = std::min(min_z, (double)p.z()); max_z = std::max(max_z, (double)p.z());
      for (int i = 0; i < 26; ++i) {
        Key nbk(cur[0]+deltas[i][0], cur[1]+deltas[i][1], cur[2]+deltas[i][2]);
        if (occ.count(nbk) && !vis.count(nbk)) {
          vis.insert(nbk);
          stack.push_back(nbk);
        }
      }
    }

    if ((int)cluster.size() < MIN_VOXELS) continue;

    const double height = (max_z - min_z) + OCT_RES;
    T3Object d;
    d.min_x = min_x; d.max_x = max_x; d.min_y = min_y; d.max_y = max_y;
    d.min_z = min_z; d.max_z = max_z;
    d.voxel_keys = cluster;

    if (height > 0.05)       d.category = "obstacle";
    else if (height >= 0.03) d.category = "basket";
    else                     d.category = "object";

    d.shape = "N/A";
    d.centroid.x = d.centroid.y = d.centroid.z = 0.0;

    if (d.category == "object") {
      Key top_ref_key = latest_octree_->coordToKey(0, 0, max_z);
      std::vector<octomap::point3d> surf;
      for (const Key &k : cluster)
        if (k[2] == top_ref_key[2]) surf.push_back(latest_octree_->keyToCoord(k));
      if (surf.empty()) continue;
      for (const auto &p : surf) {
        d.centroid.x += p.x(); d.centroid.y += p.y(); d.centroid.z += p.z();
      }
      d.centroid.x /= surf.size();
      d.centroid.y /= surf.size();
      d.centroid.z /= surf.size();

      Key ck = latest_octree_->coordToKey(d.centroid.x, d.centroid.y, max_z);
      bool occupied = false;
      for (int dx = -1; dx <= 1 && !occupied; ++dx)
        for (int dy = -1; dy <= 1 && !occupied; ++dy) {
          Key q(ck[0]+dx, ck[1]+dy, ck[2]);
          if (cluster.count(q)) occupied = true;
        }
      d.shape = occupied ? "cross" : "nought";
    } else if (d.category == "basket") {
      d.centroid.x = 0.5 * (min_x + max_x);
      d.centroid.y = 0.5 * (min_y + max_y);
      d.centroid.z = 0.5 * (min_z + max_z);
    }

    detected.push_back(d);
  }

  // ── COUNT SHAPES & CHOOSE TARGET ───────────────────────────────────────
  int n_noughts = 0, n_crosses = 0;
  for (const auto &d : detected)
    if (d.category == "object") {
      if (d.shape == "nought") ++n_noughts;
      else if (d.shape == "cross") ++n_crosses;
    }

  if (n_noughts + n_crosses == 0) {
    RCLCPP_ERROR(node_->get_logger(), "T3: no shapes detected");
    goToInitial(); return;
  }

  std::string target_shape = (n_noughts > n_crosses) ? "nought" :
                             (n_crosses > n_noughts) ? "cross" :
                             (std::rand() % 2 ? "nought" : "cross");

  response->total_num_shapes = static_cast<int64_t>(n_noughts + n_crosses);
  response->num_most_common_shape = (n_noughts == n_crosses) ?
    static_cast<int64_t>(n_noughts) : static_cast<int64_t>(std::max(n_noughts, n_crosses));

  // Find basket and target object
  T3Object basket, target;
  bool basket_ok = false, target_ok = false;
  for (auto &d : detected) {
    if (!basket_ok && d.category == "basket") { basket = d; basket_ok = true; }
    if (!target_ok && d.category == "object" && d.shape == target_shape) { target = d; target_ok = true; }
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

  // ── ADD OBSTACLES TO PLANNING SCENE (prevents hitting tall obstacles) ─────
  std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
  int obs_idx = 0;
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

    // ── FIXED GRASP + CENTRED PLACEMENT ─────────────────────────────────────
  goToInitial();
  double grasp_x = target.centroid.x;
  double grasp_y = target.centroid.y;
  if (target_shape == "nought") {
      grasp_y += 0.08;          // your working value
  } else {
      grasp_x += 0.06;
  }
  double grasp_z = target.centroid.z + 0.015;

  // === NEW: always place in the exact centre of the basket ===
  double place_x = basket.centroid.x;
  double place_y = basket.centroid.y;

  RCLCPP_INFO(node_->get_logger(),
    "T3: grasp=(%.3f,%.3f,%.3f) place=(%.3f,%.3f) [centred in basket]",
    grasp_x, grasp_y, grasp_z, place_x, place_y);

  // ── PICK-AND-PLACE (Cartesian, safe orientation) ───────────────────────
  openGripper();
  moveToPose(makePose(grasp_x, grasp_y, grasp_z + 0.15));

  // Cartesian descend
  {
    std::vector<geometry_msgs::msg::Pose> wps = {makePose(grasp_x, grasp_y, grasp_z + 0.08)};
    moveit_msgs::msg::RobotTrajectory traj;
    arm_group_->setStartStateToCurrentState();
    double frac = arm_group_->computeCartesianPath(wps, 0.01, 0.0, traj);
    if (frac >= 0.95) arm_group_->execute(traj);
    else moveToPose(makePose(grasp_x, grasp_y, grasp_z + 0.08));
  }

  closeGripper();

  // Lift → Y → X → lower
  {
    geometry_msgs::msg::PoseStamped cur = arm_group_->getCurrentPose();
    geometry_msgs::msg::Pose base_pose = cur.pose;

    std::vector<geometry_msgs::msg::Pose> wps;
    geometry_msgs::msg::Pose lift = base_pose; lift.position.z = grasp_z + 0.40; wps.push_back(lift);
    geometry_msgs::msg::Pose mvy = lift; mvy.position.y = place_y; wps.push_back(mvy);
    geometry_msgs::msg::Pose mvx = mvy; mvx.position.x = place_x; wps.push_back(mvx);
    geometry_msgs::msg::Pose lower = mvx; lower.position.z = basket.centroid.z + 0.08; wps.push_back(lower);

    moveit_msgs::msg::RobotTrajectory traj;
    arm_group_->setStartStateToCurrentState();
    double frac = arm_group_->computeCartesianPath(wps, 0.01, 0.0, traj);
    if (frac >= 0.95) {
      arm_group_->execute(traj);
    } else {
      moveToPose(makePose(grasp_x, grasp_y, grasp_z + 0.40));
      moveToPose(makePose(place_x, place_y, basket.centroid.z + 0.40));
    }
  }

  openGripper();
  goToInitial();

  RCLCPP_INFO(node_->get_logger(),
    "=== Task 3 complete: picked a %s total=%lld most_common=%lld ===",
    target_shape.c_str(),
    static_cast<long long>(response->total_num_shapes),
    static_cast<long long>(response->num_most_common_shape));
}