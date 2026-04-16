/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#include <cw2_class.h>

#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>

// Extra headers for Task 3 scene scanning
#include <algorithm>
#include <thread>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>

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

void cw2::moveToNamedPose(const std::string &pose_name)
{
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
  arm_group_->setPoseTarget(target_pose);
  arm_group_->move();
}

bool cw2::computeAndExecuteCartesianPath(const geometry_msgs::msg::Pose &target)
{
  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(target);

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

// ─────────────────────────────────────────────────────────────────────────────
// Task 1 – Pick and Place (unchanged from teammates)
// ─────────────────────────────────────────────────────────────────────────────
void cw2::t1_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response)
{
  (void)response;

  const auto &object     = request->object_point.point;
  const auto &basket     = request->goal_point.point;
  const auto &shape_type = request->shape_type;

  tf2::Quaternion orientation;
  orientation.setRPY(M_PI, 0, -M_PI / 4); // Change when T1_ANY_ORIENTATION = True

  // 1. Move to ready position
  arm_group_->setNamedTarget("ready");

  // 2. Move to pre-grasp position, just above the object
  moveToPose(makeAGraspOffset(object, shape_type, 0.5, orientation));

  // 3. Open gripper
  openGripper();

  // 4. Descend to grasp the object
  computeAndExecuteCartesianPath(makeAGraspOffset(object, shape_type, 0.15, orientation));

  // 5. Close gripper to grasp the object
  closeGripper();

  // 6. Lift the object up
  computeAndExecuteCartesianPath(makeAGraspOffset(object, shape_type, 0.5, orientation));

  // 7. Move to above the basket
  if (shape_type == "nought") {
    moveToPose(makeAGraspOffset(basket, "nought", 0.5, orientation));
  } else {
    moveToPose(makeAGraspOffset(basket, " ", 0.5, orientation));
  }

  // 8. Descend to place the object in the basket
  if (shape_type == "nought") {
    computeAndExecuteCartesianPath(makeAGraspOffset(basket, "nought", 0.17, orientation));
  } else {
    computeAndExecuteCartesianPath(makeAGraspOffset(basket, " ", 0.17, orientation));
  }

  // 9. Open gripper to release the object
  openGripper();

  // 10. Move back up after releasing the object
  computeAndExecuteCartesianPath(makeAGraspOffset(basket, " ", 0.5, orientation));

  // 11. Move back to ready position
  moveToNamedPose("ready");
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 2 – Shape Detection (unchanged from teammates)
// ─────────────────────────────────────────────────────────────────────────────
void cw2::t2_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task2Service::Response> response)
{
  response->mystery_object_num = 1;

  tf2::Quaternion view_orientation;
  view_orientation.setRPY(M_PI, 0.0, -M_PI / 4.0);

  auto classify_shape_at_point =
    [&](const geometry_msgs::msg::PointStamped &query_point) -> std::string {
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
        RCLCPP_WARN(node_->get_logger(), "Task2 TF lookup failed: %s", ex.what());
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
    };

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

      return classify_shape_at_point(point_stamped);
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

// ─────────────────────────────────────────────────────────────────────────────
// Task 3 – Planning and Execution
//
//
// Never mind guys i just wanna locate it 
//
//
// ─────────────────────────────────────────────────────────────────────────────
void cw2::t3_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  response->total_num_shapes      = 0;
  response->num_most_common_shape = 0;

  RCLCPP_INFO(node_->get_logger(), "Task 3 started");

  tf2::Quaternion grasp_q;
  grasp_q.setRPY(M_PI, 0.0, -M_PI / 4.0);

  auto classify_shape_at_point =
    [&](const geometry_msgs::msg::PointStamped &query_point) -> std::string
    {
      PointCPtr cloud_snapshot;
      std::string cloud_frame;
      {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        if (!g_cloud_ptr || g_cloud_ptr->empty() || g_input_pc_frame_id_.empty())
          return "unknown";
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
        RCLCPP_WARN(node_->get_logger(), "T3 classify TF: %s", ex.what());
        return "unknown";
      }

      tf2::Transform tf_cloud_to_target;
      tf2::fromMsg(tf_msg.transform, tf_cloud_to_target);

      std::vector<geometry_msgs::msg::Point> local_points;
      local_points.reserve(4096);

      const double xy_radius_sq = 0.12 * 0.12;
      const double z_min        = query_point.point.z - 0.03;
      const double z_max_bound  = query_point.point.z + 0.12;
      double observed_z_max     = -std::numeric_limits<double>::max();

      for (const auto &pt : cloud_snapshot->points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;

        const tf2::Vector3 p_t = tf_cloud_to_target * tf2::Vector3(pt.x, pt.y, pt.z);
        const double dx = p_t.x() - query_point.point.x;
        const double dy = p_t.y() - query_point.point.y;

        if (dx * dx + dy * dy > xy_radius_sq) continue;
        if (p_t.z() < z_min || p_t.z() > z_max_bound) continue;

        geometry_msgs::msg::Point p;
        p.x = p_t.x();
        p.y = p_t.y();
        p.z = p_t.z();
        local_points.push_back(p);

        observed_z_max = std::max(observed_z_max, p.z);
      }

      if (local_points.size() < 80) return "unknown";

      const double top_z_min = observed_z_max - 0.035;
      std::vector<geometry_msgs::msg::Point> top_points;
      for (const auto &p : local_points) {
        if (p.z >= top_z_min) top_points.push_back(p);
      }

      if (top_points.size() < 30) return "unknown";

      double cx = 0.0;
      double cy = 0.0;
      for (const auto &p : top_points) {
        cx += p.x;
        cy += p.y;
      }
      cx /= static_cast<double>(top_points.size());
      cy /= static_cast<double>(top_points.size());

      const double r_sq = 0.005 * 0.005;
      int center_count = 0;
      for (const auto &p : top_points) {
        const double dx = p.x - cx;
        const double dy = p.y - cy;
        if (dx * dx + dy * dy <= r_sq) ++center_count;
      }

      return (center_count < 20) ? "nought" : "cross";
    };

  auto classify_at_centroid =
    [&](const geometry_msgs::msg::Point &centroid) -> std::string
    {
      geometry_msgs::msg::Pose view_pose;
      view_pose.position.x  = centroid.x;
      view_pose.position.y  = centroid.y;
      view_pose.position.z  = centroid.z + 0.50;
      view_pose.orientation = tf2::toMsg(grasp_q);
      moveToPose(view_pose);

      std::uint64_t before_seq;
      {
        std::lock_guard<std::mutex> lk(cloud_mutex_);
        before_seq = g_cloud_sequence_;
      }

      const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(900);

      while (std::chrono::steady_clock::now() < deadline) {
        {
          std::lock_guard<std::mutex> lk(cloud_mutex_);
          if (g_cloud_sequence_ > before_seq) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
      }

      geometry_msgs::msg::PointStamped ps;
      ps.header.frame_id = "panda_link0";
      ps.header.stamp    = node_->get_clock()->now();
      ps.point           = centroid;

      return classify_shape_at_point(ps);
    };

  openGripper();
  rclcpp::sleep_for(std::chrono::milliseconds(300));
  moveToNamedPose("ready");

  struct PaletteEntry { float r, g, b; };
  const std::array<PaletteEntry, 3> palette = {{
    {0.8f, 0.1f, 0.1f},
    {0.1f, 0.1f, 0.8f},
    {0.8f, 0.1f, 0.8f},
  }};
  const float max_col_dist_sq = 0.25f * 0.25f * 3.0f;

  auto transformCloudToBase = [&](const PointCPtr &cloud, const std::string &cloud_frame) -> PointCPtr
  {
    if (!cloud || cloud->empty()) return nullptr;

    geometry_msgs::msg::TransformStamped tf_msg;
    try {
      tf_msg = tf_buffer_.lookupTransform(
        "panda_link0",
        cloud_frame,
        tf2::TimePointZero,
        tf2::durationFromSec(0.2));
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(node_->get_logger(), "T3 scan TF failed: %s", ex.what());
      return nullptr;
    }

    tf2::Transform tf;
    tf2::fromMsg(tf_msg.transform, tf);

    PointCPtr out(new PointC);
    out->reserve(cloud->size());

    for (const auto &pt : cloud->points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;

      const tf2::Vector3 v = tf * tf2::Vector3(pt.x, pt.y, pt.z);

      if (v.z() < 0.01 || v.z() > 0.60) continue;
      if (v.x() < -0.75 || v.x() > 0.85) continue;
      if (v.y() < -0.65 || v.y() > 0.65) continue;

      PointT p = pt;
      p.x = static_cast<float>(v.x());
      p.y = static_cast<float>(v.y());
      p.z = static_cast<float>(v.z());
      out->push_back(p);
    }

    out->width = out->size();
    out->height = 1;
    out->is_dense = false;
    return out;
  };

  auto filterShapeColours = [&](const PointCPtr &cloud) -> PointCPtr
  {
    if (!cloud || cloud->empty()) return nullptr;

    PointCPtr out(new PointC);
    out->reserve(cloud->size() / 3);

    for (const auto &pt : cloud->points) {
      const float r = pt.r / 255.f;
      const float g = pt.g / 255.f;
      const float b = pt.b / 255.f;

      if (std::max({r, g, b}) < 0.20f) continue;

      bool colour_match = false;
      for (const auto &kc : palette) {
        const float dr = r - kc.r;
        const float dg = g - kc.g;
        const float db = b - kc.b;
        if (dr * dr + dg * dg + db * db < max_col_dist_sq) {
          colour_match = true;
          break;
        }
      }

      if (colour_match) out->push_back(pt);
    }

    out->width = out->size();
    out->height = 1;
    out->is_dense = false;
    return out;
  };

  const double VIEW_Z = 0.65;
  const std::vector<std::pair<double,double>> viewpoints = {
    {-0.30, -0.35}, {-0.30, 0.00}, {-0.30,  0.35},
    { 0.25, -0.35}, { 0.25, 0.00}, { 0.25,  0.35},
  };

  PointCPtr raw_world_cloud(new PointC);
  PointCPtr shape_world_cloud(new PointC);

  for (const auto &vp : viewpoints) {
    geometry_msgs::msg::Pose vpose;
    vpose.position.x  = vp.first;
    vpose.position.y  = vp.second;
    vpose.position.z  = VIEW_Z;
    vpose.orientation = tf2::toMsg(grasp_q);
    moveToPose(vpose);

    std::uint64_t seq_before;
    std::string frame;
    {
      std::lock_guard<std::mutex> lk(cloud_mutex_);
      seq_before = g_cloud_sequence_;
      frame      = g_input_pc_frame_id_;
    }

    const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);

    while (std::chrono::steady_clock::now() < deadline) {
      {
        std::lock_guard<std::mutex> lk(cloud_mutex_);
        if (g_cloud_sequence_ > seq_before) {
          frame = g_input_pc_frame_id_;
          break;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    PointCPtr snap;
    {
      std::lock_guard<std::mutex> lk(cloud_mutex_);
      if (g_cloud_ptr && !g_cloud_ptr->empty()) {
        snap = PointCPtr(new PointC(*g_cloud_ptr));
      }
    }

    PointCPtr base_cloud = transformCloudToBase(snap, frame);
    if (!base_cloud) continue;

    *raw_world_cloud += *base_cloud;

    PointCPtr shapes_only = filterShapeColours(base_cloud);
    if (shapes_only) {
      *shape_world_cloud += *shapes_only;
    }
  }

  PointCPtr raw_ds(new PointC);
  {
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(raw_world_cloud);
    vg.setLeafSize(0.005f, 0.005f, 0.005f);
    vg.filter(*raw_ds);
  }

  PointCPtr shape_ds(new PointC);
  {
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(shape_world_cloud);
    vg.setLeafSize(0.003f, 0.003f, 0.003f);
    vg.filter(*shape_ds);
  }

  RCLCPP_INFO(node_->get_logger(), "T3 raw points after voxel: %zu", raw_ds->size());
  RCLCPP_INFO(node_->get_logger(), "T3 shape points after voxel: %zu", shape_ds->size());

  if (shape_ds->size() < 50) {
    RCLCPP_ERROR(node_->get_logger(), "T3: not enough shape points after scan");
    moveToNamedPose("ready");
    return;
  }

  auto classifyClusterDirect = [&](const PointCPtr &cluster) -> std::string
  {
    if (!cluster || cluster->size() < 30) return "unknown";

    double z_max = -std::numeric_limits<double>::max();
    for (const auto &p : cluster->points) {
      z_max = std::max(z_max, static_cast<double>(p.z));
    }

    const double top_z_min = z_max - 0.035;
    std::vector<std::pair<double,double>> top_xy;

    for (const auto &p : cluster->points) {
      if (p.z >= top_z_min) top_xy.emplace_back(p.x, p.y);
    }

    if (top_xy.size() < 20) return "unknown";

    double cx = 0.0;
    double cy = 0.0;
    for (const auto &xy : top_xy) {
      cx += xy.first;
      cy += xy.second;
    }
    cx /= static_cast<double>(top_xy.size());
    cy /= static_cast<double>(top_xy.size());

    const double r_sq = 0.005 * 0.005;
    int cnt = 0;
    for (const auto &xy : top_xy) {
      const double dx = xy.first - cx;
      const double dy = xy.second - cy;
      if (dx * dx + dy * dy <= r_sq) ++cnt;
    }

    return (cnt < 20) ? "nought" : "cross";
  };

  struct ShapeInfo
  {
    std::string type;
    geometry_msgs::msg::Point centroid;
  };

  struct ObstacleInfo
  {
    geometry_msgs::msg::Point center;
    double sx;
    double sy;
    double sz;
  };

  std::vector<ShapeInfo> shapes;
  std::vector<ObstacleInfo> obstacles;

  geometry_msgs::msg::Point basket_pos;
  bool basket_found = false;

  if (!raw_ds->empty()) {
    pcl::search::KdTree<PointT>::Ptr raw_tree(new pcl::search::KdTree<PointT>);
    raw_tree->setInputCloud(raw_ds);

    std::vector<pcl::PointIndices> raw_cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> raw_ec;
    raw_ec.setClusterTolerance(0.025);
    raw_ec.setMinClusterSize(80);
    raw_ec.setMaxClusterSize(100000);
    raw_ec.setSearchMethod(raw_tree);
    raw_ec.setInputCloud(raw_ds);
    raw_ec.extract(raw_cluster_indices);

    double best_basket_score = std::numeric_limits<double>::max();

    for (const auto &indices : raw_cluster_indices) {
      PointCPtr cluster(new PointC);
      cluster->reserve(indices.indices.size());
      for (int idx : indices.indices) {
        cluster->push_back((*raw_ds)[idx]);
      }

      float xmin = 1e6f, xmax = -1e6f;
      float ymin = 1e6f, ymax = -1e6f;
      float zmin = 1e6f, zmax = -1e6f;
      double sr = 0.0, sg = 0.0, sb = 0.0;

      for (const auto &p : cluster->points) {
        xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
        zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z);

        sr += p.r / 255.0;
        sg += p.g / 255.0;
        sb += p.b / 255.0;
      }

      const double n = static_cast<double>(cluster->size());
      const double mr = sr / n;
      const double mg = sg / n;
      const double mb = sb / n;

      const double sx = static_cast<double>(xmax - xmin);
      const double sy = static_cast<double>(ymax - ymin);
      const double sz = static_cast<double>(zmax - zmin);

      geometry_msgs::msg::Point box_center;
      box_center.x = 0.5 * (xmin + xmax);
      box_center.y = 0.5 * (ymin + ymax);
      box_center.z = 0.5 * (zmin + zmax);

      const double color_score =
        std::pow(mr - 0.5, 2) +
        std::pow(mg - 0.2, 2) +
        std::pow(mb - 0.2, 2);

      const double size_score =
        std::pow(sx - 0.35, 2) +
        std::pow(sy - 0.35, 2) +
        std::pow(sz - 0.05, 2);

      const double ground_score = std::pow(box_center.z - 0.025, 2);
      const double basket_score = color_score + 2.0 * size_score + 3.0 * ground_score;

      const bool basket_colour_ok =
        mr > 0.30 && mr < 0.70 &&
        mg > 0.10 && mg < 0.35 &&
        mb > 0.10 && mb < 0.35;

      const bool basket_size_ok =
        sx > 0.22 && sx < 0.50 &&
        sy > 0.22 && sy < 0.50 &&
        sz > 0.02 && sz < 0.12;

      if (basket_colour_ok && basket_size_ok) {
        if (basket_score < best_basket_score) {
          best_basket_score = basket_score;
          basket_pos = box_center;
          basket_found = true;

          RCLCPP_INFO(node_->get_logger(),
            "T3 basket candidate accepted at center=(%.3f, %.3f, %.3f) "
            "size=(%.3f, %.3f, %.3f) color=(%.2f, %.2f, %.2f) score=%.4f",
            basket_pos.x, basket_pos.y, basket_pos.z,
            sx, sy, sz, mr, mg, mb, basket_score);
        }
        continue;
      }

      if (mr < 0.25 && mg < 0.25 && mb < 0.25 &&
          sx > 0.03 && sx < 0.30 &&
          sy > 0.03 && sy < 0.30 &&
          sz > 0.03) {
        ObstacleInfo obs;
        obs.center = box_center;
        obs.sx = std::max(0.03, sx);
        obs.sy = std::max(0.03, sy);
        obs.sz = std::max(0.03, sz);
        obstacles.push_back(obs);
      }
    }
  }

  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud(shape_ds);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(0.025);
  ec.setMinClusterSize(40);
  ec.setMaxClusterSize(50000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(shape_ds);
  ec.extract(cluster_indices);

  RCLCPP_INFO(node_->get_logger(), "T3: %zu shape clusters", cluster_indices.size());

  for (const auto &indices : cluster_indices) {
    PointCPtr cluster(new PointC);
    cluster->reserve(indices.indices.size());
    for (int idx : indices.indices) {
      cluster->push_back((*shape_ds)[idx]);
    }

    Eigen::Vector4f c4;
    pcl::compute3DCentroid(*cluster, c4);

    geometry_msgs::msg::Point centroid;
    centroid.x = c4[0];
    centroid.y = c4[1];
    centroid.z = c4[2];

    std::string type = classifyClusterDirect(cluster);
    const std::string confirmed = classify_at_centroid(centroid);
    if (confirmed != "unknown") type = confirmed;

    if (type == "unknown") {
      RCLCPP_WARN(node_->get_logger(),
        "T3: unclassified cluster at (%.3f, %.3f), skipped",
        centroid.x, centroid.y);
      continue;
    }

    shapes.push_back({type, centroid});

    RCLCPP_INFO(node_->get_logger(),
      "T3: shape #%zu -> %s at (%.3f, %.3f, %.3f)",
      shapes.size(), type.c_str(), centroid.x, centroid.y, centroid.z);
  }

  int n_noughts = 0;
  int n_crosses = 0;
  for (const auto &s : shapes) {
    if (s.type == "nought") ++n_noughts;
    else ++n_crosses;
  }

  const int total = n_noughts + n_crosses;
  const int most_common_cnt = std::max(n_noughts, n_crosses);
  const std::string most_common_type = (n_noughts >= n_crosses) ? "nought" : "cross";

  response->total_num_shapes = static_cast<int64_t>(total);
  response->num_most_common_shape = static_cast<int64_t>(most_common_cnt);

  RCLCPP_INFO(node_->get_logger(),
    "T3: total=%d noughts=%d crosses=%d most_common=%s(%d) obstacles=%zu",
    total, n_noughts, n_crosses, most_common_type.c_str(), most_common_cnt, obstacles.size());

  if (total == 0) {
    RCLCPP_ERROR(node_->get_logger(), "T3: no shapes detected");
    moveToNamedPose("ready");
    return;
  }

  if (!basket_found) {
    RCLCPP_WARN(node_->get_logger(),
      "T3: basket not detected from merged raw cloud, using fallback position");
    basket_pos.x = -0.41;
    basket_pos.y = -0.36;
    basket_pos.z = 0.02;
  } else {
    RCLCPP_INFO(node_->get_logger(),
      "T3: final basket position = (%.3f, %.3f, %.3f)",
      basket_pos.x, basket_pos.y, basket_pos.z);
  }

  const double MIN_GRASP_RADIUS = 0.20;
  const double MAX_GRASP_RADIUS = 0.65;
  const double OBSTACLE_CLEARANCE = 0.10;

  const ShapeInfo *target = nullptr;
  double best_dist = std::numeric_limits<double>::max();

  for (const auto &s : shapes) {
    if (s.type != most_common_type) continue;

    const double bx = s.centroid.x - basket_pos.x;
    const double by = s.centroid.y - basket_pos.y;
    if (bx * bx + by * by < 0.08 * 0.08) continue;

    const double d = std::hypot(s.centroid.x, s.centroid.y);
    if (d < MIN_GRASP_RADIUS || d > MAX_GRASP_RADIUS) continue;

    bool too_close_to_obstacle = false;
    for (const auto &obs : obstacles) {
      const double ox = s.centroid.x - obs.center.x;
      const double oy = s.centroid.y - obs.center.y;
      const double clearance =
        0.5 * std::max(obs.sx, obs.sy) + OBSTACLE_CLEARANCE;
      if (ox * ox + oy * oy < clearance * clearance) {
        too_close_to_obstacle = true;
        break;
      }
    }
    if (too_close_to_obstacle) continue;

    if (d < best_dist) {
      best_dist = d;
      target = &s;
    }
  }

  if (!target) {
    for (const auto &s : shapes) {
      const double d = std::hypot(s.centroid.x, s.centroid.y);
      if (d < MIN_GRASP_RADIUS || d > MAX_GRASP_RADIUS) continue;

      const double bx = s.centroid.x - basket_pos.x;
      const double by = s.centroid.y - basket_pos.y;
      if (bx * bx + by * by < 0.08 * 0.08) continue;

      bool too_close_to_obstacle = false;
      for (const auto &obs : obstacles) {
        const double ox = s.centroid.x - obs.center.x;
        const double oy = s.centroid.y - obs.center.y;
        const double clearance =
          0.5 * std::max(obs.sx, obs.sy) + OBSTACLE_CLEARANCE;
        if (ox * ox + oy * oy < clearance * clearance) {
          too_close_to_obstacle = true;
          break;
        }
      }
      if (too_close_to_obstacle) continue;

      if (d < best_dist) {
        best_dist = d;
        target = &s;
      }
    }
  }

  if (!target) {
    RCLCPP_ERROR(node_->get_logger(), "T3: no reachable target found");
    moveToNamedPose("ready");
    return;
  }

  geometry_msgs::msg::Point grasp_pt;
  grasp_pt.z = 0.02;

  if (target->type == "nought") {
    const double dist = std::hypot(target->centroid.x, target->centroid.y);
    if (dist > 0.01) {
      const double ux = -target->centroid.x / dist;
      const double uy = -target->centroid.y / dist;
      const double rim_offset = 0.05;
      grasp_pt.x = target->centroid.x + ux * rim_offset;
      grasp_pt.y = target->centroid.y + uy * rim_offset;
    } else {
      grasp_pt.x = target->centroid.x;
      grasp_pt.y = target->centroid.y;
    }
  } else {
    grasp_pt.x = target->centroid.x;
    grasp_pt.y = target->centroid.y;
  }

  geometry_msgs::msg::Point drop_pt;
  drop_pt.x = basket_pos.x;
  drop_pt.y = basket_pos.y;
  drop_pt.z = 0.02;

  auto safeDescend = [&](const geometry_msgs::msg::Pose &tp) {
    if (!computeAndExecuteCartesianPath(tp)) {
      RCLCPP_WARN(node_->get_logger(), "T3: Cartesian failed, using moveToPose");
      moveToPose(tp);
    }
  };

  const double TRAVEL_Z = 0.55;
  const double GRASP_Z  = 0.10;
  const double DROP_Z   = 0.15;

  moveToPose(makeAGraspOffset(grasp_pt, " ", TRAVEL_Z, grasp_q));
  openGripper();
  rclcpp::sleep_for(std::chrono::milliseconds(300));

  safeDescend(makeAGraspOffset(grasp_pt, " ", GRASP_Z, grasp_q));

  closeGripper();
  rclcpp::sleep_for(std::chrono::milliseconds(300));

  safeDescend(makeAGraspOffset(grasp_pt, " ", TRAVEL_Z, grasp_q));

  moveToPose(makeAGraspOffset(drop_pt, " ", TRAVEL_Z, grasp_q));
  safeDescend(makeAGraspOffset(drop_pt, " ", DROP_Z, grasp_q));

  openGripper();
  rclcpp::sleep_for(std::chrono::milliseconds(300));

  safeDescend(makeAGraspOffset(drop_pt, " ", TRAVEL_Z, grasp_q));
  moveToNamedPose("ready");

  RCLCPP_INFO(node_->get_logger(),
    "Task 3 complete: total=%lld  most_common=%lld  type=%s",
    static_cast<long long>(response->total_num_shapes),
    static_cast<long long>(response->num_most_common_shape),
    most_common_type.c_str());
}