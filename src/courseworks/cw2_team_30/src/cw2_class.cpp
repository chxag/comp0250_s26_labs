/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#include <cw2_class.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>  // for tf2::toMsg
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>

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

void cw2::t1_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response)
{
  (void)response;

  const auto &object = request->object_point.point;
  const auto &basket = request->goal_point.point;
  const auto &shape_type = request->shape_type;

  tf2:: Quaternion orientation;
  orientation.setRPY(M_PI, 0, -M_PI / 4); // Change when T1_ANY_ORIENTATION = True

  // 1. Move to ready position
  arm_group_->setNamedTarget("ready");

  // 2. Move to pre-grasp position, just above the object
  moveToPose(makeAGraspOffset(object, shape_type, 0.5, orientation ));

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
    moveToPose(makeAGraspOffset(basket, " ",  0.5, orientation));
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
        cloud_frame = g_input_pc_frame_id_;
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

      const double xy_radius = 0.12;
      const double xy_radius_sq = xy_radius * xy_radius;
      const double z_min = query_point.point.z - 0.03;
      const double z_max = query_point.point.z + 0.12;

      double observed_z_max = -std::numeric_limits<double>::max();
      for (const auto &pt : cloud_snapshot->points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
          continue;
        }

        const tf2::Vector3 p_t = tf_cloud_to_target * tf2::Vector3(pt.x, pt.y, pt.z);
        const double dx = p_t.x() - query_point.point.x;
        const double dy = p_t.y() - query_point.point.y;
        if ((dx * dx + dy * dy) > xy_radius_sq) {
          continue;
        }
        if (p_t.z() < z_min || p_t.z() > z_max) {
          continue;
        }

        geometry_msgs::msg::Point p;
        p.x = p_t.x();
        p.y = p_t.y();
        p.z = p_t.z();
        local_points.push_back(p);
        observed_z_max = std::max(observed_z_max, p.z);
      }

      if (local_points.size() < 80) {
        return "unknown";
      }

      const double top_z_min = observed_z_max - 0.035;
      std::vector<geometry_msgs::msg::Point> top_points;
      top_points.reserve(local_points.size());
      for (const auto &p : local_points) {
        if (p.z >= top_z_min) {
          top_points.push_back(p);
        }
      }
      if (top_points.size() < 30) {
        return "unknown";
      }

      double cx = 0.0;
      double cy = 0.0;
      for (const auto &p : top_points) {
        cx += p.x;
        cy += p.y;
      }
      cx /= static_cast<double>(top_points.size());
      cy /= static_cast<double>(top_points.size());

      const double center_radius = 0.005;
      const double center_radius_sq = center_radius * center_radius;
      int center_count = 0;
      for (const auto &p : top_points) {
        const double dx = p.x - cx;
        const double dy = p.y - cy;
        if (dx * dx + dy * dy <= center_radius_sq) {
          ++center_count;
        }
      }

      return (center_count < 20) ? "nought" : "cross";
    };

  auto classify_with_viewpoint =
    [&](const geometry_msgs::msg::PointStamped &point_stamped) -> std::string {
      // Move above candidate to reduce occlusion and get a clear local view.
      geometry_msgs::msg::Pose view_pose;
      view_pose.position.x = point_stamped.point.x;
      view_pose.position.y = point_stamped.point.y;
      view_pose.position.z = point_stamped.point.z + 0.50;
      view_pose.orientation = tf2::toMsg(view_orientation);
      moveToPose(view_pose);

      std::uint64_t before_seq = 0;
      {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        before_seq = g_cloud_sequence_;
      }

      const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(900);
      while (std::chrono::steady_clock::now() < deadline) {
        {
          std::lock_guard<std::mutex> lock(cloud_mutex_);
          if (g_cloud_sequence_ > before_seq) {
            break;
          }
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
    RCLCPP_INFO(
      node_->get_logger(),
      "Task2 reference #%zu classified as: %s",
      i + 1,
      shape.c_str());
  }

  const auto mystery_shape = classify_with_viewpoint(request->mystery_object_point);
  RCLCPP_INFO(node_->get_logger(), "Task2 mystery classified as: %s", mystery_shape.c_str());

  // Return 1 if mystery matches ref1, 2 if mystery matches ref2, 
  int64_t match_index = 1;  
  if (ref_shapes.size() >= 2) {
    const bool match_ref1 = (mystery_shape != "unknown" && mystery_shape == ref_shapes[0]);
    const bool match_ref2 = (mystery_shape != "unknown" && mystery_shape == ref_shapes[1]);

    if (match_ref1 && !match_ref2) {
      match_index = 1;
    } else if (match_ref2 && !match_ref1) {
      match_index = 2;
    } else if (ref_shapes[0] == "unknown" && ref_shapes[1] != "unknown") {
      match_index = 2;
    } else {
      // ambiguous case: keep deterministic fallback to 1
      match_index = 1;
    }
  }
  response->mystery_object_num = match_index;

  RCLCPP_INFO(
    node_->get_logger(),
    "Task2 final output: ref1=%s ref2=%s mystery=%s -> mystery_object_num=%lld",
    ref_shapes.size() > 0 ? ref_shapes[0].c_str() : "n/a",
    ref_shapes.size() > 1 ? ref_shapes[1].c_str() : "n/a",
    mystery_shape.c_str(),
    static_cast<long long>(response->mystery_object_num));

  moveToNamedPose("ready");
}

void cw2::t3_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  response->total_num_shapes = 0;
  response->num_most_common_shape = 0;
  response->most_common_shape_vector.clear();

  std::string frame_id;
  std::size_t point_count = 0;
  std::uint64_t sequence = 0;
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    frame_id = g_input_pc_frame_id_;
    point_count = g_cloud_ptr ? g_cloud_ptr->size() : 0;
    sequence = g_cloud_sequence_;
  }

  RCLCPP_WARN(
    node_->get_logger(),
    "Task 3 is not implemented in cw2_team_30. Latest cloud: seq=%llu frame='%s' points=%zu",
    static_cast<unsigned long long>(sequence),
    frame_id.c_str(),
    point_count);
}
