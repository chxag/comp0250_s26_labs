/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire 
solution is contained within the cw1_team_<your_team_number> package */

#ifndef CW1_CLASS_H_
#define CW1_CLASS_H_

// system includes
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "cw1_world_spawner/srv/task1_service.hpp"
#include "cw1_world_spawner/srv/task2_service.hpp"
#include "cw1_world_spawner/srv/task3_service.hpp"

/* MoveIt! includes */
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

/* TF2 */
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

/* PCL */
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class cw1
{
public:

  /* ----- class member functions ----- */

  explicit cw1(const rclcpp::Node::SharedPtr &node);

  void t1_callback(
    const std::shared_ptr<cw1_world_spawner::srv::Task1Service::Request> request,
    std::shared_ptr<cw1_world_spawner::srv::Task1Service::Response> response);
  void t2_callback(
    const std::shared_ptr<cw1_world_spawner::srv::Task2Service::Request> request,
    std::shared_ptr<cw1_world_spawner::srv::Task2Service::Response> response);
  void t3_callback(
    const std::shared_ptr<cw1_world_spawner::srv::Task3Service::Request> request,
    std::shared_ptr<cw1_world_spawner::srv::Task3Service::Response> response);

  /* ----- class member variables ----- */

  rclcpp::Node::SharedPtr node_;
  rclcpp::Service<cw1_world_spawner::srv::Task1Service>::SharedPtr t1_service_;
  rclcpp::Service<cw1_world_spawner::srv::Task2Service>::SharedPtr t2_service_;
  rclcpp::Service<cw1_world_spawner::srv::Task3Service>::SharedPtr t3_service_;
  rclcpp::CallbackGroup::SharedPtr service_cb_group_;
  rclcpp::CallbackGroup::SharedPtr sensor_cb_group_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;

  std::atomic<int64_t> latest_joint_state_stamp_ns_{0};
  std::atomic<uint64_t> joint_state_msg_count_{0};
  std::atomic<int64_t> latest_cloud_stamp_ns_{0};
  std::atomic<uint64_t> cloud_msg_count_{0};

  bool enable_cloud_viewer_ = false;
  bool move_home_on_start_ = false;
  bool use_path_constraints_ = false;
  bool use_cartesian_reach_ = false;
  bool allow_position_only_fallback_ = false;
  bool publish_programmatic_debug_ = false;
  bool enable_task1_snap_ = false;
  bool return_home_between_pick_place_ = false;
  bool return_home_after_pick_place_ = false;
  bool task2_capture_enabled_ = false;

  double cartesian_eef_step_ = 0.005;
  double cartesian_jump_threshold_ = 0.0;
  double cartesian_min_fraction_ = 0.98;
  double pick_offset_z_ = 0.12;
  double task3_pick_offset_z_ = 0.13;
  double place_offset_z_ = 0.35;
  double grasp_approach_offset_z_ = 0.015;
  double post_grasp_lift_z_ = 0.05;
  double gripper_grasp_width_ = 0.03;
  double joint_state_wait_timeout_sec_ = 2.0;

  std::string task2_capture_dir_ = "/tmp/cw1_task2_capture";

  /* MoveIt! interfaces */
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_group;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> hand_group;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

  /* TF2 */
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  /* Latest point cloud protected by mutex */
  sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_cloud_;
  std::mutex cloud_mutex_;

  /* ----- motion helpers ----- */
  bool moveToPose(const geometry_msgs::msg::Pose target_pose);
  bool moveToLiftXY(double x, double y);
  bool moveToGraspZ(double x, double y, double z);
  bool setGripper(double width);
  bool pickUpObject(const geometry_msgs::msg::PoseStamped &object_loc);
  bool placeObject(const geometry_msgs::msg::PoseStamped &goal_loc);

  /* ----- Task 2 helpers ----- */
  bool moveToScanPose();
  sensor_msgs::msg::PointCloud2::ConstSharedPtr waitForCloud(double timeout_sec = 5.0);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cropAroundBasket(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const geometry_msgs::msg::PointStamped & basket_world_loc);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr removeNoiseAndFloor(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud);

  geometry_msgs::msg::PointStamped transformToCameraFrame(
    const geometry_msgs::msg::PointStamped & point_in_world,
    const std::string & target_frame);

  std::string detectBasketColour(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const geometry_msgs::msg::PointStamped & basket_world_loc);
};

#endif // CW1_CLASS_H_