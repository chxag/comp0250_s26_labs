/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirement is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#ifndef CW2_CLASS_H_
#define CW2_CLASS_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>
#include <atomic>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <octomap/OcTree.h>
#include <octomap/OcTreeKey.h>

#include "cw2_world_spawner/srv/task1_service.hpp"
#include "cw2_world_spawner/srv/task2_service.hpp"
#include "cw2_world_spawner/srv/task3_service.hpp"

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointC;
typedef PointC::Ptr PointCPtr;

// Hash function for octomap::OcTreeKey to use in unordered_set
struct KeyHash {
  std::size_t operator()(const octomap::OcTreeKey& k) const {
    return ((static_cast<std::size_t>(k.k[0]) * 73856093) ^ 
            (static_cast<std::size_t>(k.k[1]) * 19349663) ^ 
            (static_cast<std::size_t>(k.k[2]) * 83492791));
  }
};

// Structure to hold detected object information
struct DetectedObj {
  std::string category;      // "object" or "basket"
  std::string shape;         // "nought" or "cross"
  geometry_msgs::msg::Point centroid;
  std::unordered_set<octomap::OcTreeKey, KeyHash> voxel_keys;
  double min_x, max_x, min_y, max_y, min_z, max_z;
};

class cw2
{
public:
  explicit cw2(const rclcpp::Node::SharedPtr &node);

  // Task service callbacks
  void t1_callback(
    const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request> request,
    std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response);
  void t2_callback(
    const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request> request,
    std::shared_ptr<cw2_world_spawner::srv::Task2Service::Response> response);
  void t3_callback(
    const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
    std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response);

  // Point cloud callback
  void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);

  // Arm movement functions
  void moveToNamedPose(const std::string &pose_name);
  void openGripper();
  void closeGripper();
  void moveToPose(const geometry_msgs::msg::Pose &target_pose);
  bool computeAndExecuteCartesianPath(const geometry_msgs::msg::Pose &target);
  
  // Helper for grasp offset
  geometry_msgs::msg::Pose makeAGraspOffset(
    const geometry_msgs::msg::Point &point,
    const std::string &shape_type,
    double z_offset,
    const tf2::Quaternion &orientation, 
    double shape_yaw);
  
  double computeShapeOrientation(const geometry_msgs::msg::PointStamped &query_point);

  void waitForFreshCloud(int frames_to_wait = 2, double timeout_sec = 2.0);
  
  // OctoMap functions for Task 3 (unused but kept for compatibility)
  void buildOctomapFromAccumulatedCloud();
  bool extractObjectsFromOctomap(std::vector<DetectedObj>& out_objects);

  // PCL filtering helpers
  template <typename PointT>
  typename pcl::PointCloud<PointT>::Ptr filterPassThrough(
      const typename pcl::PointCloud<PointT>::Ptr& cloud,
      const std::string& axis, float min_val, float max_val);
  
  PointCPtr filterTopLayer(const PointCPtr& cloud);

  // TF helper
  geometry_msgs::msg::PointStamped makePointStamped(double x, double y, double z);

private:
  rclcpp::Node::SharedPtr node_;
  
  // Services
  rclcpp::Service<cw2_world_spawner::srv::Task1Service>::SharedPtr t1_service_;
  rclcpp::Service<cw2_world_spawner::srv::Task2Service>::SharedPtr t2_service_;
  rclcpp::Service<cw2_world_spawner::srv::Task3Service>::SharedPtr t3_service_;

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr color_cloud_sub_;
  rclcpp::CallbackGroup::SharedPtr pointcloud_callback_group_;

  // MoveIt interfaces
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_group_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> hand_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

  // TF
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Point cloud data (protected by mutex)
  std::mutex cloud_mutex_;
  PointCPtr g_cloud_ptr;
  std::uint64_t g_cloud_sequence_ = 0;
  std::string g_input_pc_frame_id_;

  // Parameters
  std::string pointcloud_topic_;
  bool pointcloud_qos_reliable_ = false;

  // Task 3 specific members
  std::shared_ptr<octomap::OcTree> latest_octree_;          
  PointCPtr accumulated_cloud_;
  std::mutex accumulated_cloud_mutex_;                      
  std::atomic<bool> is_scanning_{false};                    

  // Shape classification helper 
  std::string classifyShapeAtPoint(const geometry_msgs::msg::PointStamped &query_point);
};

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr cw2::filterPassThrough(
    const typename pcl::PointCloud<PointT>::Ptr& cloud,
    const std::string& axis, float min_val, float max_val)
{
  typename pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
  pcl::PassThrough<PointT> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName(axis);
  pass.setFilterLimits(min_val, max_val);
  pass.filter(*filtered);
  return filtered;
}

#endif  // CW2_CLASS_H_