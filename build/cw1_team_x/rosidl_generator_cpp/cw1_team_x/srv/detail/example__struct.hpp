// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from cw1_team_x:srv/Example.idl
// generated code does not contain a copyright notice

#ifndef CW1_TEAM_X__SRV__DETAIL__EXAMPLE__STRUCT_HPP_
#define CW1_TEAM_X__SRV__DETAIL__EXAMPLE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'example_pose'
// Member 'example_std_vector_of_poses'
#include "geometry_msgs/msg/detail/pose__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__cw1_team_x__srv__Example_Request __attribute__((deprecated))
#else
# define DEPRECATED__cw1_team_x__srv__Example_Request __declspec(deprecated)
#endif

namespace cw1_team_x
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Example_Request_
{
  using Type = Example_Request_<ContainerAllocator>;

  explicit Example_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : example_pose(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->example_float = 0.0;
      this->example_int = 0l;
      this->example_string = "";
    }
  }

  explicit Example_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : example_string(_alloc),
    example_pose(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->example_float = 0.0;
      this->example_int = 0l;
      this->example_string = "";
    }
  }

  // field types and members
  using _example_float_type =
    double;
  _example_float_type example_float;
  using _example_int_type =
    int32_t;
  _example_int_type example_int;
  using _example_string_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _example_string_type example_string;
  using _example_pose_type =
    geometry_msgs::msg::Pose_<ContainerAllocator>;
  _example_pose_type example_pose;
  using _example_std_vector_of_poses_type =
    std::vector<geometry_msgs::msg::Pose_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Pose_<ContainerAllocator>>>;
  _example_std_vector_of_poses_type example_std_vector_of_poses;

  // setters for named parameter idiom
  Type & set__example_float(
    const double & _arg)
  {
    this->example_float = _arg;
    return *this;
  }
  Type & set__example_int(
    const int32_t & _arg)
  {
    this->example_int = _arg;
    return *this;
  }
  Type & set__example_string(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->example_string = _arg;
    return *this;
  }
  Type & set__example_pose(
    const geometry_msgs::msg::Pose_<ContainerAllocator> & _arg)
  {
    this->example_pose = _arg;
    return *this;
  }
  Type & set__example_std_vector_of_poses(
    const std::vector<geometry_msgs::msg::Pose_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Pose_<ContainerAllocator>>> & _arg)
  {
    this->example_std_vector_of_poses = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    cw1_team_x::srv::Example_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const cw1_team_x::srv::Example_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cw1_team_x::srv::Example_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cw1_team_x::srv::Example_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cw1_team_x::srv::Example_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cw1_team_x::srv::Example_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cw1_team_x::srv::Example_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cw1_team_x::srv::Example_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cw1_team_x::srv::Example_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cw1_team_x::srv::Example_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cw1_team_x__srv__Example_Request
    std::shared_ptr<cw1_team_x::srv::Example_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cw1_team_x__srv__Example_Request
    std::shared_ptr<cw1_team_x::srv::Example_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Example_Request_ & other) const
  {
    if (this->example_float != other.example_float) {
      return false;
    }
    if (this->example_int != other.example_int) {
      return false;
    }
    if (this->example_string != other.example_string) {
      return false;
    }
    if (this->example_pose != other.example_pose) {
      return false;
    }
    if (this->example_std_vector_of_poses != other.example_std_vector_of_poses) {
      return false;
    }
    return true;
  }
  bool operator!=(const Example_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Example_Request_

// alias to use template instance with default allocator
using Example_Request =
  cw1_team_x::srv::Example_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace cw1_team_x


// Include directives for member types
// Member 'example_std_vector_of_points'
#include "geometry_msgs/msg/detail/point__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__cw1_team_x__srv__Example_Response __attribute__((deprecated))
#else
# define DEPRECATED__cw1_team_x__srv__Example_Response __declspec(deprecated)
#endif

namespace cw1_team_x
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Example_Response_
{
  using Type = Example_Response_<ContainerAllocator>;

  explicit Example_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit Example_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;
  using _example_std_vector_of_points_type =
    std::vector<geometry_msgs::msg::Point_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Point_<ContainerAllocator>>>;
  _example_std_vector_of_points_type example_std_vector_of_points;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }
  Type & set__example_std_vector_of_points(
    const std::vector<geometry_msgs::msg::Point_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Point_<ContainerAllocator>>> & _arg)
  {
    this->example_std_vector_of_points = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    cw1_team_x::srv::Example_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const cw1_team_x::srv::Example_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cw1_team_x::srv::Example_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cw1_team_x::srv::Example_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cw1_team_x::srv::Example_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cw1_team_x::srv::Example_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cw1_team_x::srv::Example_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cw1_team_x::srv::Example_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cw1_team_x::srv::Example_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cw1_team_x::srv::Example_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cw1_team_x__srv__Example_Response
    std::shared_ptr<cw1_team_x::srv::Example_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cw1_team_x__srv__Example_Response
    std::shared_ptr<cw1_team_x::srv::Example_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Example_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    if (this->example_std_vector_of_points != other.example_std_vector_of_points) {
      return false;
    }
    return true;
  }
  bool operator!=(const Example_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Example_Response_

// alias to use template instance with default allocator
using Example_Response =
  cw1_team_x::srv::Example_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace cw1_team_x

namespace cw1_team_x
{

namespace srv
{

struct Example
{
  using Request = cw1_team_x::srv::Example_Request;
  using Response = cw1_team_x::srv::Example_Response;
};

}  // namespace srv

}  // namespace cw1_team_x

#endif  // CW1_TEAM_X__SRV__DETAIL__EXAMPLE__STRUCT_HPP_
