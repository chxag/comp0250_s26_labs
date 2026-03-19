// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from cw1_world_spawner:srv/TaskSetup.idl
// generated code does not contain a copyright notice

#ifndef CW1_WORLD_SPAWNER__SRV__DETAIL__TASK_SETUP__STRUCT_HPP_
#define CW1_WORLD_SPAWNER__SRV__DETAIL__TASK_SETUP__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__cw1_world_spawner__srv__TaskSetup_Request __attribute__((deprecated))
#else
# define DEPRECATED__cw1_world_spawner__srv__TaskSetup_Request __declspec(deprecated)
#endif

namespace cw1_world_spawner
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct TaskSetup_Request_
{
  using Type = TaskSetup_Request_<ContainerAllocator>;

  explicit TaskSetup_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->task_index = 0;
    }
  }

  explicit TaskSetup_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->task_index = 0;
    }
  }

  // field types and members
  using _task_index_type =
    int8_t;
  _task_index_type task_index;

  // setters for named parameter idiom
  Type & set__task_index(
    const int8_t & _arg)
  {
    this->task_index = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    cw1_world_spawner::srv::TaskSetup_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const cw1_world_spawner::srv::TaskSetup_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cw1_world_spawner::srv::TaskSetup_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cw1_world_spawner::srv::TaskSetup_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cw1_world_spawner::srv::TaskSetup_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cw1_world_spawner::srv::TaskSetup_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cw1_world_spawner::srv::TaskSetup_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cw1_world_spawner::srv::TaskSetup_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cw1_world_spawner::srv::TaskSetup_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cw1_world_spawner::srv::TaskSetup_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cw1_world_spawner__srv__TaskSetup_Request
    std::shared_ptr<cw1_world_spawner::srv::TaskSetup_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cw1_world_spawner__srv__TaskSetup_Request
    std::shared_ptr<cw1_world_spawner::srv::TaskSetup_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const TaskSetup_Request_ & other) const
  {
    if (this->task_index != other.task_index) {
      return false;
    }
    return true;
  }
  bool operator!=(const TaskSetup_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct TaskSetup_Request_

// alias to use template instance with default allocator
using TaskSetup_Request =
  cw1_world_spawner::srv::TaskSetup_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace cw1_world_spawner


#ifndef _WIN32
# define DEPRECATED__cw1_world_spawner__srv__TaskSetup_Response __attribute__((deprecated))
#else
# define DEPRECATED__cw1_world_spawner__srv__TaskSetup_Response __declspec(deprecated)
#endif

namespace cw1_world_spawner
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct TaskSetup_Response_
{
  using Type = TaskSetup_Response_<ContainerAllocator>;

  explicit TaskSetup_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  explicit TaskSetup_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  // field types and members
  using _structure_needs_at_least_one_member_type =
    uint8_t;
  _structure_needs_at_least_one_member_type structure_needs_at_least_one_member;


  // constant declarations

  // pointer types
  using RawPtr =
    cw1_world_spawner::srv::TaskSetup_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const cw1_world_spawner::srv::TaskSetup_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cw1_world_spawner::srv::TaskSetup_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cw1_world_spawner::srv::TaskSetup_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cw1_world_spawner::srv::TaskSetup_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cw1_world_spawner::srv::TaskSetup_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cw1_world_spawner::srv::TaskSetup_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cw1_world_spawner::srv::TaskSetup_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cw1_world_spawner::srv::TaskSetup_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cw1_world_spawner::srv::TaskSetup_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cw1_world_spawner__srv__TaskSetup_Response
    std::shared_ptr<cw1_world_spawner::srv::TaskSetup_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cw1_world_spawner__srv__TaskSetup_Response
    std::shared_ptr<cw1_world_spawner::srv::TaskSetup_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const TaskSetup_Response_ & other) const
  {
    if (this->structure_needs_at_least_one_member != other.structure_needs_at_least_one_member) {
      return false;
    }
    return true;
  }
  bool operator!=(const TaskSetup_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct TaskSetup_Response_

// alias to use template instance with default allocator
using TaskSetup_Response =
  cw1_world_spawner::srv::TaskSetup_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace cw1_world_spawner

namespace cw1_world_spawner
{

namespace srv
{

struct TaskSetup
{
  using Request = cw1_world_spawner::srv::TaskSetup_Request;
  using Response = cw1_world_spawner::srv::TaskSetup_Response;
};

}  // namespace srv

}  // namespace cw1_world_spawner

#endif  // CW1_WORLD_SPAWNER__SRV__DETAIL__TASK_SETUP__STRUCT_HPP_
