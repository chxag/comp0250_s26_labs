// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cw1_world_spawner:srv/Task1Service.idl
// generated code does not contain a copyright notice

#ifndef CW1_WORLD_SPAWNER__SRV__DETAIL__TASK1_SERVICE__BUILDER_HPP_
#define CW1_WORLD_SPAWNER__SRV__DETAIL__TASK1_SERVICE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "cw1_world_spawner/srv/detail/task1_service__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace cw1_world_spawner
{

namespace srv
{

namespace builder
{

class Init_Task1Service_Request_goal_loc
{
public:
  explicit Init_Task1Service_Request_goal_loc(::cw1_world_spawner::srv::Task1Service_Request & msg)
  : msg_(msg)
  {}
  ::cw1_world_spawner::srv::Task1Service_Request goal_loc(::cw1_world_spawner::srv::Task1Service_Request::_goal_loc_type arg)
  {
    msg_.goal_loc = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cw1_world_spawner::srv::Task1Service_Request msg_;
};

class Init_Task1Service_Request_object_loc
{
public:
  Init_Task1Service_Request_object_loc()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Task1Service_Request_goal_loc object_loc(::cw1_world_spawner::srv::Task1Service_Request::_object_loc_type arg)
  {
    msg_.object_loc = std::move(arg);
    return Init_Task1Service_Request_goal_loc(msg_);
  }

private:
  ::cw1_world_spawner::srv::Task1Service_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::cw1_world_spawner::srv::Task1Service_Request>()
{
  return cw1_world_spawner::srv::builder::Init_Task1Service_Request_object_loc();
}

}  // namespace cw1_world_spawner


namespace cw1_world_spawner
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::cw1_world_spawner::srv::Task1Service_Response>()
{
  return ::cw1_world_spawner::srv::Task1Service_Response(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace cw1_world_spawner

#endif  // CW1_WORLD_SPAWNER__SRV__DETAIL__TASK1_SERVICE__BUILDER_HPP_
