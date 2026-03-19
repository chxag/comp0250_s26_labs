// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cw1_world_spawner:srv/TaskSetup.idl
// generated code does not contain a copyright notice

#ifndef CW1_WORLD_SPAWNER__SRV__DETAIL__TASK_SETUP__BUILDER_HPP_
#define CW1_WORLD_SPAWNER__SRV__DETAIL__TASK_SETUP__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "cw1_world_spawner/srv/detail/task_setup__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace cw1_world_spawner
{

namespace srv
{

namespace builder
{

class Init_TaskSetup_Request_task_index
{
public:
  Init_TaskSetup_Request_task_index()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::cw1_world_spawner::srv::TaskSetup_Request task_index(::cw1_world_spawner::srv::TaskSetup_Request::_task_index_type arg)
  {
    msg_.task_index = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cw1_world_spawner::srv::TaskSetup_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::cw1_world_spawner::srv::TaskSetup_Request>()
{
  return cw1_world_spawner::srv::builder::Init_TaskSetup_Request_task_index();
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
auto build<::cw1_world_spawner::srv::TaskSetup_Response>()
{
  return ::cw1_world_spawner::srv::TaskSetup_Response(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace cw1_world_spawner

#endif  // CW1_WORLD_SPAWNER__SRV__DETAIL__TASK_SETUP__BUILDER_HPP_
