// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cw1_world_spawner:srv/Task3Service.idl
// generated code does not contain a copyright notice

#ifndef CW1_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__BUILDER_HPP_
#define CW1_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "cw1_world_spawner/srv/detail/task3_service__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace cw1_world_spawner
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::cw1_world_spawner::srv::Task3Service_Request>()
{
  return ::cw1_world_spawner::srv::Task3Service_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
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
auto build<::cw1_world_spawner::srv::Task3Service_Response>()
{
  return ::cw1_world_spawner::srv::Task3Service_Response(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace cw1_world_spawner

#endif  // CW1_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__BUILDER_HPP_
