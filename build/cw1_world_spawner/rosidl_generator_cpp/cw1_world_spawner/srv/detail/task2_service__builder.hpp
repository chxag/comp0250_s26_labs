// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cw1_world_spawner:srv/Task2Service.idl
// generated code does not contain a copyright notice

#ifndef CW1_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__BUILDER_HPP_
#define CW1_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "cw1_world_spawner/srv/detail/task2_service__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace cw1_world_spawner
{

namespace srv
{

namespace builder
{

class Init_Task2Service_Request_basket_locs
{
public:
  Init_Task2Service_Request_basket_locs()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::cw1_world_spawner::srv::Task2Service_Request basket_locs(::cw1_world_spawner::srv::Task2Service_Request::_basket_locs_type arg)
  {
    msg_.basket_locs = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cw1_world_spawner::srv::Task2Service_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::cw1_world_spawner::srv::Task2Service_Request>()
{
  return cw1_world_spawner::srv::builder::Init_Task2Service_Request_basket_locs();
}

}  // namespace cw1_world_spawner


namespace cw1_world_spawner
{

namespace srv
{

namespace builder
{

class Init_Task2Service_Response_basket_colours
{
public:
  Init_Task2Service_Response_basket_colours()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::cw1_world_spawner::srv::Task2Service_Response basket_colours(::cw1_world_spawner::srv::Task2Service_Response::_basket_colours_type arg)
  {
    msg_.basket_colours = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cw1_world_spawner::srv::Task2Service_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::cw1_world_spawner::srv::Task2Service_Response>()
{
  return cw1_world_spawner::srv::builder::Init_Task2Service_Response_basket_colours();
}

}  // namespace cw1_world_spawner

#endif  // CW1_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__BUILDER_HPP_
