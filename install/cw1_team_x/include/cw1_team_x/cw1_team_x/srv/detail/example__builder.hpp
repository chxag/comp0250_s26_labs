// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cw1_team_x:srv/Example.idl
// generated code does not contain a copyright notice

#ifndef CW1_TEAM_X__SRV__DETAIL__EXAMPLE__BUILDER_HPP_
#define CW1_TEAM_X__SRV__DETAIL__EXAMPLE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "cw1_team_x/srv/detail/example__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace cw1_team_x
{

namespace srv
{

namespace builder
{

class Init_Example_Request_example_std_vector_of_poses
{
public:
  explicit Init_Example_Request_example_std_vector_of_poses(::cw1_team_x::srv::Example_Request & msg)
  : msg_(msg)
  {}
  ::cw1_team_x::srv::Example_Request example_std_vector_of_poses(::cw1_team_x::srv::Example_Request::_example_std_vector_of_poses_type arg)
  {
    msg_.example_std_vector_of_poses = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cw1_team_x::srv::Example_Request msg_;
};

class Init_Example_Request_example_pose
{
public:
  explicit Init_Example_Request_example_pose(::cw1_team_x::srv::Example_Request & msg)
  : msg_(msg)
  {}
  Init_Example_Request_example_std_vector_of_poses example_pose(::cw1_team_x::srv::Example_Request::_example_pose_type arg)
  {
    msg_.example_pose = std::move(arg);
    return Init_Example_Request_example_std_vector_of_poses(msg_);
  }

private:
  ::cw1_team_x::srv::Example_Request msg_;
};

class Init_Example_Request_example_string
{
public:
  explicit Init_Example_Request_example_string(::cw1_team_x::srv::Example_Request & msg)
  : msg_(msg)
  {}
  Init_Example_Request_example_pose example_string(::cw1_team_x::srv::Example_Request::_example_string_type arg)
  {
    msg_.example_string = std::move(arg);
    return Init_Example_Request_example_pose(msg_);
  }

private:
  ::cw1_team_x::srv::Example_Request msg_;
};

class Init_Example_Request_example_int
{
public:
  explicit Init_Example_Request_example_int(::cw1_team_x::srv::Example_Request & msg)
  : msg_(msg)
  {}
  Init_Example_Request_example_string example_int(::cw1_team_x::srv::Example_Request::_example_int_type arg)
  {
    msg_.example_int = std::move(arg);
    return Init_Example_Request_example_string(msg_);
  }

private:
  ::cw1_team_x::srv::Example_Request msg_;
};

class Init_Example_Request_example_float
{
public:
  Init_Example_Request_example_float()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Example_Request_example_int example_float(::cw1_team_x::srv::Example_Request::_example_float_type arg)
  {
    msg_.example_float = std::move(arg);
    return Init_Example_Request_example_int(msg_);
  }

private:
  ::cw1_team_x::srv::Example_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::cw1_team_x::srv::Example_Request>()
{
  return cw1_team_x::srv::builder::Init_Example_Request_example_float();
}

}  // namespace cw1_team_x


namespace cw1_team_x
{

namespace srv
{

namespace builder
{

class Init_Example_Response_example_std_vector_of_points
{
public:
  explicit Init_Example_Response_example_std_vector_of_points(::cw1_team_x::srv::Example_Response & msg)
  : msg_(msg)
  {}
  ::cw1_team_x::srv::Example_Response example_std_vector_of_points(::cw1_team_x::srv::Example_Response::_example_std_vector_of_points_type arg)
  {
    msg_.example_std_vector_of_points = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cw1_team_x::srv::Example_Response msg_;
};

class Init_Example_Response_success
{
public:
  Init_Example_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Example_Response_example_std_vector_of_points success(::cw1_team_x::srv::Example_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_Example_Response_example_std_vector_of_points(msg_);
  }

private:
  ::cw1_team_x::srv::Example_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::cw1_team_x::srv::Example_Response>()
{
  return cw1_team_x::srv::builder::Init_Example_Response_success();
}

}  // namespace cw1_team_x

#endif  // CW1_TEAM_X__SRV__DETAIL__EXAMPLE__BUILDER_HPP_
