// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from cw1_world_spawner:srv/TaskSetup.idl
// generated code does not contain a copyright notice

#ifndef CW1_WORLD_SPAWNER__SRV__DETAIL__TASK_SETUP__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define CW1_WORLD_SPAWNER__SRV__DETAIL__TASK_SETUP__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "cw1_world_spawner/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "cw1_world_spawner/srv/detail/task_setup__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

#include "fastcdr/Cdr.h"

namespace cw1_world_spawner
{

namespace srv
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cw1_world_spawner
cdr_serialize(
  const cw1_world_spawner::srv::TaskSetup_Request & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cw1_world_spawner
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  cw1_world_spawner::srv::TaskSetup_Request & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cw1_world_spawner
get_serialized_size(
  const cw1_world_spawner::srv::TaskSetup_Request & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cw1_world_spawner
max_serialized_size_TaskSetup_Request(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace srv

}  // namespace cw1_world_spawner

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cw1_world_spawner
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, cw1_world_spawner, srv, TaskSetup_Request)();

#ifdef __cplusplus
}
#endif

// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "cw1_world_spawner/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
// already included above
// #include "cw1_world_spawner/srv/detail/task_setup__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// already included above
// #include "fastcdr/Cdr.h"

namespace cw1_world_spawner
{

namespace srv
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cw1_world_spawner
cdr_serialize(
  const cw1_world_spawner::srv::TaskSetup_Response & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cw1_world_spawner
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  cw1_world_spawner::srv::TaskSetup_Response & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cw1_world_spawner
get_serialized_size(
  const cw1_world_spawner::srv::TaskSetup_Response & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cw1_world_spawner
max_serialized_size_TaskSetup_Response(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace srv

}  // namespace cw1_world_spawner

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cw1_world_spawner
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, cw1_world_spawner, srv, TaskSetup_Response)();

#ifdef __cplusplus
}
#endif

#include "rmw/types.h"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "cw1_world_spawner/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cw1_world_spawner
const rosidl_service_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, cw1_world_spawner, srv, TaskSetup)();

#ifdef __cplusplus
}
#endif

#endif  // CW1_WORLD_SPAWNER__SRV__DETAIL__TASK_SETUP__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
