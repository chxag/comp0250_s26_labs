// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cw1_team_x:srv/Example.idl
// generated code does not contain a copyright notice

#ifndef CW1_TEAM_X__SRV__DETAIL__EXAMPLE__STRUCT_H_
#define CW1_TEAM_X__SRV__DETAIL__EXAMPLE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'example_string'
#include "rosidl_runtime_c/string.h"
// Member 'example_pose'
// Member 'example_std_vector_of_poses'
#include "geometry_msgs/msg/detail/pose__struct.h"

/// Struct defined in srv/Example in the package cw1_team_x.
typedef struct cw1_team_x__srv__Example_Request
{
  double example_float;
  int32_t example_int;
  rosidl_runtime_c__String example_string;
  geometry_msgs__msg__Pose example_pose;
  geometry_msgs__msg__Pose__Sequence example_std_vector_of_poses;
} cw1_team_x__srv__Example_Request;

// Struct for a sequence of cw1_team_x__srv__Example_Request.
typedef struct cw1_team_x__srv__Example_Request__Sequence
{
  cw1_team_x__srv__Example_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cw1_team_x__srv__Example_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'example_std_vector_of_points'
#include "geometry_msgs/msg/detail/point__struct.h"

/// Struct defined in srv/Example in the package cw1_team_x.
typedef struct cw1_team_x__srv__Example_Response
{
  bool success;
  geometry_msgs__msg__Point__Sequence example_std_vector_of_points;
} cw1_team_x__srv__Example_Response;

// Struct for a sequence of cw1_team_x__srv__Example_Response.
typedef struct cw1_team_x__srv__Example_Response__Sequence
{
  cw1_team_x__srv__Example_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cw1_team_x__srv__Example_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CW1_TEAM_X__SRV__DETAIL__EXAMPLE__STRUCT_H_
