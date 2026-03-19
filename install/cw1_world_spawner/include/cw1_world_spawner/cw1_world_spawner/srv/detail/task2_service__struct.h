// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cw1_world_spawner:srv/Task2Service.idl
// generated code does not contain a copyright notice

#ifndef CW1_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__STRUCT_H_
#define CW1_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'basket_locs'
#include "geometry_msgs/msg/detail/point_stamped__struct.h"

/// Struct defined in srv/Task2Service in the package cw1_world_spawner.
typedef struct cw1_world_spawner__srv__Task2Service_Request
{
  geometry_msgs__msg__PointStamped__Sequence basket_locs;
} cw1_world_spawner__srv__Task2Service_Request;

// Struct for a sequence of cw1_world_spawner__srv__Task2Service_Request.
typedef struct cw1_world_spawner__srv__Task2Service_Request__Sequence
{
  cw1_world_spawner__srv__Task2Service_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cw1_world_spawner__srv__Task2Service_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'basket_colours'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/Task2Service in the package cw1_world_spawner.
typedef struct cw1_world_spawner__srv__Task2Service_Response
{
  rosidl_runtime_c__String__Sequence basket_colours;
} cw1_world_spawner__srv__Task2Service_Response;

// Struct for a sequence of cw1_world_spawner__srv__Task2Service_Response.
typedef struct cw1_world_spawner__srv__Task2Service_Response__Sequence
{
  cw1_world_spawner__srv__Task2Service_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cw1_world_spawner__srv__Task2Service_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CW1_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__STRUCT_H_
