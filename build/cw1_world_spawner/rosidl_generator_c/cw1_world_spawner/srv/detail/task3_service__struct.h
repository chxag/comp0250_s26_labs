// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cw1_world_spawner:srv/Task3Service.idl
// generated code does not contain a copyright notice

#ifndef CW1_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__STRUCT_H_
#define CW1_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/Task3Service in the package cw1_world_spawner.
typedef struct cw1_world_spawner__srv__Task3Service_Request
{
  uint8_t structure_needs_at_least_one_member;
} cw1_world_spawner__srv__Task3Service_Request;

// Struct for a sequence of cw1_world_spawner__srv__Task3Service_Request.
typedef struct cw1_world_spawner__srv__Task3Service_Request__Sequence
{
  cw1_world_spawner__srv__Task3Service_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cw1_world_spawner__srv__Task3Service_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/Task3Service in the package cw1_world_spawner.
typedef struct cw1_world_spawner__srv__Task3Service_Response
{
  uint8_t structure_needs_at_least_one_member;
} cw1_world_spawner__srv__Task3Service_Response;

// Struct for a sequence of cw1_world_spawner__srv__Task3Service_Response.
typedef struct cw1_world_spawner__srv__Task3Service_Response__Sequence
{
  cw1_world_spawner__srv__Task3Service_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cw1_world_spawner__srv__Task3Service_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CW1_WORLD_SPAWNER__SRV__DETAIL__TASK3_SERVICE__STRUCT_H_
