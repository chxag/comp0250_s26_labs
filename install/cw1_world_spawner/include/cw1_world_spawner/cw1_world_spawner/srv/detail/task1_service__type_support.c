// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from cw1_world_spawner:srv/Task1Service.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "cw1_world_spawner/srv/detail/task1_service__rosidl_typesupport_introspection_c.h"
#include "cw1_world_spawner/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "cw1_world_spawner/srv/detail/task1_service__functions.h"
#include "cw1_world_spawner/srv/detail/task1_service__struct.h"


// Include directives for member types
// Member `object_loc`
#include "geometry_msgs/msg/pose_stamped.h"
// Member `object_loc`
#include "geometry_msgs/msg/detail/pose_stamped__rosidl_typesupport_introspection_c.h"
// Member `goal_loc`
#include "geometry_msgs/msg/point_stamped.h"
// Member `goal_loc`
#include "geometry_msgs/msg/detail/point_stamped__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void cw1_world_spawner__srv__Task1Service_Request__rosidl_typesupport_introspection_c__Task1Service_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  cw1_world_spawner__srv__Task1Service_Request__init(message_memory);
}

void cw1_world_spawner__srv__Task1Service_Request__rosidl_typesupport_introspection_c__Task1Service_Request_fini_function(void * message_memory)
{
  cw1_world_spawner__srv__Task1Service_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember cw1_world_spawner__srv__Task1Service_Request__rosidl_typesupport_introspection_c__Task1Service_Request_message_member_array[2] = {
  {
    "object_loc",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw1_world_spawner__srv__Task1Service_Request, object_loc),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "goal_loc",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw1_world_spawner__srv__Task1Service_Request, goal_loc),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers cw1_world_spawner__srv__Task1Service_Request__rosidl_typesupport_introspection_c__Task1Service_Request_message_members = {
  "cw1_world_spawner__srv",  // message namespace
  "Task1Service_Request",  // message name
  2,  // number of fields
  sizeof(cw1_world_spawner__srv__Task1Service_Request),
  cw1_world_spawner__srv__Task1Service_Request__rosidl_typesupport_introspection_c__Task1Service_Request_message_member_array,  // message members
  cw1_world_spawner__srv__Task1Service_Request__rosidl_typesupport_introspection_c__Task1Service_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  cw1_world_spawner__srv__Task1Service_Request__rosidl_typesupport_introspection_c__Task1Service_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t cw1_world_spawner__srv__Task1Service_Request__rosidl_typesupport_introspection_c__Task1Service_Request_message_type_support_handle = {
  0,
  &cw1_world_spawner__srv__Task1Service_Request__rosidl_typesupport_introspection_c__Task1Service_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cw1_world_spawner
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_world_spawner, srv, Task1Service_Request)() {
  cw1_world_spawner__srv__Task1Service_Request__rosidl_typesupport_introspection_c__Task1Service_Request_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, PoseStamped)();
  cw1_world_spawner__srv__Task1Service_Request__rosidl_typesupport_introspection_c__Task1Service_Request_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, PointStamped)();
  if (!cw1_world_spawner__srv__Task1Service_Request__rosidl_typesupport_introspection_c__Task1Service_Request_message_type_support_handle.typesupport_identifier) {
    cw1_world_spawner__srv__Task1Service_Request__rosidl_typesupport_introspection_c__Task1Service_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &cw1_world_spawner__srv__Task1Service_Request__rosidl_typesupport_introspection_c__Task1Service_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "cw1_world_spawner/srv/detail/task1_service__rosidl_typesupport_introspection_c.h"
// already included above
// #include "cw1_world_spawner/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "cw1_world_spawner/srv/detail/task1_service__functions.h"
// already included above
// #include "cw1_world_spawner/srv/detail/task1_service__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void cw1_world_spawner__srv__Task1Service_Response__rosidl_typesupport_introspection_c__Task1Service_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  cw1_world_spawner__srv__Task1Service_Response__init(message_memory);
}

void cw1_world_spawner__srv__Task1Service_Response__rosidl_typesupport_introspection_c__Task1Service_Response_fini_function(void * message_memory)
{
  cw1_world_spawner__srv__Task1Service_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember cw1_world_spawner__srv__Task1Service_Response__rosidl_typesupport_introspection_c__Task1Service_Response_message_member_array[1] = {
  {
    "structure_needs_at_least_one_member",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw1_world_spawner__srv__Task1Service_Response, structure_needs_at_least_one_member),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers cw1_world_spawner__srv__Task1Service_Response__rosidl_typesupport_introspection_c__Task1Service_Response_message_members = {
  "cw1_world_spawner__srv",  // message namespace
  "Task1Service_Response",  // message name
  1,  // number of fields
  sizeof(cw1_world_spawner__srv__Task1Service_Response),
  cw1_world_spawner__srv__Task1Service_Response__rosidl_typesupport_introspection_c__Task1Service_Response_message_member_array,  // message members
  cw1_world_spawner__srv__Task1Service_Response__rosidl_typesupport_introspection_c__Task1Service_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  cw1_world_spawner__srv__Task1Service_Response__rosidl_typesupport_introspection_c__Task1Service_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t cw1_world_spawner__srv__Task1Service_Response__rosidl_typesupport_introspection_c__Task1Service_Response_message_type_support_handle = {
  0,
  &cw1_world_spawner__srv__Task1Service_Response__rosidl_typesupport_introspection_c__Task1Service_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cw1_world_spawner
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_world_spawner, srv, Task1Service_Response)() {
  if (!cw1_world_spawner__srv__Task1Service_Response__rosidl_typesupport_introspection_c__Task1Service_Response_message_type_support_handle.typesupport_identifier) {
    cw1_world_spawner__srv__Task1Service_Response__rosidl_typesupport_introspection_c__Task1Service_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &cw1_world_spawner__srv__Task1Service_Response__rosidl_typesupport_introspection_c__Task1Service_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "cw1_world_spawner/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "cw1_world_spawner/srv/detail/task1_service__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers cw1_world_spawner__srv__detail__task1_service__rosidl_typesupport_introspection_c__Task1Service_service_members = {
  "cw1_world_spawner__srv",  // service namespace
  "Task1Service",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // cw1_world_spawner__srv__detail__task1_service__rosidl_typesupport_introspection_c__Task1Service_Request_message_type_support_handle,
  NULL  // response message
  // cw1_world_spawner__srv__detail__task1_service__rosidl_typesupport_introspection_c__Task1Service_Response_message_type_support_handle
};

static rosidl_service_type_support_t cw1_world_spawner__srv__detail__task1_service__rosidl_typesupport_introspection_c__Task1Service_service_type_support_handle = {
  0,
  &cw1_world_spawner__srv__detail__task1_service__rosidl_typesupport_introspection_c__Task1Service_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_world_spawner, srv, Task1Service_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_world_spawner, srv, Task1Service_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cw1_world_spawner
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_world_spawner, srv, Task1Service)() {
  if (!cw1_world_spawner__srv__detail__task1_service__rosidl_typesupport_introspection_c__Task1Service_service_type_support_handle.typesupport_identifier) {
    cw1_world_spawner__srv__detail__task1_service__rosidl_typesupport_introspection_c__Task1Service_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)cw1_world_spawner__srv__detail__task1_service__rosidl_typesupport_introspection_c__Task1Service_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_world_spawner, srv, Task1Service_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_world_spawner, srv, Task1Service_Response)()->data;
  }

  return &cw1_world_spawner__srv__detail__task1_service__rosidl_typesupport_introspection_c__Task1Service_service_type_support_handle;
}
