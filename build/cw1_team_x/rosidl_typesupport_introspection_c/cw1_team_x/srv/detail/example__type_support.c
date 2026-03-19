// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from cw1_team_x:srv/Example.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "cw1_team_x/srv/detail/example__rosidl_typesupport_introspection_c.h"
#include "cw1_team_x/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "cw1_team_x/srv/detail/example__functions.h"
#include "cw1_team_x/srv/detail/example__struct.h"


// Include directives for member types
// Member `example_string`
#include "rosidl_runtime_c/string_functions.h"
// Member `example_pose`
// Member `example_std_vector_of_poses`
#include "geometry_msgs/msg/pose.h"
// Member `example_pose`
// Member `example_std_vector_of_poses`
#include "geometry_msgs/msg/detail/pose__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__Example_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  cw1_team_x__srv__Example_Request__init(message_memory);
}

void cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__Example_Request_fini_function(void * message_memory)
{
  cw1_team_x__srv__Example_Request__fini(message_memory);
}

size_t cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__size_function__Example_Request__example_std_vector_of_poses(
  const void * untyped_member)
{
  const geometry_msgs__msg__Pose__Sequence * member =
    (const geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  return member->size;
}

const void * cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__get_const_function__Example_Request__example_std_vector_of_poses(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Pose__Sequence * member =
    (const geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  return &member->data[index];
}

void * cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__get_function__Example_Request__example_std_vector_of_poses(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Pose__Sequence * member =
    (geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  return &member->data[index];
}

void cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__fetch_function__Example_Request__example_std_vector_of_poses(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const geometry_msgs__msg__Pose * item =
    ((const geometry_msgs__msg__Pose *)
    cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__get_const_function__Example_Request__example_std_vector_of_poses(untyped_member, index));
  geometry_msgs__msg__Pose * value =
    (geometry_msgs__msg__Pose *)(untyped_value);
  *value = *item;
}

void cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__assign_function__Example_Request__example_std_vector_of_poses(
  void * untyped_member, size_t index, const void * untyped_value)
{
  geometry_msgs__msg__Pose * item =
    ((geometry_msgs__msg__Pose *)
    cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__get_function__Example_Request__example_std_vector_of_poses(untyped_member, index));
  const geometry_msgs__msg__Pose * value =
    (const geometry_msgs__msg__Pose *)(untyped_value);
  *item = *value;
}

bool cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__resize_function__Example_Request__example_std_vector_of_poses(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Pose__Sequence * member =
    (geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  geometry_msgs__msg__Pose__Sequence__fini(member);
  return geometry_msgs__msg__Pose__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__Example_Request_message_member_array[5] = {
  {
    "example_float",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw1_team_x__srv__Example_Request, example_float),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "example_int",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw1_team_x__srv__Example_Request, example_int),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "example_string",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw1_team_x__srv__Example_Request, example_string),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "example_pose",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw1_team_x__srv__Example_Request, example_pose),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "example_std_vector_of_poses",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw1_team_x__srv__Example_Request, example_std_vector_of_poses),  // bytes offset in struct
    NULL,  // default value
    cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__size_function__Example_Request__example_std_vector_of_poses,  // size() function pointer
    cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__get_const_function__Example_Request__example_std_vector_of_poses,  // get_const(index) function pointer
    cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__get_function__Example_Request__example_std_vector_of_poses,  // get(index) function pointer
    cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__fetch_function__Example_Request__example_std_vector_of_poses,  // fetch(index, &value) function pointer
    cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__assign_function__Example_Request__example_std_vector_of_poses,  // assign(index, value) function pointer
    cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__resize_function__Example_Request__example_std_vector_of_poses  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__Example_Request_message_members = {
  "cw1_team_x__srv",  // message namespace
  "Example_Request",  // message name
  5,  // number of fields
  sizeof(cw1_team_x__srv__Example_Request),
  cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__Example_Request_message_member_array,  // message members
  cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__Example_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__Example_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__Example_Request_message_type_support_handle = {
  0,
  &cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__Example_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cw1_team_x
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_team_x, srv, Example_Request)() {
  cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__Example_Request_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Pose)();
  cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__Example_Request_message_member_array[4].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Pose)();
  if (!cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__Example_Request_message_type_support_handle.typesupport_identifier) {
    cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__Example_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &cw1_team_x__srv__Example_Request__rosidl_typesupport_introspection_c__Example_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "cw1_team_x/srv/detail/example__rosidl_typesupport_introspection_c.h"
// already included above
// #include "cw1_team_x/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "cw1_team_x/srv/detail/example__functions.h"
// already included above
// #include "cw1_team_x/srv/detail/example__struct.h"


// Include directives for member types
// Member `example_std_vector_of_points`
#include "geometry_msgs/msg/point.h"
// Member `example_std_vector_of_points`
#include "geometry_msgs/msg/detail/point__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__Example_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  cw1_team_x__srv__Example_Response__init(message_memory);
}

void cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__Example_Response_fini_function(void * message_memory)
{
  cw1_team_x__srv__Example_Response__fini(message_memory);
}

size_t cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__size_function__Example_Response__example_std_vector_of_points(
  const void * untyped_member)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return member->size;
}

const void * cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__get_const_function__Example_Response__example_std_vector_of_points(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void * cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__get_function__Example_Response__example_std_vector_of_points(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__fetch_function__Example_Response__example_std_vector_of_points(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const geometry_msgs__msg__Point * item =
    ((const geometry_msgs__msg__Point *)
    cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__get_const_function__Example_Response__example_std_vector_of_points(untyped_member, index));
  geometry_msgs__msg__Point * value =
    (geometry_msgs__msg__Point *)(untyped_value);
  *value = *item;
}

void cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__assign_function__Example_Response__example_std_vector_of_points(
  void * untyped_member, size_t index, const void * untyped_value)
{
  geometry_msgs__msg__Point * item =
    ((geometry_msgs__msg__Point *)
    cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__get_function__Example_Response__example_std_vector_of_points(untyped_member, index));
  const geometry_msgs__msg__Point * value =
    (const geometry_msgs__msg__Point *)(untyped_value);
  *item = *value;
}

bool cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__resize_function__Example_Response__example_std_vector_of_points(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  geometry_msgs__msg__Point__Sequence__fini(member);
  return geometry_msgs__msg__Point__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__Example_Response_message_member_array[2] = {
  {
    "success",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw1_team_x__srv__Example_Response, success),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "example_std_vector_of_points",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw1_team_x__srv__Example_Response, example_std_vector_of_points),  // bytes offset in struct
    NULL,  // default value
    cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__size_function__Example_Response__example_std_vector_of_points,  // size() function pointer
    cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__get_const_function__Example_Response__example_std_vector_of_points,  // get_const(index) function pointer
    cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__get_function__Example_Response__example_std_vector_of_points,  // get(index) function pointer
    cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__fetch_function__Example_Response__example_std_vector_of_points,  // fetch(index, &value) function pointer
    cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__assign_function__Example_Response__example_std_vector_of_points,  // assign(index, value) function pointer
    cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__resize_function__Example_Response__example_std_vector_of_points  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__Example_Response_message_members = {
  "cw1_team_x__srv",  // message namespace
  "Example_Response",  // message name
  2,  // number of fields
  sizeof(cw1_team_x__srv__Example_Response),
  cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__Example_Response_message_member_array,  // message members
  cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__Example_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__Example_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__Example_Response_message_type_support_handle = {
  0,
  &cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__Example_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cw1_team_x
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_team_x, srv, Example_Response)() {
  cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__Example_Response_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  if (!cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__Example_Response_message_type_support_handle.typesupport_identifier) {
    cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__Example_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &cw1_team_x__srv__Example_Response__rosidl_typesupport_introspection_c__Example_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "cw1_team_x/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "cw1_team_x/srv/detail/example__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers cw1_team_x__srv__detail__example__rosidl_typesupport_introspection_c__Example_service_members = {
  "cw1_team_x__srv",  // service namespace
  "Example",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // cw1_team_x__srv__detail__example__rosidl_typesupport_introspection_c__Example_Request_message_type_support_handle,
  NULL  // response message
  // cw1_team_x__srv__detail__example__rosidl_typesupport_introspection_c__Example_Response_message_type_support_handle
};

static rosidl_service_type_support_t cw1_team_x__srv__detail__example__rosidl_typesupport_introspection_c__Example_service_type_support_handle = {
  0,
  &cw1_team_x__srv__detail__example__rosidl_typesupport_introspection_c__Example_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_team_x, srv, Example_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_team_x, srv, Example_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cw1_team_x
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_team_x, srv, Example)() {
  if (!cw1_team_x__srv__detail__example__rosidl_typesupport_introspection_c__Example_service_type_support_handle.typesupport_identifier) {
    cw1_team_x__srv__detail__example__rosidl_typesupport_introspection_c__Example_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)cw1_team_x__srv__detail__example__rosidl_typesupport_introspection_c__Example_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_team_x, srv, Example_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_team_x, srv, Example_Response)()->data;
  }

  return &cw1_team_x__srv__detail__example__rosidl_typesupport_introspection_c__Example_service_type_support_handle;
}
