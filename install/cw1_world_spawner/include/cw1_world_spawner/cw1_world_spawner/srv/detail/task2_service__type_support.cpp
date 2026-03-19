// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from cw1_world_spawner:srv/Task2Service.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "cw1_world_spawner/srv/detail/task2_service__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace cw1_world_spawner
{

namespace srv
{

namespace rosidl_typesupport_introspection_cpp
{

void Task2Service_Request_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) cw1_world_spawner::srv::Task2Service_Request(_init);
}

void Task2Service_Request_fini_function(void * message_memory)
{
  auto typed_message = static_cast<cw1_world_spawner::srv::Task2Service_Request *>(message_memory);
  typed_message->~Task2Service_Request();
}

size_t size_function__Task2Service_Request__basket_locs(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<geometry_msgs::msg::PointStamped> *>(untyped_member);
  return member->size();
}

const void * get_const_function__Task2Service_Request__basket_locs(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<geometry_msgs::msg::PointStamped> *>(untyped_member);
  return &member[index];
}

void * get_function__Task2Service_Request__basket_locs(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<geometry_msgs::msg::PointStamped> *>(untyped_member);
  return &member[index];
}

void fetch_function__Task2Service_Request__basket_locs(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const geometry_msgs::msg::PointStamped *>(
    get_const_function__Task2Service_Request__basket_locs(untyped_member, index));
  auto & value = *reinterpret_cast<geometry_msgs::msg::PointStamped *>(untyped_value);
  value = item;
}

void assign_function__Task2Service_Request__basket_locs(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<geometry_msgs::msg::PointStamped *>(
    get_function__Task2Service_Request__basket_locs(untyped_member, index));
  const auto & value = *reinterpret_cast<const geometry_msgs::msg::PointStamped *>(untyped_value);
  item = value;
}

void resize_function__Task2Service_Request__basket_locs(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<geometry_msgs::msg::PointStamped> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember Task2Service_Request_message_member_array[1] = {
  {
    "basket_locs",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<geometry_msgs::msg::PointStamped>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw1_world_spawner::srv::Task2Service_Request, basket_locs),  // bytes offset in struct
    nullptr,  // default value
    size_function__Task2Service_Request__basket_locs,  // size() function pointer
    get_const_function__Task2Service_Request__basket_locs,  // get_const(index) function pointer
    get_function__Task2Service_Request__basket_locs,  // get(index) function pointer
    fetch_function__Task2Service_Request__basket_locs,  // fetch(index, &value) function pointer
    assign_function__Task2Service_Request__basket_locs,  // assign(index, value) function pointer
    resize_function__Task2Service_Request__basket_locs  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers Task2Service_Request_message_members = {
  "cw1_world_spawner::srv",  // message namespace
  "Task2Service_Request",  // message name
  1,  // number of fields
  sizeof(cw1_world_spawner::srv::Task2Service_Request),
  Task2Service_Request_message_member_array,  // message members
  Task2Service_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  Task2Service_Request_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t Task2Service_Request_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &Task2Service_Request_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace srv

}  // namespace cw1_world_spawner


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<cw1_world_spawner::srv::Task2Service_Request>()
{
  return &::cw1_world_spawner::srv::rosidl_typesupport_introspection_cpp::Task2Service_Request_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, cw1_world_spawner, srv, Task2Service_Request)() {
  return &::cw1_world_spawner::srv::rosidl_typesupport_introspection_cpp::Task2Service_Request_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "array"
// already included above
// #include "cstddef"
// already included above
// #include "string"
// already included above
// #include "vector"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "cw1_world_spawner/srv/detail/task2_service__struct.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/field_types.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace cw1_world_spawner
{

namespace srv
{

namespace rosidl_typesupport_introspection_cpp
{

void Task2Service_Response_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) cw1_world_spawner::srv::Task2Service_Response(_init);
}

void Task2Service_Response_fini_function(void * message_memory)
{
  auto typed_message = static_cast<cw1_world_spawner::srv::Task2Service_Response *>(message_memory);
  typed_message->~Task2Service_Response();
}

size_t size_function__Task2Service_Response__basket_colours(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<std::string> *>(untyped_member);
  return member->size();
}

const void * get_const_function__Task2Service_Response__basket_colours(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<std::string> *>(untyped_member);
  return &member[index];
}

void * get_function__Task2Service_Response__basket_colours(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<std::string> *>(untyped_member);
  return &member[index];
}

void fetch_function__Task2Service_Response__basket_colours(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const std::string *>(
    get_const_function__Task2Service_Response__basket_colours(untyped_member, index));
  auto & value = *reinterpret_cast<std::string *>(untyped_value);
  value = item;
}

void assign_function__Task2Service_Response__basket_colours(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<std::string *>(
    get_function__Task2Service_Response__basket_colours(untyped_member, index));
  const auto & value = *reinterpret_cast<const std::string *>(untyped_value);
  item = value;
}

void resize_function__Task2Service_Response__basket_colours(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<std::string> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember Task2Service_Response_message_member_array[1] = {
  {
    "basket_colours",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cw1_world_spawner::srv::Task2Service_Response, basket_colours),  // bytes offset in struct
    nullptr,  // default value
    size_function__Task2Service_Response__basket_colours,  // size() function pointer
    get_const_function__Task2Service_Response__basket_colours,  // get_const(index) function pointer
    get_function__Task2Service_Response__basket_colours,  // get(index) function pointer
    fetch_function__Task2Service_Response__basket_colours,  // fetch(index, &value) function pointer
    assign_function__Task2Service_Response__basket_colours,  // assign(index, value) function pointer
    resize_function__Task2Service_Response__basket_colours  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers Task2Service_Response_message_members = {
  "cw1_world_spawner::srv",  // message namespace
  "Task2Service_Response",  // message name
  1,  // number of fields
  sizeof(cw1_world_spawner::srv::Task2Service_Response),
  Task2Service_Response_message_member_array,  // message members
  Task2Service_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  Task2Service_Response_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t Task2Service_Response_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &Task2Service_Response_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace srv

}  // namespace cw1_world_spawner


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<cw1_world_spawner::srv::Task2Service_Response>()
{
  return &::cw1_world_spawner::srv::rosidl_typesupport_introspection_cpp::Task2Service_Response_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, cw1_world_spawner, srv, Task2Service_Response)() {
  return &::cw1_world_spawner::srv::rosidl_typesupport_introspection_cpp::Task2Service_Response_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "rosidl_typesupport_introspection_cpp/visibility_control.h"
// already included above
// #include "cw1_world_spawner/srv/detail/task2_service__struct.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/service_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/service_type_support_decl.hpp"

namespace cw1_world_spawner
{

namespace srv
{

namespace rosidl_typesupport_introspection_cpp
{

// this is intentionally not const to allow initialization later to prevent an initialization race
static ::rosidl_typesupport_introspection_cpp::ServiceMembers Task2Service_service_members = {
  "cw1_world_spawner::srv",  // service namespace
  "Task2Service",  // service name
  // these two fields are initialized below on the first access
  // see get_service_type_support_handle<cw1_world_spawner::srv::Task2Service>()
  nullptr,  // request message
  nullptr  // response message
};

static const rosidl_service_type_support_t Task2Service_service_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &Task2Service_service_members,
  get_service_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace srv

}  // namespace cw1_world_spawner


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_service_type_support_t *
get_service_type_support_handle<cw1_world_spawner::srv::Task2Service>()
{
  // get a handle to the value to be returned
  auto service_type_support =
    &::cw1_world_spawner::srv::rosidl_typesupport_introspection_cpp::Task2Service_service_type_support_handle;
  // get a non-const and properly typed version of the data void *
  auto service_members = const_cast<::rosidl_typesupport_introspection_cpp::ServiceMembers *>(
    static_cast<const ::rosidl_typesupport_introspection_cpp::ServiceMembers *>(
      service_type_support->data));
  // make sure that both the request_members_ and the response_members_ are initialized
  // if they are not, initialize them
  if (
    service_members->request_members_ == nullptr ||
    service_members->response_members_ == nullptr)
  {
    // initialize the request_members_ with the static function from the external library
    service_members->request_members_ = static_cast<
      const ::rosidl_typesupport_introspection_cpp::MessageMembers *
      >(
      ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<
        ::cw1_world_spawner::srv::Task2Service_Request
      >()->data
      );
    // initialize the response_members_ with the static function from the external library
    service_members->response_members_ = static_cast<
      const ::rosidl_typesupport_introspection_cpp::MessageMembers *
      >(
      ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<
        ::cw1_world_spawner::srv::Task2Service_Response
      >()->data
      );
  }
  // finally return the properly initialized service_type_support handle
  return service_type_support;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, cw1_world_spawner, srv, Task2Service)() {
  return ::rosidl_typesupport_introspection_cpp::get_service_type_support_handle<cw1_world_spawner::srv::Task2Service>();
}

#ifdef __cplusplus
}
#endif
