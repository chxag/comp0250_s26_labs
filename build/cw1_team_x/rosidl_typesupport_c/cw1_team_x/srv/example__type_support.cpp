// generated from rosidl_typesupport_c/resource/idl__type_support.cpp.em
// with input from cw1_team_x:srv/Example.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "cw1_team_x/srv/detail/example__struct.h"
#include "cw1_team_x/srv/detail/example__type_support.h"
#include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/message_type_support_dispatch.h"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_c/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace cw1_team_x
{

namespace srv
{

namespace rosidl_typesupport_c
{

typedef struct _Example_Request_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Example_Request_type_support_ids_t;

static const _Example_Request_type_support_ids_t _Example_Request_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _Example_Request_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Example_Request_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Example_Request_type_support_symbol_names_t _Example_Request_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, cw1_team_x, srv, Example_Request)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_team_x, srv, Example_Request)),
  }
};

typedef struct _Example_Request_type_support_data_t
{
  void * data[2];
} _Example_Request_type_support_data_t;

static _Example_Request_type_support_data_t _Example_Request_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Example_Request_message_typesupport_map = {
  2,
  "cw1_team_x",
  &_Example_Request_message_typesupport_ids.typesupport_identifier[0],
  &_Example_Request_message_typesupport_symbol_names.symbol_name[0],
  &_Example_Request_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t Example_Request_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Example_Request_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace cw1_team_x

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, cw1_team_x, srv, Example_Request)() {
  return &::cw1_team_x::srv::rosidl_typesupport_c::Example_Request_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "cw1_team_x/srv/detail/example__struct.h"
// already included above
// #include "cw1_team_x/srv/detail/example__type_support.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace cw1_team_x
{

namespace srv
{

namespace rosidl_typesupport_c
{

typedef struct _Example_Response_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Example_Response_type_support_ids_t;

static const _Example_Response_type_support_ids_t _Example_Response_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _Example_Response_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Example_Response_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Example_Response_type_support_symbol_names_t _Example_Response_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, cw1_team_x, srv, Example_Response)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_team_x, srv, Example_Response)),
  }
};

typedef struct _Example_Response_type_support_data_t
{
  void * data[2];
} _Example_Response_type_support_data_t;

static _Example_Response_type_support_data_t _Example_Response_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Example_Response_message_typesupport_map = {
  2,
  "cw1_team_x",
  &_Example_Response_message_typesupport_ids.typesupport_identifier[0],
  &_Example_Response_message_typesupport_symbol_names.symbol_name[0],
  &_Example_Response_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t Example_Response_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Example_Response_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace cw1_team_x

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, cw1_team_x, srv, Example_Response)() {
  return &::cw1_team_x::srv::rosidl_typesupport_c::Example_Response_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "cw1_team_x/srv/detail/example__type_support.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/service_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace cw1_team_x
{

namespace srv
{

namespace rosidl_typesupport_c
{

typedef struct _Example_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _Example_type_support_ids_t;

static const _Example_type_support_ids_t _Example_service_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _Example_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _Example_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _Example_type_support_symbol_names_t _Example_service_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, cw1_team_x, srv, Example)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cw1_team_x, srv, Example)),
  }
};

typedef struct _Example_type_support_data_t
{
  void * data[2];
} _Example_type_support_data_t;

static _Example_type_support_data_t _Example_service_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _Example_service_typesupport_map = {
  2,
  "cw1_team_x",
  &_Example_service_typesupport_ids.typesupport_identifier[0],
  &_Example_service_typesupport_symbol_names.symbol_name[0],
  &_Example_service_typesupport_data.data[0],
};

static const rosidl_service_type_support_t Example_service_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_Example_service_typesupport_map),
  rosidl_typesupport_c__get_service_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace cw1_team_x

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_c, cw1_team_x, srv, Example)() {
  return &::cw1_team_x::srv::rosidl_typesupport_c::Example_service_type_support_handle;
}

#ifdef __cplusplus
}
#endif
