// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cw1_team_x:srv/Example.idl
// generated code does not contain a copyright notice

#ifndef CW1_TEAM_X__SRV__DETAIL__EXAMPLE__TRAITS_HPP_
#define CW1_TEAM_X__SRV__DETAIL__EXAMPLE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "cw1_team_x/srv/detail/example__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'example_pose'
// Member 'example_std_vector_of_poses'
#include "geometry_msgs/msg/detail/pose__traits.hpp"

namespace cw1_team_x
{

namespace srv
{

inline void to_flow_style_yaml(
  const Example_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: example_float
  {
    out << "example_float: ";
    rosidl_generator_traits::value_to_yaml(msg.example_float, out);
    out << ", ";
  }

  // member: example_int
  {
    out << "example_int: ";
    rosidl_generator_traits::value_to_yaml(msg.example_int, out);
    out << ", ";
  }

  // member: example_string
  {
    out << "example_string: ";
    rosidl_generator_traits::value_to_yaml(msg.example_string, out);
    out << ", ";
  }

  // member: example_pose
  {
    out << "example_pose: ";
    to_flow_style_yaml(msg.example_pose, out);
    out << ", ";
  }

  // member: example_std_vector_of_poses
  {
    if (msg.example_std_vector_of_poses.size() == 0) {
      out << "example_std_vector_of_poses: []";
    } else {
      out << "example_std_vector_of_poses: [";
      size_t pending_items = msg.example_std_vector_of_poses.size();
      for (auto item : msg.example_std_vector_of_poses) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Example_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: example_float
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "example_float: ";
    rosidl_generator_traits::value_to_yaml(msg.example_float, out);
    out << "\n";
  }

  // member: example_int
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "example_int: ";
    rosidl_generator_traits::value_to_yaml(msg.example_int, out);
    out << "\n";
  }

  // member: example_string
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "example_string: ";
    rosidl_generator_traits::value_to_yaml(msg.example_string, out);
    out << "\n";
  }

  // member: example_pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "example_pose:\n";
    to_block_style_yaml(msg.example_pose, out, indentation + 2);
  }

  // member: example_std_vector_of_poses
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.example_std_vector_of_poses.size() == 0) {
      out << "example_std_vector_of_poses: []\n";
    } else {
      out << "example_std_vector_of_poses:\n";
      for (auto item : msg.example_std_vector_of_poses) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Example_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace cw1_team_x

namespace rosidl_generator_traits
{

[[deprecated("use cw1_team_x::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const cw1_team_x::srv::Example_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  cw1_team_x::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use cw1_team_x::srv::to_yaml() instead")]]
inline std::string to_yaml(const cw1_team_x::srv::Example_Request & msg)
{
  return cw1_team_x::srv::to_yaml(msg);
}

template<>
inline const char * data_type<cw1_team_x::srv::Example_Request>()
{
  return "cw1_team_x::srv::Example_Request";
}

template<>
inline const char * name<cw1_team_x::srv::Example_Request>()
{
  return "cw1_team_x/srv/Example_Request";
}

template<>
struct has_fixed_size<cw1_team_x::srv::Example_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<cw1_team_x::srv::Example_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<cw1_team_x::srv::Example_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'example_std_vector_of_points'
#include "geometry_msgs/msg/detail/point__traits.hpp"

namespace cw1_team_x
{

namespace srv
{

inline void to_flow_style_yaml(
  const Example_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: example_std_vector_of_points
  {
    if (msg.example_std_vector_of_points.size() == 0) {
      out << "example_std_vector_of_points: []";
    } else {
      out << "example_std_vector_of_points: [";
      size_t pending_items = msg.example_std_vector_of_points.size();
      for (auto item : msg.example_std_vector_of_points) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Example_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }

  // member: example_std_vector_of_points
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.example_std_vector_of_points.size() == 0) {
      out << "example_std_vector_of_points: []\n";
    } else {
      out << "example_std_vector_of_points:\n";
      for (auto item : msg.example_std_vector_of_points) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Example_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace cw1_team_x

namespace rosidl_generator_traits
{

[[deprecated("use cw1_team_x::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const cw1_team_x::srv::Example_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  cw1_team_x::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use cw1_team_x::srv::to_yaml() instead")]]
inline std::string to_yaml(const cw1_team_x::srv::Example_Response & msg)
{
  return cw1_team_x::srv::to_yaml(msg);
}

template<>
inline const char * data_type<cw1_team_x::srv::Example_Response>()
{
  return "cw1_team_x::srv::Example_Response";
}

template<>
inline const char * name<cw1_team_x::srv::Example_Response>()
{
  return "cw1_team_x/srv/Example_Response";
}

template<>
struct has_fixed_size<cw1_team_x::srv::Example_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<cw1_team_x::srv::Example_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<cw1_team_x::srv::Example_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<cw1_team_x::srv::Example>()
{
  return "cw1_team_x::srv::Example";
}

template<>
inline const char * name<cw1_team_x::srv::Example>()
{
  return "cw1_team_x/srv/Example";
}

template<>
struct has_fixed_size<cw1_team_x::srv::Example>
  : std::integral_constant<
    bool,
    has_fixed_size<cw1_team_x::srv::Example_Request>::value &&
    has_fixed_size<cw1_team_x::srv::Example_Response>::value
  >
{
};

template<>
struct has_bounded_size<cw1_team_x::srv::Example>
  : std::integral_constant<
    bool,
    has_bounded_size<cw1_team_x::srv::Example_Request>::value &&
    has_bounded_size<cw1_team_x::srv::Example_Response>::value
  >
{
};

template<>
struct is_service<cw1_team_x::srv::Example>
  : std::true_type
{
};

template<>
struct is_service_request<cw1_team_x::srv::Example_Request>
  : std::true_type
{
};

template<>
struct is_service_response<cw1_team_x::srv::Example_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // CW1_TEAM_X__SRV__DETAIL__EXAMPLE__TRAITS_HPP_
