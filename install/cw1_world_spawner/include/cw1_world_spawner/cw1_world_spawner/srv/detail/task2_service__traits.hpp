// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cw1_world_spawner:srv/Task2Service.idl
// generated code does not contain a copyright notice

#ifndef CW1_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__TRAITS_HPP_
#define CW1_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "cw1_world_spawner/srv/detail/task2_service__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'basket_locs'
#include "geometry_msgs/msg/detail/point_stamped__traits.hpp"

namespace cw1_world_spawner
{

namespace srv
{

inline void to_flow_style_yaml(
  const Task2Service_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: basket_locs
  {
    if (msg.basket_locs.size() == 0) {
      out << "basket_locs: []";
    } else {
      out << "basket_locs: [";
      size_t pending_items = msg.basket_locs.size();
      for (auto item : msg.basket_locs) {
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
  const Task2Service_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: basket_locs
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.basket_locs.size() == 0) {
      out << "basket_locs: []\n";
    } else {
      out << "basket_locs:\n";
      for (auto item : msg.basket_locs) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Task2Service_Request & msg, bool use_flow_style = false)
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

}  // namespace cw1_world_spawner

namespace rosidl_generator_traits
{

[[deprecated("use cw1_world_spawner::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const cw1_world_spawner::srv::Task2Service_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  cw1_world_spawner::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use cw1_world_spawner::srv::to_yaml() instead")]]
inline std::string to_yaml(const cw1_world_spawner::srv::Task2Service_Request & msg)
{
  return cw1_world_spawner::srv::to_yaml(msg);
}

template<>
inline const char * data_type<cw1_world_spawner::srv::Task2Service_Request>()
{
  return "cw1_world_spawner::srv::Task2Service_Request";
}

template<>
inline const char * name<cw1_world_spawner::srv::Task2Service_Request>()
{
  return "cw1_world_spawner/srv/Task2Service_Request";
}

template<>
struct has_fixed_size<cw1_world_spawner::srv::Task2Service_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<cw1_world_spawner::srv::Task2Service_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<cw1_world_spawner::srv::Task2Service_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace cw1_world_spawner
{

namespace srv
{

inline void to_flow_style_yaml(
  const Task2Service_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: basket_colours
  {
    if (msg.basket_colours.size() == 0) {
      out << "basket_colours: []";
    } else {
      out << "basket_colours: [";
      size_t pending_items = msg.basket_colours.size();
      for (auto item : msg.basket_colours) {
        rosidl_generator_traits::value_to_yaml(item, out);
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
  const Task2Service_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: basket_colours
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.basket_colours.size() == 0) {
      out << "basket_colours: []\n";
    } else {
      out << "basket_colours:\n";
      for (auto item : msg.basket_colours) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Task2Service_Response & msg, bool use_flow_style = false)
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

}  // namespace cw1_world_spawner

namespace rosidl_generator_traits
{

[[deprecated("use cw1_world_spawner::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const cw1_world_spawner::srv::Task2Service_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  cw1_world_spawner::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use cw1_world_spawner::srv::to_yaml() instead")]]
inline std::string to_yaml(const cw1_world_spawner::srv::Task2Service_Response & msg)
{
  return cw1_world_spawner::srv::to_yaml(msg);
}

template<>
inline const char * data_type<cw1_world_spawner::srv::Task2Service_Response>()
{
  return "cw1_world_spawner::srv::Task2Service_Response";
}

template<>
inline const char * name<cw1_world_spawner::srv::Task2Service_Response>()
{
  return "cw1_world_spawner/srv/Task2Service_Response";
}

template<>
struct has_fixed_size<cw1_world_spawner::srv::Task2Service_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<cw1_world_spawner::srv::Task2Service_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<cw1_world_spawner::srv::Task2Service_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<cw1_world_spawner::srv::Task2Service>()
{
  return "cw1_world_spawner::srv::Task2Service";
}

template<>
inline const char * name<cw1_world_spawner::srv::Task2Service>()
{
  return "cw1_world_spawner/srv/Task2Service";
}

template<>
struct has_fixed_size<cw1_world_spawner::srv::Task2Service>
  : std::integral_constant<
    bool,
    has_fixed_size<cw1_world_spawner::srv::Task2Service_Request>::value &&
    has_fixed_size<cw1_world_spawner::srv::Task2Service_Response>::value
  >
{
};

template<>
struct has_bounded_size<cw1_world_spawner::srv::Task2Service>
  : std::integral_constant<
    bool,
    has_bounded_size<cw1_world_spawner::srv::Task2Service_Request>::value &&
    has_bounded_size<cw1_world_spawner::srv::Task2Service_Response>::value
  >
{
};

template<>
struct is_service<cw1_world_spawner::srv::Task2Service>
  : std::true_type
{
};

template<>
struct is_service_request<cw1_world_spawner::srv::Task2Service_Request>
  : std::true_type
{
};

template<>
struct is_service_response<cw1_world_spawner::srv::Task2Service_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // CW1_WORLD_SPAWNER__SRV__DETAIL__TASK2_SERVICE__TRAITS_HPP_
