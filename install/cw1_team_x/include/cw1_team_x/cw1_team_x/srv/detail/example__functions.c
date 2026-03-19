// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from cw1_team_x:srv/Example.idl
// generated code does not contain a copyright notice
#include "cw1_team_x/srv/detail/example__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `example_string`
#include "rosidl_runtime_c/string_functions.h"
// Member `example_pose`
// Member `example_std_vector_of_poses`
#include "geometry_msgs/msg/detail/pose__functions.h"

bool
cw1_team_x__srv__Example_Request__init(cw1_team_x__srv__Example_Request * msg)
{
  if (!msg) {
    return false;
  }
  // example_float
  // example_int
  // example_string
  if (!rosidl_runtime_c__String__init(&msg->example_string)) {
    cw1_team_x__srv__Example_Request__fini(msg);
    return false;
  }
  // example_pose
  if (!geometry_msgs__msg__Pose__init(&msg->example_pose)) {
    cw1_team_x__srv__Example_Request__fini(msg);
    return false;
  }
  // example_std_vector_of_poses
  if (!geometry_msgs__msg__Pose__Sequence__init(&msg->example_std_vector_of_poses, 0)) {
    cw1_team_x__srv__Example_Request__fini(msg);
    return false;
  }
  return true;
}

void
cw1_team_x__srv__Example_Request__fini(cw1_team_x__srv__Example_Request * msg)
{
  if (!msg) {
    return;
  }
  // example_float
  // example_int
  // example_string
  rosidl_runtime_c__String__fini(&msg->example_string);
  // example_pose
  geometry_msgs__msg__Pose__fini(&msg->example_pose);
  // example_std_vector_of_poses
  geometry_msgs__msg__Pose__Sequence__fini(&msg->example_std_vector_of_poses);
}

bool
cw1_team_x__srv__Example_Request__are_equal(const cw1_team_x__srv__Example_Request * lhs, const cw1_team_x__srv__Example_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // example_float
  if (lhs->example_float != rhs->example_float) {
    return false;
  }
  // example_int
  if (lhs->example_int != rhs->example_int) {
    return false;
  }
  // example_string
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->example_string), &(rhs->example_string)))
  {
    return false;
  }
  // example_pose
  if (!geometry_msgs__msg__Pose__are_equal(
      &(lhs->example_pose), &(rhs->example_pose)))
  {
    return false;
  }
  // example_std_vector_of_poses
  if (!geometry_msgs__msg__Pose__Sequence__are_equal(
      &(lhs->example_std_vector_of_poses), &(rhs->example_std_vector_of_poses)))
  {
    return false;
  }
  return true;
}

bool
cw1_team_x__srv__Example_Request__copy(
  const cw1_team_x__srv__Example_Request * input,
  cw1_team_x__srv__Example_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // example_float
  output->example_float = input->example_float;
  // example_int
  output->example_int = input->example_int;
  // example_string
  if (!rosidl_runtime_c__String__copy(
      &(input->example_string), &(output->example_string)))
  {
    return false;
  }
  // example_pose
  if (!geometry_msgs__msg__Pose__copy(
      &(input->example_pose), &(output->example_pose)))
  {
    return false;
  }
  // example_std_vector_of_poses
  if (!geometry_msgs__msg__Pose__Sequence__copy(
      &(input->example_std_vector_of_poses), &(output->example_std_vector_of_poses)))
  {
    return false;
  }
  return true;
}

cw1_team_x__srv__Example_Request *
cw1_team_x__srv__Example_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cw1_team_x__srv__Example_Request * msg = (cw1_team_x__srv__Example_Request *)allocator.allocate(sizeof(cw1_team_x__srv__Example_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(cw1_team_x__srv__Example_Request));
  bool success = cw1_team_x__srv__Example_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
cw1_team_x__srv__Example_Request__destroy(cw1_team_x__srv__Example_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    cw1_team_x__srv__Example_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
cw1_team_x__srv__Example_Request__Sequence__init(cw1_team_x__srv__Example_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cw1_team_x__srv__Example_Request * data = NULL;

  if (size) {
    data = (cw1_team_x__srv__Example_Request *)allocator.zero_allocate(size, sizeof(cw1_team_x__srv__Example_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = cw1_team_x__srv__Example_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        cw1_team_x__srv__Example_Request__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
cw1_team_x__srv__Example_Request__Sequence__fini(cw1_team_x__srv__Example_Request__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      cw1_team_x__srv__Example_Request__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

cw1_team_x__srv__Example_Request__Sequence *
cw1_team_x__srv__Example_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cw1_team_x__srv__Example_Request__Sequence * array = (cw1_team_x__srv__Example_Request__Sequence *)allocator.allocate(sizeof(cw1_team_x__srv__Example_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = cw1_team_x__srv__Example_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
cw1_team_x__srv__Example_Request__Sequence__destroy(cw1_team_x__srv__Example_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    cw1_team_x__srv__Example_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
cw1_team_x__srv__Example_Request__Sequence__are_equal(const cw1_team_x__srv__Example_Request__Sequence * lhs, const cw1_team_x__srv__Example_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!cw1_team_x__srv__Example_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
cw1_team_x__srv__Example_Request__Sequence__copy(
  const cw1_team_x__srv__Example_Request__Sequence * input,
  cw1_team_x__srv__Example_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(cw1_team_x__srv__Example_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    cw1_team_x__srv__Example_Request * data =
      (cw1_team_x__srv__Example_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!cw1_team_x__srv__Example_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          cw1_team_x__srv__Example_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!cw1_team_x__srv__Example_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `example_std_vector_of_points`
#include "geometry_msgs/msg/detail/point__functions.h"

bool
cw1_team_x__srv__Example_Response__init(cw1_team_x__srv__Example_Response * msg)
{
  if (!msg) {
    return false;
  }
  // success
  // example_std_vector_of_points
  if (!geometry_msgs__msg__Point__Sequence__init(&msg->example_std_vector_of_points, 0)) {
    cw1_team_x__srv__Example_Response__fini(msg);
    return false;
  }
  return true;
}

void
cw1_team_x__srv__Example_Response__fini(cw1_team_x__srv__Example_Response * msg)
{
  if (!msg) {
    return;
  }
  // success
  // example_std_vector_of_points
  geometry_msgs__msg__Point__Sequence__fini(&msg->example_std_vector_of_points);
}

bool
cw1_team_x__srv__Example_Response__are_equal(const cw1_team_x__srv__Example_Response * lhs, const cw1_team_x__srv__Example_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  // example_std_vector_of_points
  if (!geometry_msgs__msg__Point__Sequence__are_equal(
      &(lhs->example_std_vector_of_points), &(rhs->example_std_vector_of_points)))
  {
    return false;
  }
  return true;
}

bool
cw1_team_x__srv__Example_Response__copy(
  const cw1_team_x__srv__Example_Response * input,
  cw1_team_x__srv__Example_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  // example_std_vector_of_points
  if (!geometry_msgs__msg__Point__Sequence__copy(
      &(input->example_std_vector_of_points), &(output->example_std_vector_of_points)))
  {
    return false;
  }
  return true;
}

cw1_team_x__srv__Example_Response *
cw1_team_x__srv__Example_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cw1_team_x__srv__Example_Response * msg = (cw1_team_x__srv__Example_Response *)allocator.allocate(sizeof(cw1_team_x__srv__Example_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(cw1_team_x__srv__Example_Response));
  bool success = cw1_team_x__srv__Example_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
cw1_team_x__srv__Example_Response__destroy(cw1_team_x__srv__Example_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    cw1_team_x__srv__Example_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
cw1_team_x__srv__Example_Response__Sequence__init(cw1_team_x__srv__Example_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cw1_team_x__srv__Example_Response * data = NULL;

  if (size) {
    data = (cw1_team_x__srv__Example_Response *)allocator.zero_allocate(size, sizeof(cw1_team_x__srv__Example_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = cw1_team_x__srv__Example_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        cw1_team_x__srv__Example_Response__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
cw1_team_x__srv__Example_Response__Sequence__fini(cw1_team_x__srv__Example_Response__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      cw1_team_x__srv__Example_Response__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

cw1_team_x__srv__Example_Response__Sequence *
cw1_team_x__srv__Example_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cw1_team_x__srv__Example_Response__Sequence * array = (cw1_team_x__srv__Example_Response__Sequence *)allocator.allocate(sizeof(cw1_team_x__srv__Example_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = cw1_team_x__srv__Example_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
cw1_team_x__srv__Example_Response__Sequence__destroy(cw1_team_x__srv__Example_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    cw1_team_x__srv__Example_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
cw1_team_x__srv__Example_Response__Sequence__are_equal(const cw1_team_x__srv__Example_Response__Sequence * lhs, const cw1_team_x__srv__Example_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!cw1_team_x__srv__Example_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
cw1_team_x__srv__Example_Response__Sequence__copy(
  const cw1_team_x__srv__Example_Response__Sequence * input,
  cw1_team_x__srv__Example_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(cw1_team_x__srv__Example_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    cw1_team_x__srv__Example_Response * data =
      (cw1_team_x__srv__Example_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!cw1_team_x__srv__Example_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          cw1_team_x__srv__Example_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!cw1_team_x__srv__Example_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
