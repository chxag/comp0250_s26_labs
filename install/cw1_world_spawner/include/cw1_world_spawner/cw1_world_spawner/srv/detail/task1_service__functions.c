// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from cw1_world_spawner:srv/Task1Service.idl
// generated code does not contain a copyright notice
#include "cw1_world_spawner/srv/detail/task1_service__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `object_loc`
#include "geometry_msgs/msg/detail/pose_stamped__functions.h"
// Member `goal_loc`
#include "geometry_msgs/msg/detail/point_stamped__functions.h"

bool
cw1_world_spawner__srv__Task1Service_Request__init(cw1_world_spawner__srv__Task1Service_Request * msg)
{
  if (!msg) {
    return false;
  }
  // object_loc
  if (!geometry_msgs__msg__PoseStamped__init(&msg->object_loc)) {
    cw1_world_spawner__srv__Task1Service_Request__fini(msg);
    return false;
  }
  // goal_loc
  if (!geometry_msgs__msg__PointStamped__init(&msg->goal_loc)) {
    cw1_world_spawner__srv__Task1Service_Request__fini(msg);
    return false;
  }
  return true;
}

void
cw1_world_spawner__srv__Task1Service_Request__fini(cw1_world_spawner__srv__Task1Service_Request * msg)
{
  if (!msg) {
    return;
  }
  // object_loc
  geometry_msgs__msg__PoseStamped__fini(&msg->object_loc);
  // goal_loc
  geometry_msgs__msg__PointStamped__fini(&msg->goal_loc);
}

bool
cw1_world_spawner__srv__Task1Service_Request__are_equal(const cw1_world_spawner__srv__Task1Service_Request * lhs, const cw1_world_spawner__srv__Task1Service_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // object_loc
  if (!geometry_msgs__msg__PoseStamped__are_equal(
      &(lhs->object_loc), &(rhs->object_loc)))
  {
    return false;
  }
  // goal_loc
  if (!geometry_msgs__msg__PointStamped__are_equal(
      &(lhs->goal_loc), &(rhs->goal_loc)))
  {
    return false;
  }
  return true;
}

bool
cw1_world_spawner__srv__Task1Service_Request__copy(
  const cw1_world_spawner__srv__Task1Service_Request * input,
  cw1_world_spawner__srv__Task1Service_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // object_loc
  if (!geometry_msgs__msg__PoseStamped__copy(
      &(input->object_loc), &(output->object_loc)))
  {
    return false;
  }
  // goal_loc
  if (!geometry_msgs__msg__PointStamped__copy(
      &(input->goal_loc), &(output->goal_loc)))
  {
    return false;
  }
  return true;
}

cw1_world_spawner__srv__Task1Service_Request *
cw1_world_spawner__srv__Task1Service_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cw1_world_spawner__srv__Task1Service_Request * msg = (cw1_world_spawner__srv__Task1Service_Request *)allocator.allocate(sizeof(cw1_world_spawner__srv__Task1Service_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(cw1_world_spawner__srv__Task1Service_Request));
  bool success = cw1_world_spawner__srv__Task1Service_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
cw1_world_spawner__srv__Task1Service_Request__destroy(cw1_world_spawner__srv__Task1Service_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    cw1_world_spawner__srv__Task1Service_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
cw1_world_spawner__srv__Task1Service_Request__Sequence__init(cw1_world_spawner__srv__Task1Service_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cw1_world_spawner__srv__Task1Service_Request * data = NULL;

  if (size) {
    data = (cw1_world_spawner__srv__Task1Service_Request *)allocator.zero_allocate(size, sizeof(cw1_world_spawner__srv__Task1Service_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = cw1_world_spawner__srv__Task1Service_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        cw1_world_spawner__srv__Task1Service_Request__fini(&data[i - 1]);
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
cw1_world_spawner__srv__Task1Service_Request__Sequence__fini(cw1_world_spawner__srv__Task1Service_Request__Sequence * array)
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
      cw1_world_spawner__srv__Task1Service_Request__fini(&array->data[i]);
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

cw1_world_spawner__srv__Task1Service_Request__Sequence *
cw1_world_spawner__srv__Task1Service_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cw1_world_spawner__srv__Task1Service_Request__Sequence * array = (cw1_world_spawner__srv__Task1Service_Request__Sequence *)allocator.allocate(sizeof(cw1_world_spawner__srv__Task1Service_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = cw1_world_spawner__srv__Task1Service_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
cw1_world_spawner__srv__Task1Service_Request__Sequence__destroy(cw1_world_spawner__srv__Task1Service_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    cw1_world_spawner__srv__Task1Service_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
cw1_world_spawner__srv__Task1Service_Request__Sequence__are_equal(const cw1_world_spawner__srv__Task1Service_Request__Sequence * lhs, const cw1_world_spawner__srv__Task1Service_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!cw1_world_spawner__srv__Task1Service_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
cw1_world_spawner__srv__Task1Service_Request__Sequence__copy(
  const cw1_world_spawner__srv__Task1Service_Request__Sequence * input,
  cw1_world_spawner__srv__Task1Service_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(cw1_world_spawner__srv__Task1Service_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    cw1_world_spawner__srv__Task1Service_Request * data =
      (cw1_world_spawner__srv__Task1Service_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!cw1_world_spawner__srv__Task1Service_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          cw1_world_spawner__srv__Task1Service_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!cw1_world_spawner__srv__Task1Service_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
cw1_world_spawner__srv__Task1Service_Response__init(cw1_world_spawner__srv__Task1Service_Response * msg)
{
  if (!msg) {
    return false;
  }
  // structure_needs_at_least_one_member
  return true;
}

void
cw1_world_spawner__srv__Task1Service_Response__fini(cw1_world_spawner__srv__Task1Service_Response * msg)
{
  if (!msg) {
    return;
  }
  // structure_needs_at_least_one_member
}

bool
cw1_world_spawner__srv__Task1Service_Response__are_equal(const cw1_world_spawner__srv__Task1Service_Response * lhs, const cw1_world_spawner__srv__Task1Service_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // structure_needs_at_least_one_member
  if (lhs->structure_needs_at_least_one_member != rhs->structure_needs_at_least_one_member) {
    return false;
  }
  return true;
}

bool
cw1_world_spawner__srv__Task1Service_Response__copy(
  const cw1_world_spawner__srv__Task1Service_Response * input,
  cw1_world_spawner__srv__Task1Service_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // structure_needs_at_least_one_member
  output->structure_needs_at_least_one_member = input->structure_needs_at_least_one_member;
  return true;
}

cw1_world_spawner__srv__Task1Service_Response *
cw1_world_spawner__srv__Task1Service_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cw1_world_spawner__srv__Task1Service_Response * msg = (cw1_world_spawner__srv__Task1Service_Response *)allocator.allocate(sizeof(cw1_world_spawner__srv__Task1Service_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(cw1_world_spawner__srv__Task1Service_Response));
  bool success = cw1_world_spawner__srv__Task1Service_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
cw1_world_spawner__srv__Task1Service_Response__destroy(cw1_world_spawner__srv__Task1Service_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    cw1_world_spawner__srv__Task1Service_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
cw1_world_spawner__srv__Task1Service_Response__Sequence__init(cw1_world_spawner__srv__Task1Service_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cw1_world_spawner__srv__Task1Service_Response * data = NULL;

  if (size) {
    data = (cw1_world_spawner__srv__Task1Service_Response *)allocator.zero_allocate(size, sizeof(cw1_world_spawner__srv__Task1Service_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = cw1_world_spawner__srv__Task1Service_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        cw1_world_spawner__srv__Task1Service_Response__fini(&data[i - 1]);
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
cw1_world_spawner__srv__Task1Service_Response__Sequence__fini(cw1_world_spawner__srv__Task1Service_Response__Sequence * array)
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
      cw1_world_spawner__srv__Task1Service_Response__fini(&array->data[i]);
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

cw1_world_spawner__srv__Task1Service_Response__Sequence *
cw1_world_spawner__srv__Task1Service_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cw1_world_spawner__srv__Task1Service_Response__Sequence * array = (cw1_world_spawner__srv__Task1Service_Response__Sequence *)allocator.allocate(sizeof(cw1_world_spawner__srv__Task1Service_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = cw1_world_spawner__srv__Task1Service_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
cw1_world_spawner__srv__Task1Service_Response__Sequence__destroy(cw1_world_spawner__srv__Task1Service_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    cw1_world_spawner__srv__Task1Service_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
cw1_world_spawner__srv__Task1Service_Response__Sequence__are_equal(const cw1_world_spawner__srv__Task1Service_Response__Sequence * lhs, const cw1_world_spawner__srv__Task1Service_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!cw1_world_spawner__srv__Task1Service_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
cw1_world_spawner__srv__Task1Service_Response__Sequence__copy(
  const cw1_world_spawner__srv__Task1Service_Response__Sequence * input,
  cw1_world_spawner__srv__Task1Service_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(cw1_world_spawner__srv__Task1Service_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    cw1_world_spawner__srv__Task1Service_Response * data =
      (cw1_world_spawner__srv__Task1Service_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!cw1_world_spawner__srv__Task1Service_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          cw1_world_spawner__srv__Task1Service_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!cw1_world_spawner__srv__Task1Service_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
