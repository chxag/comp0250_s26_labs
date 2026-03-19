#!/usr/bin/env python3

"""
This code contains the world spawner. This file accomplishes two main goals:

  1. Create and manage the objects in the gazebo world using the World() class
  2. Spawn and monitor the completion of coursework tasks using the Task() class

The coursework contains three tasks, and each of them are defined here. There
are three classes derived from the Task() base class, Task1(), Task2(), and
Task3().
"""

import time
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, PointStamped, PoseStamped
from cw1_world_spawner.srv import TaskSetup, Task1Service, Task2Service, Task3Service
from coursework_world_spawner_lib.coursework_world_spawner import (
  Model,
  WorldSpawner,
  call_service_sync,
  random_position_in_area,
)

# ----- key coursework task parameters ----- #

# task 1 parameters
T1_BOX_X_LIMS = [0.3, 0.5]            # xrange a box can spawn
T1_BOX_Y_LIMS = [-0.10, 0.11]         # yrange a box can spawn

# task 2 parameters
T2_N_BASKETS = 3                      # number of baskets to spawn
T2_BASKET_COLOUR_UNIQUE = False       # can there only be one basket of each colour
T2_BASKET_NOISE = 50e-3               # possible noise on (x, y) for basket location

# task 3 parameters
T3_N_BOXES = 4                        # number of boxes spawn
T3_N_BASKETS = 3                      # number of baskets to spawn
T3_BASKET_COLOUR_UNIQUE = True        # can there only be one basket of each colour
T3_BOX_X_LIMS = [0.35, 0.66]          # xrange a box can spawn
T3_BOX_Y_LIMS = [-0.10, 0.11]         # yrange a box can spawn
T3_BASKET_NOISE = 50e-3               # possible noise on (x, y) for basket location

# possible goal basket locations (x, y)
BASKET_LOCATIONS = [(0.59, -0.34),
                    (0.59,  0.34),
                    (0.31, -0.34),
                    (0.31,  0.34)]

# possible spawned box colours, do not edit as rgb for models set in sdf file
BOX_COLORS = {'purple': [0.8, 0.1, 0.8],
              'red':    [0.8, 0.1, 0.1],
              'blue':   [0.1, 0.1, 0.8]}

# possible goal basket colours, do not edit as rgb for models set in sdf file
BASKET_COLORS = {'purple': [0.8, 0.1, 0.8],
                 'red':    [0.8, 0.1, 0.1],
                 'blue':   [0.1, 0.1, 0.8]}

GLOBAL_NODE = None
GLOBAL_WORLD_SPAWNER = None
GLOBAL_WORLD = None


def set_context(node, world_spawner, world):
  global GLOBAL_NODE, GLOBAL_WORLD_SPAWNER, GLOBAL_WORLD
  GLOBAL_NODE = node
  GLOBAL_WORLD_SPAWNER = world_spawner
  GLOBAL_WORLD = world


class World(object):

  # lengths in metres for world objects, resultant from sdf model files, don't change
  tile_side_length = 100e-3
  tile_thickness = 20e-3
  basket_side_length = 120e-3
  robot_safety_radius = 280e-3

  def __init__(self, node, world_spawner):
    self.node = node
    self.world_spawner = world_spawner
    self.world_spawner.despawn_all(exceptions="object-all-golf-tiles")
    self.spawn_tiles()

  def spawn_tiles(self):
    model = Model(model_name="all_tiles",
                  instance_name='object-all-golf-tiles',
                  model_type='sdf',
                  position=[0, 0, 0],
                  spawner=self.world_spawner)
    if not self.world_spawner.spawn(model):
      self.node.get_logger().error("Failed to spawn base tiles model 'object-all-golf-tiles'")

  def spawn_goal_basket(self, goal_loc=None, colour=None, name=None):
    if goal_loc is None:
      raise RuntimeError("no goal location given to spawn_goal_basket()")
    if colour is None:
      colour = np.random.choice(list(BASKET_COLORS.keys()))
    if name is None:
      name = "object-goal-basket"

    goal_pt = Point(x=float(goal_loc[0]), y=float(goal_loc[1]), z=float(self.tile_thickness))

    if colour != "empty":
      model = Model(model_name=f"basket_{colour}",
                    instance_name=name,
                    model_type='sdf',
                    position=[goal_pt.x, goal_pt.y, goal_pt.z],
                    spawner=self.world_spawner)
      self.world_spawner.spawn(model)

    return goal_pt

  def reset(self):
    self.world_spawner.despawn_all(exceptions="object-all-golf-tiles")


class Task(object):

  def __init__(self, mode='coursework', validation_scenario=0,
               node=None, world_spawner=None, world=None):
    self.node = node or GLOBAL_NODE
    self.world_spawner = world_spawner or GLOBAL_WORLD_SPAWNER
    self.world = world or GLOBAL_WORLD
    if self.node is None or self.world_spawner is None or self.world is None:
      raise RuntimeError("Task context not initialized")

    self.models = {}
    if mode == 'coursework':
      self.spawn_task_objects()
      self.begin_task()
    else:
      self.spawn_test_objects(validation_scenario)
      self.begin_test(validation_scenario)

  def spawn_box_object(self, name='boxobject1',
                       xlims=None, ylims=None, zlims=None,
                       color='blue'):
    if xlims is None:
      xlims = [0.3, 0.5]
    if ylims is None:
      ylims = [-0.12, 0.12]
    if zlims is None:
      zlims = [0.1, 0.101]

    if color == 'blue':
      model_name = 'box_blue'
    elif color == 'red':
      model_name = 'box_red'
    elif color == 'purple':
      model_name = 'box_purple'
    else:
      model_name = 'box'
    self.node.get_logger().debug(f"Spawning {model_name}")

    model = Model(model_name=model_name,
                  instance_name=name,
                  model_type='sdf',
                  position=random_position_in_area(xlims, ylims, zlims),
                  spawner=self.world_spawner)
    self.world_spawner.spawn(model)
    self.models[name] = model

  def spawn_random_goal_baskets(self, num=1, save_empty=False,
                                unique_colours=True, noise=0.0):
    if num < 1:
      raise RuntimeError("spawn_random_goal_baskets(...) given num < 1")
    if num > len(BASKET_LOCATIONS):
      raise RuntimeError("spawn_random_goal_baskets(...) given num greater than len(BASKET_LOCATIONS)")

    self.basket_points = []
    self.basket_colours = []

    if unique_colours:
      goal_colours = list(BASKET_COLORS.keys())
    else:
      goal_colours = list(BASKET_COLORS.keys()) * 10

    if num < len(goal_colours):
      goal_colours = list(np.random.permutation(goal_colours)[:num])
    elif num > len(goal_colours):
      raise RuntimeError("spawn_random_goal_baskets(...) has num > len(goal_colours) but unique_colours=True")

    if save_empty:
      num_empties = len(BASKET_LOCATIONS) - len(goal_colours)
      if num_empties > 0:
        goal_colours += ["empty"] * num_empties

    goal_colours = np.random.permutation(goal_colours)
    goal_locs = np.random.permutation(BASKET_LOCATIONS)

    for i, colour in enumerate(goal_colours):
      basket_loc = goal_locs[i] + noise * (np.random.random() * 2 - 1)
      basket_point = self.world.spawn_goal_basket(goal_loc=basket_loc, colour=colour,
                                                  name=f"object-goal-basket-{i}")
      self.basket_points.append(basket_point)
      self.basket_colours.append(colour)

  def prepare_for_task_request(self, srv_type, service_name, timeout=60.0):
    self.node.get_logger().debug(f"Attempting to connect to {service_name} Service...")
    client_node = self.world_spawner.client_node
    client = client_node.create_client(srv_type, service_name)
    if not client.wait_for_service(timeout_sec=timeout):
      self.node.get_logger().warn(f"{service_name} Request failed - not advertised")
      return None
    return client

  def reset_task(self):
    self.world_spawner.despawn_all(keyword='object', exceptions="tile")

  def get_position_from_point(self, pt):
    return np.asarray([pt.x, pt.y, pt.z])

  def get_position_from_point_stamped(self, ptst):
    pt = ptst.point
    return np.asarray([pt.x, pt.y, pt.z])

  def get_position_from_pose(self, pose):
    pos_np = self.get_position_from_point(pose.position)
    return pos_np

  def get_euclidean_distance(self, a, b):
    return np.sqrt(np.sum(np.power(a - b, 2)))

  def spawn_task_objects(self):
    raise NotImplementedError

  def begin_task(self):
    raise NotImplementedError

  def spawn_test_objects(self, validation_scenario):
    raise NotImplementedError

  def begin_test(self, validation_scenario):
    raise NotImplementedError


class Task1(Task):
  service_to_request = "/task1_start"

  def __init__(self, mode='coursework', validation_scenario=0,
               node=None, world_spawner=None, world=None):
    if node is None:
      node = GLOBAL_NODE
    node.get_logger().info('================Starting Task1==============')
    super().__init__(mode, validation_scenario, node, world_spawner, world)

  def spawn_task_objects(self):
    self.reset_task()
    self.spawn_random_goal_baskets(num=1)
    self.spawn_box_object(name='boxobject1', xlims=T1_BOX_X_LIMS, ylims=T1_BOX_Y_LIMS)

  def send_task1_request(self, pose):
    client_node = self.world_spawner.client_node
    client = client_node.create_client(Task1Service, self.service_to_request)
    if not client.wait_for_service(timeout_sec=60.0):
      self.node.get_logger().warn("Task1 service not available")
      return False

    pose_st = PoseStamped()
    pose_st.pose = pose
    pose_st.header.frame_id = "panda_link0"
    pose_st.header.stamp = self.node.get_clock().now().to_msg()

    point_st = PointStamped()
    point_st.point = self.basket_points[0]
    point_st.header.frame_id = "panda_link0"
    point_st.header.stamp = self.node.get_clock().now().to_msg()

    req = Task1Service.Request()
    req.object_loc = pose_st
    req.goal_loc = point_st
    resp = call_service_sync(client_node, client, req, timeout_sec=600.0)
    return resp is not None

  def begin_task(self):
    success = self.prepare_for_task_request(Task1Service, self.service_to_request)
    time.sleep(1.0)
    state = self.models['boxobject1'].get_model_state()
    if state is None:
      self.node.get_logger().error("Failed to fetch model state")
      return
    init_pose = state.pose
    if success:
      self.send_task1_request(init_pose)
    else:
      self.node.get_logger().error("Task Request failed - not advertised")


class Task2(Task):
  service_to_request = "/task2_start"

  def __init__(self, mode='coursework', validation_scenario=0,
               node=None, world_spawner=None, world=None):
    if node is None:
      node = GLOBAL_NODE
    node.get_logger().info('================Starting Task2==============')
    super().__init__(mode, validation_scenario, node, world_spawner, world)

  def spawn_task_objects(self):
    self.reset_task()
    self.spawn_random_goal_baskets(num=T2_N_BASKETS, unique_colours=T2_BASKET_COLOUR_UNIQUE,
                                   noise=T2_BASKET_NOISE, save_empty=True)

  def send_task2_request(self):
    client_node = self.world_spawner.client_node
    client = client_node.create_client(Task2Service, self.service_to_request)
    if not client.wait_for_service(timeout_sec=60.0):
      self.node.get_logger().warn("Task2 service not available")
      return None

    basket_locs = []
    for i in range(len(BASKET_LOCATIONS)):
      point_st = PointStamped()
      point_st.point = self.basket_points[i]
      point_st.header.frame_id = "panda_link0"
      point_st.header.stamp = self.node.get_clock().now().to_msg()
      basket_locs.append(point_st)

    req = Task2Service.Request()
    req.basket_locs = basket_locs
    return call_service_sync(client_node, client, req, timeout_sec=600.0)

  def begin_task(self):
    success = self.prepare_for_task_request(Task2Service, self.service_to_request)
    if success:
      self.send_task2_request()
    else:
      self.node.get_logger().error("Task Request failed - not advertised")


class Task3(Task):
  service_to_request = "/task3_start"

  def __init__(self, mode='coursework', validation_scenario=0,
               node=None, world_spawner=None, world=None):
    if node is None:
      node = GLOBAL_NODE
    node.get_logger().info('================Starting Task3==============')
    super().__init__(mode, validation_scenario, node, world_spawner, world)

  def spawn_task_objects(self):
    self.reset_task()
    self.spawn_random_goal_baskets(num=T3_N_BASKETS, unique_colours=T3_BASKET_COLOUR_UNIQUE,
                                   noise=T3_BASKET_NOISE, save_empty=True)

    self.box_locs = []
    ys = np.arange(T3_BOX_Y_LIMS[0], T3_BOX_Y_LIMS[1], step=self.world.tile_side_length)
    xs = np.arange(T3_BOX_X_LIMS[0], T3_BOX_X_LIMS[1], step=self.world.tile_side_length)
    for x_i in xs:
      for y_j in ys:
        if (np.abs(x_i) < self.world.robot_safety_radius and
            np.abs(y_j) < self.world.robot_safety_radius):
          continue
        self.box_locs.append([x_i, y_j])

    np.random.shuffle(self.box_locs)
    self.box_locs = self.box_locs[:T3_N_BOXES]
    for i, loc in enumerate(self.box_locs):
      box_colour = list(BOX_COLORS.keys())[np.random.randint(0, len(BOX_COLORS.keys()))]
      mname = 'boxobject%02d' % i
      self.spawn_box_object(name=mname, color=box_colour,
                            xlims=[loc[0], loc[0]], ylims=[loc[1], loc[1]])

  def send_task3_request(self):
    client_node = self.world_spawner.client_node
    client = client_node.create_client(Task3Service, self.service_to_request)
    if not client.wait_for_service(timeout_sec=60.0):
      self.node.get_logger().warn("Task3 service not available")
      return None

    req = Task3Service.Request()
    return call_service_sync(client_node, client, req, timeout_sec=600.0)

  def begin_task(self):
    success = self.prepare_for_task_request(Task3Service, self.service_to_request)
    if success:
      self.send_task3_request()
    else:
      self.node.get_logger().error("Task Request failed - not advertised")


def handle_task_request(request, response):
  if request.task_index == 1:
    Task1(mode="coursework")
  elif request.task_index == 2:
    Task2(mode="coursework")
  elif request.task_index == 3:
    Task3(mode="coursework")
  else:
    GLOBAL_NODE.get_logger().warn("Unrecognized task requested")
  return response


def main():
  rclpy.init()
  node = Node('coursework1_wrapper')
  world_spawner = WorldSpawner(node)
  world = World(node, world_spawner)
  set_context(node, world_spawner, world)

  node.create_service(TaskSetup, '/task', handle_task_request)
  node.get_logger().info("Ready to initiate task.")
  node.get_logger().info("Use ros2 service call /task cw1_world_spawner/srv/TaskSetup '{task_index: <INDEX>}' to start a task")

  rclpy.spin(node)
  world_spawner.destroy()
  node.destroy_node()
  rclpy.shutdown()


if __name__ == "__main__":
  main()
