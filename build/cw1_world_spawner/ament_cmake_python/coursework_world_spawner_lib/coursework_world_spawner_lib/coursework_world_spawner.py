import os
from glob import glob
import math
import re
import subprocess
import time
from types import SimpleNamespace
import numpy as np

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, GetEntityState, GetModelList


def quaternion_from_euler(roll, pitch, yaw):
  cr = math.cos(roll * 0.5)
  sr = math.sin(roll * 0.5)
  cp = math.cos(pitch * 0.5)
  sp = math.sin(pitch * 0.5)
  cy = math.cos(yaw * 0.5)
  sy = math.sin(yaw * 0.5)
  return [
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
    cr * cp * cy + sr * sp * sy,
  ]


def vectors_to_ros_pose(pos, q):
  pose = Pose()
  pose.position.x = float(pos[0])
  pose.position.y = float(pos[1])
  pose.position.z = float(pos[2])
  pose.orientation.x = float(q[0])
  pose.orientation.y = float(q[1])
  pose.orientation.z = float(q[2])
  pose.orientation.w = float(q[3])
  return pose


def random_position_in_area(xlims, ylims, zlims):
  x = np.random.uniform(xlims[0], xlims[1])
  y = np.random.uniform(ylims[0], ylims[1])
  z = np.random.uniform(zlims[0], zlims[1])
  return np.asarray([x, y, z])


def random_orientation(roll_lims, pitch_lims, yaw_lims):
  roll = np.random.uniform(roll_lims[0], roll_lims[1])
  pitch = np.random.uniform(pitch_lims[0], pitch_lims[1])
  yaw = np.random.uniform(yaw_lims[0], yaw_lims[1])
  q = quaternion_from_euler(roll, pitch, yaw)
  return q


def random_pose_in_area(xlims, ylims, zlims,
                        roll_lims, pitch_lims, yaw_lims):
  xyz = random_position_in_area(xlims, ylims, zlims)
  q = random_orientation(roll_lims, pitch_lims, yaw_lims)
  pose = vectors_to_ros_pose(xyz, q)
  return pose


def get_random_upright_pose(xlims, ylims, zlims):
  xyz = random_position_in_area(xlims, ylims, zlims)
  pose = vectors_to_ros_pose(xyz, [0, 0, 0, 1])
  return pose


def call_service_sync(node, client, request, timeout_sec=120.0):
  future = client.call_async(request)
  rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
  if not future.done():
    node.get_logger().warn("Service call timed out")
    return None
  return future.result()


class Model(object):
  def __init__(self, model_name, instance_name, model_type,
               position, orientation=None, scale=None, spawner=None):
    if orientation is None:
      orientation = [0, 0, 0, 1]
    self.spawner = spawner
    self.mdict = self.create_model_dict(model_name, instance_name,
                                        model_type, position,
                                        orientation, scale)

  def create_model_dict(self, model_name, instance_name, model_type,
                        position, orientation=None, scale=None):
    if orientation is None:
      orientation = [0, 0, 0, 1]
    if model_type == 'sdf':
      pose = vectors_to_ros_pose(position, orientation)
      mdict = dict(mtype=model_type, model_name=model_name,
                   instance_name=instance_name, pose=pose)
    elif model_type == 'primitive':
      mdict = None
    else:
      mdict = None
      if self.spawner is not None:
        self.spawner.node.get_logger().error('Wrong model type. Should be [model|primitive]')
    return mdict

  def get_model_state(self, relname="world"):
    if self.spawner is None:
      raise RuntimeError("Model has no spawner reference")
    resp = self.spawner.get_model_state_by_name(self.mdict["instance_name"], relname)
    if resp is None:
      return None
    if hasattr(resp, "state"):
      return resp.state
    return resp

  def despawn(self):
    if self.spawner is None:
      raise RuntimeError("Model has no spawner reference")
    self.spawner.despawn_by_name(self.mdict["instance_name"])


class WorldSpawner(object):
  def __init__(self, node, extra_model_dirs=None):
    if extra_model_dirs is None:
      extra_model_dirs = []
    self.node = node
    helper_name = f"{node.get_name()}_spawner_client_{os.getpid()}"
    self.client_node = rclpy.create_node(helper_name)
    # Keep service clients on a dedicated node so synchronous calls work even
    # when the main node is currently handling a service callback.
    self.spawn_client = self.client_node.create_client(SpawnEntity, '/spawn_entity')
    self.delete_client = self.client_node.create_client(DeleteEntity, '/delete_entity')
    self.get_state_client = self.client_node.create_client(GetEntityState, '/gazebo/get_entity_state')
    self.model_list_client = self.client_node.create_client(GetModelList, '/get_model_list')
    self.init_model_dirs(extra_model_dirs)
    self.init_model_names()

  def init_model_dirs(self, extra_model_dirs):
    self.models_dirs = []
    pkg_share = get_package_share_directory("cw1_world_spawner")
    pkg_models_path = os.path.join(pkg_share, "models")
    self.models_dirs += [pkg_models_path]
    self.models_dirs += list(extra_model_dirs)

    self.node.get_logger().info("Model libraries used by cw1_world_spawner:")
    for mdir in self.models_dirs:
      self.node.get_logger().info(mdir)

  def init_model_names(self):
    self.model_names = set()
    for mdir in self.models_dirs:
      if not os.path.isdir(mdir):
        continue
      subfolders = os.listdir(mdir)
      self.model_names |= set(subfolders)

  def find_model(self, model_name):
    if model_name not in self.model_names:
      self.node.get_logger().warn(f"Requested model_name {model_name} is not found")
      return None

    for mdir in self.models_dirs:
      glob_query = os.path.join(mdir, model_name, "*.sdf")
      model_paths = glob(glob_query)
      if len(model_paths) > 0:
        return model_paths[0]

    self.node.get_logger().warn(f"No .sdf found for model {model_name}")
    return None

  def spawn(self, model):
    model_dict = model.mdict
    if model_dict['mtype'] == 'sdf':
      model_name = model_dict['model_name']
      instance_name = model_dict['instance_name']
      pose = model_dict['pose']
      return self.spawn_model(model_name, instance_name, pose)
    if model_dict['mtype'] == 'primitive':
      return False
    self.node.get_logger().error(f"Unsupported model spawn type {model_dict['mtype']}")
    return False

  def spawn_model(self, model_name, instance_name, pose):
    path_to_model = self.find_model(model_name)
    if path_to_model is None:
      return False

    # Match ROS 1 behavior (rospy.wait_for_service): block until Gazebo spawn
    # service is up, so startup model spawns (tiles/baseboard) are not dropped.
    wait_step_sec = 1.0
    waited_sec = 0.0
    max_wait_sec = 60.0
    while rclpy.ok() and not self.spawn_client.wait_for_service(timeout_sec=wait_step_sec):
      waited_sec += wait_step_sec
      if int(waited_sec) % 5 == 0:
        self.node.get_logger().warn("Waiting for /spawn_entity service...")
      if waited_sec >= max_wait_sec:
        self.node.get_logger().warn("/spawn_entity service not available")
        return False

    if not rclpy.ok():
      return False

    self.node.get_logger().debug(f"Spawning model of type {model_name}")
    with open(path_to_model, 'r') as handle:
      sdff = handle.read()

    request = SpawnEntity.Request()
    request.name = instance_name
    request.xml = sdff
    request.robot_namespace = '/WorldSpawner'
    request.initial_pose = pose
    request.reference_frame = 'world'

    # Retry transient failures so a slow Gazebo startup does not permanently
    # lose world geometry (for example object-all-golf-tiles).
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
      resp = call_service_sync(self.client_node, self.spawn_client, request)
      if resp is None:
        self.node.get_logger().warn(
          f"Spawn call failed for '{instance_name}' (attempt {attempt}/{max_attempts})")
      elif getattr(resp, "success", True):
        return True
      else:
        status_message = getattr(resp, "status_message", "unknown error")
        if "exist" in status_message.lower():
          return True
        self.node.get_logger().warn(
          f"Failed to spawn '{instance_name}' (attempt {attempt}/{max_attempts}): {status_message}")

      if attempt < max_attempts:
        time.sleep(0.5)

    return False

  def get_model_state_by_name(self, name, relname="world"):
    if self.get_state_client.wait_for_service(timeout_sec=0.2):
      request = GetEntityState.Request()
      request.name = name
      request.reference_frame = relname
      return call_service_sync(self.client_node, self.get_state_client, request)

    # Gazebo Classic on ROS 2 Humble does not always expose get_entity_state.
    # Fall back to the `gz model -i` CLI so coursework validation can still
    # query object poses without changing user-facing behavior.
    return self.get_model_state_via_gz(name)

  def get_model_state_via_gz(self, name):
    try:
      output = subprocess.check_output(
        ['gz', 'model', '-m', name, '-i'],
        stderr=subprocess.STDOUT,
        text=True,
        timeout=3.0,
      )
    except Exception:
      self.node.get_logger().warn(
        f"Unable to query model state for '{name}' via /gazebo/get_entity_state or gz model")
      return None

    match = re.search(
      r'pose\s*\{\s*position\s*\{\s*x:\s*([^\s]+)\s*y:\s*([^\s]+)\s*z:\s*([^\s]+)\s*\}\s*'
      r'orientation\s*\{\s*x:\s*([^\s]+)\s*y:\s*([^\s]+)\s*z:\s*([^\s]+)\s*w:\s*([^\s]+)',
      output,
      re.S,
    )
    if match is None:
      self.node.get_logger().warn(f"Unable to parse pose from gz model output for '{name}'")
      return None

    pose = Pose()
    pose.position.x = float(match.group(1))
    pose.position.y = float(match.group(2))
    pose.position.z = float(match.group(3))
    pose.orientation.x = float(match.group(4))
    pose.orientation.y = float(match.group(5))
    pose.orientation.z = float(match.group(6))
    pose.orientation.w = float(match.group(7))
    return SimpleNamespace(pose=pose)

  def despawn_by_name(self, instance_name):
    if not self.delete_client.wait_for_service(timeout_sec=10.0):
      self.node.get_logger().warn("/delete_entity service not available")
      return None

    request = DeleteEntity.Request()
    request.name = instance_name
    return call_service_sync(self.client_node, self.delete_client, request)

  def get_world_properties(self):
    if not self.model_list_client.wait_for_service(timeout_sec=10.0):
      self.node.get_logger().warn("/get_model_list service not available")
      return None

    request = GetModelList.Request()
    return call_service_sync(self.client_node, self.model_list_client, request)

  def despawn_all(self, keyword='object', exceptions='exception'):
    props = self.get_world_properties()
    if props is None:
      return
    for name in props.model_names:
      if keyword in name and exceptions not in name:
        self.despawn_by_name(name)

  def destroy(self):
    self.client_node.destroy_node()
