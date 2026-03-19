#!/usr/bin/env python3
import math

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from controller_manager_msgs.srv import ListControllers
from std_srvs.srv import Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def _normalize_topic(name: str) -> str:
    if not name:
        return name
    return name if name.startswith('/') else f'/{name}'


class InitPandaPose(Node):
    def __init__(self) -> None:
        super().__init__('init_panda_pose')
        self.declare_parameter('controller_manager', '/controller_manager')
        self.declare_parameter('arm_controller', 'panda_arm_controller')
        self.declare_parameter('hand_controller', 'panda_hand_controller')
        self.declare_parameter(
            'arm_joints',
            [
                'panda_joint1',
                'panda_joint2',
                'panda_joint3',
                'panda_joint4',
                'panda_joint5',
                'panda_joint6',
                'panda_joint7',
            ],
        )
        self.declare_parameter(
            'arm_positions',
            [
                0.0,
                -0.785,
                0.0,
                -2.356,
                0.0,
                1.571,
                0.785,
            ],
        )
        self.declare_parameter('hand_joints', ['panda_finger_joint1', 'panda_finger_joint2'])
        self.declare_parameter('hand_positions', [0.035, 0.035])
        self.declare_parameter('time_from_start', 2.0)
        self.declare_parameter('unpause_physics_service', '/unpause_physics')
        self.declare_parameter('max_wait_seconds', 30.0)

        controller_manager = self.get_parameter('controller_manager').value
        self._arm_controller = self.get_parameter('arm_controller').value
        self._hand_controller = self.get_parameter('hand_controller').value
        self._arm_joints = list(self.get_parameter('arm_joints').value)
        self._arm_positions = list(self.get_parameter('arm_positions').value)
        self._hand_joints = list(self.get_parameter('hand_joints').value)
        self._hand_positions = list(self.get_parameter('hand_positions').value)
        self._time_from_start = float(self.get_parameter('time_from_start').value)
        self._max_wait_seconds = float(self.get_parameter('max_wait_seconds').value)
        self._unpause_service = _normalize_topic(
            self.get_parameter('unpause_physics_service').value
        )
        self._start_time = self.get_clock().now()

        service_name = _normalize_topic(controller_manager).rstrip('/') + '/list_controllers'
        self._client = self.create_client(ListControllers, service_name)
        self._arm_pub = self.create_publisher(
            JointTrajectory,
            _normalize_topic(self._arm_controller) + '/joint_trajectory',
            10,
        )
        self._hand_pub = self.create_publisher(
            JointTrajectory,
            _normalize_topic(self._hand_controller) + '/joint_trajectory',
            10,
        )
        self._unpause_client = self.create_client(Empty, self._unpause_service)
        self._in_flight = False
        self._published = False
        self._timer = self.create_timer(0.5, self._tick)

    def _tick(self) -> None:
        if self._published or self._in_flight:
            return
        elapsed = (self.get_clock().now() - self._start_time).nanoseconds / 1e9
        if elapsed > self._max_wait_seconds:
            self.get_logger().warn(
                f'Controllers did not become ready in {self._max_wait_seconds:.1f}s; exiting.'
            )
            self.create_timer(0.1, self._shutdown)
            self._published = True
            return
        if not self._client.service_is_ready():
            self.get_logger().debug('Waiting for controller_manager service...')
            return
        self._in_flight = True
        future = self._client.call_async(ListControllers.Request())
        future.add_done_callback(self._handle_list)

    def _handle_list(self, future) -> None:
        self._in_flight = False
        if future.exception() is not None:
            self.get_logger().warn(f'ListControllers failed: {future.exception()}')
            return
        response = future.result()
        if response is None:
            return
        active = {c.name for c in response.controller if c.state == 'active'}
        if self._arm_controller not in active or self._hand_controller not in active:
            self.get_logger().info('Waiting for controllers to become active...')
            return
        self._publish_initial_pose()
        self._request_unpause()
        self._published = True
        self.get_logger().info('Published initial panda pose.')
        self.destroy_timer(self._timer)
        self._timer = None
        self.create_timer(0.5, self._shutdown)

    def _publish_initial_pose(self) -> None:
        duration_msg = Duration(seconds=self._time_from_start).to_msg()

        arm_msg = JointTrajectory()
        arm_msg.joint_names = self._arm_joints
        arm_point = JointTrajectoryPoint()
        arm_point.positions = self._arm_positions
        arm_point.velocities = [0.0] * len(self._arm_positions)
        arm_point.time_from_start = duration_msg
        arm_msg.points = [arm_point]
        self._arm_pub.publish(arm_msg)

        hand_msg = JointTrajectory()
        hand_msg.joint_names = self._hand_joints
        hand_point = JointTrajectoryPoint()
        hand_point.positions = self._hand_positions
        hand_point.velocities = [0.0] * len(self._hand_positions)
        hand_point.time_from_start = duration_msg
        hand_msg.points = [hand_point]
        self._hand_pub.publish(hand_msg)

    def _request_unpause(self) -> None:
        if not self._unpause_client.service_is_ready():
            self.get_logger().warn(
                'Unpause service is not ready; physics may remain paused.'
            )
            return
        future = self._unpause_client.call_async(Empty.Request())
        future.add_done_callback(self._handle_unpause)

    def _handle_unpause(self, future) -> None:
        if future.exception() is not None:
            self.get_logger().warn(f'Failed to unpause physics: {future.exception()}')
            return
        self.get_logger().info('Requested Gazebo unpause after initial pose command.')

    def _shutdown(self) -> None:
        rclpy.shutdown()


def main() -> None:
    rclpy.init()
    node = InitPandaPose()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
