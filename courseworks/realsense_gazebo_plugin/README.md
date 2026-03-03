# Intel RealSense Gazebo ROS plugin (ROS 2)

This package uses the PAL Robotics ROS 2 branch of the RealSense Gazebo plugin (D435).
In this workspace we keep the `/r200/...` topic namespace for coursework compatibility by
configuring the plugin prefix.

## Build

```bash
colcon build --packages-select realsense_gazebo_plugin
```

## Usage

Include the macro and set the xacro args:

- `use_gazebo` (default: `true`)
- `sensor_prefix` (default: `r200`)
- `link_prefix` (default: empty)

Example (see `urdf/d435_simulation.xacro` for a minimal demo).

## Point cloud

The D435 macro disables the plugin point cloud by default to avoid duplicate publishers.
The coursework launch (`rpl_panda_with_rs`) uses `depth_image_proc` to publish
`/<sensor_prefix>/camera/depth_registered/points`.
