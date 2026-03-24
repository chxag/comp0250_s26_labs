# CW1 Team 30 — ROS 2 / MoveIt Solution README

## Acknowledgement:
We acknowledge the use of Claude and ChatGPT for brainstorming and code debugging.

## 1) Project Summary
This package contains Team 30's solution for COMP0250 CW1 in ROS 2 Humble.

Main executable:
- `cw1_solution_node` (implemented by `cw1_class.cpp` + `cw1_node.cpp`)

Main launch file:
- `launch/run_solution.launch.py`

The solution handles all three coursework tasks:
- **Task 1**: Pick one cube and place it in a basket from known coordinates.
- **Task 2**: Classify basket colors from point clouds.
- **Task 3**: Full-scene scan, detect objects (cube/basket + color), then pick-and-place cubes into matching baskets.

---

## 2) High-Level Design Choices
Our implementation prioritizes **determinism** and **stability** over fully generic planning:

1. **Task 1/Task 3 manipulation** uses analytical joint-space hover + Cartesian vertical descent.
2. **Task 2/Task 3 perception** uses PCL point cloud filtering + color-distance classification.
3. **Task 3 scene understanding** uses Euclidean clustering and a point-density heuristic to separate cubes vs baskets.

Why this approach:
- Reduces random behavior and orientation jitter from unconstrained pose planning.
- Keeps the gripper consistently downward for robust top-down grasp attempts.
- Gives predictable behavior in repeated simulation runs.

---

## 3) Code Structure
- `include/cw1_class.h`  
  Class API, ROS interfaces, helper declarations.
- `src/cw1_node.cpp`  
  Node bootstrap and multithreaded executor.
- `src/cw1_class.cpp`  
  Full implementation of Task 1/2/3 and helper methods.
- `launch/run_solution.launch.py`  
  Launch pipeline for simulator + world spawner + solution node.

---

## 4) Task-by-Task Methodology

## Task 1 — Deterministic Pick-and-Place (Known Coordinates)
### Input
- Cube pose from request
- Basket location from request

### Core motion pipeline
1. Open gripper.
2. Move above cube using `moveToLiftXY(x, y)`.
3. Descend vertically using `moveToGraspZ(x, y, z)`.
4. Close gripper.
5. Lift to safe height.
6. Move above basket.
7. Open gripper to release.

### Technical details
- `moveToLiftXY` uses analytical equations for selected joints to produce a stable downward gripper posture.
- `moveToGraspZ` uses `computeCartesianPath` for a pure Z-axis descent.
- A fixed forward offset (`0.03 m`) is applied during descent to compensate practical alignment bias in simulation.

---

## Task 2 — Basket Color Classification
### Input
- Candidate basket positions

### Pipeline
1. Move the camera above each basket candidate.
2. Wait for fresh point cloud.
3. Transform basket center from world frame to camera frame.
4. Crop local cloud around basket center:
   - XY radius crop
   - Z-band crop (to reject floor/background)
5. Remove dark/noisy points.
6. Compute mean RGB.
7. Classify color by nearest reference in RGB space (`red`, `blue`, `purple`).

### Robustness rules
- Minimum point threshold before classifying.
- Maximum allowed color distance; otherwise return `none`.

---

## Task 3 — Full Scene Scan, Detection, and Sorting
### Goal
Detect all cubes/baskets and place each cube into the basket of the same color.

### Pipeline
1. **Multi-view scanning** from 5 overhead poses to reduce occlusion blind spots.
2. Filter colored points and transform them into world frame.
3. Merge all scans into one cloud.
4. Run Euclidean clustering in world frame.
5. For each cluster:
   - Compute centroid and mean RGB.
   - Classify color by nearest reference.
6. Separate `basket` vs `cube` by point count threshold:
   - Large cluster => basket
   - Smaller cluster => cube
7. For each cube, find matching-color basket and execute Task 1-style pick-and-place.

### Key heuristic
- `BASKET_PT_THRESHOLD = 8000` (empirical, simulation-specific).

---

## 5) Build and Run

## Prerequisites
- Ubuntu + ROS 2 Humble
- MoveIt 2 + PCL ROS packages
- Workspace root: `comp0250_s26_labs`

## Build
From workspace root:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

## Launch solution

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export PATH=/usr/bin:$PATH
export RMW_FASTRTPS_USE_SHM=0
ros2 launch cw1_team_30 run_solution.launch.py \
  use_gazebo_gui:=true use_rviz:=true \
  enable_realsense:=true enable_camera_processing:=false \
  control_mode:=effort
```

## Trigger tasks

```bash
# Task 1
ros2 service call /task cw1_world_spawner/srv/TaskSetup "{task_index: 1}"

# Task 1 scenario 2 only:
ros2 service call /task cw1_world_spawner/srv/TaskSetup "{task_index: 111}"

# Task 2
ros2 service call /task cw1_world_spawner/srv/TaskSetup "{task_index: 2}"

# Task 3
ros2 service call /task cw1_world_spawner/srv/TaskSetup "{task_index: 3}"
```

---

## 6) Tunable Parameters (Launch)
Important runtime parameters from `run_solution.launch.py`:
- `pick_offset_z`
- `task3_pick_offset_z`
- `place_offset_z`
- `grasp_approach_offset_z`
- `post_grasp_lift_z`
- `gripper_grasp_width`
- `cartesian_eef_step`
- `cartesian_jump_threshold`
- `cartesian_min_fraction`

These can be overridden at launch time for tuning grasp stability.

---

## 7) Limitations and Assumptions
1. Several thresholds are empirical and tuned for provided simulation worlds.
2. The fixed forward grasp offset improves consistency in our setup but is not universally optimal.
3. No online visual servoing during final descent; grasp precision depends on detection quality + kinematic consistency.

---

## 8) Suggested Future Improvements
1. Learn or calibrate grasp XY offsets online (per run).
2. Add local pre-grasp micro-scan before final descent.
3. Replace fixed basket/cube point threshold with adaptive model-based classification.
4. Add grasp success checks and retry policy.

---

## 9) Team Notes
This repository version reflects a deterministic engineering strategy optimized for reproducibility in the coursework simulation environment.
