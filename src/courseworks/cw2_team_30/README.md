# CW2 Team 30 — ROS 2 / MoveIt Solution README

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

## Acknowledgement
We acknowledge the use of Claude and ChatGPT for brainstorming and code debugging.

---

## Authors and Contribution

| Task / Name | Charvi Agarwal | Yuanzhe Chang | Shelton Darvin |
|---|---:|---:|---:|
| Task 1 | 3–4 hours | 3–4 hours | 3–4 hours |
| Task 2 | 3–4 hours | 3–4 hours | 3–4 hours |
| Task 3 | 4–5 hours | 4–5 hours | 4–5 hours |
| Percentage | 33% | 33% | 33% |

---

## 1) Project Summary

This package contains Team 30's solution for COMP0250 CW2 in ROS 2 Humble.

Main executable:
- `cw2_solution_node` (implemented by `cw2_class.cpp`)

Main launch file:
- `launch/run_solution.launch.py`

The solution handles all three coursework tasks:
- **Task 1**: Pick a nought or cross shape from a known location and place it in a basket, with automatic orientation estimation for reliable grasping.
- **Task 2**: Classify an unknown "mystery" shape as nought or cross by comparing it against two reference shapes using point cloud analysis.
- **Task 3**: Autonomously scan the full workspace, detect and count all shapes (noughts and crosses), determine the most common shape, then pick and place one instance into the basket while avoiding obstacles.

---

## 2) High-Level Design Choices

Our implementation prioritises **robust perception** and **proper failure handling** over speed:

1. **Orientation estimation (Tasks 1 & 3)** uses a KD-Tree edge filter followed by PCL Moment of Inertia / OBB fitting to reliably extract the in-plane yaw of each shape, even for the symmetric nought whose internal covariance is near-isotropic.
2. **Shape classification (Tasks 2 & 3)** uses a centre-density test on the top-surface point cloud: crosses have a solid centre (high point density near centroid) while noughts have a hollow centre (near-zero density near centroid).
3. **Scene understanding (Task 3)** uses a two-phase scan strategy: a coarse discovery scan to register obstacles before motion planning, followed by a fine 36-viewpoint grid scan in order to reduce occlusion and prevent collisions during scanning.
4. **Collision avoidance (Task 3)** adds detected obstacles as MoveIt `CollisionObject` boxes before the pick-and-place phase so that the motion planner can route around them automatically.

---

## 3) Code Structure

- `include/cw2_class.h`  
  Class API, ROS interfaces, member variable declarations, and type aliases.
- `src/cw2_class.cpp`  
  Full implementation of all three tasks and all helper methods.
- `launch/run_solution.launch.py`  
  Launch pipeline for simulator + world spawner + solution node.

Key methods:

| Method | Purpose |
|---|---|
| `computeShapeOrientation` | Estimates in-plane yaw via KD-Tree edge filter + OBB |
| `classifyShapeAtPoint` | Classifies nought vs cross via centre-density test |
| `makeAGraspOffset` | Computes grasp/place pose offset aligned to shape yaw and size |
| `classifyAccumulatedCloud` | Clusters and categorises accumulated point cloud for Task 3 |
| `waitForFreshCloud` | Blocks until N new point cloud frames have arrived |
| `computeAndExecuteCartesianPath` | Straight-line Cartesian motion with fraction guard |

---

## 4) Task-by-Task Methodology

### Task 1 — Pick-and-Place with Orientation Estimation

#### Input
- Object centroid (`object_point`) and basket centroid (`goal_point`) from the service request
- Shape type string (`"nought"` or `"cross"`)

#### Pipeline
1. Move to `"ready"` named pose.
2. Position the camera directly above the object at 0.5 m for a clean top-down view.
3. Wait for a fresh point cloud frame (`waitForFreshCloud`).
4. Estimate the shape's yaw with `computeShapeOrientation` (OBB on edge-filtered top-surface cloud).
5. Combine the estimated yaw with the default gripper orientation (`RPY = π, 0, -π/4 + yaw`).
6. Move to pre-grasp pose above the shape, offset along the shape's wall/arm direction.
7. Open gripper → descend via Cartesian path → close gripper.
8. Lift vertically via Cartesian path.
9. Move above basket with default (axis-aligned) orientation; descend to place.
10. Open gripper → lift → return to `"ready"`.

#### Grasp offset design (`makeAGraspOffset`)
- For **nought**: offset 8 cm along the local Y-axis (towards a wall), rotated by `shape_yaw`, then scaled by the measured object size.
- For **cross**: offset 6 cm along the local X-axis (towards an arm), rotated by `shape_yaw`.
- No offset is applied for basket placement (axis-aligned basket).

---

### Task 2 — Shape Classification and Matching

#### Input
- Two reference object positions (`ref_object_points`) and one mystery object position (`mystery_object_point`)

#### Pipeline
1. Define a fixed top-down viewpoint orientation (`RPY = π, 0, -π/4`).
2. For each object in order (ref1 → ref2 → mystery):
   - Move the arm directly above the object at 0.5 m.
   - Wait up to 900 ms for a new point cloud frame to arrive.
   - Run `classifyShapeAtPoint` to determine `"nought"`, `"cross"`, or `"unknown"`.
3. Match the mystery shape against the two reference classifications using the decision table below.
4. Return to `"ready"`.

#### Matching decision table

| Mystery == Ref1 | Mystery == Ref2 | Result |
|:---:|:---:|:---:|
| ✓ | ✗ | 1 |
| ✗ | ✓ | 2 |
| — | — | 1 (fallback) |
| Ref1 = unknown | Ref2 ≠ unknown | 2 (robustness fallback) |

#### `classifyShapeAtPoint` — centre density test
1. Take a thread-safe snapshot of the latest point cloud.
2. Transform all points from the sensor frame into the arm base frame (`panda_link0`).
3. Crop a cylinder of radius 12 cm and height ±3–12 cm around the query point.
4. Keep only the top 3.5 cm layer (top-surface points only, removes side walls).
5. Compute the 2D centroid `(cx, cy)` of the top-surface points.
6. Count points within a 5 mm radius of the centroid (`center_count`).
7. `center_count < 20` → **nought** (hollow centre); otherwise → **cross** (solid centre).

---

### Task 3 — Full Scene Scan, Detection, Counting, and Pick-and-Place

#### Goal
Detect and count all shapes (noughts and crosses) in the scene without any prior knowledge of their locations, determine the most common shape, and pick one instance and place it in the basket.

#### Phase 1 — Coarse Discovery Scan
- Visit 4 overhead viewpoints at `z = 0.65 m` (high enough to clear any obstacle).
- Accumulate point cloud and run `classifyAccumulatedCloud` to find obstacles.
- Register all detected obstacles as MoveIt `CollisionObject` boxes **before** the fine scan so the planner can avoid them during subsequent motion.

#### Phase 2 — Fine Grid Scan (36 viewpoints)
- Clear the accumulated cloud.
- Sweep the workspace in 8 rows × 5 columns at `z = 0.55 m` with `goToInitial()` between rows to prevent awkward joint configurations.
- The `observeAt` helper: stops scanning → moves to pose → settles for 700 ms → accumulates for 800 ms → stops. This eliminates motion-blur artefacts in the point cloud.

#### Phase 3 — Cluster and Categorise (`classifyAccumulatedCloud`)
1. **Colour filter**: remove green floor pixels (`g > 1.3r && g > 1.3b && g > 60`).
2. **Voxel grid** downsample at 5 mm leaf size.
3. **Radius outlier removal** (radius = 2 cm, min 4 neighbours) to discard residual arm/sensor noise.
4. **Euclidean clustering** (tolerance = 1.2 cm, min 200 pts).
5. Per cluster: compute centroid, bounding box, mean RGB, and point count.
6. **Category classification**:
   - Very dark (`r, g, b < 0.10`) → `"obstacle"`
   - Large, reddish, near a known basket spawn location → `"basket"`
   - Otherwise → `"object"` (candidate nought or cross)

#### Phase 4 — Per-Object Shape Classification
- For each `"object"` cluster, move to a viewpoint 0.5 m above its centroid, wait for 3 fresh frames, and run `classifyShapeAtPoint` (same algorithm as Task 2).

#### Phase 5 — Counting and Response
- Count noughts and crosses; populate `total_num_shapes` and `num_most_common_shape`.
- Tie-break: pick nought if counts are equal.

#### Phase 6 — Pick and Place
- Identify the basket (from fine-scan detection).
- Collect all candidates of the most common shape; sort by distance from arm base (closest = most reachable).
- Execute the same Task 1 pick-and-place sequence (orientation estimate → pre-grasp → descend → grasp → lift → place → release).
- Refine the obstacle list from the fine scan and update the planning scene before grasping.
- Clean up all collision objects from MoveIt after completion.

---

## 5) Build and Run

### Prerequisites
- Ubuntu + ROS 2 Humble
- MoveIt 2, PCL ROS, OctoMap packages
- Workspace root: `comp0250_s26_labs`

### Build
```bash
cd ~/comp0250_s26_labs
source /opt/ros/humble/setup.bash
colcon build --packages-select cw2_team_30
source install/setup.bash
```

### Launch solution
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch cw2_team_30 run_solution.launch.py
```

### Trigger tasks
```bash
# Task 1
ros2 service call /task1_start cw2_world_spawner/srv/Task1Service "{1}"

# Task 2
ros2 service call /task2_start cw2_world_spawner/srv/Task2Service "{2}"

# Task 3
ros2 service call /task3_start cw2_world_spawner/srv/Task3Service "{3}"
```

---

## 6) Tunable Parameters

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `xy_radius` | `classifyShapeAtPoint` / `computeShapeOrientation` | 0.12 m | Horizontal crop radius around query point |
| `top_z_min` offset | both classify functions | 0.035 m | Top-layer thickness kept for classification |
| `center_radius` | `classifyShapeAtPoint` | 0.005 m | Radius used for centre-density vote |
| `search_radius` / `density_threshold` | `computeShapeOrientation` | 8 mm / 15 pts | KD-Tree edge filter parameters |
| `scan_z` | `t3_callback` | 0.55 m | Scan height for the fine grid |
| `preliminary_scan_z` | `t3_callback` | 0.65 m | Scan height for the coarse discovery pass |
| `BASKET_SPAWN_LOCATIONS` | `classifyAccumulatedCloud` | `(-0.41, ±0.36)` | Known basket spawn positions for basket detection |
| `MIN_CLUSTER_SIZE` | `classifyAccumulatedCloud` | 200 pts (EC) / 600 pts (filter) | Noise rejection thresholds |

---

## 7) Limitations and Assumptions

1. Basket spawn locations are hard-coded to the two positions used in the provided simulation worlds. A different world layout would require updating `BASKET_SPAWN_LOCATIONS`.
2. The centre-density threshold (`center_count < 20`) is empirically tuned to the simulation's point cloud density. Real sensors or different heights may require re-tuning.
3. No grasp success verification: the pipeline assumes the gripper has successfully picked the object if no exception is thrown. A failed grasp (e.g. due to slight mis-alignment) will result in placing an empty gripper.


---

## License

This project is licensed under the MIT License — see the top-level `LICENSE` file for details.
