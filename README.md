# comp0250_s26_labs
Labs and coursework code for COMP0250

Currently contains the completed mtc_tutorial from [the MoveIt! website](https://moveit.picknik.ai/humble/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html) and the pcl_tutorial

More content will be added each week.

In your home directory (locally or on the GPU servers) type
```
git clone https://github.com/surgical-vision/comp0250_s26_labs.git
```

Then each week do the following to pull code that has been added (in the master branch):

```
cd ~/comp0250_s26_labs
git pull origin master
```

If you are building on the GPU servers, the following should work (otherwise use your local installation of ws_moveit2)
The lab itself should be compiled separately from the MoveIt! tutorials

```
source /cs/student/msc/rai_containers/ws_moveit2/install/setup.bash
cd ~/comp0250_s26_labs
colcon build --mixin release
```

Further instructions to follow in each lab



