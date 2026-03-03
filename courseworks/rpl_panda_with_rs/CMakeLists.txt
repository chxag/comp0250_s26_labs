cmake_minimum_required(VERSION 3.8)
project(rpl_panda_with_rs)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch rviz srdf urdf
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/init_panda_pose.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
