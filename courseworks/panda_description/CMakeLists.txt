cmake_minimum_required(VERSION 3.8)
project(panda_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  meshes
  rviz
  sdf
  srdf
  urdf
  xacro
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "*.launch" EXCLUDE
)

ament_package()
