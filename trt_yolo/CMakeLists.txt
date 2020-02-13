cmake_minimum_required(VERSION 2.8.3)
project(trt_yolo)

find_package(catkin REQUIRED
  COMPONENTS
)

catkin_package()

# catkin_install_python(
#   PROGRAMS
#     scripts/node.py
#   DESTINATION
#     ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

catkin_python_setup()

