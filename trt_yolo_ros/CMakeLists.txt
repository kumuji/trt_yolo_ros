cmake_minimum_required(VERSION 2.8.3)
project(trt_yolo_ros)

find_package(catkin REQUIRED
  COMPONENTS
    cv_bridge
    darknet_ros_msgs
)

catkin_package()

# catkin_install_python(
#   PROGRAMS
#     scripts/node.py
#   DESTINATION
#     ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

catkin_python_setup()
