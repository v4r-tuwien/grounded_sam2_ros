cmake_minimum_required(VERSION 3.0.2)
project(grounded_sam_2)

find_package(catkin REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES grounded_sam_2
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/grounded_sam_2_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

