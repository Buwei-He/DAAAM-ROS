cmake_minimum_required(VERSION 3.8)
project(daaam_ros)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(hydra_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  src/daaam_ros/nodes/daaam_node.py
  src/daaam_ros/nodes/rerun_visualizer_node.py
  src/daaam_ros/nodes/dataloader/dataloader_node.py
  src/daaam_ros/utils/colmap_camera_info_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()