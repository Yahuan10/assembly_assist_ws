cmake_minimum_required(VERSION 3.5)
project(gesture_processor_node)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})
ament_package()
