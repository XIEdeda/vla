cmake_minimum_required(VERSION 3.8)
project(topic_pub)  # 替换为你的功能包名

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclpy REQUIRED)  # 原代码依赖rclpy

# 2. 声明自定义消息（指定msg文件路径）
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArmData.msg"  # 你的自定义消息文件
  "msg/MotorErrors.msg"
  DEPENDENCIES std_msgs  # 依赖std_msgs（因用了Header）
)

# 3. 安装Python节点（确保原代码能被找到）
add_executable(camera_topic_pub src/camera_topic_pub.cpp)
ament_target_dependencies(camera_topic_pub rclcpp sensor_msgs cv_bridge OpenCV)

install(TARGETS
  camera_topic_pub
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/arm_topic_pub.py  # 替换为你的Python代码路径（如scripts/mock_arms.py）
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()