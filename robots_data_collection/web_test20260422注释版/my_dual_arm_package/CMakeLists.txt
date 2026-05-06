# =============================================
# 双臂机器人数据采集系统 - 混合构建配置
# 同时支持 Python 脚本运行和 C++ 可执行文件运行
# Python运行: ros2 run my_dual_arm_package data_collector.py / websocket_server.py
# C++运行:   ros2 run my_dual_arm_package websocket_server_node
# =============================================

cmake_minimum_required(VERSION 3.8)
project(my_dual_arm_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ========================================
# 1. 查找依赖包
# ========================================

# 构建系统
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)  # Python脚本安装支持

# C++ ROS2 依赖
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sc_ros2 REQUIRED)
# find_package(topic_pub REQUIRED)

# C++ 第三方库
find_package(nlohmann_json REQUIRED)
find_package(OpenCV REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(Boost REQUIRED COMPONENTS system)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(OpenSSL REQUIRED)

# 添加cpp-httplib支持
include_directories(include/my_dual_arm_package/external)

include_directories(include ${HDF5_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# ========================================
# 2. C++ 构建目标
# ========================================

# 2.1 C++ 共享库: 核心逻辑（配置、ROS2组件、数据采集器）
add_library(my_dual_arm_lib
  src/utils_config.cpp
  src/ros2_components.cpp
  src/data_collector.cpp
)
ament_target_dependencies(my_dual_arm_lib
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  sc_ros2
  rosbag2_cpp
  rosbag2_storage
  # topic_pub
)
target_link_libraries(my_dual_arm_lib
  nlohmann_json::nlohmann_json
  ${OpenCV_LIBS}
  ${HDF5_LIBRARIES}
  ${Boost_LIBRARIES}
)

# 2.2 C++ 可执行文件: WebSocket服务器节点（内含数据处理器单例）
add_executable(websocket_server_node src/websocket_server_node.cpp)
target_link_libraries(websocket_server_node
  my_dual_arm_lib
  nlohmann_json::nlohmann_json
  OpenSSL::SSL
  OpenSSL::Crypto
)

# 2.3 安装 C++ 可执行文件到 lib/<pkg> 目录
install(TARGETS
  websocket_server_node
  DESTINATION lib/${PROJECT_NAME}
)

# ========================================
# 3. Python 脚本安装（安装到 lib/<pkg> 目录，ros2 run 可调用）
# ========================================
install(PROGRAMS
  scripts/utils_config.py
  scripts/ros2_components.py
  scripts/data_collector.py
  scripts/websocket_server.py
  DESTINATION lib/${PROJECT_NAME}
)

# ========================================
# 4. 安装launch文件
# ========================================
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# ========================================
# 5. 测试配置
# ========================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
