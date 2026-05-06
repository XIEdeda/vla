# 双臂机器人数据采集系统 — 仅 C++（ament_cmake）
# 运行: ros2 run guodi websocket_server_node

cmake_minimum_required(VERSION 3.8)
project(guodi)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sc_ros2 REQUIRED)
find_package(topic_pub REQUIRED)

# C++ 第三方库
find_package(nlohmann_json REQUIRED)
find_package(OpenCV REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(Boost REQUIRED COMPONENTS system)

# 添加cpp-httplib支持
include_directories(include/guodi/external)

include_directories(include ${HDF5_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# ========================================
# 2. C++ 构建目标
# ========================================

# 2.1 C++ 共享库: 核心逻辑（配置、ROS2组件、数据采集器）
add_library(guodi_lib
  src/utils_config.cpp
  src/ros2_components.cpp
  src/data_collector.cpp
)
ament_target_dependencies(guodi_lib
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  sc_ros2
  topic_pub
)
target_link_libraries(guodi_lib
  nlohmann_json::nlohmann_json
  ${OpenCV_LIBS}
  ${HDF5_LIBRARIES}
  ${Boost_LIBRARIES}
)

# 2.2 C++ 可执行文件: WebSocket服务器节点（内含数据处理器单例）
add_executable(websocket_server_node src/websocket_server_node.cpp)
target_link_libraries(websocket_server_node guodi_lib)

install(TARGETS
  websocket_server_node
  DESTINATION lib/${PROJECT_NAME}
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
