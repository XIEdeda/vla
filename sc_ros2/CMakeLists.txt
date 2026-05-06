cmake_minimum_required(VERSION 3.8)
project(sc_ros2)

# 默认使用 C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -------------------------------
# 查找 ROS 2 依赖
# -------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# -------------------------------
# 查找其它依赖
# -------------------------------
find_package(Eigen3 REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Threads REQUIRED)
find_library(MODBUS_LIB modbus REQUIRED)
find_package(yaml-cpp REQUIRED)

# -------------------------------
# 包含目录
# -------------------------------
include_directories(
  include
  include/can
  ${Boost_INCLUDE_DIRS}
  ${OPENSSL_INCLUDE_DIR}
  /usr/include/eigen3
  ${rclcpp_INCLUDE_DIRS}
)

# 库目录
link_directories(
  ${PROJECT_SOURCE_DIR}/lib
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# 设置库输出目录
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# -------------------------------
# 编译库
# -------------------------------

# 在 target_link_libraries 之前添加：
set(QLROBOT_CONTROL_STATIC ${PROJECT_SOURCE_DIR}/lib/libqlrobot_control.a)
set(WEB_SOCKET_SERVER_LIB ${PROJECT_SOURCE_DIR}/lib/libweb_socket_server.a)
# 然后修改 QLROBOT 变量：
set(QLROBOT 
  Eigen3::Eigen 
  ${QLROBOT_CONTROL_STATIC}  
  ${WEB_SOCKET_SERVER_LIB} 
  ${rclcpp_LIBRARIES}
)

# -------------------------------
# 生成 service
# -------------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetJointAngles.srv"
  "srv/MulSetJointAngles.srv"
  "msg/JointSequence.msg"
  "msg/ArmData.msg"
  "msg/MotorErrors.msg"
  DEPENDENCIES std_msgs
)

# -------------------------------
# 编译可执行文件
# -------------------------------

# ROS 2 节点
# add_executable(remote_control_data src/remote_control_data.cpp)
# ament_target_dependencies(remote_control_data
#  rclcpp
#  std_msgs
# )

# #添加ROS 2接口的包含目录
# target_include_directories(remote_control_data PUBLIC
#   ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp
# )

# #链接ROS 2接口库
# target_link_libraries(remote_control_data
#  ${Boost_LIBRARIES} 
#  ${OPENSSL_LIBRARIES} 
#  ${QLROBOT} 
#  yaml-cpp  
#  ${MODBUS_LIB}
# ${PROJECT_SOURCE_DIR}/lib/libKalmanFilter.so
#  ${PROJECT_NAME}__rosidl_typesupport_cpp
#  ${PROJECT_NAME}__rosidl_generator_cpp
# )

# ROS 2 节点 - velocity version
# add_executable(remote_control_data_velocity src/remote_control_data_velocity.cpp)
# ament_target_dependencies(remote_control_data_velocity
#   rclcpp
#   std_msgs
# )

# 添加ROS 2接口的包含目录
# target_include_directories(remote_control_data_velocity PUBLIC
#   ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp
# )

# 链接ROS 2接口库
# target_link_libraries(remote_control_data_velocity
#   ${Boost_LIBRARIES} 
#   ${OPENSSL_LIBRARIES} 
#   ${QLROBOT} 
#   yaml-cpp  
#   ${MODBUS_LIB}
#   ${PROJECT_SOURCE_DIR}/lib/libKalmanFilter.so
#   ${PROJECT_NAME}__rosidl_typesupport_cpp
#   ${PROJECT_NAME}__rosidl_generator_cpp
# )

add_executable(teaching_mode src/teaching_mode.cpp)
target_link_libraries(teaching_mode ${QLROBOT})

add_executable(stand src/stand.cpp)
target_link_libraries(stand ${QLROBOT})

add_executable(squat src/squat.cpp)
target_link_libraries(squat ${QLROBOT})

add_executable(head_control src/head_control.cpp)
target_link_libraries(head_control ${QLROBOT})

add_executable(songlingtest src/songlingtest.cpp)
target_link_libraries(songlingtest ${QLROBOT})

add_executable(vla_infer src/vla_infer.cpp)
ament_target_dependencies(vla_infer
  rclcpp
  std_msgs
)
rosidl_target_interfaces(vla_infer
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)
target_link_libraries(vla_infer
  ${Boost_LIBRARIES} 
  ${OPENSSL_LIBRARIES} 
  ${QLROBOT} 
  yaml-cpp  
  ${MODBUS_LIB}
  ${PROJECT_SOURCE_DIR}/lib/libKalmanFilter.so
  ${PROJECT_NAME}__rosidl_typesupport_cpp
  ${PROJECT_NAME}__rosidl_generator_cpp
)

add_executable(robot_control src/robot_control.cpp)
ament_target_dependencies(robot_control
  rclcpp
  std_msgs
)
rosidl_target_interfaces(robot_control
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)
target_link_libraries(robot_control
  ${Boost_LIBRARIES} 
  ${OPENSSL_LIBRARIES} 
  ${QLROBOT} 
  yaml-cpp  
  ${MODBUS_LIB}
  ${PROJECT_SOURCE_DIR}/lib/libKalmanFilter.so
  ${PROJECT_NAME}__rosidl_typesupport_cpp
  ${PROJECT_NAME}__rosidl_generator_cpp
)

# -------------------------------
# 安装库和可执行文件
# -------------------------------

install(TARGETS
  # web_socket_server
  # KalmanFilter
  # clamp
  vla_infer
  robot_control

  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS 
# remote_control_data
# remote_control_data_velocity
teaching_mode
stand
squat
head_control
songlingtest
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(FILES ${PROJECT_SOURCE_DIR}/lib/libKalmanFilter.so DESTINATION lib)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

# -------------------------------
# 测试与 lint
# -------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
