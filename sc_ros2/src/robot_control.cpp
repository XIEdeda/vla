#include "rclcpp/rclcpp.hpp"
#include "sc_ros2/srv/mul_set_joint_angles.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "sc_ros2/msg/arm_data.hpp"

#include <iostream>
#include <vector>
#include <pthread.h>
#include <unistd.h>
#include <chrono>
#include <iomanip>
#include <thread>
#include <array>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "qlrobotarm_canfd.h"

#include <fstream>
#include <sstream>
#include <Eigen/Dense>

#define PI 3.14159265358979323846
#define DEGREE_TO_RAD(x) ((x)*PI / 180.0)
#define RAD_TO_DEGREE(x) ((x)*180.0 / PI)
#define JOINT_SPEED 0.6
#define JOINT_INIT_SPEED 0.1
#define LEFT_GRIPPER_FORCE 0.6
#define RIGHT_GRIPPER_FORCE 0.4

// 全局变量
// std::vector<double> zero_angle = {0, 0, 0, 0, 0, 0, 0};
std::vector<double> zero_angle_r = {-24.459 /180*M_PI , 54.611/180*M_PI , 13.531/180*M_PI , 111.269/180*M_PI, -55.829/180*M_PI , 12.532/180*M_PI , 1.082/180*M_PI};
std::vector<double> zero_angle_l = {23.628 /180*M_PI , -55.034/180*M_PI , -13.216/180*M_PI , -112.641/180*M_PI, 54.613/180*M_PI , -12.182/180*M_PI , -3.7888/180*M_PI};

// 保护所有 qlrobotarm_canfd 底层 API 的全局互斥锁
// 原因：
//   - 底层通过 CANFD 串行通信，SDK 非线程安全；
//   - MultiThreadedExecutor 下同一 topic 的回调、以及不同 topic 的回调
//     （left/right/dual_arm/gripper/publisher 线程）都可能并发进入；
//   - 未加锁时多线程调用会在 CAN 字节流层面交错，造成读到脏数据或执行半成品指令，
//     表现为关节抖动、夹爪跳动等。
static std::mutex g_hw_mutex;

constexpr int ARM_JOINT_DIM = 7;
constexpr int ARM_MSG_MIN_SIZE = 9;  // 前7维关节 + 第9维夹爪
constexpr double JOINT_MIN_RAD = -PI;
constexpr double JOINT_MAX_RAD = PI;
constexpr double GRIPPER_MIN = 0.0;
constexpr double GRIPPER_MAX = 70.0;


// 从 ArmData 中提取夹爪值，优先使用第9维(data[8])，兼容仅8维时使用data[7]
bool extract_gripper_command(const std::vector<double>& data, double& gripper_value) {
    if (data.size() >= ARM_MSG_MIN_SIZE) {
        gripper_value = data[8];
        return true;
    }
    if (data.size() > ARM_JOINT_DIM) {
        gripper_value = data[7];
        return true;
    }
    return false;
}

bool validate_joint_angles(const std::vector<double>& joint_angles, const char* topic_name) {
    if (joint_angles.size() != ARM_JOINT_DIM) {
        RCLCPP_ERROR(
            rclcpp::get_logger("RobotControl"),
            "[%s] Invalid joint dimension: %zu, expected exactly %d",
            topic_name, joint_angles.size(), ARM_JOINT_DIM);
        return false;
    }
    for (size_t i = 0; i < joint_angles.size(); ++i) {
        if (joint_angles[i] < JOINT_MIN_RAD || joint_angles[i] > JOINT_MAX_RAD) {
            RCLCPP_ERROR(
                rclcpp::get_logger("RobotControl"),
                "[%s] Joint[%zu] out of range: %.6f, expected in [%.6f, %.6f]",
                topic_name, i, joint_angles[i], JOINT_MIN_RAD, JOINT_MAX_RAD);
            return false;
        }
    }
    return true;
}

bool validate_gripper_value(double gripper_value, const char* topic_name) {
    if (gripper_value < GRIPPER_MIN || gripper_value > GRIPPER_MAX) {
        RCLCPP_ERROR(
            rclcpp::get_logger("RobotControl"),
            "[%s] Gripper value out of range: %.6f, expected in [%.6f, %.6f]",
            topic_name, gripper_value, GRIPPER_MIN, GRIPPER_MAX);
        return false;
    }
    return true;
}
/*
订阅器回调函数，用于下发关节/夹爪指令
*/
void handle_left_data_topic(const sc_ros2::msg::ArmData::SharedPtr msg) {
    if (msg->data.size() < ARM_JOINT_DIM) {
        RCLCPP_ERROR(rclcpp::get_logger("RobotControl"), "Invalid /left_data size: %zu, expected at least %d", msg->data.size(), ARM_JOINT_DIM);
        return;
    }

    std::vector<double> left_raw_joint_angles(msg->data.begin(), msg->data.begin() + ARM_JOINT_DIM);
    if (!validate_joint_angles(left_raw_joint_angles, "/left_data")) {
        return;
    }

    double left_gripper_value = 0.0;
    bool has_gripper = extract_gripper_command(msg->data, left_gripper_value);
    if (has_gripper && !validate_gripper_value(left_gripper_value, "/left_data")) {
        return;
    }

    {
        std::lock_guard<std::mutex> lk(g_hw_mutex);
        moveJ_l(left_raw_joint_angles, JOINT_SPEED);
        if (has_gripper) {
            controlLeftClamp(left_gripper_value, LEFT_GRIPPER_FORCE);
        }
    }
    RCLCPP_INFO(rclcpp::get_logger("RobotControl"), "Controlling left joint%s", has_gripper ? " + gripper" : "");
    if (!has_gripper) {
        RCLCPP_WARN(rclcpp::get_logger("RobotControl"), "/left_data has no gripper value (expected data[8], fallback data[7])");
    }
}

void handle_right_data_topic(const sc_ros2::msg::ArmData::SharedPtr msg) {
    if (msg->data.size() < ARM_JOINT_DIM) {
        RCLCPP_ERROR(rclcpp::get_logger("RobotControl"), "Invalid /right_data size: %zu, expected at least %d", msg->data.size(), ARM_JOINT_DIM);
        return;
    }

    std::vector<double> right_raw_joint_angles(msg->data.begin(), msg->data.begin() + ARM_JOINT_DIM);
    if (!validate_joint_angles(right_raw_joint_angles, "/right_data")) {
        return;
    }

    double right_gripper_value = 0.0;
    bool has_gripper = extract_gripper_command(msg->data, right_gripper_value);
    if (has_gripper && !validate_gripper_value(right_gripper_value, "/right_data")) {
        return;
    }

    {
        std::lock_guard<std::mutex> lk(g_hw_mutex);
        moveJ_r(right_raw_joint_angles, JOINT_SPEED);
        if (has_gripper) {
            controlRightClamp(right_gripper_value, RIGHT_GRIPPER_FORCE);
        }
    }
    RCLCPP_INFO(rclcpp::get_logger("RobotControl"), "Controlling right joint%s", has_gripper ? " + gripper" : "");
    if (!has_gripper) {
        RCLCPP_WARN(rclcpp::get_logger("RobotControl"), "/right_data has no gripper value (expected data[8], fallback data[7])");
    }
}

void handle_set_dual_arm_topic(const sc_ros2::msg::ArmData::SharedPtr msg) {
    if (msg->data.size() != 2*ARM_JOINT_DIM+2) {
        RCLCPP_ERROR(rclcpp::get_logger("RobotControl"), "[/set_dual_arm] Invalid data size: %zu, expected %d", msg->data.size(), 2*ARM_JOINT_DIM+2);
        return;
    }
    std::vector<double> left_joint_angles(msg->data.begin(), msg->data.begin() + ARM_JOINT_DIM);
    float left_gripper_value = msg->data[ARM_JOINT_DIM];
    std::vector<double> right_joint_angles(msg->data.begin() + ARM_JOINT_DIM+1, msg->data.begin() + 2*ARM_JOINT_DIM+1);
    float right_gripper_value = msg->data[2*ARM_JOINT_DIM+1];
    if (!validate_joint_angles(left_joint_angles, "/set_dual_arm") || !validate_gripper_value(left_gripper_value, "/set_dual_arm") || !validate_joint_angles(right_joint_angles, "/set_dual_arm") || !validate_gripper_value(right_gripper_value, "/set_dual_arm")) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_hw_mutex);
        moveJ_l(left_joint_angles, JOINT_SPEED);
        moveJ_r(right_joint_angles, JOINT_SPEED);
        controlLeftClamp(left_gripper_value, LEFT_GRIPPER_FORCE);
        controlRightClamp(right_gripper_value, RIGHT_GRIPPER_FORCE);
    }
    RCLCPP_INFO(rclcpp::get_logger("RobotControl"), "Controlling dual arm");
}

void handle_set_left_arm_topic(const sc_ros2::msg::ArmData::SharedPtr msg) {
    if (msg->data.size() != ARM_JOINT_DIM+1) {
        RCLCPP_ERROR(rclcpp::get_logger("RobotControl"), "[/set_left_arm] Invalid data size: %zu, expected %d", msg->data.size(), ARM_JOINT_DIM+1);
        return;
    }
    std::vector<double> left_joint_angles(msg->data.begin(), msg->data.begin() + ARM_JOINT_DIM);
    float left_gripper_value = msg->data[ARM_JOINT_DIM];
    if (!validate_joint_angles(left_joint_angles, "/set_left_arm") || !validate_gripper_value(left_gripper_value, "/set_left_arm")) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_hw_mutex);
        moveJ_l(left_joint_angles, JOINT_SPEED);
        controlLeftClamp(left_gripper_value, LEFT_GRIPPER_FORCE);
    }
    RCLCPP_INFO(rclcpp::get_logger("RobotControl"), "Controlling left arm");
}

void handle_set_right_arm_topic(const sc_ros2::msg::ArmData::SharedPtr msg) {
    if (msg->data.size() != ARM_JOINT_DIM+1) {
        RCLCPP_ERROR(rclcpp::get_logger("RobotControl"), "[/set_right_arm] Invalid data size: %zu, expected %d", msg->data.size(), ARM_JOINT_DIM+1);
        return;
    }
    std::vector<double> right_joint_angles(msg->data.begin(), msg->data.begin() + ARM_JOINT_DIM);
    float right_gripper_value = msg->data[ARM_JOINT_DIM];
    if (!validate_joint_angles(right_joint_angles, "/set_right_arm") || !validate_gripper_value(right_gripper_value, "/set_right_arm")) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_hw_mutex);
        moveJ_r(right_joint_angles, JOINT_SPEED);
        controlRightClamp(right_gripper_value, RIGHT_GRIPPER_FORCE);
    }
    RCLCPP_INFO(rclcpp::get_logger("RobotControl"), "Controlling right arm");
}

void handle_set_left_joint_topic(const sc_ros2::msg::ArmData::SharedPtr msg) {
    if (msg->data.size() != ARM_JOINT_DIM) {
        RCLCPP_ERROR(rclcpp::get_logger("RobotControl"), "[/set_left_joint] Invalid data size: %zu, expected %d", msg->data.size(), ARM_JOINT_DIM);
        return;
    }
    std::vector<double> left_joint_angles(msg->data.begin(), msg->data.begin() + ARM_JOINT_DIM);
    if (!validate_joint_angles(left_joint_angles, "/set_left_joint")) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_hw_mutex);
        moveJ_l(left_joint_angles, JOINT_SPEED);
    }
    RCLCPP_INFO(rclcpp::get_logger("RobotControl"), "Controlling left joint");
}

void handle_set_right_joint_topic(const sc_ros2::msg::ArmData::SharedPtr msg) {
    if (msg->data.size() != ARM_JOINT_DIM) {
        RCLCPP_ERROR(rclcpp::get_logger("RobotControl"), "[/set_right_joint] Invalid data size: %zu, expected %d", msg->data.size(), ARM_JOINT_DIM);
        return;
    }
    std::vector<double> right_joint_angles(msg->data.begin(), msg->data.begin() + ARM_JOINT_DIM);
    if (!validate_joint_angles(right_joint_angles, "/set_right_joint")) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_hw_mutex);
        moveJ_r(right_joint_angles, JOINT_SPEED);
    }
    RCLCPP_INFO(rclcpp::get_logger("RobotControl"), "Controlling right joint");
}

void handle_set_left_gripper_topic(const sc_ros2::msg::ArmData::SharedPtr msg) {
    if (msg->data.size() != 1 ) {
        RCLCPP_ERROR(rclcpp::get_logger("RobotControl"), "[/set_left_gripper] Invalid data size: %zu, expected %d", msg->data.size(), 1);
        return;
    }
    double left_gripper_value = msg->data[0];
    if (!validate_gripper_value(left_gripper_value, "/set_left_gripper")) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_hw_mutex);
        controlLeftClamp(left_gripper_value, LEFT_GRIPPER_FORCE);
    }
    RCLCPP_INFO(rclcpp::get_logger("RobotControl"), "Controlling left gripper");
}

void handle_set_right_gripper_topic(const sc_ros2::msg::ArmData::SharedPtr msg) {
    if (msg->data.size() != 1 ) {
        RCLCPP_ERROR(rclcpp::get_logger("RobotControl"), "[/set_right_gripper] Invalid data size: %zu, expected %d", msg->data.size(), 1);
        return;
    }
    double right_gripper_value = msg->data[0];
    if (!validate_gripper_value(right_gripper_value, "/set_right_gripper")) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_hw_mutex);
        controlRightClamp(right_gripper_value, RIGHT_GRIPPER_FORCE);
    }
    RCLCPP_INFO(rclcpp::get_logger("RobotControl"), "Controlling right gripper");
}

/*
发布线程，发布各种关节状态
*/
void ArmPublisherThread(rclcpp::Node::SharedPtr node) {
    // 16维数据，7左关节+1左夹爪+7右关节+1右夹爪
    auto publisher_get_dual_arm = node->create_publisher<sc_ros2::msg::ArmData>("/get_dual_arm", 10);
    // 8维数据，7左关节+1左夹爪
    auto publisher_get_left_arm = node->create_publisher<sc_ros2::msg::ArmData>("/get_left_arm", 10);
    // 8维数据，7右关节+1右夹爪
    auto publisher_get_right_arm = node->create_publisher<sc_ros2::msg::ArmData>("/get_right_arm", 10);
    // 7维数据，7左关节
    auto publisher_get_left_joint = node->create_publisher<sc_ros2::msg::ArmData>("/get_left_joint", 10);
    // 7维数据，7右关节
    auto publisher_get_right_joint = node->create_publisher<sc_ros2::msg::ArmData>("/get_right_joint", 10);
    // 1维数据，1左夹爪
    auto publisher_get_left_gripper = node->create_publisher<sc_ros2::msg::ArmData>("/get_left_gripper", 10);
    // 1维数据，1右夹爪
    auto publisher_get_right_gripper = node->create_publisher<sc_ros2::msg::ArmData>("/get_right_gripper", 10);
    // 左臂末端位姿
    auto publisher_get_left_wrist = node->create_publisher<sc_ros2::msg::ArmData>("/get_left_wrist", 10);
    // 右臂末端位姿
    auto publisher_get_right_wrist = node->create_publisher<sc_ros2::msg::ArmData>("/get_right_wrist", 10);

    RCLCPP_INFO(rclcpp::get_logger("RobotControl"), "Node started and publishers created");

    while(rclcpp::ok()) {
        std::vector<double> joint_right;
        std::vector<double> joint_left;
        double rightdata = 0.0;
        double leftdata = 0.0;
        std::vector<double> wrist_left;
        std::vector<double> wrist_right;
        {
            std::lock_guard<std::mutex> lk(g_hw_mutex);
            joint_right = getJointAngle_r();
            joint_left = getJointAngle_l();
            rightdata = getclampposition_r();
            leftdata = getclampposition_l();
            wrist_left = getRobotPos_l();
            wrist_right = getRobotPos_r();
        }

        auto stamp = node->now();
        for(int i = 0; i < 7; i++) {
            joint_right[i] = joint_right[i] / 180.0 * M_PI;
            joint_left[i] = joint_left[i] / 180.0 * M_PI;
        }

        auto dual_arm_message = sc_ros2::msg::ArmData();
        dual_arm_message.header.stamp = stamp;
        dual_arm_message.data.insert(dual_arm_message.data.end(), joint_left.begin(), joint_left.end());
        dual_arm_message.data.push_back(leftdata);
        dual_arm_message.data.insert(dual_arm_message.data.end(), joint_right.begin(), joint_right.end());
        dual_arm_message.data.push_back(rightdata);
        publisher_get_dual_arm->publish(dual_arm_message);

        auto left_arm_message = sc_ros2::msg::ArmData();
        left_arm_message.header.stamp = stamp;
        left_arm_message.data.insert(left_arm_message.data.end(), joint_left.begin(), joint_left.end());
        left_arm_message.data.push_back(leftdata);
        publisher_get_left_arm->publish(left_arm_message);

        auto right_arm_message = sc_ros2::msg::ArmData();
        right_arm_message.header.stamp = stamp;
        right_arm_message.data.insert(right_arm_message.data.end(), joint_right.begin(), joint_right.end());
        right_arm_message.data.push_back(rightdata);
        publisher_get_right_arm->publish(right_arm_message);

        auto left_joint_message = sc_ros2::msg::ArmData();
        left_joint_message.header.stamp = stamp;
        left_joint_message.data.insert(left_joint_message.data.end(), joint_left.begin(), joint_left.end());
        publisher_get_left_joint->publish(left_joint_message);

        auto right_joint_message = sc_ros2::msg::ArmData();
        right_joint_message.header.stamp = stamp;
        right_joint_message.data.insert(right_joint_message.data.end(), joint_right.begin(), joint_right.end());
        publisher_get_right_joint->publish(right_joint_message);

        auto left_gripper_message = sc_ros2::msg::ArmData();
        left_gripper_message.header.stamp = stamp;
        left_gripper_message.data.push_back(leftdata);
        publisher_get_left_gripper->publish(left_gripper_message);

        auto right_gripper_message = sc_ros2::msg::ArmData();
        right_gripper_message.header.stamp = stamp;
        right_gripper_message.data.push_back(rightdata);
        publisher_get_right_gripper->publish(right_gripper_message);

        auto left_wrist_message = sc_ros2::msg::ArmData();
        left_wrist_message.header.stamp = stamp;
        left_wrist_message.data.insert(left_wrist_message.data.end(), wrist_left.begin(), wrist_left.end());
        publisher_get_left_wrist->publish(left_wrist_message);

        auto right_wrist_message = sc_ros2::msg::ArmData();
        right_wrist_message.header.stamp = stamp;
        right_wrist_message.data.insert(right_wrist_message.data.end(), wrist_right.begin(), wrist_right.end());
        publisher_get_right_wrist->publish(right_wrist_message);

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int main(int argc, char **argv) {
    {
        std::lock_guard<std::mutex> lk(g_hw_mutex);
        robot_init(1);

        controlRightClamp(69, 1.0);
        controlLeftClamp(69, 1.0);

        moveJ_r(zero_angle_r, JOINT_INIT_SPEED);
        moveJ_l(zero_angle_l, JOINT_INIT_SPEED);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(3000));


    rclcpp::init(argc, argv);

    auto node = rclcpp::Node::make_shared("RobotControl");
    
    // 发布话题线程
    std::thread arm_publisher_thread(ArmPublisherThread, node);

    // 订阅话题
    auto left_data_sub = node->create_subscription<sc_ros2::msg::ArmData>("/left_data", 10, handle_left_data_topic); // 仅用于重播mcap
    auto right_data_sub = node->create_subscription<sc_ros2::msg::ArmData>("/right_data", 10, handle_right_data_topic); // 仅用于重播mcap
    auto set_dual_arm_sub = node->create_subscription<sc_ros2::msg::ArmData>("/set_dual_arm", 10, handle_set_dual_arm_topic); // 16维数据，7左关节+1左夹爪+7右关节+1右夹爪
    auto set_left_arm_sub = node->create_subscription<sc_ros2::msg::ArmData>("/set_left_arm", 10, handle_set_left_arm_topic); // 8维数据，7左关节+1左夹爪
    auto set_right_arm_sub = node->create_subscription<sc_ros2::msg::ArmData>("/set_right_arm", 10, handle_set_right_arm_topic); // 8维数据，7右关节+1右夹爪
    auto set_left_joint_sub = node->create_subscription<sc_ros2::msg::ArmData>("/set_left_joint", 10, handle_set_left_joint_topic); // 7维数据，7左关节
    auto set_right_joint_sub = node->create_subscription<sc_ros2::msg::ArmData>("/set_right_joint", 10, handle_set_right_joint_topic); // 7维数据，7右关节
    auto set_left_gripper_sub = node->create_subscription<sc_ros2::msg::ArmData>("/set_left_gripper", 10, handle_set_left_gripper_topic); // 1维数据，1左夹爪
    auto set_right_gripper_sub = node->create_subscription<sc_ros2::msg::ArmData>("/set_right_gripper", 10, handle_set_right_gripper_topic); // 1维数据，1右夹爪
    RCLCPP_INFO(rclcpp::get_logger("RobotControl"), "Robot control node ready");
    
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    
    if (arm_publisher_thread.joinable()) arm_publisher_thread.join();
    
    rclcpp::shutdown();
    {
        std::lock_guard<std::mutex> lk(g_hw_mutex);
        robot_end();
    }

    return 0;
}
