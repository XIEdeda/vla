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

// 基类卡尔曼滤波器
class KalmanFilterBase {
public:
    KalmanFilterBase(int state_dim, int meas_dim) : 
        n(state_dim), m(meas_dim),
        x(state_dim), P(state_dim, state_dim),
        F(state_dim, state_dim), H(meas_dim, state_dim),
        Q(state_dim, state_dim), R(meas_dim, meas_dim),
        initialized(false)
    {
        x.setZero();
        F.setIdentity();
        H.setIdentity();
    }
    
    virtual ~KalmanFilterBase() = default;
    
    void init(const std::vector<double>& initial_angles) {
        if (initial_angles.size() != n) {
            RCLCPP_ERROR(rclcpp::get_logger("KalmanFilterBase"), 
                        "Initial angles size mismatch: expected %d, got %zu", 
                        n, initial_angles.size());
            return;
        }
        
        for (int i = 0; i < n; ++i) {
            x(i) = initial_angles[i];
        }
        initialized = true;
    }
    
    std::vector<double> filter(const std::vector<double>& raw_angles) {
        if (!initialized) {
            RCLCPP_WARN(rclcpp::get_logger("KalmanFilterBase"), 
                       "Filter not initialized, using raw measurements");
            return raw_angles;
        }
        
        if (raw_angles.size() != m) {
            RCLCPP_ERROR(rclcpp::get_logger("KalmanFilterBase"), 
                        "Raw angles size mismatch: expected %d, got %zu", 
                        m, raw_angles.size());
            return raw_angles;
        }
        
        // 预测步骤
        predict();
        
        // 更新步骤
        Eigen::VectorXd measurement(m);
        for (int i = 0; i < m; ++i) {
            measurement(i) = raw_angles[i];
        }
        update(measurement);
        
        // 返回滤波后的状态
        std::vector<double> result(n);
        for (int i = 0; i < n; ++i) {
            result[i] = x(i);
        }
        
        return result;
    }
    
    bool isInitialized() const {
        return initialized;
    }
    
    const std::vector<double> getState() const {
        std::vector<double> state(n);
        for (int i = 0; i < n; ++i) {
            state[i] = x(i);
        }
        return state;
    }
    
protected:
    virtual void predict() {
        x = F * x;
        P = F * P * F.transpose() + Q;
    }
    
    virtual void update(const Eigen::VectorXd& z) {
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();
        x = x + K * (z - H * x);
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
        P = (I - K * H) * P;
    }
    
    int n;
    int m;
    Eigen::VectorXd x;
    Eigen::MatrixXd P;
    Eigen::MatrixXd F;
    Eigen::MatrixXd H;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    bool initialized;
};

// 右臂卡尔曼滤波器类
class KalmanFilterRight : public KalmanFilterBase {
public:
    KalmanFilterRight(int state_dim = 7, int meas_dim = 7) : 
        KalmanFilterBase(state_dim, meas_dim) 
    {
        // 右臂特有参数
        P.setIdentity();
        P *= 1000.0;    
        Q.setIdentity();
        Q *= 0.01;   
        R.setIdentity();
        R *= 0.5;      
    }
    
    void init(const std::vector<double>& initial_angles) {
        KalmanFilterBase::init(initial_angles);
        if (initialized) {
            RCLCPP_INFO(rclcpp::get_logger("KalmanFilterRight"), 
                       "Right arm Kalman filter initialized");
        }
    }
};

// 左臂卡尔曼滤波器类
class KalmanFilterLeft : public KalmanFilterBase {
public:
    KalmanFilterLeft(int state_dim = 7, int meas_dim = 7) : 
        KalmanFilterBase(state_dim, meas_dim) 
    {
        // 左臂特有参数
        P.setIdentity();
        P *= 1000.0;    
        Q.setIdentity();
        Q *= 0.01;     
        R.setIdentity();
        R *= 0.5;      
    }
    
    void init(const std::vector<double>& initial_angles) {
        KalmanFilterBase::init(initial_angles);
        if (initialized) {
            RCLCPP_INFO(rclcpp::get_logger("KalmanFilterLeft"), 
                       "Left arm Kalman filter initialized");
        }
    }
};

// 全局变量
std::vector<double> zero_angle_r0 = {0, 0, 0, 0, 0, 0, 0};
std::vector<double> send_angle_r = {0, 0, 0, 0, 0, 0, 0};
//std::vector<double> zero_angle_r = {21.8795/180*M_PI , 13.3083/180*M_PI , -48.7721/180*M_PI , 101.017/180*M_PI, -23.3799/180*M_PI , 18.3073/180*M_PI , -0.384521/180*M_PI};   16.2045 , 21.4456 , -56.4347 , 83.7871 , -40.6408 , 18.2799 , -24.4542 , 
std::vector<double> zero_angle_r = {-24.459 /180*M_PI , 54.611/180*M_PI , 13.531/180*M_PI , 111.269/180*M_PI, -55.829/180*M_PI , 12.532/180*M_PI , 1.082/180*M_PI};
std::vector<double> zero_angle_l = {23.628 /180*M_PI , -55.034/180*M_PI , -13.216/180*M_PI , -112.641/180*M_PI, 54.613/180*M_PI , -12.182/180*M_PI , -3.7888/180*M_PI};

// 添加夹爪互斥锁和值
std::mutex gripper_right_mutex;
std::mutex gripper_left_mutex;
double gripper_right_value = 1.0;
double gripper_left_value = 1.0;

// 添加机械臂和夹爪控制相关的全局变量
std::queue<std::vector<double>> right_robot_arm_queue;
std::queue<std::vector<double>> left_robot_arm_queue;
std::queue<double> gripper_right_command_queue;
std::queue<double> gripper_left_command_queue;
std::mutex robot_arm_mutex;
std::mutex gripper_right_queue_mutex;
std::mutex gripper_left_queue_mutex;
std::condition_variable robot_arm_cv;
std::condition_variable gripper_right_cv;
std::condition_variable gripper_left_cv;
bool robot_arm_thread_running = true;
bool gripper_right_thread_running = true;
bool gripper_left_thread_running = true;

// 卡尔曼滤波相关 - 使用两个独立的类
KalmanFilterRight* right_kalman_filter = nullptr;
KalmanFilterLeft* left_kalman_filter = nullptr;

std::vector<double> filtered_joint_angles_right(7, 0.0);
std::vector<double> filtered_joint_angles_left(7, 0.0);

bool right_kalman_initialized = false;
bool left_kalman_initialized = false;

// 初始化右臂卡尔曼滤波器
void initialize_kalman_filter_right(const std::vector<double>& initial_angles) {
    if (right_kalman_filter == nullptr) {
        right_kalman_filter = new KalmanFilterRight(7, 7);
        right_kalman_filter->init(initial_angles);
        filtered_joint_angles_right = initial_angles;
        right_kalman_initialized = true;
        RCLCPP_INFO(rclcpp::get_logger("KalmanFilter"), "Right arm Kalman filter initialized");
    }
}

// 初始化左臂卡尔曼滤波器
void initialize_kalman_filter_left(const std::vector<double>& initial_angles) {
    if (left_kalman_filter == nullptr) {
        left_kalman_filter = new KalmanFilterLeft(7, 7);
        left_kalman_filter->init(initial_angles);
        filtered_joint_angles_left = initial_angles;
        left_kalman_initialized = true;
        RCLCPP_INFO(rclcpp::get_logger("KalmanFilter"), "Left arm Kalman filter initialized");
    }
}

// 应用右臂卡尔曼滤波
std::vector<double> apply_kalman_filter_right(const std::vector<double>& raw_angles) {
    if (!right_kalman_initialized) {
        initialize_kalman_filter_right(raw_angles);
        return raw_angles;
    }
    
    if (right_kalman_filter == nullptr) {
        RCLCPP_ERROR(rclcpp::get_logger("KalmanFilter"), "Right arm Kalman filter is null");
        return raw_angles;
    }
    
    return right_kalman_filter->filter(raw_angles);
}

// 应用左臂卡尔曼滤波
std::vector<double> apply_kalman_filter_left(const std::vector<double>& raw_angles) {
    if (!left_kalman_initialized) {
        initialize_kalman_filter_left(raw_angles);
        return raw_angles;
    }
    
    if (left_kalman_filter == nullptr) {
        RCLCPP_ERROR(rclcpp::get_logger("KalmanFilter"), "Left arm Kalman filter is null");
        return raw_angles;
    }
    
    return left_kalman_filter->filter(raw_angles);
}

// 机械臂控制线程
void right_robot_arm_control_thread() {
    RCLCPP_INFO(rclcpp::get_logger("RobotArmControl"), "Right robot arm control thread started");
    while (robot_arm_thread_running) {
        std::vector<double> angles;
        {
            std::unique_lock<std::mutex> lock(robot_arm_mutex);
            robot_arm_cv.wait(lock, [] {
                return !right_robot_arm_queue.empty() || !robot_arm_thread_running;
            });
            
            if (!robot_arm_thread_running && right_robot_arm_queue.empty()) {
                break;
            }
            
            if (!right_robot_arm_queue.empty()) {
                angles = right_robot_arm_queue.front();
                right_robot_arm_queue.pop();
            }
        }
        
        if (!angles.empty()) {
            RCLCPP_INFO(rclcpp::get_logger("RobotArmControl"), "Moving right robot arm");
            moveJ_r(angles, 0.5);
        }
    }
    
    RCLCPP_INFO(rclcpp::get_logger("RobotArmControl"), "Right robot arm control thread stopped");
}

void left_robot_arm_control_thread() {
    RCLCPP_INFO(rclcpp::get_logger("RobotArmControl"), "Left robot arm control thread started");
    while (robot_arm_thread_running) {
        std::vector<double> angles;
        {
            std::unique_lock<std::mutex> lock(robot_arm_mutex);
            robot_arm_cv.wait(lock, [] {
                return !left_robot_arm_queue.empty() || !robot_arm_thread_running;
            });
            
            if (!robot_arm_thread_running && left_robot_arm_queue.empty()) {
                break;
            }
            
            if (!left_robot_arm_queue.empty()) {
                angles = left_robot_arm_queue.front();
                left_robot_arm_queue.pop();
            }
        }
        
        if (!angles.empty()) {
            RCLCPP_INFO(rclcpp::get_logger("RobotArmControl"), "Moving left robot arm");
            moveJ_l(angles, 0.5);
        }
    }
    
    RCLCPP_INFO(rclcpp::get_logger("RobotArmControl"), "Left robot arm control thread stopped");
}

// 右夹爪控制线程
void gripper_right_control_thread() {
    RCLCPP_INFO(rclcpp::get_logger("GripperControl"), "Right gripper control thread started");
    
    while (gripper_right_thread_running) {
        double command = 0.0;
        
        {
            std::unique_lock<std::mutex> lock(gripper_right_queue_mutex);
            gripper_right_cv.wait(lock, [] {
                return !gripper_right_command_queue.empty() || !gripper_right_thread_running;
            });
            
            if (!gripper_right_thread_running && gripper_right_command_queue.empty()) {
                break;
            }

            // 夹爪线程改为和机械臂一样的FIFO逻辑
            if (!gripper_right_command_queue.empty()) {
                command = gripper_right_command_queue.front();
                gripper_right_command_queue.pop();
            }
        }
        double command_gripper_width = command ;
        // double command_gripper_width = command * 1000;
        RCLCPP_INFO(rclcpp::get_logger("GripperControl"), "Controlling right gripper");
        controlRightClamp(command_gripper_width,0.32);
        // clamp(mb_r,command_gripper_width); // 修改为右夹爪控制
        std::lock_guard<std::mutex> gripper_lock(gripper_right_mutex);
        gripper_right_value = command;
        std::cout << "Right gripper clamp===========================================： " << command_gripper_width << std::endl;
    }
    
    RCLCPP_INFO(rclcpp::get_logger("GripperControl"), "Right gripper control thread stopped");
}

// 左夹爪控制线程
void gripper_left_control_thread() {
    RCLCPP_INFO(rclcpp::get_logger("GripperControl"), "Left gripper control thread started");
    
    while (gripper_left_thread_running) {
        double command = 0.0;
        
        {
            std::unique_lock<std::mutex> lock(gripper_left_queue_mutex);
            gripper_left_cv.wait(lock, [] {
                return !gripper_left_command_queue.empty() || !gripper_left_thread_running;
            });
            
            if (!gripper_left_thread_running && gripper_left_command_queue.empty()) {
                break;
            }

            // 夹爪线程改为和机械臂一样的FIFO逻辑
            if (!gripper_left_command_queue.empty()) {
                command = gripper_left_command_queue.front();
                gripper_left_command_queue.pop();
            }
        }
        
        double command_gripper_width = command;
        RCLCPP_INFO(rclcpp::get_logger("GripperControl"), "Controlling left gripper");
        controlLeftClamp(command_gripper_width,0.4);
        std::lock_guard<std::mutex> gripper_lock(gripper_left_mutex);
        gripper_left_value = command;
        std::cout << "Left gripper clamp===========================================： " << command_gripper_width << std::endl;
    }
    
    RCLCPP_INFO(rclcpp::get_logger("GripperControl"), "Left gripper control thread stopped");
}

// 处理请求的函数
void handle_request(const std::shared_ptr<sc_ros2::srv::MulSetJointAngles::Request> request,
                    std::shared_ptr<sc_ros2::srv::MulSetJointAngles::Response> response) {
  
    if (request->sequences.empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("MulSetJointAngles"), "Received empty sequences array");
        response->success = false;
        return;
    }
    
    size_t num_sequences = request->sequences.size();
    RCLCPP_INFO(rclcpp::get_logger("MulSetJointAngles"), "Received %zu sequences", num_sequences);
    
    bool all_success = true;

    for (size_t seq = 0; seq < num_sequences; ++seq) {
        const auto& current_sequence = request->sequences[seq];
        const auto& joints_data = current_sequence.joints;
        
        if (joints_data.size() < 16) {
            RCLCPP_ERROR(rclcpp::get_logger("MulSetJointAngles"), 
                        "Sequence %zu has only %zu elements, expected at least 8", 
                        seq, joints_data.size());
            all_success = false;
            continue;
        }
        
        // 获取原始关节角度
        std::vector<double> left_raw_joint_angles(joints_data.begin(), joints_data.begin() + 7);
        std::vector<double> right_raw_joint_angles(joints_data.begin()+8, joints_data.begin() + 15);

        // 分别应用卡尔曼滤波
        std::vector<double> filtered_angles_right = apply_kalman_filter_right(right_raw_joint_angles);
        std::vector<double> filtered_angles_left = apply_kalman_filter_left(left_raw_joint_angles);

        // 更新全局滤波后的角度
        filtered_joint_angles_right = filtered_angles_right;
        filtered_joint_angles_left = filtered_angles_left;
        
        double left_current_gripper_value = joints_data[7];
        double right_current_gripper_value = joints_data[15];

        // 发送机械臂角度到机械臂控制线程
        {
            std::lock_guard<std::mutex> lock(robot_arm_mutex);
            right_robot_arm_queue.push(filtered_angles_right);
            left_robot_arm_queue.push(filtered_angles_left);
        }
        robot_arm_cv.notify_one();
        
        // 发送右夹爪命令到右夹爪控制线程
        {
            std::lock_guard<std::mutex> lock(gripper_right_queue_mutex);
            gripper_right_command_queue.push(right_current_gripper_value);
        }
        gripper_right_cv.notify_one();
        
        // 发送左夹爪命令到左夹爪控制线程
        {
            std::lock_guard<std::mutex> lock(gripper_left_queue_mutex);
            gripper_left_command_queue.push(left_current_gripper_value);
        }
        gripper_left_cv.notify_one();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 避免忙等待

    }
    
    response->success = all_success;
}

// monitorthread函数
void monitorthread() {
    auto node = rclcpp::Node::make_shared("remote_control_publisher");
    auto publisher_dual = node->create_publisher<sc_ros2::msg::ArmData>("dual_data", 10);
    // auto publisher_left = node->create_publisher<std_msgs::msg::Float64MultiArray>("left_data", 10);
    RCLCPP_INFO(node->get_logger(), "Node started and publishers created");

    while(rclcpp::ok()) {
    std::vector<double> joint_right = getJointAngle_r();
    std::vector<double> joint_left = getJointAngle_l();
    double rightdata = getclampposition_r();
    double leftdata = getclampposition_l();


        auto stamp = node->now();
        for(int i = 0; i < 7; i++) {
            joint_right[i] = joint_right[i] / 180.0 * M_PI;
            joint_left[i] = joint_left[i] / 180.0 * M_PI;
        }

        auto message_data = sc_ros2::msg::ArmData();
        message_data.header.stamp = stamp;
        
        message_data.data.insert(message_data.data.end(), 
                                joint_left.begin(), joint_left.end());
        
        {
            std::lock_guard<std::mutex> lock(gripper_left_mutex);
            message_data.data.push_back(leftdata);
        }
        
        message_data.data.insert(message_data.data.end(), 
                                joint_right.begin(), joint_right.end());
        
        {
            std::lock_guard<std::mutex> lock(gripper_right_mutex);
            message_data.data.push_back(rightdata);
        }
        
        publisher_dual->publish(message_data);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int main(int argc, char **argv) {
    robot_init(1);

    controlRightClamp(69,1.0);
    controlLeftClamp(69,1.0);
    
    int retr = moveJ_r(zero_angle_r, 0.1);
    int retl = moveJ_l(zero_angle_l, 0.1);

    std::this_thread::sleep_for(std::chrono::milliseconds(3000));


    // 分别初始化左右臂的卡尔曼滤波器
    initialize_kalman_filter_right(zero_angle_r);
    initialize_kalman_filter_left(zero_angle_l);

    rclcpp::init(argc, argv);
    
    // 启动控制线程
    std::thread right_robot_arm_thread(right_robot_arm_control_thread);
    std::thread left_robot_arm_thread(left_robot_arm_control_thread);
    std::thread gripper_right_thread(gripper_right_control_thread);
    std::thread gripper_left_thread(gripper_left_control_thread);
    std::thread monitor_thread(monitorthread);
    
    auto node = rclcpp::Node::make_shared("MulSetJointAngles");
    auto service = node->create_service<sc_ros2::srv::MulSetJointAngles>("MulSetJointAngles", handle_request);
    RCLCPP_INFO(rclcpp::get_logger("MulSetJointAngles"), "Dual Arm Move Service ready");
    
    //rclcpp::spin(node);
    // add by xiededa
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    
    // 清理资源
    {
        std::lock_guard<std::mutex> lock1(robot_arm_mutex);
        std::lock_guard<std::mutex> lock2(gripper_right_queue_mutex);
        std::lock_guard<std::mutex> lock3(gripper_left_queue_mutex);
        robot_arm_thread_running = false;
        gripper_right_thread_running = false;
        gripper_left_thread_running = false;
    }
    robot_arm_cv.notify_one();
    gripper_right_cv.notify_one();
    gripper_left_cv.notify_one();
    
    if (right_robot_arm_thread.joinable()) right_robot_arm_thread.join();
    if (left_robot_arm_thread.joinable()) left_robot_arm_thread.join();
    if (gripper_right_thread.joinable()) gripper_right_thread.join();
    if (gripper_left_thread.joinable()) gripper_left_thread.join();
    if (monitor_thread.joinable()) monitor_thread.join();
    
    // 清理卡尔曼滤波器
    if (right_kalman_filter != nullptr) {
        delete right_kalman_filter;
        right_kalman_filter = nullptr;
    }
    if (left_kalman_filter != nullptr) {
        delete left_kalman_filter;
        left_kalman_filter = nullptr;
    }
    
    rclcpp::shutdown();
    robot_end();


    return 0;
}
