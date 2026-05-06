#ifndef MY_DUAL_ARM_PACKAGE_UTILS_CONFIG_HPP
#define MY_DUAL_ARM_PACKAGE_UTILS_CONFIG_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <filesystem>
#include <rclcpp/rclcpp.hpp>
#include <nlohmann/json.hpp>

namespace my_dual_arm_package {

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

// ==============================
// 2. 相机ROS2话题映射
// 对应Python: ROS2_CAMERA_TOPIC_MAP
// 每个相机(head/hand_left/hand_right)对应的color和depth话题
// ==============================
extern const std::map<std::string, std::map<std::string, std::string>> ROS2_CAMERA_TOPIC_MAP;

// ==============================
// 3. 机器人配置结构体
// 对应Python: ROBOT_CONFIGS
// 包含关节名、夹爪配置、末端名称、相机列表等
// ==============================
struct RobotConfig {
    std::map<std::string, std::vector<std::string>> joint_names;
    std::map<std::string, std::map<std::string, std::string>> gripper;
    std::vector<std::string> end_names;
    std::vector<std::string> robot_base_name;
    std::vector<std::string> cameras;
    std::vector<std::string> state_fields;
    std::vector<std::string> action_fields;
};

extern const std::map<std::string, RobotConfig> ROBOT_CONFIGS;

// ==============================
// 1. 全局相机参数(默认值与launch对应,后续被ROS参数覆盖)
// 对应Python: realsense_left_color_width 等全局变量
// ==============================
extern int realsense_left_color_width;
extern int realsense_left_color_height;
extern int realsense_left_color_fps;
extern int realsense_left_depth_width;
extern int realsense_left_depth_height;
extern int realsense_left_depth_fps;

extern int realsense_right_color_width;
extern int realsense_right_color_height;
extern int realsense_right_color_fps;
extern int realsense_right_depth_width;
extern int realsense_right_depth_height;
extern int realsense_right_depth_fps;

extern int orbbec_color_width;
extern int orbbec_color_height;
extern int orbbec_color_fps;
extern int orbbec_depth_width;
extern int orbbec_depth_height;
extern int orbbec_depth_fps;

extern int frequency;
extern int camera_width;
extern int camera_height;

struct CameraConfig {
    std::string color_format;
    std::string depth_format;
    int frame_timeout_ms;
    int fps;
    int width;
    int height;
};

extern CameraConfig CAMERA_CONFIG;

// ==============================
// 5. 时间同步配置
// 对应Python: SYNC_CONFIG
// ==============================
struct SyncConfig {
    bool allow_older_data;
    int max_time_diff_ms;
};

extern SyncConfig SYNC_CONFIG;

// ==============================
// 6. 双臂数据长度与分片
// 单臂数据: 7关节 + 2夹爪 + 3末端位置 + 4末端姿态 = 16
// 双臂数据: 16 x 2 = 32
// ==============================
const int SINGLE_ARM_STATE_LEN = 16;
const int DOUBLE_ARM_STATE_LEN = 32;

struct SplitRange {
    int start;
    int end;
};

extern const std::map<std::string, SplitRange> SINGLE_ARM_STATE_SPLIT;
extern const std::map<std::string, SplitRange> DOUBLE_ARM_STATE_SPLIT;

// ==============================
// 7. 全局状态管理类
// 对应Python: ServerState
// 用于同步采集进度(相机/关节帧数)
// ==============================
class ServerState {
public:
    ServerState();
    std::map<std::string, std::map<std::string, int>> camera_frames_count;
    std::map<std::string, int> joint_frames_count;
    bool is_recording;
    std::mutex lock;
};

extern std::shared_ptr<ServerState> global_server_state;

// ==============================
// 工具函数声明
// ==============================

/// 9. 动态获取指定相机、指定数据类型(color/depth)的参数(width/height/fps), 参数从3cameras.sh解析
int get_camera_param(const std::string& cam_name, const std::string& data_type, const std::string& param);

/// 10. ROS时间戳(秒级浮点数)转换为19位纳秒时间戳(int64)
int64_t ros_time_to_ns(double ros_time);

/// 初始化meta_info.json(采集元数据:作者、相机列表、创建时间等)
void init_meta_info(const std::map<std::string, fs::path>& dirs, const std::vector<std::string>& camera_names);

/// 11. 初始化parameters文件夹(相机内参/外参、关节范围、URDF等)
void init_parameters(const std::map<std::string, fs::path>& dirs, rclcpp::Node::SharedPtr node);

/// 初始化端侧落盘目录结构(record、camera、parameters等子目录)
std::map<std::string, fs::path> init_task_directories(const fs::path& root_dir);

/// 创建raw_joints.h5的完整结构(state/action组,包含arm/end/effector/robot子组)
void create_h5_structure(const fs::path& h5_path, const std::string& robot_type = "Keenon_F1_DoubleArm");

} // namespace my_dual_arm_package

#endif // MY_DUAL_ARM_PACKAGE_UTILS_CONFIG_HPP
