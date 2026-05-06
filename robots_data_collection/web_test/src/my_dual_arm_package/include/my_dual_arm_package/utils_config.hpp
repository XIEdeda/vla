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

extern const std::map<std::string, std::map<std::string, std::string>> ROS2_CAMERA_TOPIC_MAP;

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

struct SyncConfig {
    bool allow_older_data;
    int max_time_diff_ms;
};

extern SyncConfig SYNC_CONFIG;

const int SINGLE_ARM_STATE_LEN = 16;
const int DOUBLE_ARM_STATE_LEN = 32;

struct SplitRange {
    int start;
    int end;
};

extern const std::map<std::string, SplitRange> SINGLE_ARM_STATE_SPLIT;
extern const std::map<std::string, SplitRange> DOUBLE_ARM_STATE_SPLIT;


class ServerState {
public:
    ServerState();
    std::map<std::string, std::map<std::string, int>> camera_frames_count;
    std::map<std::string, int> joint_frames_count;
    bool is_recording;
    std::mutex lock;
};

extern std::shared_ptr<ServerState> global_server_state;


int get_camera_param(const std::string& cam_name, const std::string& data_type, const std::string& param);

int64_t ros_time_to_ns(double ros_time);

void init_meta_info(const std::map<std::string, fs::path>& dirs, const std::vector<std::string>& camera_names);


void init_parameters(const std::map<std::string, fs::path>& dirs, rclcpp::Node::SharedPtr node);

std::map<std::string, fs::path> init_task_directories(const fs::path& root_dir);


void create_h5_structure(const fs::path& h5_path, const std::string& robot_type = "Keenon_F1_DoubleArm");

} 

#endif 
