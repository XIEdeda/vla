#include "my_dual_arm_package/utils_config.hpp"
#include <fstream>
#include <tuple>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <future>
#include <regex>
#include <cstdlib>
#include <H5Cpp.h>
#include <nlohmann/json.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <rclcpp/logger.hpp>

namespace my_dual_arm_package {

const std::map<std::string, std::map<std::string, std::string>> ROS2_CAMERA_TOPIC_MAP = {
    {"head", {
        {"color", "/camera3/camera_head/color/image_raw"},
        {"depth", "/camera3/camera_head/depth/image_rect_raw"},
        {"color_compressed", "/camera3/camera_head/color/image_raw/compressed"},
        {"depth_compressed", "/camera3/camera_head/depth/image_rect_raw/compressedDepth"}
    }},
    {"hand_left", {
        {"color", "/camera1/camera_left/color/image_rect_raw"},
        {"depth", "/camera1/camera_left/depth/image_rect_raw"},
        {"color_compressed", "/camera1/camera_left/color/image_rect_raw/compressed"},
        {"depth_compressed", "/camera1/camera_left/depth/image_rect_raw/compressedDepth"}
    }},
    {"hand_right", {
        {"color", "/camera2/camera_right/color/image_rect_raw"},
        {"depth", "/camera2/camera_right/depth/image_rect_raw"},
        {"color_compressed", "/camera2/camera_right/color/image_rect_raw/compressed"},
        {"depth_compressed", "/camera2/camera_right/depth/image_rect_raw/compressedDepth"}
    }}
};

static const std::map<std::string, std::string> CAM_INFO_TOPIC_MAP = {
    {"hand_left", "/camera1/camera_left/color/camera_info"},
    {"hand_right", "/camera2/camera_right/color/camera_info"},
    {"head", "/camera3/camera_head/color/camera_info"}
};

const std::map<std::string, RobotConfig> ROBOT_CONFIGS = {
    {"Keenon_F1_DoubleArm", {
        {{"left_arm", {"left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow_roll", "left_elbow_yaw", "left_wrist_pitch", "left_wrist_roll"}},
         {"right_arm", {"right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow_roll", "right_elbow_yaw", "right_wrist_pitch", "right_wrist_roll"}}},
        {{"left_gripper", {{"0", "closed"}, {"1", "open"}, {"category", "binary"}}},
         {"right_gripper", {{"0", "closed"}, {"1", "open"}, {"category", "binary"}}}},
        {"left_end_link", "right_end_link"},
        {"base_link"},
        {"head", "hand_left", "hand_right"},
        {"left_arm_joints", "right_arm_joints", "left_gripper", "right_gripper"},
        {"left_arm_joints", "right_arm_joints", "left_gripper", "right_gripper"}
    }}
};

//realsensed405
int realsense_left_color_width = 640;
int realsense_left_color_height = 480;
int realsense_left_color_fps = 30;
int realsense_left_depth_width = 640;
int realsense_left_depth_height = 480;
int realsense_left_depth_fps = 30;

int realsense_right_color_width = 640;
int realsense_right_color_height = 480;
int realsense_right_color_fps = 30;
int realsense_right_depth_width = 640;
int realsense_right_depth_height = 480;
int realsense_right_depth_fps = 30;

//实际为realsensed435
int orbbec_color_width = 640;
int orbbec_color_height = 480;
int orbbec_color_fps = 30;
int orbbec_depth_width = 640;
int orbbec_depth_height = 480;
int orbbec_depth_fps = 30;

int frequency = 30;
int camera_width = 640;
int camera_height = 480;

CameraConfig CAMERA_CONFIG = {"bgr8", "16UC1", 100, 30, 640, 480};

SyncConfig SYNC_CONFIG = {true, 100};

const std::map<std::string, SplitRange> SINGLE_ARM_STATE_SPLIT = {
    {"arm_joints", {0, 7}},
    {"gripper", {7, 9}},
    {"end_position", {9, 12}},
    {"end_orientation", {12, 16}}
};

const std::map<std::string, SplitRange> DOUBLE_ARM_STATE_SPLIT = {
    {"left_arm_joints", {0, 7}},
    {"left_gripper", {7, 9}},
    {"left_end_position", {9, 12}},
    {"left_end_orientation", {12, 16}},
    {"right_arm_joints", {16, 23}},
    {"right_gripper", {23, 25}},
    {"right_end_position", {25, 28}},
    {"right_end_orientation", {28, 32}}
};

ServerState::ServerState() {
    for (auto const& [name, topics] : ROS2_CAMERA_TOPIC_MAP) {
        camera_frames_count[name] = {{"color", 0}, {"depth", 0}};
    }
    joint_frames_count = {{"left_arm", 0}, {"right_arm", 0}};
    is_recording = false;
}

std::shared_ptr<ServerState> global_server_state = std::make_shared<ServerState>();

static std::map<int, std::map<std::string, std::tuple<int, int, int>>> parse_profile_from_script(
    const fs::path& script_path, rclcpp::Logger logger) {
    std::map<int, std::map<std::string, std::tuple<int, int, int>>> profile_map;
    std::ifstream f(script_path);
    if (!f) {
        RCLCPP_WARN(logger, "无法打开相机启动脚本: %s", script_path.string().c_str());
        return profile_map;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    std::regex re_depth(R"(depth_module\.(color|depth)_profile(\d+)\s*:=\s*(\d+)x(\d+)x(\d+))");
    std::regex re_rgb(R"(rgb_camera\.color_profile(\d+)\s*:=\s*(\d+)x(\d+)x(\d+))");
    std::smatch m;
    std::string::const_iterator start, end;

    start = content.cbegin();
    end = content.cend();
    while (std::regex_search(start, end, m, re_depth)) {
        std::string type_str = m[1].str();
        int idx = std::stoi(m[2].str());
        int w = std::stoi(m[3].str());
        int h = std::stoi(m[4].str());
        int fps_val = std::stoi(m[5].str());
        profile_map[idx][type_str] = {w, h, fps_val};
        start = m[0].second;
    }
    start = content.cbegin();
    end = content.cend();
    while (std::regex_search(start, end, m, re_rgb)) {
        int idx = std::stoi(m[1].str());
        int w = std::stoi(m[2].str());
        int h = std::stoi(m[3].str());
        int fps_val = std::stoi(m[4].str());
        profile_map[idx]["color"] = {w, h, fps_val};
        start = m[0].second;
    }
    return profile_map;
}

static void init_camera_params_from_script() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    fs::path script_path;
    const char* env_path = std::getenv("CAMERA_SN_SCRIPT");
    if (env_path && env_path[0]) script_path = fs::path(env_path);
    if (script_path.empty() || !fs::exists(script_path)) return;

    auto logger = rclcpp::get_logger("utils_config");
    auto profile_map = parse_profile_from_script(script_path, logger);

    auto apply = [&profile_map](int idx, int& cw, int& ch, int& cf, int& dw, int& dh, int& df) {
        if (profile_map.count(idx)) {
            auto& pm = profile_map[idx];
            if (pm.count("color")) {
                auto [w, h, f] = pm["color"];
                cw = w; ch = h; cf = f;
            }
            if (pm.count("depth")) {
                auto [w, h, f] = pm["depth"];
                dw = w; dh = h; df = f;
            }
        }
    };
    apply(1, realsense_left_color_width, realsense_left_color_height, realsense_left_color_fps,
          realsense_left_depth_width, realsense_left_depth_height, realsense_left_depth_fps);
    apply(2, realsense_right_color_width, realsense_right_color_height, realsense_right_color_fps,
          realsense_right_depth_width, realsense_right_depth_height, realsense_right_depth_fps);
    apply(3, orbbec_color_width, orbbec_color_height, orbbec_color_fps,
          orbbec_depth_width, orbbec_depth_height, orbbec_depth_fps);

    frequency = realsense_left_color_fps;
    camera_width = realsense_left_color_width;
    camera_height = realsense_left_color_height;
    CAMERA_CONFIG.fps = frequency;
    CAMERA_CONFIG.width = camera_width;
    CAMERA_CONFIG.height = camera_height;

    RCLCPP_INFO(logger, "==================================================");
    RCLCPP_INFO(logger, "utils 相机参数已从3cameras.sh加载:");
    RCLCPP_INFO(logger, "1. hand_left: %dx%d@%dfps (Depth: %dx%d@%dfps)",
                realsense_left_color_width, realsense_left_color_height, realsense_left_color_fps,
                realsense_left_depth_width, realsense_left_depth_height, realsense_left_depth_fps);
    RCLCPP_INFO(logger, "2. hand_right: %dx%d@%dfps (Depth: %dx%d@%dfps)",
                realsense_right_color_width, realsense_right_color_height, realsense_right_color_fps,
                realsense_right_depth_width, realsense_right_depth_height, realsense_right_depth_fps);
    RCLCPP_INFO(logger, "3. head: %dx%d@%dfps (Depth: %dx%d@%dfps)",
                orbbec_color_width, orbbec_color_height, orbbec_color_fps,
                orbbec_depth_width, orbbec_depth_height, orbbec_depth_fps);
    RCLCPP_INFO(logger, "==================================================");
}

int get_camera_param(const std::string& cam_name, const std::string& data_type, const std::string& param) {
    init_camera_params_from_script();  // 懒加载: 首次调用时从3cameras.sh解析
    if (cam_name == "hand_left") {
        if (data_type == "color") {
            if (param == "width") return realsense_left_color_width;
            if (param == "height") return realsense_left_color_height;
            if (param == "fps") return realsense_left_color_fps;
        } else {
            if (param == "width") return realsense_left_depth_width;
            if (param == "height") return realsense_left_depth_height;
            if (param == "fps") return realsense_left_depth_fps;
        }
    } else if (cam_name == "hand_right") {
        if (data_type == "color") {
            if (param == "width") return realsense_right_color_width;
            if (param == "height") return realsense_right_color_height;
            if (param == "fps") return realsense_right_color_fps;
        } else {
            if (param == "width") return realsense_right_depth_width;
            if (param == "height") return realsense_right_depth_height;
            if (param == "fps") return realsense_right_depth_fps;
        }
    } else if (cam_name == "head") {
        if (data_type == "color") {
            if (param == "width") return orbbec_color_width;
            if (param == "height") return orbbec_color_height;
            if (param == "fps") return orbbec_color_fps;
        } else {
            if (param == "width") return orbbec_depth_width;
            if (param == "height") return orbbec_depth_height;
            if (param == "fps") return orbbec_depth_fps;
        }
    }
    return 0;
}


int64_t ros_time_to_ns(double ros_time) {
    return static_cast<int64_t>(ros_time * 1e9);
}

void init_meta_info(const std::map<std::string, fs::path>& dirs, const std::vector<std::string>& camera_names) {
    auto meta_info_path = dirs.at("meta_info");
    if (fs::exists(meta_info_path)) return;

    json meta_data;
    meta_data["author"] = "Keenon";
    meta_data["robot_type"] = "R1";
    meta_data["SN"] = "qlznxman001";
    meta_data["version"] = "0.0.1";
    meta_data["ee_type"] = "gripper";
    meta_data["ee_list"] = {
        {{"name", "left_gripper"}, {"type", "gripper"}, {"category", "binary"}, {"unit", "radian"}},
        {{"name", "right_gripper"}, {"type", "gripper"}, {"category", "binary"}, {"unit", "radian"}}
    };
    
    json cameras = json::array();
    for (const auto& cam_name : camera_names) {
        cameras.push_back({
            {"name", cam_name},
            {"type", "rgbd"},
            {"camera_fps", std::to_string(get_camera_param(cam_name, "color", "fps"))},
            {"rgb_data", {{"format", "jpg"}}},
            {"depth_data", {{"format", "png"}}}
        });
    }
    meta_data["cameras"] = cameras;

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S +08:00");
    meta_data["create_time"] = ss.str();
    meta_data["durationInMs"] = 0;

    std::ofstream f(meta_info_path);
    f << meta_data.dump(4);
}

static std::map<std::string, std::string> parse_serial_numbers_from_script(const fs::path& script_path, rclcpp::Logger logger) {
    std::map<std::string, std::string> sn_map;
    std::ifstream f(script_path);
    if (!f) {
        RCLCPP_WARN(logger, "无法打开相机启动脚本: %s", script_path.string().c_str());
        return sn_map;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    std::regex re(R"(serial_no(\d+)\s*:=\s*_?(\d+))");
    std::smatch m;
    std::string::const_iterator start = content.cbegin();
    std::string::const_iterator end = content.cend();
    std::map<int, std::string> by_num;
    while (std::regex_search(start, end, m, re)) {
        int idx = std::stoi(m[1].str());
        std::string sn = m[2].str();  
        by_num[idx] = sn;
        start = m[0].second;
    }

    if (by_num.count(1)) sn_map["hand_left"] = by_num[1];
    if (by_num.count(2)) sn_map["hand_right"] = by_num[2];
    if (by_num.count(3)) sn_map["head"] = by_num[3];
    return sn_map;
}


static json get_default_hand_right_intrinsic() {
    return {
        {"fx", 394.4433288574219}, {"fy", 394.2029113769531}, {"ppx", 320.6365966796875}, {"ppy", 240.33468627929688},
        {"distortion_model", "plumb_bob"},
        {"k1", -0.05233052372932434}, {"k2", 0.06378290802240372}, {"k3", -0.022098379209637642},
        {"p1", 0.001027168007567525}, {"p2", 0.0009821165585890412}
    };
}
static json get_default_hand_left_intrinsic() {
    return {
        {"fx", 392.3494873046875}, {"fy", 391.842041015625}, {"ppx", 321.5123596191406}, {"ppy", 237.52915954589844},
        {"distortion_model", "plumb_bob"},
        {"k1", -0.0518009215593338}, {"k2", 0.06316935271024704}, {"k3", -0.021872593089938164},
        {"p1", 0.0003664149553515017}, {"p2", 0.00007883293437771499}
    };
}
static json get_default_d435_intrinsic() {
    return {
        {"fx", 606.0679321289062}, {"fy", 604.723876953125}, {"ppx", 318.74200439453125}, {"ppy", 251.1754455664062},
        {"distortion_model", "plumb_bob"},
        {"k1", 0.0}, {"k2", 0.0}, {"k3", 0.0}, {"p1", 0.0}, {"p2", 0.0}
    };
}

static json camera_info_to_intrinsic_json(const sensor_msgs::msg::CameraInfo::SharedPtr& msg) {
    if (!msg || msg->k.size() < 9 || msg->k[0] == 0.0) return json();
    if (msg->d.size() < 5) return json();
    return {
        {"fx", msg->k[0]}, {"fy", msg->k[4]}, {"ppx", msg->k[2]}, {"ppy", msg->k[5]},
        {"distortion_model", msg->distortion_model.empty() ? "plumb_bob" : msg->distortion_model},
        {"k1", msg->d[0]}, {"k2", msg->d[1]}, {"k3", msg->d[4]}, {"p1", msg->d[2]}, {"p2", msg->d[3]}
    };
}

static sensor_msgs::msg::CameraInfo::SharedPtr fetch_camera_info_once(
    rclcpp::Node::SharedPtr node, const std::string& topic, rclcpp::Logger logger) {
    auto prom = std::make_shared<std::promise<sensor_msgs::msg::CameraInfo::SharedPtr>>();
    auto fut = prom->get_future();
    auto sub = node->create_subscription<sensor_msgs::msg::CameraInfo>(
        topic, 10, [prom](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
            try {
                prom->set_value(msg);
            } catch (...) {}
        });
    auto status = fut.wait_for(std::chrono::seconds(5));
    if (status == std::future_status::ready) {
        return fut.get();
    }
    RCLCPP_WARN(logger, "获取 camera_info 超时: %s", topic.c_str());
    return nullptr;
}

static const std::map<std::string, std::string> DEFAULT_SN = {
    {"hand_right", "230322270825"},
    {"hand_left", "230422271369"},
    {"head", "213622073652"}
};


void init_parameters(const std::map<std::string, fs::path>& dirs, rclcpp::Node::SharedPtr node) {

    static std::map<std::string, json> cached_intrinsics;
    static std::map<std::string, std::string> cached_serial_numbers;
    static bool cache_populated = false;
    static bool sn_cache_populated = false;

    auto logger = node ? node->get_logger() : rclcpp::get_logger("utils_config");

    if (!cache_populated) {
        if (!node) {
            RCLCPP_WARN(logger, "node 为空,使用默认内参");
        }
        for (const auto& [cam_name, topic] : CAM_INFO_TOPIC_MAP) {
            auto msg = node ? fetch_camera_info_once(node, topic, logger) : nullptr;
            json intrinsic;
            if (msg) {
                intrinsic = camera_info_to_intrinsic_json(msg);
            }
            if (intrinsic.empty()) {
                if (cam_name == "head") {
                    intrinsic = get_default_d435_intrinsic();
                } else if (cam_name == "hand_left") {
                    intrinsic = get_default_hand_left_intrinsic();
                } else {
                    intrinsic = get_default_hand_right_intrinsic();
                }
                RCLCPP_WARN(logger, "使用默认内参: %s", cam_name.c_str());
            }
            cached_intrinsics[cam_name] = intrinsic;
        }
        cache_populated = true;
    }

    if (!sn_cache_populated) {
        fs::path script_path;
        const char* env_path = std::getenv("CAMERA_SN_SCRIPT");
        if (env_path && env_path[0]) {
            script_path = fs::path(env_path);
        } else if (node) {
            try {
                node->declare_parameter("camera_sn_script_path", "");
                std::string param_path = node->get_parameter("camera_sn_script_path").as_string();
                if (!param_path.empty()) script_path = fs::path(param_path);
            } catch (...) {}
        }
        if (!script_path.empty() && fs::exists(script_path)) {
            auto sn_map = parse_serial_numbers_from_script(script_path, logger);
            for (const auto& [cam_name, sn] : sn_map) {
                cached_serial_numbers[cam_name] = sn;
            }
        }
        for (const auto& [cam_name, default_sn] : DEFAULT_SN) {
            if (cached_serial_numbers.find(cam_name) == cached_serial_numbers.end()) {
                cached_serial_numbers[cam_name] = default_sn;
                RCLCPP_WARN(logger, "SN 使用默认值: %s -> %s", cam_name.c_str(), default_sn.c_str());
            }
        }
        sn_cache_populated = true;
    }

    std::map<std::string, json> fixed_cam_configs = {
        {"hand_right", {{"manufacturer", "Intel"}, {"mode", "Realsense D405"}, {"SN", cached_serial_numbers["hand_right"]}, {"intrinsic", cached_intrinsics["hand_right"]}}},
        {"hand_left", {{"manufacturer", "Intel"}, {"mode", "Realsense D405"}, {"SN", cached_serial_numbers["hand_left"]}, {"intrinsic", cached_intrinsics["hand_left"]}}},
        {"head", {{"manufacturer", "Intel"}, {"mode", "Realsense D435"}, {"SN", cached_serial_numbers["head"]}, {"intrinsic", cached_intrinsics["head"]}}}
    };

    for (auto const& [cam_name, config] : fixed_cam_configs) {
        auto path = dirs.at("params_camera") / (cam_name + ".json");
        if (fs::exists(path)) continue;

        json extrinsic = {
            {"rotation_matrix", {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}},
            {"translation_vector", {0.0, 0.0, 1.0}}
        };
        json full_config = {
            {"manufacturer", config["manufacturer"]},
            {"mode", config["mode"]},
            {"SN", config["SN"]},
            {"fps", std::to_string(get_camera_param(cam_name, "color", "fps"))},
            {"width", std::to_string(get_camera_param(cam_name, "color", "width"))},
            {"height", std::to_string(get_camera_param(cam_name, "color", "height"))},
            {"intrinsic", config["intrinsic"]},
            {"extrinsic", extrinsic}
        };

        std::ofstream f(path);
        f << full_config.dump(4);
    }

    auto joint_desc_path = dirs.at("params_hardware") / "Joint_data_description.json";
    if (!fs::exists(joint_desc_path)) {
        json joint_desc = {
            {"state/arm/position/1", {-3.14159, 3.14159}}, {"state/arm/position/2", {-1.57080, 1.57080}},
            {"state/arm/position/3", {-3.14159, 3.14159}}, {"state/arm/position/4", {-2.09440, 2.09440}},
            {"state/arm/position/5", {-3.14159, 3.14159}}, {"state/arm/position/6", {-1.43117, 1.43117}},
            {"state/arm/position/7", {-0.43633, 2.26893}}, {"state/arm/position/8", {-3.14159, 3.14159}},
            {"state/arm/position/9", {-1.57080, 1.57080}}, {"state/arm/position/10", {-3.14159, 3.14159}},
            {"state/arm/position/11", {-2.09440, 2.09440}}, {"state/arm/position/12", {-3.14159, 3.14159}},
            {"state/arm/position/13", {-1.43117, 1.43117}}, {"state/arm/position/14", {-0.43633, 2.26893}},
            {"state/arm", 100}, {"state/robot", 100}, {"action/robot", 100}
        };
        std::ofstream f(joint_desc_path);
        f << joint_desc.dump(4);
    }

    auto effector_desc_path = dirs.at("params_hardware") / "Robot_end_effector_description.json";
    if (!fs::exists(effector_desc_path)) {
        json effector_desc = {
            {"left_gripper", {{"type", "gripper"}, {"category", "binary"}, {"unit", "radian"}}},
            {"right_gripper", {{"type", "gripper"}, {"category", "binary"}, {"unit", "radian"}}}
        };
        std::ofstream f(effector_desc_path);
        f << effector_desc.dump(4);
    }

}

std::map<std::string, fs::path> init_task_directories(const fs::path& root_dir) {
    std::map<std::string, fs::path> dirs;
    dirs["root"] = root_dir;
    auto record_dir = root_dir / "record";
    fs::create_directories(record_dir);
    dirs["raw_joints_h5"] = record_dir / "raw_joints.h5";
    dirs["meta_info"] = root_dir / "meta_info.json";

    dirs["camera_root"] = root_dir / "camera";
    for (auto const& [cam_name, topics] : ROS2_CAMERA_TOPIC_MAP) {
        dirs["cam_" + cam_name + "_color"] = dirs["camera_root"] / cam_name / "color";
        dirs["cam_" + cam_name + "_depth"] = dirs["camera_root"] / cam_name / "depth";
        fs::create_directories(dirs["cam_" + cam_name + "_color"]);
        fs::create_directories(dirs["cam_" + cam_name + "_depth"]);
    }

    dirs["params_root"] = root_dir / "parameters";
    dirs["params_hardware"] = dirs["params_root"] / "hardware";
    dirs["params_camera"] = dirs["params_root"] / "camera";
    fs::create_directories(dirs["params_hardware"]);
    fs::create_directories(dirs["params_camera"]);

    return dirs;
}

void create_h5_structure(const fs::path& h5_path, const std::string& robot_type) {
    using namespace H5;
    if (fs::exists(h5_path)) {
        throw std::runtime_error("raw_joints.h5已存在: " + h5_path.string());
    }

    H5File file(h5_path.string(), H5F_ACC_TRUNC);
    auto robot_cfg = ROBOT_CONFIGS.at(robot_type);

    auto create_extendable_ds = [&](Group& group, const std::string& name, const std::vector<hsize_t>& dims, const DataType& type) {
        std::vector<hsize_t> max_dims = dims;
        max_dims[0] = H5S_UNLIMITED;
        DataSpace dataspace(dims.size(), dims.data(), max_dims.data());
        
        DSetCreatPropList prop;
        std::vector<hsize_t> chunk_dims = dims;
        if (chunk_dims[0] == 0) chunk_dims[0] = 1;
        prop.setChunk(chunk_dims.size(), chunk_dims.data());
        
        group.createDataSet(name, type, dataspace, prop);
    };

    auto set_str_attr = [&](Group& group, const std::string& name, const std::vector<std::string>& values) {
        hsize_t attr_dims[1] = { values.size() };
        DataSpace attr_space(1, attr_dims);
        StrType str_type(PredType::C_S1, H5T_VARIABLE);
        Attribute attr = group.createAttribute(name, str_type, attr_space);
        
        std::vector<const char*> c_strs;
        for (const auto& s : values) c_strs.push_back(s.c_str());
        attr.write(str_type, c_strs.data());
    };

    auto create_group_structure = [&](Group& parent) {
        Group arm_grp = parent.createGroup("arm");
        create_extendable_ds(arm_grp, "timestamp", {0, 1}, PredType::NATIVE_INT64);
        create_extendable_ds(arm_grp, "effort", {0, 14}, PredType::NATIVE_DOUBLE);
        create_extendable_ds(arm_grp, "velocity", {0, 14}, PredType::NATIVE_DOUBLE);
        create_extendable_ds(arm_grp, "position", {0, 14}, PredType::NATIVE_DOUBLE);
        
        std::vector<std::string> arm_names = robot_cfg.joint_names.at("left_arm");
        auto right_names = robot_cfg.joint_names.at("right_arm");
        arm_names.insert(arm_names.end(), right_names.begin(), right_names.end());
        set_str_attr(arm_grp, "name", arm_names);

        Group end_grp = parent.createGroup("end");
        create_extendable_ds(end_grp, "timestamp", {0, 1}, PredType::NATIVE_INT64);
        create_extendable_ds(end_grp, "velocity", {0, 2, 3}, PredType::NATIVE_DOUBLE);
        create_extendable_ds(end_grp, "angular", {0, 2, 3}, PredType::NATIVE_DOUBLE);
        create_extendable_ds(end_grp, "position", {0, 2, 3}, PredType::NATIVE_DOUBLE);
        create_extendable_ds(end_grp, "orientation", {0, 2, 4}, PredType::NATIVE_DOUBLE);
        set_str_attr(end_grp, "name", robot_cfg.end_names);

        Group eff_grp = parent.createGroup("effector");
        create_extendable_ds(eff_grp, "timestamp", {0, 1}, PredType::NATIVE_INT64);
        create_extendable_ds(eff_grp, "force", {0, 2}, PredType::NATIVE_DOUBLE);
        create_extendable_ds(eff_grp, "position", {0, 2}, PredType::NATIVE_DOUBLE);
        
        std::vector<std::string> eff_names;
        std::vector<std::string> eff_cats;
        for (auto const& [name, cfg] : robot_cfg.gripper) {
            eff_names.push_back(name);
            eff_cats.push_back(cfg.at("category"));
        }
        set_str_attr(eff_grp, "name", eff_names);
        set_str_attr(eff_grp, "category", eff_cats);

        Group rob_grp = parent.createGroup("robot");
        create_extendable_ds(rob_grp, "timestamp", {0, 1}, PredType::NATIVE_INT64);
        create_extendable_ds(rob_grp, "velocity", {0, 3}, PredType::NATIVE_DOUBLE);
        create_extendable_ds(rob_grp, "angular", {0, 3}, PredType::NATIVE_DOUBLE);
        create_extendable_ds(rob_grp, "position", {0, 3}, PredType::NATIVE_DOUBLE);
        create_extendable_ds(rob_grp, "orientation", {0, 4}, PredType::NATIVE_DOUBLE);
        set_str_attr(rob_grp, "name", robot_cfg.robot_base_name);
    };

    Group state_grp = file.createGroup("state");
    create_group_structure(state_grp);

    Group action_grp = file.createGroup("action");
    create_group_structure(action_grp);
}

} // namespace my_dual_arm_package
