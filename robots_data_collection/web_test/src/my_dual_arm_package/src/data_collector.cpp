#include "my_dual_arm_package/data_collector.hpp"
#include <fstream>
#include <H5Cpp.h>  
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_storage/topic_metadata.hpp>

namespace my_dual_arm_package {

std::shared_ptr<DoubleArmRGBDDataProcessor> DoubleArmRGBDDataProcessor::_instance = nullptr;
std::mutex DoubleArmRGBDDataProcessor::_instance_lock;

DoubleArmRGBDDataProcessor::DoubleArmRGBDDataProcessor(const std::string& robot_type, const std::vector<std::string>& preview_camera_names)
    : is_preview_mode(!preview_camera_names.empty()), is_recording(false), robot_type(robot_type), is_recording_thread_running(false), stop_executor_flag(false) {
    
    auto robot_cfg = ROBOT_CONFIGS.at(robot_type);
    camera_names = is_preview_mode ? preview_camera_names : robot_cfg.cameras;
    frequency = !camera_names.empty()
        ? static_cast<double>(get_camera_param(camera_names[0], "color", "fps"))
        : 30.0;
    record_interval = 1.0 / frequency;

    for (const auto& cam : ROS2_CAMERA_TOPIC_MAP) camera_frames_count[cam.first] = 0;
    joint_frames_count = {{"left", 0}, {"right", 0}};

    camera_sub = std::make_shared<ROS2RGBDCameraSubscriber>(camera_names, is_preview_mode);
    executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
    executor->add_node(camera_sub);
    executor_thread = std::thread(&DoubleArmRGBDDataProcessor::_spin_executor, this);

    if (!is_preview_mode) {
        arm_interface = std::make_shared<ROS2DoubleArmInterface>(robot_type, is_preview_mode);
        arm_executor = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
        arm_executor->add_node(arm_interface);
        arm_executor_thread = std::thread(&DoubleArmRGBDDataProcessor::_spin_arm_executor, this);
    }

    camera_sub->start();
    
    RCLCPP_INFO(camera_sub->get_logger(), "双臂RGBD数据处理器初始化完成(单例模式)");
}

void DoubleArmRGBDDataProcessor::_spin_executor() {
    while (rclcpp::ok() && !stop_executor_flag) {
        try {
            executor->spin();
        } catch (...) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

void DoubleArmRGBDDataProcessor::_spin_arm_executor() {
    while (rclcpp::ok() && !stop_executor_flag) {
        try {
            arm_executor->spin();
        } catch (...) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

std::shared_ptr<DoubleArmRGBDDataProcessor> DoubleArmRGBDDataProcessor::get_instance(const std::string& robot_type, const std::vector<std::string>& preview_camera_names) {
    std::lock_guard<std::mutex> lock(_instance_lock);
    if (!_instance) {
        _instance = std::shared_ptr<DoubleArmRGBDDataProcessor>(new DoubleArmRGBDDataProcessor(robot_type, preview_camera_names));
    } else {
        bool new_is_preview = !preview_camera_names.empty();
        auto robot_cfg = ROBOT_CONFIGS.at(robot_type);
        std::vector<std::string> new_camera_names = new_is_preview ? preview_camera_names : robot_cfg.cameras;
        
        if (new_camera_names != _instance->camera_names) {
            RCLCPP_INFO(_instance->camera_sub->get_logger(), "单例复用: 相机列表发生变化, 重新初始化订阅器");
            _instance->executor->remove_node(_instance->camera_sub);

            if (_instance->camera_sub->running) {
                _instance->camera_sub->stop(true, false);  
            }
            _instance->camera_sub.reset();
            _instance->camera_names = new_camera_names;
            _instance->camera_sub = std::make_shared<ROS2RGBDCameraSubscriber>(new_camera_names, new_is_preview);
            _instance->executor->add_node(_instance->camera_sub);
        }
        _instance->is_preview_mode = new_is_preview;

        if (!new_is_preview && !_instance->arm_interface) {
            RCLCPP_INFO(_instance->camera_sub->get_logger(), "单例复用: 采集模式缺失机械臂接口，开始补全初始化");
            _instance->arm_interface = std::make_shared<ROS2DoubleArmInterface>(robot_type, new_is_preview);
            _instance->arm_executor = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
            _instance->arm_executor->add_node(_instance->arm_interface);
            _instance->arm_executor_thread = std::thread(&DoubleArmRGBDDataProcessor::_spin_arm_executor, _instance.get());
        }
    }
    return _instance;
}

std::shared_ptr<DoubleArmRGBDDataProcessor> DoubleArmRGBDDataProcessor::get_current_instance() {
    std::lock_guard<std::mutex> lock(_instance_lock);
    return _instance;
}

void DoubleArmRGBDDataProcessor::reset_instance() {
    std::lock_guard<std::mutex> lock(_instance_lock);
    if (_instance) {
        try {
            _instance->close();
        } catch (const std::exception& e) {
            RCLCPP_ERROR(rclcpp::get_logger("data_collector"), "reset_instance: close异常 %s", e.what());
        } catch (...) {
            RCLCPP_ERROR(rclcpp::get_logger("data_collector"), "reset_instance: close未知异常");
        }
    }
    _instance.reset();
}


void DoubleArmRGBDDataProcessor::_start_record_async_task() {
    camera_frames_count.clear();
    for (const auto& cam : camera_names) camera_frames_count[cam] = 0;
    joint_frames_count = {{"left", 0}, {"right", 0}};
    is_recording = true;
    is_recording_thread_running = true;
    record_thread = std::thread(&DoubleArmRGBDDataProcessor::_async_record_loop, this);
    RCLCPP_INFO(camera_sub->get_logger(), "异步采集任务已启动(帧率:%.0fHz, 间隔:%.3f秒)", frequency, record_interval);
}

void DoubleArmRGBDDataProcessor::_stop_record_async_task() {
    is_recording_thread_running = false;
    is_recording = false;
    if (record_thread.joinable()) record_thread.join();
    RCLCPP_INFO(camera_sub->get_logger(), "异步采集任务已停止");
}

void DoubleArmRGBDDataProcessor::_async_record_loop() {
    RCLCPP_INFO(camera_sub->get_logger(), "采集循环: 已进入循环");
    while (is_recording_thread_running) {
        auto start_time = std::chrono::steady_clock::now();
        
        if (is_preview_mode) {
            RCLCPP_INFO(camera_sub->get_logger(), "预览模式: 退出采集循环");
            break;
        }

        _record_frame();
        
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        auto sleep_time = std::chrono::duration<double>(record_interval) - elapsed;
        if (sleep_time.count() > 0) {
            std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::microseconds>(sleep_time));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    RCLCPP_INFO(camera_sub->get_logger(), "采集循环已退出");
}


bool DoubleArmRGBDDataProcessor::start_episode(const std::string& /*temp_folder*/, const std::string& data_folder) {
    std::lock_guard<std::mutex> lk(lock);
    if (is_recording) {
        RCLCPP_WARN(camera_sub->get_logger(), "当前周期未结束,无法启动新周期");
        return false;
    }

    root_dir = fs::path(data_folder);
    dirs = init_task_directories(root_dir);
    RCLCPP_INFO(camera_sub->get_logger(), "目录初始化完成: %s", root_dir.string().c_str());

    use_mcap_mode_ = camera_sub->get_parameter("use_mcap").as_bool();
    bool save_depth = camera_sub->get_parameter("save_depth").as_bool();

    if (use_mcap_mode_) {

        auto writer = std::make_shared<rosbag2_cpp::Writer>();
        rosbag2_storage::StorageOptions storage_opts;
        storage_opts.uri        = (root_dir / "raw_bag").string();
        storage_opts.storage_id = "mcap";
        writer->open(storage_opts);

        auto reg_topic = [&](const std::string& topic, const std::string& type) {
            rosbag2_storage::TopicMetadata meta;
            meta.name                 = topic;
            meta.type                 = type;
            meta.serialization_format = "cdr";
            writer->create_topic(meta);
        };
        for (const auto& [cam_name, topics] : ROS2_CAMERA_TOPIC_MAP) {
            reg_topic(topics.at("color_compressed"), "sensor_msgs/msg/CompressedImage");
            if (save_depth) reg_topic(topics.at("depth_compressed"), "sensor_msgs/msg/CompressedImage");
        }
        reg_topic("/left_data",  "sc_ros2/msg/ArmData");
        reg_topic("/right_data", "sc_ros2/msg/ArmData");

        auto queue = std::make_shared<McapWriteQueue>(writer);
        bag_queue_ = queue;

        std::atomic_store(&camera_sub->bag_queue, queue);
        if (arm_interface) std::atomic_store(&arm_interface->bag_queue, queue);

        RCLCPP_INFO(camera_sub->get_logger(),
            "McapWriteQueue 已启动: %s/raw_bag [save_depth=%s]",
            root_dir.string().c_str(), save_depth ? "true" : "false");

        camera_sub->clear_queues();
        camera_sub->start();
        init_meta_info(dirs, camera_names);
        init_parameters(dirs, camera_sub);

        camera_sub->stop_receiving      = false;
        camera_sub->is_recording_active = false;
        camera_sub->is_preview_active   = true;
        is_recording = true;

        episode_start_ros_time = arm_interface
            ? arm_interface->get_clock()->now().seconds()
            : camera_sub->now().seconds();

        RCLCPP_INFO(camera_sub->get_logger(),
            "mcap episode 开始 (进程内写入, 零 DDS 延迟, 独立写线程)");
        return true;
    }

    if (arm_interface) {
        arm_interface->reset_cache();
        arm_interface->start_arm_recording();
    }
    
    camera_sub->clear_queues();
    camera_sub->start(dirs);
    
    init_meta_info(dirs, camera_names);
    init_parameters(dirs, camera_sub);

    if (!is_preview_mode) {
        camera_sub->stop_receiving = false;
        camera_sub->is_recording_active = true;
        camera_sub->is_preview_active = true;
        RCLCPP_INFO(camera_sub->get_logger(), "已激活相机 Record + Preview 处理流");
    } else {
        camera_sub->stop_receiving = false;
        camera_sub->is_preview_active = true;
        RCLCPP_INFO(camera_sub->get_logger(), "预览模式: 仅激活 Preview 处理流");
    }

    if (!is_preview_mode && arm_interface) {
        episode_start_ros_time = arm_interface->get_clock()->now().seconds();
    } else {
        episode_start_ros_time = camera_sub->now().seconds();
    }
    RCLCPP_INFO(camera_sub->get_logger(), "采集开始时间记录完成: %.3fs", episode_start_ros_time);

    cam_episode_data.clear();
    arm_episode_data.clear();
    _start_record_async_task();
    
    return true;
}


bool DoubleArmRGBDDataProcessor::_record_frame() {
    if (!camera_sub) return false;

    auto [frames, ts] = camera_sub->get_frames(is_preview_mode);
    if (frames.empty()) return false;

    for (const auto& cam_name : camera_names) {
        if (frames.find(cam_name) == frames.end()) return false;
        if (frames[cam_name].find("color") == frames[cam_name].end() || 
            frames[cam_name].find("depth") == frames[cam_name].end()) return false;
    }

    if (!is_preview_mode) {
        for (const auto& cam_name : camera_names) {
            camera_sub->save_unified_frame(cam_name, "color", frames[cam_name]["color"], ts[cam_name]["color"]);
            camera_sub->save_unified_frame(cam_name, "depth", frames[cam_name]["depth"], ts[cam_name]["depth"]);
        }
    }

    json cam_data;
    cam_data["sync_stats"] = {{"total_frames", (int)cam_episode_data.size() + 1}};
    json timestamps;
    for (const auto& cam_name : camera_names) {
        timestamps[cam_name] = {{"color", ts[cam_name]["color"]}, {"depth", ts[cam_name]["depth"]}};
    }
    cam_data["camera_timestamps"] = timestamps;
    cam_episode_data.push_back(cam_data);

    std::lock_guard<std::mutex> lk(lock);
    for (const auto& cam_name : camera_names) {
        camera_frames_count[cam_name]++;
    }
    if (arm_interface) {
        joint_frames_count["left"] = arm_interface->joint_frames_count["left"];
        joint_frames_count["right"] = arm_interface->joint_frames_count["right"];
    }

    return true;
}


void DoubleArmRGBDDataProcessor::end_episode() {

    {
        std::lock_guard<std::mutex> lk(lock);
        if (!is_recording) {
            RCLCPP_WARN(camera_sub->get_logger(), "无活跃采集周期,无需结束");
            return;
        }
    }

    if (use_mcap_mode_) {

        auto null_q = std::shared_ptr<McapWriteQueue>{};
        if (camera_sub) std::atomic_store(&camera_sub->bag_queue, null_q);
        if (arm_interface) std::atomic_store(&arm_interface->bag_queue, null_q);

        if (bag_queue_) {
            bag_queue_->stop();
            bag_queue_.reset();
        }
        RCLCPP_INFO(camera_sub->get_logger(),
            "mcap bag 已关闭并刷盘: %s/raw_bag/", root_dir.string().c_str());

        if (camera_sub) {
            camera_sub->stop_receiving    = true;
            camera_sub->is_preview_active = false;
            camera_sub->stop(is_preview_mode, false);
        }
        is_recording   = false;
        use_mcap_mode_ = false;
        return;
    }

    if (camera_sub) {
        camera_sub->stop_receiving = true;
        RCLCPP_INFO(camera_sub->get_logger(), "已置位 stop_receiving,相机停止接收新帧");
    }

    double episode_end_ros_time;
    if (!is_preview_mode && arm_interface) {
        episode_end_ros_time = arm_interface->get_clock()->now().seconds();
        RCLCPP_INFO(camera_sub->get_logger(), "end_episode: 步骤2 - 停止机械臂采集并获取数据");
        arm_episode_data = arm_interface->stop_arm_recording();
        RCLCPP_INFO(camera_sub->get_logger(), "获取机械臂100Hz数据共%zu帧", arm_episode_data.size());
    } else {
        episode_end_ros_time = camera_sub->now().seconds();
    }

    if (camera_sub) {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        camera_sub->is_recording_active = false;
        camera_sub->is_preview_active = false;
        RCLCPP_INFO(camera_sub->get_logger(), "已关闭录制标志,后续ROS回调不会再将帧放入record队列");
    }

    _stop_record_async_task();
    RCLCPP_INFO(camera_sub->get_logger(), "采集循环已停止");

    RCLCPP_INFO(camera_sub->get_logger(), "end_episode: 步骤4 - 执行相机数据排空");
    camera_sub->stop(is_preview_mode, false);
    RCLCPP_INFO(camera_sub->get_logger(), "end_episode: 相机数据排空完成,订阅保持运行");

    bool h5_success = false;
    if (!is_preview_mode) {
        RCLCPP_INFO(camera_sub->get_logger(), "end_episode: 步骤5 - 开始H5/TXT数据写入");
        // _write_h5_data();
        _write_arm_data_txt(); 
        h5_success = true;
        RCLCPP_INFO(camera_sub->get_logger(), "end_episode: H5/TXT写入完成");
    } else {
        RCLCPP_INFO(camera_sub->get_logger(), "end_episode: 预览模式,跳过H5/TXT写入");
    }

    int duration_ms = static_cast<int>((episode_end_ros_time - episode_start_ros_time) * 1000);
    RCLCPP_INFO(camera_sub->get_logger(), "采集时长: %dms", duration_ms);

    RCLCPP_INFO(camera_sub->get_logger(), "end_episode: 步骤7 - 更新meta_info.json");
    auto meta_path = dirs["meta_info"];
    if (fs::exists(meta_path)) {
        try {
            std::ifstream f_in(meta_path);
            json meta_data = json::parse(f_in);
            f_in.close();

            meta_data["durationInMs"] = duration_ms;
            meta_data["h5_write_success"] = h5_success;

            json camera_details = json::object();
            int total_camera_frames = 0;
            for (const auto& name : camera_names) {

                int color_cnt = camera_sub->get_frame_count(name, "color");
                int final_count = color_cnt;
                camera_details[name] = final_count;
                total_camera_frames += final_count;

                camera_frames_count[name] = final_count;
            }
            meta_data["camera_frame_details"] = camera_details;
            meta_data["camera_frame_total_count"] = total_camera_frames;

            meta_data["arm_frame_details"] = {
                {"left_arm", arm_episode_data.size()},
                {"right_arm", arm_episode_data.size()}
            };
            meta_data["arm_frame_total_count"] = arm_episode_data.size() * 2;

            std::ofstream f_out(meta_path);
            f_out << meta_data.dump(4);
            RCLCPP_INFO(camera_sub->get_logger(), "meta_info更新完成");
        } catch (const std::exception& e) {
            RCLCPP_ERROR(camera_sub->get_logger(), "meta_info更新失败: %s", e.what());
        }
    }

    RCLCPP_INFO(camera_sub->get_logger(),
        "=== 采集周期结束统计 ===\n"
        "  采集时长: %dms\n"
        "  H5写入: %s\n"
        "  机械臂帧数: %zu",
        duration_ms, h5_success ? "成功" : "失败", arm_episode_data.size());

    is_recording = false;
}

void DoubleArmRGBDDataProcessor::_write_arm_data_txt() {
    if (is_preview_mode || arm_episode_data.empty()) {
        RCLCPP_INFO(camera_sub->get_logger(), "预览模式或无机械臂数据,跳过TXT写入");
        return;
    }

    auto record_dir = dirs["root"] / "record";
    fs::create_directories(record_dir);
    
    std::ofstream f_left(record_dir / "left_data.txt");
    std::ofstream f_right(record_dir / "right_data.txt");

    for (const auto& frame : arm_episode_data) {
        int64_t ts = frame["timestamp"];
        auto state = frame["arm_state"];
        
        auto format_data = [](const json& s, const std::string& prefix) {
            std::stringstream ss;
            auto joints = s[prefix + "_arm_joints"].get<std::vector<double>>();
            auto gripper = s[prefix + "_gripper"].get<std::vector<double>>();
            auto pos = s[prefix + "_end_position"].get<std::vector<double>>();
            auto ori = s[prefix + "_end_orientation"].get<std::vector<double>>();
            
            for (size_t i = 0; i < joints.size(); ++i) ss << joints[i] << ",";
            for (size_t i = 0; i < gripper.size(); ++i) ss << gripper[i] << ",";
            for (size_t i = 0; i < pos.size(); ++i) ss << pos[i] << ",";
            for (size_t i = 0; i < ori.size(); ++i) ss << ori[i] << (i == ori.size() - 1 ? "" : ",");
            return ss.str();
        };

        f_left << ts << " " << format_data(state, "left") << "\n";
        f_right << ts << " " << format_data(state, "right") << "\n";
    }
}

void DoubleArmRGBDDataProcessor::_write_h5_data() {
    if (is_preview_mode || arm_episode_data.empty()) {
        RCLCPP_INFO(camera_sub->get_logger(), "预览模式或无机械臂数据,跳过H5写入");
        return;
    }

    std::string h5_path = dirs["raw_joints_h5"].string();
    size_t total_frames = arm_episode_data.size();
    RCLCPP_INFO(camera_sub->get_logger(), "开始H5写入(共%zu帧)", total_frames);
    
    try {
        using namespace H5;
        create_h5_structure(h5_path, robot_type);
        H5File file(h5_path, H5F_ACC_RDWR);

        auto write_data = [&](const std::string& group_name, const std::string& dataset_name, const std::vector<double>& data, size_t row_idx, const std::vector<hsize_t>& item_dims) {
            Group group = file.openGroup(group_name);
            DataSet dataset = group.openDataSet(dataset_name);
            DataSpace filespace = dataset.getSpace();
            
            hsize_t count[3];
            count[0] = 1;
            for (size_t i = 0; i < item_dims.size(); ++i) count[i+1] = item_dims[i];

            hsize_t dims[3];
            filespace.getSimpleExtentDims(dims);
            dims[0] = row_idx + 1;
            dataset.extend(dims);

            filespace = dataset.getSpace();
            hsize_t offset[3] = {row_idx, 0, 0};
            filespace.selectHyperslab(H5S_SELECT_SET, count, offset);

            DataSpace memspace(item_dims.size() + 1, count);
            dataset.write(data.data(), PredType::NATIVE_DOUBLE, memspace, filespace);
        };

        auto write_ts = [&](const std::string& group_name, int64_t ts, size_t row_idx) {
            Group group = file.openGroup(group_name);
            DataSet dataset = group.openDataSet("timestamp");
            DataSpace filespace = dataset.getSpace();
            
            hsize_t count[2] = {1, 1};
            hsize_t dims[2];
            filespace.getSimpleExtentDims(dims);
            dims[0] = row_idx + 1;
            dataset.extend(dims);

            filespace = dataset.getSpace();
            hsize_t offset[2] = {row_idx, 0};
            filespace.selectHyperslab(H5S_SELECT_SET, count, offset);

            DataSpace memspace(2, count);
            dataset.write(&ts, PredType::NATIVE_INT64, memspace, filespace);
        };

        for (size_t i = 0; i < total_frames; ++i) {
            const auto& frame = arm_episode_data[i];
            int64_t ts = frame["timestamp"];
            const auto& state = frame["arm_state"];
            bool is_last_frame = (i == total_frames - 1);

            std::vector<double> joints;
            auto l_j = state["left_arm_joints"].get<std::vector<double>>();
            auto r_j = state["right_arm_joints"].get<std::vector<double>>();
            joints.insert(joints.end(), l_j.begin(), l_j.end());
            joints.insert(joints.end(), r_j.begin(), r_j.end());

            write_ts("state/arm", ts, i);
            write_data("state/arm", "position", joints, i, {14});

            auto l_p = state["left_end_position"].get<std::vector<double>>();
            auto r_p = state["right_end_position"].get<std::vector<double>>();
            std::vector<double> pos;
            pos.insert(pos.end(), l_p.begin(), l_p.end());
            pos.insert(pos.end(), r_p.begin(), r_p.end());

            auto l_o = state["left_end_orientation"].get<std::vector<double>>();
            auto r_o = state["right_end_orientation"].get<std::vector<double>>();
            std::vector<double> ori;
            ori.insert(ori.end(), l_o.begin(), l_o.end());
            ori.insert(ori.end(), r_o.begin(), r_o.end());

            write_ts("state/end", ts, i);
            write_data("state/end", "position", pos, i, {2, 3});
            write_data("state/end", "orientation", ori, i, {2, 4});

            auto l_g = state["left_gripper"].get<std::vector<double>>();
            auto r_g = state["right_gripper"].get<std::vector<double>>();
            std::vector<double> state_gripper = {l_g[1], r_g[1]};

            write_ts("state/effector", ts, i);
            write_data("state/effector", "position", state_gripper, i, {2});

            std::vector<double> action_joints;
            std::vector<double> action_pos;
            std::vector<double> action_ori;
            std::vector<double> action_gripper;

            if (!is_last_frame) {

                const auto& next_state = arm_episode_data[i + 1]["arm_state"];
                auto nl_j = next_state["left_arm_joints"].get<std::vector<double>>();
                auto nr_j = next_state["right_arm_joints"].get<std::vector<double>>();
                action_joints.insert(action_joints.end(), nl_j.begin(), nl_j.end());
                action_joints.insert(action_joints.end(), nr_j.begin(), nr_j.end());

                auto nl_p = next_state["left_end_position"].get<std::vector<double>>();
                auto nr_p = next_state["right_end_position"].get<std::vector<double>>();
                action_pos.insert(action_pos.end(), nl_p.begin(), nl_p.end());
                action_pos.insert(action_pos.end(), nr_p.begin(), nr_p.end());

                auto nl_o = next_state["left_end_orientation"].get<std::vector<double>>();
                auto nr_o = next_state["right_end_orientation"].get<std::vector<double>>();
                action_ori.insert(action_ori.end(), nl_o.begin(), nl_o.end());
                action_ori.insert(action_ori.end(), nr_o.begin(), nr_o.end());

                auto nl_g = next_state["left_gripper"].get<std::vector<double>>();
                auto nr_g = next_state["right_gripper"].get<std::vector<double>>();
                action_gripper = {nl_g[0], nr_g[0]};
            } else {

                action_joints = joints;
                action_pos = pos;
                action_ori = ori;
                action_gripper = {l_g[0], r_g[0]};
            }

            write_ts("action/arm", ts, i);
            write_data("action/arm", "position", action_joints, i, {14});

            write_ts("action/end", ts, i);
            write_data("action/end", "position", action_pos, i, {2, 3});
            write_data("action/end", "orientation", action_ori, i, {2, 4});

            write_ts("action/effector", ts, i);
            write_data("action/effector", "position", action_gripper, i, {2});
        }
        RCLCPP_INFO(camera_sub->get_logger(), "H5写入完成(%zu帧), 文件路径: %s", total_frames, h5_path.c_str());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(camera_sub->get_logger(), "H5写入失败: %s", e.what());
    }
}

void DoubleArmRGBDDataProcessor::data_postprocess(const std::string& temp_folder, const std::string& data_folder) {
    fs::path temp_path(temp_folder);
    fs::path data_path(data_folder);
    if (!fs::exists(temp_path) || !fs::exists(data_path)) {
        throw std::runtime_error("临时文件夹或数据文件夹不存在");
    }

    auto copy_options = fs::copy_options::overwrite_existing | fs::copy_options::recursive;
    
    if (fs::exists(temp_path / "camera")) {
        fs::create_directories(data_path / "camera");
        fs::copy(temp_path / "camera", data_path / "camera", copy_options);
    }

    if (fs::exists(temp_path / "record/raw_joints.h5")) {
        fs::copy(temp_path / "record/raw_joints.h5", data_path / "record/raw_joints.h5", copy_options);
    }
}

void DoubleArmRGBDDataProcessor::recover() {
    std::lock_guard<std::mutex> lk(lock);
    RCLCPP_INFO(camera_sub->get_logger(), "RECOVER: 系统状态恢复中...");
    if (arm_interface) arm_interface->reset_cache();
    if (camera_sub && camera_sub->running) camera_sub->stop(true, false);
    is_recording = false;
    RCLCPP_INFO(camera_sub->get_logger(), "RECOVER: 系统状态恢复完成");
}


void DoubleArmRGBDDataProcessor::close() {
    if (!camera_sub) return;
    RCLCPP_INFO(camera_sub->get_logger(), "开始释放双臂RGBD数据处理器资源...");

    _stop_record_async_task();

    stop_executor_flag = true;
    if (executor) {
        executor->cancel();
    }
    if (arm_executor) {
        arm_executor->cancel();
    }

    if (executor_thread.joinable()) {
        executor_thread.join();
        RCLCPP_INFO(camera_sub->get_logger(), "主执行器线程已退出");
    }
    if (arm_executor_thread.joinable()) {
        arm_executor_thread.join();
        RCLCPP_INFO(camera_sub->get_logger(), "机械臂执行器线程已退出");
    }
    
    RCLCPP_INFO(camera_sub->get_logger(), "双臂RGBD数据处理器资源释放完成");
}

} // namespace my_dual_arm_package
