#include "my_dual_arm_package/data_collector.hpp"
#include <fstream>
#include <H5Cpp.h>  // HDF5 C++ API (H5File, Group, DataSet, DataSpace等)
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_storage/topic_metadata.hpp>

namespace my_dual_arm_package {

// ==============================
// 单例模式静态成员初始化
// ==============================
std::shared_ptr<DoubleArmRGBDDataProcessor> DoubleArmRGBDDataProcessor::_instance = nullptr;
std::mutex DoubleArmRGBDDataProcessor::_instance_lock;

/// 构造函数: 初始化相机订阅器、机械臂接口、ROS2执行器
DoubleArmRGBDDataProcessor::DoubleArmRGBDDataProcessor(const std::string& robot_type, const std::vector<std::string>& preview_camera_names)
    : is_preview_mode(!preview_camera_names.empty()), is_recording(false), robot_type(robot_type), is_recording_thread_running(false), stop_executor_flag(false) {
    
    auto robot_cfg = ROBOT_CONFIGS.at(robot_type);
    camera_names = is_preview_mode ? preview_camera_names : robot_cfg.cameras;
    frequency = !camera_names.empty()
        ? static_cast<double>(get_camera_param(camera_names[0], "color", "fps"))
        : 30.0;
    record_interval = 1.0 / frequency;

    // 初始化帧计数
    for (const auto& cam : ROS2_CAMERA_TOPIC_MAP) camera_frames_count[cam.first] = 0;
    joint_frames_count = {{"left", 0}, {"right", 0}};

    // 创建相机订阅器并加入多线程执行器
    camera_sub = std::make_shared<ROS2RGBDCameraSubscriber>(camera_names, is_preview_mode);
    executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
    executor->add_node(camera_sub);
    executor_thread = std::thread(&DoubleArmRGBDDataProcessor::_spin_executor, this);

    // 非预览模式: 创建双臂接口并加入单线程执行器
    if (!is_preview_mode) {
        arm_interface = std::make_shared<ROS2DoubleArmInterface>(robot_type, is_preview_mode);
        arm_executor = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
        arm_executor->add_node(arm_interface);
        arm_executor_thread = std::thread(&DoubleArmRGBDDataProcessor::_spin_arm_executor, this);
    }

    // 预启动相机订阅，使 get_camera_status() 在程序启动后即可用
    camera_sub->start();
    
    RCLCPP_INFO(camera_sub->get_logger(), "双臂RGBD数据处理器初始化完成(单例模式)");
}

/// 相机执行器自旋(多线程,崩溃后自动重启)
void DoubleArmRGBDDataProcessor::_spin_executor() {
    while (rclcpp::ok() && !stop_executor_flag) {
        try {
            executor->spin();
        } catch (...) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

/// 双臂独立执行器自旋(单线程高优先级)
void DoubleArmRGBDDataProcessor::_spin_arm_executor() {
    while (rclcpp::ok() && !stop_executor_flag) {
        try {
            arm_executor->spin();
        } catch (...) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

/// 单例获取: 首次创建实例,后续复用/更新配置(相机列表变化时重建订阅器)
std::shared_ptr<DoubleArmRGBDDataProcessor> DoubleArmRGBDDataProcessor::get_instance(const std::string& robot_type, const std::vector<std::string>& preview_camera_names) {
    std::lock_guard<std::mutex> lock(_instance_lock);
    if (!_instance) {
        _instance = std::shared_ptr<DoubleArmRGBDDataProcessor>(new DoubleArmRGBDDataProcessor(robot_type, preview_camera_names));
    } else {
        bool new_is_preview = !preview_camera_names.empty();
        auto robot_cfg = ROBOT_CONFIGS.at(robot_type);
        std::vector<std::string> new_camera_names = new_is_preview ? preview_camera_names : robot_cfg.cameras;
        
        // 相机列表变化时重建订阅器
        if (new_camera_names != _instance->camera_names) {
            RCLCPP_INFO(_instance->camera_sub->get_logger(), "单例复用: 相机列表发生变化, 重新初始化订阅器");
            _instance->executor->remove_node(_instance->camera_sub);
            // 显式调用stop()确保资源正确释放
            if (_instance->camera_sub->running) {
                _instance->camera_sub->stop(true, false);  // 停止但不销毁节点
            }
            _instance->camera_sub.reset();
            _instance->camera_names = new_camera_names;
            _instance->camera_sub = std::make_shared<ROS2RGBDCameraSubscriber>(new_camera_names, new_is_preview);
            _instance->executor->add_node(_instance->camera_sub);
        }
        _instance->is_preview_mode = new_is_preview;

        // 从预览切回采集模式,且缺失机械臂接口时补全创建
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

/// 只读获取当前实例(不触发配置更新,用于 robot_status 等查询)
std::shared_ptr<DoubleArmRGBDDataProcessor> DoubleArmRGBDDataProcessor::get_current_instance() {
    std::lock_guard<std::mutex> lock(_instance_lock);
    return _instance;
}

/// 重置单例实例(RECOVER时调用,释放旧实例以便下次get_instance创建新实例)
/// 必须先调用close()停止线程再销毁,否则析构时joinable线程会导致std::terminate
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

/// 启动异步采集任务(创建采集线程)
void DoubleArmRGBDDataProcessor::_start_record_async_task() {
    // 重置帧数统计
    camera_frames_count.clear();
    for (const auto& cam : camera_names) camera_frames_count[cam] = 0;
    joint_frames_count = {{"left", 0}, {"right", 0}};
    is_recording = true;
    is_recording_thread_running = true;
    record_thread = std::thread(&DoubleArmRGBDDataProcessor::_async_record_loop, this);
    RCLCPP_INFO(camera_sub->get_logger(), "异步采集任务已启动(帧率:%.0fHz, 间隔:%.3f秒)", frequency, record_interval);
}

/// 停止异步采集任务(等待采集线程退出)
void DoubleArmRGBDDataProcessor::_stop_record_async_task() {
    is_recording_thread_running = false;
    is_recording = false;
    if (record_thread.joinable()) record_thread.join();
    RCLCPP_INFO(camera_sub->get_logger(), "异步采集任务已停止");
}

/// 采集循环主体: 按record_interval间隔反复调用_record_frame
void DoubleArmRGBDDataProcessor::_async_record_loop() {
    RCLCPP_INFO(camera_sub->get_logger(), "采集循环: 已进入循环");
    while (is_recording_thread_running) {
        auto start_time = std::chrono::steady_clock::now();
        
        // 预览模式下直接退出循环
        if (is_preview_mode) {
            RCLCPP_INFO(camera_sub->get_logger(), "预览模式: 退出采集循环");
            break;
        }

        _record_frame();
        
        // 动态调整sleep时间,保证总周期接近目标
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        auto sleep_time = std::chrono::duration<double>(record_interval) - elapsed;
        if (sleep_time.count() > 0) {
            std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::microseconds>(sleep_time));
        } else {
            // 压力很大时极短yield,让出CPU
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    RCLCPP_INFO(camera_sub->get_logger(), "采集循环已退出");
}


/// 启动采集周期: 初始化目录、重置缓存、启动相机和机械臂采集
bool DoubleArmRGBDDataProcessor::start_episode(const std::string& /*temp_folder*/, const std::string& data_folder) {
    std::lock_guard<std::mutex> lk(lock);
    if (is_recording) {
        RCLCPP_WARN(camera_sub->get_logger(), "当前周期未结束,无法启动新周期");
        return false;
    }

    // 步骤1: 初始化目录结构
    root_dir = fs::path(data_folder);
    dirs = init_task_directories(root_dir);
    RCLCPP_INFO(camera_sub->get_logger(), "目录初始化完成: %s", root_dir.string().c_str());

    // 读取录制模式参数(由 run_cpp.sh 通过 ROS 参数传入)
    use_mcap_mode_ = camera_sub->get_parameter("use_mcap").as_bool();
    bool save_depth = camera_sub->get_parameter("save_depth").as_bool();

    if (use_mcap_mode_) {
        // ===== 进程内 McapWriteQueue 录制链路 =====
        // 回调线程只推 lambda（捕获 shared_ptr，不拷贝图像数据），
        // 专用写线程串行消费，序列化 + 写盘完全在回调线程之外。

        auto writer = std::make_shared<rosbag2_cpp::Writer>();
        rosbag2_storage::StorageOptions storage_opts;
        storage_opts.uri        = (root_dir / "raw_bag").string();
        storage_opts.storage_id = "mcap";
        writer->open(storage_opts);

        // 注册所有需要录制的 topic（相机 color 固定录，depth 受 save_depth 控制）
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

        // 创建写队列（内含专用写线程，Writer 仅被该线程访问，无额外写锁）
        auto queue = std::make_shared<McapWriteQueue>(writer);
        bag_queue_ = queue;
        // 原子写入，回调线程 atomic_load 读取，无锁可见
        std::atomic_store(&camera_sub->bag_queue, queue);
        if (arm_interface) std::atomic_store(&arm_interface->bag_queue, queue);

        RCLCPP_INFO(camera_sub->get_logger(),
            "McapWriteQueue 已启动: %s/raw_bag [save_depth=%s]",
            root_dir.string().c_str(), save_depth ? "true" : "false");

        // 仅激活预览流（供 WebSocket 推流），不启动落盘采集循环
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

    // ===== 原有落盘链路(完全不变) =====

    // 步骤2: 重置机械臂缓存并启动100Hz采集
    if (arm_interface) {
        arm_interface->reset_cache();
        arm_interface->start_arm_recording();
    }
    
    // 步骤3: 重置相机缓存并启动订阅
    camera_sub->clear_queues();
    camera_sub->start(dirs);
    
    // 步骤4: 初始化元信息和参数文件(放在机械臂和相机启动之后)
    init_meta_info(dirs, camera_names);
    init_parameters(dirs, camera_sub);

    // 步骤5: 开启处理开关(采集模式下预览和录制同时开启)
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

    // 步骤5: 记录采集开始时间
    if (!is_preview_mode && arm_interface) {
        episode_start_ros_time = arm_interface->get_clock()->now().seconds();
    } else {
        episode_start_ros_time = camera_sub->now().seconds();
    }
    RCLCPP_INFO(camera_sub->get_logger(), "采集开始时间记录完成: %.3fs", episode_start_ros_time);

    // 步骤6: 启动异步采集任务
    cam_episode_data.clear();
    arm_episode_data.clear();
    _start_record_async_task();
    
    return true;
}

// 移除_force_first_record函数，不再需要强制首次采集

/// 单次采集: 从相机取帧、校验、保存、记录时间戳
bool DoubleArmRGBDDataProcessor::_record_frame() {
    if (!camera_sub) return false;

    // 1. 获取相机帧(仅处理相机,不涉及机械臂)
    auto [frames, ts] = camera_sub->get_frames(is_preview_mode);
    if (frames.empty()) return false;

    // 2. 相机帧有效性校验
    for (const auto& cam_name : camera_names) {
        if (frames.find(cam_name) == frames.end()) return false;
        if (frames[cam_name].find("color") == frames[cam_name].end() || 
            frames[cam_name].find("depth") == frames[cam_name].end()) return false;
    }

    // 3. 非预览模式: 执行图像保存(color和depth各自使用原始时间戳)
    if (!is_preview_mode) {
        for (const auto& cam_name : camera_names) {
            camera_sub->save_unified_frame(cam_name, "color", frames[cam_name]["color"], ts[cam_name]["color"]);
            camera_sub->save_unified_frame(cam_name, "depth", frames[cam_name]["depth"], ts[cam_name]["depth"]);
        }
    }

    // 4. 记录相机数据(使用各相机自己的时间戳)
    json cam_data;
    cam_data["sync_stats"] = {{"total_frames", (int)cam_episode_data.size() + 1}};
    json timestamps;
    for (const auto& cam_name : camera_names) {
        timestamps[cam_name] = {{"color", ts[cam_name]["color"]}, {"depth", ts[cam_name]["depth"]}};
    }
    cam_data["camera_timestamps"] = timestamps;
    cam_episode_data.push_back(cam_data);

    // 5. 更新帧计数并同步机械臂计数
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

/// 结束采集周期: 停止机械臂、排空缓存、写入H5/TXT、更新meta_info
void DoubleArmRGBDDataProcessor::end_episode() {
    // 步骤0: 仅在锁内检查状态,避免长时间持锁导致与_record_frame死锁
    {
        std::lock_guard<std::mutex> lk(lock);
        if (!is_recording) {
            RCLCPP_WARN(camera_sub->get_logger(), "无活跃采集周期,无需结束");
            return;
        }
    }

    // ===== 进程内 mcap 链路早返回: 停写队列（排空后 Writer 自动关闭）=====
    if (use_mcap_mode_) {
        // 1. 原子置空回调节点的队列指针，新到的回调不再入队
        auto null_q = std::shared_ptr<McapWriteQueue>{};
        if (camera_sub) std::atomic_store(&camera_sub->bag_queue, null_q);
        if (arm_interface) std::atomic_store(&arm_interface->bag_queue, null_q);

        // 2. 停止队列：设 stopped=true → 通知写线程 → 写线程排空剩余任务后 join
        //    join 返回时所有写盘已完成，Writer 析构 → close/flush
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

    // 步骤1: 先停止相机接收新帧,确保相机和机械臂最后时间戳对齐
    // (Python版在handle_stop_telemetry中先停采集循环再end_episode,C++在此处对齐)
    if (camera_sub) {
        camera_sub->stop_receiving = true;
        RCLCPP_INFO(camera_sub->get_logger(), "已置位 stop_receiving,相机停止接收新帧");
    }

    // 步骤2: 停止机械臂采集并获取数据(此时相机已停止接收,时间戳对齐)
    double episode_end_ros_time;
    if (!is_preview_mode && arm_interface) {
        episode_end_ros_time = arm_interface->get_clock()->now().seconds();
        RCLCPP_INFO(camera_sub->get_logger(), "end_episode: 步骤2 - 停止机械臂采集并获取数据");
        arm_episode_data = arm_interface->stop_arm_recording();
        RCLCPP_INFO(camera_sub->get_logger(), "获取机械臂100Hz数据共%zu帧", arm_episode_data.size());
    } else {
        episode_end_ros_time = camera_sub->now().seconds();
    }

    // 步骤3: 等待线程池处理完stop_receiving之前已提交的帧,然后关闭录制标志
    if (camera_sub) {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        camera_sub->is_recording_active = false;
        camera_sub->is_preview_active = false;
        RCLCPP_INFO(camera_sub->get_logger(), "已关闭录制标志,后续ROS回调不会再将帧放入record队列");
    }

    // 步骤4: 停止采集循环(此时不持有lock,_record_frame可正常退出,不会死锁)
    _stop_record_async_task();
    RCLCPP_INFO(camera_sub->get_logger(), "采集循环已停止");

    // 步骤4: 执行相机数据排空(非破坏性停止)
    RCLCPP_INFO(camera_sub->get_logger(), "end_episode: 步骤4 - 执行相机数据排空");
    camera_sub->stop(is_preview_mode, false);
    RCLCPP_INFO(camera_sub->get_logger(), "end_episode: 相机数据排空完成,订阅保持运行");

    // 步骤5: 写入H5和TXT数据
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

    // 步骤6: 计算采集时长
    int duration_ms = static_cast<int>((episode_end_ros_time - episode_start_ros_time) * 1000);
    RCLCPP_INFO(camera_sub->get_logger(), "采集时长: %dms", duration_ms);
    
    // 步骤7: 更新meta_info.json
    RCLCPP_INFO(camera_sub->get_logger(), "end_episode: 步骤7 - 更新meta_info.json");
    auto meta_path = dirs["meta_info"];
    if (fs::exists(meta_path)) {
        try {
            std::ifstream f_in(meta_path);
            json meta_data = json::parse(f_in);
            f_in.close();

            meta_data["durationInMs"] = duration_ms;
            meta_data["h5_write_success"] = h5_success;
            
            // 相机帧数统计
            json camera_details = json::object();
            int total_camera_frames = 0;
            for (const auto& name : camera_names) {
                // 通过公共方法读取最终的 counter 统计
                // 这样可以包含采集循环结束后的“排空”帧数
                int color_cnt = camera_sub->get_frame_count(name, "color");
                
                // 直接使用彩色图计数作为该相机的有效帧数
                int final_count = color_cnt;
                
                camera_details[name] = final_count;
                total_camera_frames += final_count;
                
                // 同时更新内存中的计数器，确保后续 VERIFY_DATA 即使不读文件也准确
                camera_frames_count[name] = final_count;
            }
            meta_data["camera_frame_details"] = camera_details;
            meta_data["camera_frame_total_count"] = total_camera_frames;

            // 机械臂帧数统计
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

    // 最终统计
    RCLCPP_INFO(camera_sub->get_logger(),
        "=== 采集周期结束统计 ===\n"
        "  采集时长: %dms\n"
        "  H5写入: %s\n"
        "  机械臂帧数: %zu",
        duration_ms, h5_success ? "成功" : "失败", arm_episode_data.size());

    is_recording = false;
}

/// 写入TXT数据: 将左右臂原始数据分别写入left_data.txt/right_data.txt
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

/// 写入H5数据: 将机械臂数据写入raw_joints.h5
/// 核心逻辑:
///   state/effector/position = gripper[1](第2个数据)
///   action/effector/position = 下一帧的gripper[0](第1个数据)
///   action/arm/position = 下一帧的关节位置
///   action/end/ = 下一帧的末端位置/姿态
///   最后一帧: action复用当前帧(但夹爪取gripper[0])
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

        // 辅助函数: 写入数据到指定group/dataset
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

        // 辅助函数: 写入时间戳
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

            // ========================
            // state组: 当前帧数据
            // ========================
            // state/arm: 左右臂关节拼接(14维)
            std::vector<double> joints;
            auto l_j = state["left_arm_joints"].get<std::vector<double>>();
            auto r_j = state["right_arm_joints"].get<std::vector<double>>();
            joints.insert(joints.end(), l_j.begin(), l_j.end());
            joints.insert(joints.end(), r_j.begin(), r_j.end());

            write_ts("state/arm", ts, i);
            write_data("state/arm", "position", joints, i, {14});

            // state/end: 末端位置和姿态
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

            // state/effector: 夹爪取gripper[1](第2个数据)
            auto l_g = state["left_gripper"].get<std::vector<double>>();
            auto r_g = state["right_gripper"].get<std::vector<double>>();
            std::vector<double> state_gripper = {l_g[1], r_g[1]};

            write_ts("state/effector", ts, i);
            write_data("state/effector", "position", state_gripper, i, {2});

            // ========================
            // action组: 下一帧数据(最后一帧复用当前帧)
            // ========================
            std::vector<double> action_joints;
            std::vector<double> action_pos;
            std::vector<double> action_ori;
            std::vector<double> action_gripper;

            if (!is_last_frame) {
                // 下一帧数据
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

                // action夹爪取下一帧的gripper[0](第1个数据)
                auto nl_g = next_state["left_gripper"].get<std::vector<double>>();
                auto nr_g = next_state["right_gripper"].get<std::vector<double>>();
                action_gripper = {nl_g[0], nr_g[0]};
            } else {
                // 最后一帧: action复用当前帧数据
                action_joints = joints;
                action_pos = pos;
                action_ori = ori;
                // 最后一帧的action夹爪取当前帧的gripper[0]
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

/// 数据后处理: 复制相机帧和H5数据到目标目录
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

/// 系统状态恢复: 重置机械臂缓存、停止相机订阅
void DoubleArmRGBDDataProcessor::recover() {
    std::lock_guard<std::mutex> lk(lock);
    RCLCPP_INFO(camera_sub->get_logger(), "RECOVER: 系统状态恢复中...");
    if (arm_interface) arm_interface->reset_cache();
    if (camera_sub && camera_sub->running) camera_sub->stop(true, false);
    is_recording = false;
    RCLCPP_INFO(camera_sub->get_logger(), "RECOVER: 系统状态恢复完成");
}


/// 彻底释放资源: 停止采集、执行器、节点
void DoubleArmRGBDDataProcessor::close() {
    if (!camera_sub) return;
    RCLCPP_INFO(camera_sub->get_logger(), "开始释放双臂RGBD数据处理器资源...");
    
    // 1. 停止采集循环线程
    _stop_record_async_task();
    
    // 2. 停止 ROS2 执行器
    stop_executor_flag = true;
    if (executor) {
        executor->cancel();
    }
    if (arm_executor) {
        arm_executor->cancel();
    }

    // 3. 等待所有线程安全退出
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
