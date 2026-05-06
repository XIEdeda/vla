#include "my_dual_arm_package/ros2_components.hpp"
#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
// base64编码函数前向声明
std::string base64_encode(const unsigned char* bytes_to_encode, size_t in_len);

namespace my_dual_arm_package {

// ==============================
// 线程池实现
// ==============================
ROS2RGBDCameraSubscriber::ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; ++i)
        workers.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                    if (this->stop && this->tasks.empty()) return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop_front();
                }
                task();
            }
        });
}

template<class F, class... Args>
auto ROS2RGBDCameraSubscriber::ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;
    auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
        tasks.emplace_back([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

ROS2RGBDCameraSubscriber::ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) worker.join();
}

// ==============================
// 1. ROS2 RGBD相机订阅器实现
// ==============================
ROS2RGBDCameraSubscriber::ROS2RGBDCameraSubscriber(const std::vector<std::string>& camera_names, bool is_preview_mode)
    : Node("rgbd_camera_subscriber"),
      is_preview_active(false), is_recording_active(false), stop_receiving(false), running(false),
      camera_names(camera_names), is_preview_mode(is_preview_mode) {
    
    this->declare_parameter("save_depth", true);
    save_depth_ = this->get_parameter("save_depth").as_bool();
    RCLCPP_INFO(this->get_logger(), "save_depth=%s", save_depth_ ? "true" : "false");

    this->declare_parameter("use_mcap", false);
    RCLCPP_INFO(this->get_logger(), "use_mcap=%s",
        this->get_parameter("use_mcap").as_bool() ? "true" : "false");
    
    // 创建线程池: 根据12核CPU优化配置
    // Color图像处理(4线程) / Depth(4线程) / IO写入(4线程) / 预览编码(3线程)
    process_pool = std::make_unique<ThreadPool>(4);  // color专用
    depth_priority_pool = std::make_unique<ThreadPool>(4);  // depth专用
    io_pool = std::make_unique<ThreadPool>(4);
    encode_pool = std::make_unique<ThreadPool>(3);

    // 初始化预览编码缓存、帧计数器和 color 超时检测时间戳
    for (const auto& cam : camera_names) {
        auto& pc = preview_processing_cache[cam];
        pc.last_result = json();
        pc.last_processed_ts = -1;
        pc.is_processing.store(false);
        frame_counters[cam] = {{"color", 0}, {"depth", 0}};
        last_color_received_ts_[cam] = std::chrono::steady_clock::time_point{};
    }

    // 相机话题配置校验
    _validate_camera_topics();
}

/// 相机话题配置校验: 检查相机名是否在ROS2_CAMERA_TOPIC_MAP中
void ROS2RGBDCameraSubscriber::_validate_camera_topics() {
    std::vector<std::string> invalid_cams;
    for (const auto& cam : camera_names) {
        if (ROS2_CAMERA_TOPIC_MAP.find(cam) == ROS2_CAMERA_TOPIC_MAP.end()) {
            invalid_cams.push_back(cam);
        }
    }
    if (!invalid_cams.empty()) {
        std::string msg = "相机无话题配置: ";
        for (const auto& c : invalid_cams) msg += c + " ";
        throw std::runtime_error(msg);
    }
    RCLCPP_INFO(this->get_logger(), "所有相机话题配置验证通过");
}

ROS2RGBDCameraSubscriber::~ROS2RGBDCameraSubscriber() {
    stop();
}

/// 启动相机订阅(color+depth),配置QoS并绑定回调
void ROS2RGBDCameraSubscriber::start(const std::map<std::string, fs::path>& camera_dirs) {
    std::lock_guard<std::mutex> lock(dirs_mutex);
    this->camera_dirs = camera_dirs;
    // 关键: 重置"停止接收"标志,允许接收新帧
    stop_receiving = false;
    is_preview_active = true;

    if (running) {
        RCLCPP_INFO(this->get_logger(), "相机订阅器已在运行，激活预览处理流");
        return;
    }

    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();

    const bool use_mcap = this->get_parameter("use_mcap").as_bool();
    if (use_mcap) {
        // MCAP压缩链路：第二条订阅 + _compressed_frame_callback
        for (const auto& cam_name : camera_names) {
            const auto& topics = ROS2_CAMERA_TOPIC_MAP.at(cam_name);
            compressed_subscribers[cam_name]["color"] = create_subscription<sensor_msgs::msg::CompressedImage>(
                topics.at("color_compressed"), qos,
                [this, cam_name](sensor_msgs::msg::CompressedImage::SharedPtr msg) {
                    _compressed_frame_callback(msg, cam_name, "color");
                });
            compressed_subscribers[cam_name]["depth"] = create_subscription<sensor_msgs::msg::CompressedImage>(
                topics.at("depth_compressed"), qos,
                [this, cam_name](sensor_msgs::msg::CompressedImage::SharedPtr msg) {
                    _compressed_frame_callback(msg, cam_name, "depth");
                });
        }
    } else {
        // raw 链路：第一条订阅 + _frame_callback
        for (const auto& cam_name : camera_names) {
            subscribers[cam_name]["color"] = create_subscription<sensor_msgs::msg::Image>(
                ROS2_CAMERA_TOPIC_MAP.at(cam_name).at("color"), qos,
                [this, cam_name](const sensor_msgs::msg::Image::SharedPtr msg) { _frame_callback(msg, cam_name, "color"); });

            subscribers[cam_name]["depth"] = create_subscription<sensor_msgs::msg::Image>(
                ROS2_CAMERA_TOPIC_MAP.at(cam_name).at("depth"), qos,
                [this, cam_name](const sensor_msgs::msg::Image::SharedPtr msg) { _frame_callback(msg, cam_name, "depth"); });
        }
    }
    {
        std::lock_guard<std::mutex> lk(camera_status_mutex_);
        subscription_start_ts_ = std::chrono::steady_clock::now();
    }
    running = true;
    RCLCPP_INFO(this->get_logger(), "所有相机订阅器启动完成，已开启预览处理流");
}


/// 回调函数: 根据激活状态决定是否处理
void ROS2RGBDCameraSubscriber::_frame_callback(const sensor_msgs::msg::Image::SharedPtr msg, const std::string& cam_name, const std::string& img_type) {
    // 最高优先级检查"停止接收"标志
    if (stop_receiving) return;

    // color 超时检测: 记录最后一次收到数据的时间
    if (img_type == "color") {
        std::lock_guard<std::mutex> lk(camera_status_mutex_);
        last_color_received_ts_[cam_name] = std::chrono::steady_clock::now();
    }

    // depth 不参与预览：非落盘录制时直接跳过（本回调仅在 raw Image 订阅下调用；mcap 写 bag 全在 _compressed_frame_callback）
    if (img_type == "depth" && !is_recording_active) {
        return;
    }

    // 如果既不录制也不预览,直接跳过耗时的转换处理
    if (!is_recording_active && !is_preview_active) return;

    // 给depth图像分配专用的高优先级线程池
    if (img_type == "depth") {
        depth_priority_pool->enqueue([this, msg, cam_name, img_type] { 
            _process_frame(msg, cam_name, img_type); 
        });
    } else {
        // color图像使用普通处理线程池
        process_pool->enqueue([this, msg, cam_name, img_type] {
            _process_frame(msg, cam_name, img_type);
        });
    }
}

// use_mcap 时唯一相机回调。流程（配合 img_type 为 "color" / "depth"）：
// 1) stop / color 心跳 → 2) depth 且「仅预览、且无 bag」→ 丢弃 depth
// 3) 有 bag 且（color 或 save_depth）→ 把 CompressedImage 推入 McapWriteQueue；depth 写完即 return（不做 OpenCV）
// 4) 无预览 → return（mcap 图只进 bag，不进 record_queues）
// 5) 仅 color：imdecode → 更新 preview_frames
void ROS2RGBDCameraSubscriber::_compressed_frame_callback(
    const sensor_msgs::msg::CompressedImage::SharedPtr msg,
    const std::string& cam_name,
    const std::string& img_type) {
    if (stop_receiving) return;

    if (img_type == "color") {
        std::lock_guard<std::mutex> lk(camera_status_mutex_);
        last_color_received_ts_[cam_name] = std::chrono::steady_clock::now();
    }

    // 仅预览、且还未开始 mcap（无 bag_queue）时不需要 depth，直接丢帧
    if (img_type == "depth" && !is_recording_active) {
        if (!std::atomic_load(&bag_queue)) return;
    }

    if (img_type == "color" || save_depth_) {
        auto q = std::atomic_load(&bag_queue);
        if (q) {
            const std::string topic = ROS2_CAMERA_TOPIC_MAP.at(cam_name).at(
                img_type == "color" ? "color_compressed" : "depth_compressed");
            q->push([msg, topic](rosbag2_cpp::Writer& w) {
                try {
                    w.write(*msg, topic, rclcpp::Time(msg->header.stamp));
                } catch (const std::exception& e) {
                    (void)e;
                }
            });
            if (img_type == "depth") return;
        }
    }

    // mcap 链路不进 record_queues（图在 bag 里）；此处仅在为预览解码 color
    if (!is_preview_active) return;

    if (img_type != "color") return;

    process_pool->enqueue([this, msg, cam_name, img_type] {
        try {
            if (msg->data.empty()) return;
            cv::Mat buf(1, static_cast<int>(msg->data.size()), CV_8UC1,
                const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(msg->data.data())));
            cv::Mat bgr = cv::imdecode(buf, cv::IMREAD_COLOR);
            if (bgr.empty()) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "[%s] color CompressedImage imdecode 失败", cam_name.c_str());
                return;
            }
            int64_t ts_ns = msg->header.stamp.sec * 1000000000LL + msg->header.stamp.nanosec;
            FrameData data = {bgr, ts_ns, true};
            std::lock_guard<std::mutex> lock(preview_mutex);
            preview_frames[cam_name][img_type] = data;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "[%s_%s] 压缩帧处理失败: %s",
                cam_name.c_str(), img_type.c_str(), e.what());
        }
    });
}

/// 线程池中处理单帧: 写入preview + record两条路径
void ROS2RGBDCameraSubscriber::_process_frame(const sensor_msgs::msg::Image::SharedPtr msg, const std::string& cam_name, const std::string& img_type) {
    try {
        // 1. 图像转换
        cv_bridge::CvImagePtr cv_ptr;
        if (img_type == "color")
            cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        else
            cv_ptr = cv_bridge::toCvCopy(msg, "16UC1");

        // 2. 时间戳(ROS原始时间戳)
        int64_t ts_ns = msg->header.stamp.sec * 1000000000LL + msg->header.stamp.nanosec;

        FrameData data = {cv_ptr->image, ts_ns, true};

        // 3. 预览路径: 永远覆盖,只保留最新
        if (is_preview_active) {
            std::lock_guard<std::mutex> lock(preview_mutex);
            preview_frames[cam_name][img_type] = data;
        }

        // 4. 录制路径: 严格FIFO,生产一帧就入队一帧
        if (is_recording_active) {
            std::lock_guard<std::mutex> lock(queues_mutex);
            auto& q = record_queues[cam_name][img_type];
            q.push_back(data);
            // 防止record队列失控
            if (q.size() > 200) {
                q.pop_front();
                RCLCPP_WARN_ONCE(this->get_logger(), "[%s_%s] record FIFO超过上限200,开始丢弃最旧帧",
                    cam_name.c_str(), img_type.c_str());
            }
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "[%s_%s] 帧处理失败: %s", cam_name.c_str(), img_type.c_str(), e.what());
    }
}

/// 获取当前预览帧(异步编码,返回base64 JPEG)
json ROS2RGBDCameraSubscriber::get_current_frame(const std::string& cam_name, const std::string& img_type, int quality) {
    if (img_type != "color") return json();

    auto& cache_state = preview_processing_cache[cam_name];
    FrameData preview_frame;
    {
        std::lock_guard<std::mutex> lock(preview_mutex);
        if (preview_frames.count(cam_name) && preview_frames[cam_name].count(img_type)) {
            preview_frame = preview_frames[cam_name][img_type];
        } else {
            return cache_state.last_result;
        }
    }

    if (!cache_state.is_processing && preview_frame.timestamp_ns > cache_state.last_processed_ts) {
        cache_state.is_processing = true;
        encode_pool->enqueue([this, cam_name, preview_frame, quality] {
            try {
                cv::Mat resized;
                cv::resize(preview_frame.cv_image, resized,
                    cv::Size(get_camera_param(cam_name, "color", "width"), get_camera_param(cam_name, "color", "height")));
                std::vector<uchar> buf;
                cv::imencode(".jpg", resized, buf, {cv::IMWRITE_JPEG_QUALITY, quality});
                
                std::string base64_data = base64_encode(buf.data(), buf.size());
                
                json result = {
                    {"camera_name", cam_name},
                    {"image_data", base64_data},
                    {"width", get_camera_param(cam_name, "color", "width")},
                    {"height", get_camera_param(cam_name, "color", "height")}
                };
                
                auto& cs = preview_processing_cache[cam_name];
                cs.last_result = result;
                cs.last_processed_ts = preview_frame.timestamp_ns;
            } catch (...) {}
            preview_processing_cache[cam_name].is_processing = false;
        });
    }

    return cache_state.last_result;
}

/// 采集模式: 从record FIFO中取出一帧(先预检再弹出)
std::pair<std::map<std::string, std::map<std::string, cv::Mat>>, std::map<std::string, std::map<std::string, int64_t>>> ROS2RGBDCameraSubscriber::get_frames(bool /*is_preview_mode*/) {
    std::lock_guard<std::mutex> lock(queues_mutex);
    
    // 1. 第一阶段: 预检 - 确保所有相机都有帧
    for (const auto& cam : camera_names) {
        if (record_queues[cam]["color"].empty() || record_queues[cam]["depth"].empty())
            return {{}, {}};
    }

    std::map<std::string, std::map<std::string, cv::Mat>> frames;
    std::map<std::string, std::map<std::string, int64_t>> timestamps;

    // 2. 第二阶段: 统一消费 - 弹出队列头部的帧
    for (const auto& cam : camera_names) {
        auto color_data = record_queues[cam]["color"].front();
        record_queues[cam]["color"].pop_front();
        auto depth_data = record_queues[cam]["depth"].front();
        record_queues[cam]["depth"].pop_front();

        frames[cam]["color"] = color_data.cv_image;
        frames[cam]["depth"] = depth_data.cv_image;
        timestamps[cam]["color"] = color_data.timestamp_ns;
        timestamps[cam]["depth"] = depth_data.timestamp_ns;
    }

    return {frames, timestamps};
}

/// 合并逻辑: 先缓存,达到阈值后批量提交并行写入
bool ROS2RGBDCameraSubscriber::save_unified_frame(const std::string& cam_name, const std::string& img_type, const cv::Mat& img_data, int64_t unified_timestamp) {
    // 深度图保存开关: 关闭时直接返回,不缓存不写入
    if (img_type == "depth" && !save_depth_) return true;
    
    std::vector<std::pair<cv::Mat, int64_t>> batch;
    {
        std::lock_guard<std::mutex> lock(queues_mutex);
        write_cache[cam_name][img_type].push_back({img_data.clone(), unified_timestamp});
        if (write_cache[cam_name][img_type].size() >= static_cast<size_t>(batch_size)) {
            batch = std::move(write_cache[cam_name][img_type]);
            write_cache[cam_name][img_type].clear();
        }
    }

    if (!batch.empty()) {
        io_pool->enqueue([this, cam_name, img_type, batch] {
            for (const auto& item : batch) {
                _async_save(cam_name, img_type, item.first, item.second);
            }
        });
    }
    return true;
}

/// IO线程池中实际执行保存的函数
void ROS2RGBDCameraSubscriber::_async_save(const std::string& cam_name, const std::string& img_type, cv::Mat img_data, int64_t unified_timestamp) {
    std::string dir_key = "cam_" + cam_name + "_" + img_type;
    fs::path save_dir;
    {
        std::lock_guard<std::mutex> lock(dirs_mutex);
        if (camera_dirs.count(dir_key)) save_dir = camera_dirs.at(dir_key);
        else return;
    }

    std::string suffix = (img_type == "color") ? ".jpg" : ".png";
    fs::path img_path = save_dir / (std::to_string(unified_timestamp) + suffix);

    if (img_type == "color") {
        cv::imwrite(img_path.string(), img_data, {cv::IMWRITE_JPEG_QUALITY, 80});
    } else {
        cv::imwrite(img_path.string(), img_data, {cv::IMWRITE_PNG_COMPRESSION, 1});
    }

    // 更新帧计数器
    {
        std::lock_guard<std::mutex> lock(queues_mutex);
        frame_counters[cam_name][img_type]++;
    }
}

/// 停止相机订阅,排空队列并保存剩余图像
void ROS2RGBDCameraSubscriber::stop(bool is_preview_mode, bool destroy_nodes) {
    if (!running) {
        RCLCPP_WARN(this->get_logger(), "相机订阅器已停止,无需重复操作");
        return;
    }
    
    if (!stop_receiving) {
        stop_receiving = true;
        RCLCPP_INFO(this->get_logger(), "已置位 stop_receiving 标志，回调函数将拒绝所有新帧");
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    
    // 清空预览缓存，释放内存
    {
        std::lock_guard<std::mutex> lock(preview_mutex);
        preview_frames.clear();
        for (auto& [cam_name, cache] : preview_processing_cache) {
            cache.last_result = json();
            cache.last_processed_ts = -1;
            cache.is_processing = false;
        }
    }

    if (!is_preview_mode) {
        RCLCPP_INFO(this->get_logger(), "正在从record队列排空遗留帧...");
        std::map<std::string, std::map<std::string, int>> drain_counts;
        for (const auto& cam_name : camera_names) {
            drain_counts[cam_name] = {{"color", 0}, {"depth", 0}};
            for (const auto& img_type : {"color", "depth"}) {
                // 先在锁内取出所有帧,释放锁后再调用save_unified_frame,
                // 避免持有queues_mutex时调用save_unified_frame导致重复加锁死锁
                std::vector<FrameData> drained;
                {
                    std::lock_guard<std::mutex> lock(queues_mutex);
                    auto& q = record_queues[cam_name][img_type];
                    while (!q.empty()) {
                        drained.push_back(q.front());
                        q.pop_front();
                    }
                }
                drain_counts[cam_name][img_type] = static_cast<int>(drained.size());
                for (const auto& frame : drained) {
                    save_unified_frame(cam_name, img_type, frame.cv_image, frame.timestamp_ns);
                }
            }
        }
        int drain_total = 0;
        for (const auto& [cam, counts] : drain_counts) {
            for (const auto& [type, cnt] : counts) {
                drain_total += cnt;
            }
        }
        if (drain_total > 0) {
            RCLCPP_INFO(this->get_logger(), "从record队列额外排空了共%d帧:", drain_total);
            for (const auto& [cam, counts] : drain_counts) {
                RCLCPP_INFO(this->get_logger(), "  %s: color=%d, depth=%d", cam.c_str(), counts.at("color"), counts.at("depth"));
            }
        }
    }

    RCLCPP_INFO(this->get_logger(), "处理缓存中剩余图像...");
    int remaining_total = 0;
    for (const auto& cam_name : camera_names) {
        for (const auto& img_type : {"color", "depth"}) {
            std::vector<std::pair<cv::Mat, int64_t>> remaining;
            {
                std::lock_guard<std::mutex> lock(queues_mutex);
                remaining = std::move(write_cache[cam_name][img_type]);
                write_cache[cam_name][img_type].clear();
            }
            remaining_total += remaining.size();
            for (const auto& item : remaining) {
                _async_save(cam_name, img_type, item.first, item.second);
            }
        }
    }

    if (remaining_total > 0) {
        RCLCPP_INFO(this->get_logger(), "共%d帧剩余图像已触发写入流程", remaining_total);
    } else {
        RCLCPP_INFO(this->get_logger(), "缓存中无剩余图像，无需额外处理");
    }

    // 帧数统计日志
    RCLCPP_INFO(this->get_logger(), "===== 阶段帧数统计 =====");
    for (const auto& cam_name : camera_names) {
        int color_cnt = 0, depth_cnt = 0;
        if (frame_counters.count(cam_name)) {
            color_cnt = frame_counters[cam_name]["color"];
            depth_cnt = frame_counters[cam_name]["depth"];
        }
        RCLCPP_INFO(this->get_logger(), "%s: color=%d帧, depth=%d帧", cam_name.c_str(), color_cnt, depth_cnt);
    }
    RCLCPP_INFO(this->get_logger(), "=======================");

    if (destroy_nodes) {
        for (auto& cam_subs : subscribers) {
            cam_subs.second["color"].reset();
            cam_subs.second["depth"].reset();
        }
        for (auto& cam_subs : compressed_subscribers) {
            cam_subs.second["color"].reset();
            cam_subs.second["depth"].reset();
        }
        subscribers.clear();
        compressed_subscribers.clear();
        running = false;
        RCLCPP_INFO(this->get_logger(), "相机订阅器停止流程完成(destroy_nodes=true)");
    } else {
        RCLCPP_INFO(this->get_logger(), "订阅器保持运行，stop_receiving标志将在下次启动时重置");
    }
}

int ROS2RGBDCameraSubscriber::get_frame_count(const std::string& cam_name, const std::string& img_type) {
    std::lock_guard<std::mutex> lock(queues_mutex);
    if (frame_counters.count(cam_name) && frame_counters.at(cam_name).count(img_type)) {
        return frame_counters.at(cam_name).at(img_type);
    }
    return 0;
}

std::string ROS2RGBDCameraSubscriber::get_camera_status() const {
    // 1. 采集已主动停止: 不应报"异常"
    if (stop_receiving) {
        return "相机状态正常";
    }
    // 2. 订阅器未运行(如刚重建): 无法判断, 不报异常
    if (!running) {
        return "相机状态正常";
    }
    // 3. 超过2秒无数据则报异常: 曾收到过→超时; 从未收到→若订阅已超2秒也报(相机/驱动未发布)
    constexpr auto timeout = std::chrono::seconds(2);
    constexpr auto epoch = std::chrono::steady_clock::time_point{};
    auto now = std::chrono::steady_clock::now();
    std::vector<std::string> timeout_cams;
    {
        std::lock_guard<std::mutex> lock(camera_status_mutex_);
        auto subs_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - subscription_start_ts_);
        for (const auto& cam_name : camera_names) {
            auto it = last_color_received_ts_.find(cam_name);
            if (it == last_color_received_ts_.end()) continue;
            if (it->second == epoch) {
                // 从未收到首帧: 若订阅已超过2秒仍无数据, 报异常(相机/驱动未发布)
                if (subs_elapsed >= timeout) timeout_cams.push_back(cam_name);
                continue;
            }
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - it->second);
            if (elapsed >= timeout) timeout_cams.push_back(cam_name);
        }
    }
    if (timeout_cams.empty()) {
        return "相机状态正常";
    }
    std::string msg = "";
    for (size_t i = 0; i < timeout_cams.size(); ++i) {
        if (i > 0) msg += "、";
        msg += timeout_cams[i];
    }
    return msg + " 相机异常,超过2秒无数据,请停止采集，检查相机";
}

/// 清空所有缓存队列和写入缓存(用于新采集周期开始前)
void ROS2RGBDCameraSubscriber::clear_queues() {
    std::lock_guard<std::mutex> lock(queues_mutex);
    for (auto& cam_queues : record_queues) {
        cam_queues.second["color"].clear();
        cam_queues.second["depth"].clear();
    }
    // 清空写入缓存和帧计数器
    for (auto& cam_cache : write_cache) {
        cam_cache.second["color"].clear();
        cam_cache.second["depth"].clear();
    }
    for (auto& cam_cnt : frame_counters) {
        cam_cnt.second["color"] = 0;
        cam_cnt.second["depth"] = 0;
    }
    
    // 同时清空预览缓存
    std::lock_guard<std::mutex> preview_lock(preview_mutex);
    preview_frames.clear();
    for (auto& [cam_name, cache] : preview_processing_cache) {
        cache.last_result = json();
        cache.last_processed_ts = -1;
        cache.is_processing = false;
    }
    RCLCPP_INFO(this->get_logger(), "所有相机缓存队列已清空");
}

/// 更新保存目录(用于新采集周期开始)
void ROS2RGBDCameraSubscriber::update_dirs(const std::map<std::string, fs::path>& camera_dirs) {
    std::lock_guard<std::mutex> lock(dirs_mutex);
    this->camera_dirs = camera_dirs;
    RCLCPP_INFO(this->get_logger(), "相机保存目录已更新");
}

} // namespace my_dual_arm_package

// ==============================
// base64编码实现(全局作用域,JPEG帧编码为base64字符串)
// ==============================
static const std::string base64_chars = 
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

std::string base64_encode(const unsigned char* bytes_to_encode, size_t in_len) {
  std::string ret;
  int i = 0;
  int j = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  while (in_len--) {
    char_array_3[i++] = *(bytes_to_encode++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for(i = 0; (i <4) ; i++)
        ret += base64_chars[char_array_4[i]];
      i = 0;
    }
  }

  if (i)
  {
    for(j = i; j < 3; j++)
      char_array_3[j] = '\0';

    char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
    char_array_4[3] = char_array_3[2] & 0x3f;

    for (j = 0; (j < i + 1); j++)
      ret += base64_chars[char_array_4[j]];

    while((i++ < 3))
      ret += '=';
  }

  return ret;
}

namespace my_dual_arm_package {

// ==============================
// 2. ROS2 机械臂订阅实现
// ==============================

/// 双臂接口构造函数: 订阅left_data/right_data/motor_errors_topic
ROS2DoubleArmInterface::ROS2DoubleArmInterface(const std::string& robot_type, bool is_preview_mode)
    : Node("double_arm_interface"), robot_type(robot_type), is_preview_mode(is_preview_mode), arm_recording(false) {
    
    arm_cache = std::make_shared<DoubleArmDataCache>();
    
    auto qos = rclcpp::QoS(rclcpp::KeepLast(200)).reliable();

    left_sub = create_subscription<sc_ros2::msg::ArmData>(
        "left_data", qos, [this](const sc_ros2::msg::ArmData::SharedPtr msg) { _arm_callback(msg, "left_arm"); });
    // left_sub = create_subscription<topic_pub::msg::ArmData>(
    //     "left_data", qos, [this](const topic_pub::msg::ArmData::SharedPtr msg) { _arm_callback(msg, "left_arm"); });

    right_sub = create_subscription<sc_ros2::msg::ArmData>(
        "right_data", qos, [this](const sc_ros2::msg::ArmData::SharedPtr msg) { _arm_callback(msg, "right_arm"); });
    // right_sub = create_subscription<topic_pub::msg::ArmData>(
    //     "right_data", qos, [this](const topic_pub::msg::ArmData::SharedPtr msg) { _arm_callback(msg, "right_arm"); });

    motor_errors_sub = create_subscription<sc_ros2::msg::MotorErrors>(
        "motor_errors_topic", 10, [this](const sc_ros2::msg::MotorErrors::SharedPtr msg) { _motor_errors_callback(msg); });
    // motor_errors_sub = create_subscription<topic_pub::msg::MotorErrors>(
    //     "motor_errors_topic", 10, [this](const topic_pub::msg::MotorErrors::SharedPtr msg) { _motor_errors_callback(msg); });

    // 在收到首条 MotorErrors 之前不冒充「正常」；与 websocket 中 motor_pending_no_topic 判断前缀「电机状态未获取」一致
    motor_error_status = "机械臂状态未获取（请检查机械臂是否启动）";

    joint_frames_count = {{"left", 0}, {"right", 0}};
    callback_count = {{"left", 0}, {"right", 0}};
    RCLCPP_INFO(this->get_logger(), "双臂接口启动:已订阅 left_data(左臂)、right_data(右臂)");
}

std::string ROS2DoubleArmInterface::get_motor_error_status_effective() const {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_motor_errors_at_ > std::chrono::milliseconds(k_motor_errors_stale_after_ms_)) {
        return "机械臂状态未获取（请检查机械臂是否启动）";
    }
    return motor_error_status;
}

/// 机械臂回调: 校验数据长度(16维)并写入缓存
void ROS2DoubleArmInterface::_arm_callback(const sc_ros2::msg::ArmData::SharedPtr msg, const std::string& arm_type) {
    // (topic_pub::msg::ArmData::SharedPtr -> sc_ros2::msg::ArmData::SharedPtr)
    // 记录回调触发次数
    if (arm_type == "left_arm") callback_count["left"]++;
    else callback_count["right"]++;

    // 校验数据长度(7关节+2夹爪+7位姿=16维)
    if (msg->data.size() != 16) {
        RCLCPP_WARN(this->get_logger(), "%s 数据长度错误:实际=%zu | 预期=16(7关节+2夹爪+3位置+4姿态)",
            arm_type.c_str(), msg->data.size());
        return;
    }

    // 转换时间戳(ROS当前时间)
    int64_t ts = msg->header.stamp.sec * 1000000000LL + msg->header.stamp.nanosec;
    // 创建数据副本,避免引用失效
    std::vector<double> data(msg->data.begin(), msg->data.end());

    arm_cache->add_data(arm_type, ts, data);

    // 进程内 mcap 写入: 捕获 shared_ptr 推入队列，写线程完成序列化
    {
        auto q = std::atomic_load(&bag_queue);
        if (q) {
            const std::string topic = (arm_type == "left_arm") ? "/left_data" : "/right_data";
            q->push([msg, topic](rosbag2_cpp::Writer& w) {
                try {
                    w.write(*msg, topic, rclcpp::Time(msg->header.stamp));
                } catch (const std::exception& e) {
                    (void)e;
                }
            });
        }
    }

    std::lock_guard<std::mutex> lock(count_lock);
    if (arm_type == "left_arm") joint_frames_count["left"]++;
    else joint_frames_count["right"]++;
}

/// 电机错误回调: 18项约定 — [0..13]电机故障(0无,1故障); [14]右臂/[15]左臂虚拟墙(0未进入,1预警);
/// [16]右臂/[17]左臂掉电(0有电,1任关节掉电)。数组不足18时仅解析已有下标。
void ROS2DoubleArmInterface::_motor_errors_callback(const sc_ros2::msg::MotorErrors::SharedPtr msg) {
    const auto& codes = msg->error_codes;
    const size_t n = codes.size();

    bool motor_fault = false;
    for (size_t i = 0; i < n && i < 14; ++i) {
        if (codes[i] != 0) {
            motor_fault = true;
            break;
        }
    }

    auto idx = [&](size_t i) -> int32_t { return i < n ? codes[i] : 0; };
    const bool r_vwall_warn = idx(14) != 0;
    const bool l_vwall_warn = idx(15) != 0;
    const bool r_power_loss = idx(16) != 0;
    const bool l_power_loss = idx(17) != 0;

    std::vector<std::string> parts;
    if (motor_fault) parts.emplace_back("电机故障");
    if (r_vwall_warn) parts.emplace_back("右臂虚拟墙预警");
    if (l_vwall_warn) parts.emplace_back("左臂虚拟墙预警");
    if (r_power_loss) parts.emplace_back("右臂掉电");
    if (l_power_loss) parts.emplace_back("左臂掉电");

    if (parts.empty()) {
        motor_error_status = "机械臂状态正常";
    } else {
        std::string detail;
        for (size_t i = 0; i < parts.size(); ++i) {
            if (i) detail += "；";
            detail += parts[i];
        }
        const bool need_stop = motor_fault || r_power_loss || l_power_loss;
        motor_error_status =
            need_stop ? ("机械臂状态异常，请停止采集（" + detail + "）") : ("虚拟墙预警：" + detail);
    }
    last_motor_errors_at_ = std::chrono::steady_clock::now();
}

/// 获取所有双臂状态数据(left_states, right_states)
std::pair<std::vector<std::pair<int64_t, std::vector<double>>>, std::vector<std::pair<int64_t, std::vector<double>>>>
ROS2DoubleArmInterface::get_all_double_arm_states() {
    auto left_states = arm_cache->get_all_states("left_arm");
    auto right_states = arm_cache->get_all_states("right_arm");
    return {left_states, right_states};
}

/// 添加单臂数据到缓存(左右臂独立锁)
void DoubleArmDataCache::add_data(const std::string& arm_type, int64_t timestamp, const std::vector<double>& state) {
    if (arm_type == "left_arm") {
        std::lock_guard<std::mutex> lock(left_lock);
        cache["left_arm"].push_back({timestamp, state});
    } else {
        std::lock_guard<std::mutex> lock(right_lock);
        cache["right_arm"].push_back({timestamp, state});
    }
}

/// 重置双臂数据缓存
void DoubleArmDataCache::reset() {
    std::lock_guard<std::mutex> lock1(left_lock);
    std::lock_guard<std::mutex> lock2(right_lock);
    cache["left_arm"].clear();
    cache["right_arm"].clear();
}

/// 获取指定臂的所有状态数据
std::vector<std::pair<int64_t, std::vector<double>>> DoubleArmDataCache::get_all_states(const std::string& arm_type) {
    if (arm_type == "left_arm") {
        std::lock_guard<std::mutex> lock(left_lock);
        return {cache["left_arm"].begin(), cache["left_arm"].end()};
    } else {
        std::lock_guard<std::mutex> lock(right_lock);
        return {cache["right_arm"].begin(), cache["right_arm"].end()};
    }
}

DoubleArmDataCache::DoubleArmDataCache() {}

/// 启动机械臂独立采集线程(100Hz)
void ROS2DoubleArmInterface::start_arm_recording() {
    arm_recording = true;
    arm_episode_data.clear();
    std::lock_guard<std::mutex> lock(count_lock);
    joint_frames_count = {{"left", 0}, {"right", 0}};
    // 重置回调计数器
    callback_count = {{"left", 0}, {"right", 0}};
    RCLCPP_INFO(this->get_logger(), "机械臂100Hz独立采集线程启动");
}

/// 机械臂缓存统一合并: 依次顺序配对左右臂数据,时间戳取平均
void ROS2DoubleArmInterface::_arm_record_loop() {
    RCLCPP_INFO(this->get_logger(), "开始机械臂缓存统一合并...");
    
    std::vector<std::pair<int64_t, std::vector<double>>> left, right;
    {
        std::lock_guard<std::mutex> lock(arm_cache->left_lock);
        left.assign(arm_cache->cache["left_arm"].begin(), arm_cache->cache["left_arm"].end());
    }
    {
        std::lock_guard<std::mutex> lock(arm_cache->right_lock);
        right.assign(arm_cache->cache["right_arm"].begin(), arm_cache->cache["right_arm"].end());
    }

    size_t min_len = std::min(left.size(), right.size());
    std::vector<json> merged;

    for (size_t i = 0; i < min_len; ++i) {
        int64_t avg_ts = (left[i].first + right[i].first) / 2;
        json frame;
        frame["timestamp"] = avg_ts;
        auto& l_data = left[i].second;
        auto& r_data = right[i].second;
        
        frame["arm_state"] = {
            {"left_arm_joints", std::vector<double>(l_data.begin(), l_data.begin()+7)},
            {"left_gripper", std::vector<double>(l_data.begin()+7, l_data.begin()+9)},
            {"left_end_position", std::vector<double>(l_data.begin()+9, l_data.begin()+12)},
            {"left_end_orientation", std::vector<double>(l_data.begin()+12, l_data.begin()+16)},
            {"right_arm_joints", std::vector<double>(r_data.begin(), r_data.begin()+7)},
            {"right_gripper", std::vector<double>(r_data.begin()+7, r_data.begin()+9)},
            {"right_end_position", std::vector<double>(r_data.begin()+9, r_data.begin()+12)},
            {"right_end_orientation", std::vector<double>(r_data.begin()+12, r_data.begin()+16)}
        };
        merged.push_back(frame);
    }

    arm_episode_data = merged;
    {
        std::lock_guard<std::mutex> lock(count_lock);
        joint_frames_count["left"] = min_len;
        joint_frames_count["right"] = min_len;
    }

    RCLCPP_INFO(this->get_logger(), "==============================");
    RCLCPP_INFO(this->get_logger(), "机械臂缓存统一合并完成");
    RCLCPP_INFO(this->get_logger(), "成功配对组数: %zu", min_len);
    RCLCPP_INFO(this->get_logger(), "左臂计数: %d, 右臂计数: %d", 
        joint_frames_count["left"], joint_frames_count["right"]);
    RCLCPP_INFO(this->get_logger(), "==============================");

    // 清空缓存
    reset_cache();
}

/// 停止机械臂采集,一次性合并缓存数据并返回
std::vector<json> ROS2DoubleArmInterface::stop_arm_recording() {
    if (!arm_recording) return {};
    arm_recording = false;
    _arm_record_loop();
    RCLCPP_INFO(this->get_logger(), "机械臂采集停止，本次任务共%zu条数据", arm_episode_data.size());
    return arm_episode_data;
}

/// 重置双臂数据缓存
void ROS2DoubleArmInterface::reset_cache() {
    arm_cache->reset();
    RCLCPP_INFO(this->get_logger(), "双臂数据缓存已重置");
}


} // namespace my_dual_arm_package
