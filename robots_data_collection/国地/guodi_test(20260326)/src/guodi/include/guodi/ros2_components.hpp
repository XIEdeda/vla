#ifndef GUODI_ROS2_COMPONENTS_HPP
#define GUODI_ROS2_COMPONENTS_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <deque>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <vector>
#include <map>

#include "sc_ros2/msg/arm_data.hpp"
#include "sc_ros2/msg/motor_errors.hpp"
// #include "topic_pub/msg/arm_data.hpp"
// #include "topic_pub/msg/motor_errors.hpp"
#include "guodi/utils_config.hpp"

namespace guodi {

/// 帧数据结构体: 包含图像、时间戳、有效标志
struct FrameData {
    cv::Mat cv_image;
    int64_t timestamp_ns;
    bool is_valid;
};

// ==============================
// 1. ROS2 RGBD相机订阅和数据管理器
// 对应Python: ROS2RGBDCameraSubscriber
// 负责订阅相机话题、缓存帧数据、保存图片、预览编码等
// ==============================
class ROS2RGBDCameraSubscriber : public rclcpp::Node {
public:
    ROS2RGBDCameraSubscriber(const std::vector<std::string>& camera_names, bool is_preview_mode = false);
    ~ROS2RGBDCameraSubscriber();

    /// camera_dirs 可选: 采集/保存时才需传入,预启动或仅预览时可省略
    void start(const std::map<std::string, fs::path>& camera_dirs = {});
    /// 停止相机订阅,排空队列并保存剩余图像
    void stop(bool is_preview_mode = false, bool destroy_nodes = true);
    /// 更新保存目录(用于新采集周期开始)
    void update_dirs(const std::map<std::string, fs::path>& camera_dirs);
    /// 清空所有缓存队列和写入缓存(用于新采集周期开始前)
    void clear_queues();

    /// 采集模式: 从record FIFO中取出一帧(严格消费,先预检再弹出)
    std::pair<std::map<std::string, std::map<std::string, cv::Mat>>, std::map<std::string, std::map<std::string, int64_t>>> get_frames(bool is_preview_mode = false);
    /// 合并逻辑: 先缓存,达到阈值后批量提交并行写入
    bool save_unified_frame(const std::string& cam_name, const std::string& img_type, const cv::Mat& img_data, int64_t unified_timestamp);

    /// 获取当前预览帧(异步编码,返回base64 JPEG数据)
    json get_current_frame(const std::string& cam_name, const std::string& img_type, int quality = 70);

    /// 获取相机帧计数(color/depth分别计数,用于统计)
    int get_frame_count(const std::string& cam_name, const std::string& img_type);

    /// 获取相机 color 状态(超过2秒无数据则上报异常,用于WebSocket推送)
    std::string get_camera_status() const;

    /// 预览活动开关(控制是否处理预览帧)
    bool is_preview_active;
    /// 录制活动开关(控制是否将帧写入record队列)
    bool is_recording_active;
    /// 停止接收新帧的标志(不销毁订阅器,只在回调入口拒绝)
    std::atomic<bool> stop_receiving;
    /// 订阅器是否在运行
    bool running;
    /// 相机名称列表(公开用于外部遍历)
    std::vector<std::string> camera_names;

private:
    /// 相机话题配置校验(检查相机名是否在ROS2_CAMERA_TOPIC_MAP中)
    void _validate_camera_topics();
    /// 回调函数: 根据激活状态决定是否处理,提交到线程池
    void _frame_callback(const sensor_msgs::msg::Image::SharedPtr msg, const std::string& cam_name, const std::string& img_type);
    /// 线程池中处理单帧: 写入preview + record两条路径
    void _process_frame(const sensor_msgs::msg::Image::SharedPtr msg, const std::string& cam_name, const std::string& img_type);
    /// IO线程池中实际执行保存的函数
    void _async_save(const std::string& cam_name, const std::string& img_type, cv::Mat img_data, int64_t unified_timestamp);

    bool is_preview_mode;
    bool save_depth_;  ///< 是否保存深度图到磁盘(默认true,ROS参数save_depth可覆盖)
    std::map<std::string, std::map<std::string, rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr>> subscribers;
    /// record队列: 录制模式的帧FIFO
    std::map<std::string, std::map<std::string, std::deque<FrameData>>> record_queues;
    /// 预览帧: 只保留最新一帧
    std::map<std::string, std::map<std::string, FrameData>> preview_frames;
    /// 写入缓存: 批量写入磁盘前的内存缓存
    std::map<std::string, std::map<std::string, std::vector<std::pair<cv::Mat, int64_t>>>> write_cache;
    /// 批量写入阈值
    int batch_size = 3;
    /// 帧计数器(color/depth分别计数,用于统计)
    std::map<std::string, std::map<std::string, int>> frame_counters;

    std::map<std::string, fs::path> camera_dirs;

    std::mutex queues_mutex;
    std::mutex preview_mutex;
    std::mutex dirs_mutex;

    /// 各相机 color 最后一次收到数据的时间戳(用于超时检测)
    std::map<std::string, std::chrono::steady_clock::time_point> last_color_received_ts_;
    /// 订阅启动时间(首次创建订阅时记录,用于判断"从未收到"是否应报异常)
    std::chrono::steady_clock::time_point subscription_start_ts_;
    mutable std::mutex camera_status_mutex_;

    // 线程池: 图像处理/IO写入/预览编码
    class ThreadPool {
    public:
        ThreadPool(size_t threads);
        template<class F, class... Args>
        auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
        ~ThreadPool();
    private:
        std::vector<std::thread> workers;
        std::deque<std::function<void()>> tasks;
        std::mutex queue_mutex;
        std::condition_variable condition;
        bool stop;
    };

    std::unique_ptr<ThreadPool> process_pool;
    std::unique_ptr<ThreadPool> io_pool;
    std::unique_ptr<ThreadPool> encode_pool;
    std::unique_ptr<ThreadPool> depth_priority_pool;  // 专门为depth图像处理的高优先级线程池

    /// 预览编码缓存结构
    struct PreviewCache {
        json last_result;
        int64_t last_processed_ts;
        std::atomic<bool> is_processing;
    };
    std::map<std::string, PreviewCache> preview_processing_cache;
};

// ==============================
// 2. 机械臂数据缓存管理器
// 对应Python: DoubleArmDataCache
// 由于数据量小所以一直缓存,左右臂独立锁避免竞争
// ==============================
class DoubleArmDataCache {
public:
    DoubleArmDataCache();
    void add_data(const std::string& arm_type, int64_t timestamp, const std::vector<double>& state);
    void reset();
    std::vector<std::pair<int64_t, std::vector<double>>> get_all_states(const std::string& arm_type);
    
    std::map<std::string, std::deque<std::pair<int64_t, std::vector<double>>>> cache;
    std::mutex left_lock;
    std::mutex right_lock;
};

// ==============================
// 3. ROS2双臂接口(适配预览/采集模式)
// 对应Python: ROS2DoubleArmInterface
// 订阅双臂数据话题、缓存数据、合并左右臂数据
// ==============================
class ROS2DoubleArmInterface : public rclcpp::Node {
public:
    ROS2DoubleArmInterface(const std::string& robot_type = "Keenon_F1_DoubleArm", bool is_preview_mode = false);
    
    /// 启动机械臂独立采集线程(100Hz)
    void start_arm_recording();
    /// 停止机械臂采集,一次性合并缓存数据并返回
    std::vector<json> stop_arm_recording();
    /// 重置双臂数据缓存
    void reset_cache();
    /// 获取所有双臂状态数据(left_states, right_states)
    std::pair<std::vector<std::pair<int64_t, std::vector<double>>>, std::vector<std::pair<int64_t, std::vector<double>>>> get_all_double_arm_states();

    /// 电机错误状态字符串(用于WebSocket推送)
    std::string motor_error_status;
    /// 左右臂帧计数(用于WebSocket推送进度)
    std::map<std::string, int> joint_frames_count;
    /// 回调触发计数器(用于排查丢帧)
    std::map<std::string, int> callback_count;

private:
    /// 机械臂回调: 校验数据长度(16维)并写入缓存
    void _arm_callback(const sc_ros2::msg::ArmData::SharedPtr msg, const std::string& arm_type);
    // void _arm_callback(const topic_pub::msg::ArmData::SharedPtr msg, const std::string& arm_type);
    /// 电机错误回调: 检查18个int32值,非0则状态异常
    void _motor_errors_callback(const sc_ros2::msg::MotorErrors::SharedPtr msg);
    // void _motor_errors_callback(const topic_pub::msg::MotorErrors::SharedPtr msg);
    /// 机械臂缓存统一合并: 依次顺序配对左右臂数据,时间戳取平均
    void _arm_record_loop();

    std::string robot_type;
    bool is_preview_mode;
    std::shared_ptr<DoubleArmDataCache> arm_cache;
    rclcpp::Subscription<sc_ros2::msg::ArmData>::SharedPtr left_sub;
    rclcpp::Subscription<sc_ros2::msg::ArmData>::SharedPtr right_sub;
    rclcpp::Subscription<sc_ros2::msg::MotorErrors>::SharedPtr motor_errors_sub;
    // rclcpp::Subscription<topic_pub::msg::ArmData>::SharedPtr left_sub;
    // rclcpp::Subscription<topic_pub::msg::ArmData>::SharedPtr right_sub;
    // rclcpp::Subscription<topic_pub::msg::MotorErrors>::SharedPtr motor_errors_sub;

    bool arm_recording;
    std::vector<json> arm_episode_data;
    std::mutex lock;
    std::mutex count_lock;
};

} // namespace guodi

#endif // GUODI_ROS2_COMPONENTS_HPP
