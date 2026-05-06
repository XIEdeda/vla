#ifndef MY_DUAL_ARM_PACKAGE_ROS2_COMPONENTS_HPP
#define MY_DUAL_ARM_PACKAGE_ROS2_COMPONENTS_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <deque>
#include <queue>
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
#include "my_dual_arm_package/utils_config.hpp"

namespace my_dual_arm_package {


class McapWriteQueue {
public:
    using WriteTask = std::function<void(rosbag2_cpp::Writer &)>;

    explicit McapWriteQueue(std::shared_ptr<rosbag2_cpp::Writer> writer)
        : writer_(std::move(writer)), stopped_(false)
    {
        worker_ = std::thread([this] { _run(); });
    }

    ~McapWriteQueue() { stop(); }

    void push(WriteTask task) {
        if (stopped_) return;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            queue_.push(std::move(task));
        }
        cv_.notify_one();
    }

    void stop() {
        if (stopped_.exchange(true)) return;
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

private:
    void _run() {
        while (true) {
            WriteTask task;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [this] { return !queue_.empty() || stopped_; });
                if (queue_.empty() && stopped_) break;
                task = std::move(queue_.front());
                queue_.pop();
            }
            task(*writer_);
        }
    }

    std::shared_ptr<rosbag2_cpp::Writer> writer_;
    std::queue<WriteTask> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::thread worker_;
    std::atomic<bool> stopped_;
};

struct FrameData {
    cv::Mat cv_image;
    int64_t timestamp_ns;
    bool is_valid;
};


class ROS2RGBDCameraSubscriber : public rclcpp::Node {
public:
    ROS2RGBDCameraSubscriber(const std::vector<std::string>& camera_names, bool is_preview_mode = false);
    ~ROS2RGBDCameraSubscriber();


    void start(const std::map<std::string, fs::path>& camera_dirs = {});

    void stop(bool is_preview_mode = false, bool destroy_nodes = true);

    void update_dirs(const std::map<std::string, fs::path>& camera_dirs);

    void clear_queues();

    std::pair<std::map<std::string, std::map<std::string, cv::Mat>>, std::map<std::string, std::map<std::string, int64_t>>> get_frames(bool is_preview_mode = false);

    bool save_unified_frame(const std::string& cam_name, const std::string& img_type, const cv::Mat& img_data, int64_t unified_timestamp);


    json get_current_frame(const std::string& cam_name, const std::string& img_type, int quality = 70);

    int get_frame_count(const std::string& cam_name, const std::string& img_type);

    std::string get_camera_status() const;

    bool is_preview_active;

    bool is_recording_active;

    std::atomic<bool> stop_receiving;

    bool running;

    std::vector<std::string> camera_names;
    
    std::shared_ptr<McapWriteQueue> bag_queue;

private:

    void _validate_camera_topics();

    void _frame_callback(const sensor_msgs::msg::Image::SharedPtr msg, const std::string& cam_name, const std::string& img_type);

    void _compressed_frame_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg, const std::string& cam_name, const std::string& img_type);

    void _process_frame(const sensor_msgs::msg::Image::SharedPtr msg, const std::string& cam_name, const std::string& img_type);

    void _async_save(const std::string& cam_name, const std::string& img_type, cv::Mat img_data, int64_t unified_timestamp);

    bool is_preview_mode;
    bool save_depth_;  
    std::map<std::string, std::map<std::string, rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr>> subscribers;

    std::map<std::string, std::map<std::string, rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr>> compressed_subscribers;

    std::map<std::string, std::map<std::string, std::deque<FrameData>>> record_queues;

    std::map<std::string, std::map<std::string, FrameData>> preview_frames;

    std::map<std::string, std::map<std::string, std::vector<std::pair<cv::Mat, int64_t>>>> write_cache;
    int batch_size = 3;
    std::map<std::string, std::map<std::string, int>> frame_counters;

    std::map<std::string, fs::path> camera_dirs;

    std::mutex queues_mutex;
    std::mutex preview_mutex;
    std::mutex dirs_mutex;

    std::map<std::string, std::chrono::steady_clock::time_point> last_color_received_ts_;
    std::chrono::steady_clock::time_point subscription_start_ts_;
    mutable std::mutex camera_status_mutex_;

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
    std::unique_ptr<ThreadPool> depth_priority_pool;  

    struct PreviewCache {
        json last_result;
        int64_t last_processed_ts;
        std::atomic<bool> is_processing;
    };
    std::map<std::string, PreviewCache> preview_processing_cache;
};

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

class ROS2DoubleArmInterface : public rclcpp::Node {
public:
    ROS2DoubleArmInterface(const std::string& robot_type = "Keenon_F1_DoubleArm", bool is_preview_mode = false);

    void start_arm_recording();

    std::vector<json> stop_arm_recording();

    void reset_cache();

    std::pair<std::vector<std::pair<int64_t, std::vector<double>>>, std::vector<std::pair<int64_t, std::vector<double>>>> get_all_double_arm_states();

    std::string get_motor_error_status_effective() const;

    std::string motor_error_status;

    std::map<std::string, int> joint_frames_count;

    std::map<std::string, int> callback_count;

    std::shared_ptr<McapWriteQueue> bag_queue;

private:

    void _arm_callback(const sc_ros2::msg::ArmData::SharedPtr msg, const std::string& arm_type);
    
    void _motor_errors_callback(const sc_ros2::msg::MotorErrors::SharedPtr msg);


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
    
    std::chrono::steady_clock::time_point last_motor_errors_at_{};

    static constexpr int k_motor_errors_stale_after_ms_ = 1000;

};

} // namespace my_dual_arm_package

#endif // MY_DUAL_ARM_PACKAGE_ROS2_COMPONENTS_HPP
