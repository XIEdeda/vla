#ifndef MY_DUAL_ARM_PACKAGE_DATA_COLLECTOR_HPP
#define MY_DUAL_ARM_PACKAGE_DATA_COLLECTOR_HPP

#include "my_dual_arm_package/utils_config.hpp"
#include "my_dual_arm_package/ros2_components.hpp"
#include <atomic>

namespace my_dual_arm_package {


class DoubleArmRGBDDataProcessor {
public:
    static std::shared_ptr<DoubleArmRGBDDataProcessor> get_instance(const std::string& robot_type = "Keenon_F1_DoubleArm", const std::vector<std::string>& preview_camera_names = {});
    static std::shared_ptr<DoubleArmRGBDDataProcessor> get_current_instance();
    static void reset_instance();
    
    bool start_episode(const std::string& temp_folder = "", const std::string& data_folder = "");
    void end_episode();
    void close();
    void data_postprocess(const std::string& temp_folder, const std::string& data_folder);
    void recover();

    std::shared_ptr<ROS2RGBDCameraSubscriber> camera_sub;
    std::shared_ptr<ROS2DoubleArmInterface> arm_interface;
    
    std::map<std::string, int> camera_frames_count;
    std::map<std::string, int> joint_frames_count;
    bool is_preview_mode;
    std::atomic<bool> is_recording;

private:
    DoubleArmRGBDDataProcessor(const std::string& robot_type, const std::vector<std::string>& preview_camera_names);

    void _start_record_async_task();

    void _stop_record_async_task();

    void _async_record_loop();

    bool _record_frame();

    void _force_first_record();

    void _write_h5_data();

    void _write_arm_data_txt();

    void _spin_executor();

    void _spin_arm_executor();

    static std::shared_ptr<DoubleArmRGBDDataProcessor> _instance;

    static std::mutex _instance_lock;

    std::string robot_type;           
    std::vector<std::string> camera_names; 
    fs::path root_dir;                
    std::map<std::string, fs::path> dirs; 
    double frequency;                 
    double record_interval;           

    rclcpp::executors::MultiThreadedExecutor::SharedPtr executor;
    rclcpp::executors::SingleThreadedExecutor::SharedPtr arm_executor;
    std::thread executor_thread;       
    std::thread arm_executor_thread;   
    std::thread record_thread;         
    std::atomic<bool> is_recording_thread_running; 
    std::atomic<bool> stop_executor_flag;          
    
    double episode_start_ros_time;     
    std::vector<json> cam_episode_data;  
    std::vector<json> arm_episode_data;  

    bool use_mcap_mode_ = false;                          
    std::shared_ptr<McapWriteQueue> bag_queue_;           

    std::mutex lock;                   
};

} 

#endif
