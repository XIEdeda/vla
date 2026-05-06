#ifndef GUODI_DATA_COLLECTOR_HPP
#define GUODI_DATA_COLLECTOR_HPP

#include "guodi/utils_config.hpp"
#include "guodi/ros2_components.hpp"
#include <atomic>

namespace guodi {

// ==============================
// 核心数据处理器(单例模式)
// 负责: 目录创建、采集周期管理、H5写入、相机+机械臂協调
// ==============================
class DoubleArmRGBDDataProcessor {
public:
    /// 单例获取: 首次调用创建实例,后续调用复用/更新配置
    static std::shared_ptr<DoubleArmRGBDDataProcessor> get_instance(const std::string& robot_type = "Keenon_F1_DoubleArm", const std::vector<std::string>& preview_camera_names = {});
    /// 只读获取当前实例(不触发配置更新,用于 robot_status 等查询; 若未创建则返回空)
    static std::shared_ptr<DoubleArmRGBDDataProcessor> get_current_instance();
    /// 重置单例(RECOVER时调用,释放全局实例使下次get_instance创建新实例)
    static void reset_instance();
    
    /// 启动采集周期: 初始化目录、重置缓存、启动相机和机械臂采集
    bool start_episode(const std::string& temp_folder = "", const std::string& data_folder = "");
    /// 结束采集周期: 停止机械臂、排空缓存、写入H5/TXT、更新meta_info
    void end_episode();
    /// 彻底释放资源: 停止执行器、销毁节点
    void close();
    /// 数据后处理: 复制相机帧和H5数据到目标目录
    void data_postprocess(const std::string& temp_folder, const std::string& data_folder);
    /// 系统状态恢复: 重置机械臂缓存、停止相机订阅
    void recover();

    /// 相机订阅器(ROS2节点)
    std::shared_ptr<ROS2RGBDCameraSubscriber> camera_sub;
    /// 双臂接口(ROS2节点)
    std::shared_ptr<ROS2DoubleArmInterface> arm_interface;
    
    /// 相机帧计数(key=相机名, value=计数)
    std::map<std::string, int> camera_frames_count;
    /// 机械臂帧计数(key=left/right, value=计数)
    std::map<std::string, int> joint_frames_count;
    /// 是否为预览模式
    bool is_preview_mode;
    /// 采集状态标志(原子操作,线程安全)
    std::atomic<bool> is_recording;

private:
    /// 私有构造函数(单例模式,禁止外部直接构造)
    DoubleArmRGBDDataProcessor(const std::string& robot_type, const std::vector<std::string>& preview_camera_names);
    
    /// 启动异步采集任务(创建采集线程)
    void _start_record_async_task();
    /// 停止异步采集任务(等待采集线程退出)
    void _stop_record_async_task();
    /// 采集循环主体: 按record_interval间隔反复调用_record_frame
    void _async_record_loop();
    /// 单次采集: 从相机取帧、校验、保存、记录时间戳
    bool _record_frame();
    /// 强制首次采集(解决调度延迟问题)
    void _force_first_record();
    /// 写入H5数据: 将机械臂数据(state/action)写入raw_joints.h5
    void _write_h5_data();
    /// 写入TXT数据: 将左右臂原始数据分别写入left_data.txt/right_data.txt
    void _write_arm_data_txt();
    /// 相机执行器自旋(多线程,崩溃后自动重启)
    void _spin_executor();
    /// 双臂独立执行器自旋(单线程高优先级)
    void _spin_arm_executor();

    /// 单例实例指针
    static std::shared_ptr<DoubleArmRGBDDataProcessor> _instance;
    /// 单例锁
    static std::mutex _instance_lock;

    std::string robot_type;           ///< 机器人类型("Keenon_F1_DoubleArm")
    std::vector<std::string> camera_names; ///< 相机列表
    fs::path root_dir;                ///< 数据根目录
    std::map<std::string, fs::path> dirs; ///< 目录映射(raw_joints_h5、meta_info等)
    double frequency;                 ///< 采集频率(Hz)
    double record_interval;           ///< 采集间隔(秒)
    
    /// 相机执行器(多线程)
    rclcpp::executors::MultiThreadedExecutor::SharedPtr executor;
    /// 双臂独立执行器(单线程)
    rclcpp::executors::SingleThreadedExecutor::SharedPtr arm_executor;
    std::thread executor_thread;       ///< 相机执行器线程
    std::thread arm_executor_thread;   ///< 双臂执行器线程
    std::thread record_thread;         ///< 采集线程
    std::atomic<bool> is_recording_thread_running; ///< 采集线程运行状态
    std::atomic<bool> stop_executor_flag;          ///< 执行器停止标志
    
    double episode_start_ros_time;     ///< 采集开始时间(ROS时间戳,秒)
    std::vector<json> cam_episode_data;  ///< 相机采集数据(时间戳+同步统计)
    std::vector<json> arm_episode_data;  ///< 机械臂采集数据(100Hz)
    
    std::mutex lock;                   ///< 内部互斥锁
};

} // namespace guodi

#endif // GUODI_DATA_COLLECTOR_HPP
