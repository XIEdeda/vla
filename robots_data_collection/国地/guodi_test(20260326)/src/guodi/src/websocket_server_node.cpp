#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/strand.hpp>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <set>

#include "guodi/data_collector.hpp"
// #include "guodi/http_file_server.hpp"

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

#include <boost/asio/steady_timer.hpp>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include <cstdlib>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>

namespace fs = std::filesystem;

// 声明使用ros2_components.cpp中的Base64编码函数
extern std::string base64_encode(const unsigned char* bytes_to_encode, size_t in_len);

namespace guodi {

// 连接计数器
static int active_connections = 0;

/// 自动获取 eth 网卡的 IPv4 地址，若未找到则返回 "127.0.0.1"
std::string get_eth_ip_address();

// 帧率配置(frequency已在utils_config.hpp中声明)
static constexpr int joint_push_fps = 100;          // 机械臂推送帧率
static constexpr int img_quality = 70;              // 图像压缩质量

// ==============================
// WebSocket 会话类: 处理单个客户端连接
// ==============================
class Session : public std::enable_shared_from_this<Session> {
    websocket::stream<beast::tcp_stream> ws_;
    beast::flat_buffer buffer_;
    std::shared_ptr<DoubleArmRGBDDataProcessor> processor_;  ///< 数据处理器实例
    std::shared_ptr<net::steady_timer> sampler_timer_;       ///< 采集进度推送定时器
    std::shared_ptr<net::steady_timer> stream_timer_;        ///< 相机帧推送定时器
    std::deque<std::string> write_queue_;                    ///< WebSocket发送队列(串行化写入)
    bool is_writing_ = false;                                ///< 是否正在执行async_write
    int stream_push_count_ = 0;                              ///< 帧推送轮数统计
    std::string data_folder_;                                ///< 当前采集会话的数据目录(对应Python server_state.data_folder)
    bool is_data_saved_ = false;                             ///< 数据是否已保存完成(对应Python server_state.is_data_saved)
    
    // 相机帧率统计相关
    struct CameraStats {
        int success_count = 0;                               ///< 成功推送的帧数
        std::chrono::steady_clock::time_point start_time;    ///< 统计起始时间
        CameraStats() : start_time(std::chrono::steady_clock::now()) {}
    };
    std::map<std::string, CameraStats> camera_stats_;        ///< 每个相机的统计信息
    const int stats_interval_ = 30;                          ///< 统计间隔(每30次循环打印一次)
    
    /// 客户端断开时清理: 取消定时器、停止采集/预览，避免无客户端时继续推送帧和保存数据
    void cleanup_on_disconnect() {
        if (sampler_timer_) { sampler_timer_->cancel(); sampler_timer_.reset(); }
        if (stream_timer_) { stream_timer_->cancel(); stream_timer_.reset(); }

        if (processor_) {
            if (processor_->is_recording) {
                std::cout << "[WebSocket] 客户端断开: 停止采集并落盘" << std::endl;
                processor_->end_episode();
                is_data_saved_ = true;
                // 采集模式下客户端意外断开: 落盘完成后丢弃孤儿数据,避免占用磁盘
                if (!data_folder_.empty()) {
                    try {
                        fs::path target_path(data_folder_);
                        if (target_path.filename() == "data") {
                            target_path = target_path.parent_path();
                            std::cout << "[WebSocket] 客户端断开: 检测到路径为data子目录，将删除外层Session目录: " << target_path.string() << std::endl;
                        } else {
                            std::cout << "[WebSocket] 客户端断开: 丢弃采集数据，删除目录: " << target_path.string() << std::endl;
                        }
                        if (fs::exists(target_path)) {
                            fs::remove_all(target_path);
                            std::cout << "[WebSocket] 客户端断开: 孤儿数据已删除" << std::endl;
                        }
                        data_folder_.clear();
                        is_data_saved_ = false;
                    } catch (const std::exception& e) {
                        std::cout << "[WebSocket] 客户端断开: 丢弃数据失败 " << e.what() << std::endl;
                    }
                }
            } else if (processor_->camera_sub) {
                processor_->camera_sub->is_preview_active = false;
                std::cout << "[WebSocket] 客户端断开: 已关闭预览处理流" << std::endl;
            }
            processor_.reset();
        }
    }
    
public:
    explicit Session(tcp::socket&& socket) : ws_(std::move(socket)) {}

    /// 采集进度定时推送: 相机帧数、机械臂帧数、电机状态等
    void start_sampler_push(int mid) {
        sampler_timer_ = std::make_shared<net::steady_timer>(ws_.get_executor(), std::chrono::milliseconds(33));
        sampler_timer_->async_wait([this, mid, self = shared_from_this()](const boost::system::error_code& ec) {
            if (ec) return;
            
            auto proc = processor_;
            if (!proc || !proc->is_recording) return;

            json progress;
            progress["cmd"] = "SEND_SAMPLER_PROCESS";
            progress["mid"] = mid;
            progress["type"] = "SamplerProgressPayload";
            
            json cam_frames = json::array();
            for (auto const& [name, count] : proc->camera_frames_count) {
                cam_frames.push_back({{"name", name}, {"count", count}});
            }
            
            json joint_frames = {
                {{"name", "left_arm"}, {"count", proc->joint_frames_count["left"]}},
                {{"name", "right_arm"}, {"count", proc->joint_frames_count["right"]}}
            };

            progress["data"] = {
                {"camera_frames", cam_frames},
                {"joint_frames", joint_frames},
                {"motor_status", proc->arm_interface ? proc->arm_interface->motor_error_status : "未知"},
                {"camera_status", proc->camera_sub ? proc->camera_sub->get_camera_status() : "未知"},
                {"push_fps", joint_push_fps}
            };

            queue_message(progress.dump());
            start_sampler_push(mid);
        });
    }

    /// 相机帧定时推送(带写入串行化和帧率统计)
    void start_stream_push(int mid) {
        stream_timer_ = std::make_shared<net::steady_timer>(ws_.get_executor(), std::chrono::milliseconds(33));
        stream_timer_->async_wait([this, mid, self = shared_from_this()](const boost::system::error_code& ec) {
            if (ec) return;
            
            auto proc = processor_;
            if (!proc || !proc->camera_sub) return;

            int frames_found = 0;
            // 先收集所有有效的帧数据
            std::vector<std::pair<std::string, json>> valid_frames;
            for (const auto& cam_name : proc->camera_sub->camera_names) {
                json frame_data = proc->camera_sub->get_current_frame(cam_name, "color");
                if (!frame_data.empty() && frame_data.contains("image_data")) {
                    valid_frames.emplace_back(cam_name, frame_data);
                    frames_found++;
                    // 只在这里计数一次！
                    camera_stats_[cam_name].success_count++;
                }
            }
            
            // 推送有效的帧
            for (const auto& [cam_name, frame_data] : valid_frames) {
                json msg;
                msg["cmd"] = "SEND_CAMERA_FRAME";
                msg["mid"] = mid;
                msg["type"] = "CameraFramePayload";
                json msg_data;
                msg_data["image"] = frame_data["image_data"];
                msg_data["image_type"] = "BASE64_JPG";
                msg_data["camera_name"] = cam_name;
                msg_data["width"] = frame_data.value("width", get_camera_param(cam_name, "color", "width"));
                msg_data["height"] = frame_data.value("height", get_camera_param(cam_name, "color", "height"));
                msg_data["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                msg_data["source_fps"] = get_camera_param(cam_name, "color", "fps");
                msg_data["push_fps"] = get_camera_param(cam_name, "color", "fps");
                msg["data"] = msg_data;
                queue_message(msg.dump());
                // 不在这里重复计数！
            }
            
            stream_push_count_++;
            
            // 判断当前是采集模式还是预览模式
            bool is_recording_mode = proc->is_recording;
            
            if (is_recording_mode) {
                // 采集模式：不进行任何帧率统计
            } else {
                // 预览模式：按时间窗口统计帧率
                if (stream_push_count_ % stats_interval_ == 0) {
                    auto current_time = std::chrono::steady_clock::now();
                    std::vector<std::string> fps_logs;
                    
                    for (const auto& cam_name : proc->camera_sub->camera_names) {
                        auto& stats = camera_stats_[cam_name];
                        auto elapsed = std::chrono::duration<double>(current_time - stats.start_time).count();
                        
                        // 实际帧率 = 成功推送帧数 / 时间间隔
                        double fps = elapsed > 0 ? stats.success_count / elapsed : 0.0;
                        
                        // 保留1位小数（强制格式化）
                        char fps_str[10];
                        snprintf(fps_str, sizeof(fps_str), "%.1f", fps);
                        fps_logs.push_back(cam_name + ": " + std::string(fps_str));
                        
                        // 重置统计
                        stats.success_count = 0;
                        stats.start_time = current_time;
                    }
                    
                    // 打印所有相机的帧率
                    std::string fps_log_str;
                    for (size_t i = 0; i < fps_logs.size(); ++i) {
                        if (i > 0) fps_log_str += ", ";
                        fps_log_str += fps_logs[i];
                    }
                    std::cout << "[WebSocket] 预览模式相机实际推送FPS: " << fps_log_str << std::endl;
                }
            }

            start_stream_push(mid);
        });
    }

    /// 消息入队(保证WebSocket写入串行化,避免并发async_write导致崩溃)
    void queue_message(std::string msg) {
        // 使用 post 确保对队列的操作在 io_context 线程中串行执行
        net::post(ws_.get_executor(), [this, self = shared_from_this(), msg = std::move(msg)]() mutable {
            write_queue_.push_back(std::move(msg));
            if (!is_writing_) {
                do_write();
            }
        });
    }

    /// 从队列取消息并发送(写入完成后自动取下一条)
    void do_write() {
        if (write_queue_.empty()) {
            is_writing_ = false;
            return;
        }
        is_writing_ = true;
        auto msg = std::make_shared<std::string>(std::move(write_queue_.front()));
        write_queue_.pop_front();
        
        ws_.async_write(net::buffer(*msg),
            beast::bind_front_handler(&Session::on_write, shared_from_this(), msg));
    }

    void on_write(std::shared_ptr<std::string> msg, beast::error_code ec, std::size_t bytes_transferred) {
        boost::ignore_unused(msg);
        boost::ignore_unused(bytes_transferred);
        if (ec) {
            if (ec != net::error::operation_aborted) {
                std::cout << "[WebSocket] 消息发送失败: " << ec.message() << std::endl;
            }
            is_writing_ = false;
            write_queue_.clear();
            return;
        }
        do_write();
    }

    /// 连接建立后启动读取循环
    void run() {
        // 设置更大的消息大小限制（1000MB）
        ws_.read_message_max(1000 * 1024 * 1024);
        // 设置写入缓冲区大小
        ws_.write_buffer_bytes(1000 * 1024 * 1024);
        ws_.async_accept(beast::bind_front_handler(&Session::on_accept, shared_from_this()));
    }

    /// 接受连接后开始读取消息
    void on_accept(beast::error_code ec) {
        if (ec) {
            std::cout << "[WebSocket] 接受连接失败: " << ec.message() << std::endl;
            return;
        }
        active_connections++;
        std::cout << "[WebSocket] 新客户端连接已建立,当前活跃客户端数:" << active_connections << std::endl;
        do_read();
    }

    /// 异步读取客户端消息
    void do_read() {
        // 确保在读取前 buffer 是干净的
        buffer_.consume(buffer_.size());
        ws_.async_read(buffer_, beast::bind_front_handler(&Session::on_read, shared_from_this()));
    }

    /// 消息接收回调: 解析JSON指令并分发处理
    void on_read(beast::error_code ec, std::size_t bytes_transferred) {
        boost::ignore_unused(bytes_transferred);
        
        // 1. 检查错误码(客户端断开或读取出错)
        if (ec) {
            active_connections = std::max(0, active_connections - 1);
            if (ec == websocket::error::closed || 
                ec == net::error::eof || 
                ec == net::error::connection_reset) {
                std::cout << "[WebSocket] 客户端连接已正常断开,当前活跃数:" << active_connections << std::endl;
            } else {
                std::cout << "[WebSocket] 连接读取出错 (" << ec.value() << "): " << ec.message() << std::endl;
            }
            // 关键: 断开时立即清理，停止预览/采集的帧推送和数据保存
            cleanup_on_disconnect();
            return;
        }

        // 2. 处理消息
        std::string message = beast::buffers_to_string(buffer_.data());
        buffer_.consume(buffer_.size());
        
        try {
            json msg = json::parse(message);
            std::string cmd = msg.at("cmd").get<std::string>();
            int mid = msg.at("mid").get<int>();
            json data = msg.contains("data") ? msg["data"] : json::object();

            std::cout << "[WebSocket] 收到客户端指令: cmd=" << cmd << ", mid=" << mid << std::endl;
            handle_command(cmd, mid, data);
        } catch (const std::exception& e) {
            std::cout << "[WebSocket] 指令解析失败: " << e.what() << std::endl;
        }

        // 3. 继续读取下一条消息
        do_read();
    }

    /// 指令分发处理: 根据cmd字段执行对应操作
    void handle_command(const std::string& cmd, int mid, const json& data) {
        json response;
        // ====== GET_INFO: 获取机器人信息 ======
        if (cmd == "GET_INFO") {
            response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "GetInfoPayload"}, {"data", {
                {"keenon_cmd_ver", "2.0.0"},
                {"model", "Keenon_F1_DoubleArm"},
                {"data_format", "COMMON_V2"},
                {"cameras", json::array()},
                {"joint_push_fps", 100}
            }}};
            for (const auto& cam : ROS2_CAMERA_TOPIC_MAP) {
                int cam_fps = get_camera_param(cam.first, "color", "fps");
                response["data"]["cameras"].push_back({
                    {"name", cam.first},
                    {"fps", cam_fps},
                    {"push_fps", cam_fps},
                    {"support_preview", cam.first != "head_fisheye"}
                });
            }

        // ====== START_PREVIEW: 启动相机预览 ======
        } else if (cmd == "START_PREVIEW") {
            std::vector<std::string> cameras = data.contains("camera_names") ? data["camera_names"].get<std::vector<std::string>>() : std::vector<std::string>{};
            std::cout << "[WebSocket] START_PREVIEW: 相机列表=[";
            for (size_t i = 0; i < cameras.size(); ++i) { std::cout << cameras[i]; if (i+1 < cameras.size()) std::cout << ","; }
            std::cout << "]" << std::endl;

            try {
                // 步骤1: 获取/创建处理器实例(单例模式)
                processor_ = DoubleArmRGBDDataProcessor::get_instance("Keenon_F1_DoubleArm", cameras);
                std::cout << "[WebSocket] START_PREVIEW: 处理器实例已就绪" << std::endl;

                // 步骤2: 启动相机订阅(关键! 创建ROS2 topic订阅，预览不落盘，无需 dirs)
                processor_->camera_sub->start();
                std::cout << "[WebSocket] START_PREVIEW: 相机ROS2订阅已启动" << std::endl;

                // 步骤3: 启动帧推送定时器
                start_stream_push(mid);
                std::cout << "[WebSocket] START_PREVIEW: 帧推送定时器已启动(30fps)" << std::endl;

                response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}};
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] START_PREVIEW 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        // ====== STOP_PREVIEW: 停止相机预览 ======
        } else if (cmd == "STOP_PREVIEW") {
            std::cout << "[WebSocket] STOP_PREVIEW: 停止预览" << std::endl;
            if (stream_timer_) stream_timer_->cancel();
            if (processor_ && processor_->camera_sub) {
                processor_->camera_sub->is_preview_active = false;
                std::cout << "[WebSocket] STOP_PREVIEW: 已关闭预览处理流" << std::endl;
            }
            response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}};
            
        // ====== RECOVER: 系统状态恢复(重置所有状态) ======
        } else if (cmd == "RECOVER") {
            std::cout << "[WebSocket] RECOVER: 系统状态恢复中..." << std::endl;
            
            // 1. 先取消所有定时器，防止产生新的推送请求
            if (sampler_timer_) { sampler_timer_->cancel(); std::cout << "[WebSocket] RECOVER: 采集进度定时器已取消" << std::endl; }
            if (stream_timer_) { stream_timer_->cancel(); std::cout << "[WebSocket] RECOVER: 帧推送定时器已取消" << std::endl; }
            
            // 2. 释放处理器资源
            if (processor_) {
                try {
                    // 调用 close 停止内部线程池、执行器和采集循环
                    processor_->close();
                    std::cout << "[WebSocket] RECOVER: 处理器资源已释放" << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "[WebSocket] RECOVER: 处理器释放异常: " << e.what() << std::endl;
                } catch (...) {
                    std::cout << "[WebSocket] RECOVER: 处理器释放未知异常" << std::endl;
                }
                // 关键：将 Session 持有的拷贝置空，这样异步回调中 if(!proc) 就能生效
                processor_.reset();
            }
            
            // 3. 重置全局单例，确保下次 get_instance 创建全新实例
            DoubleArmRGBDDataProcessor::reset_instance();
            
            // 4. 重置 Session 内部状态
            stream_push_count_ = 0;
            camera_stats_.clear();
            data_folder_.clear();
            is_data_saved_ = false;
            
            // 注意：不要强行重置 is_writing_ 和 write_queue_
            // 这样正在进行的发送任务可以正常结束，避免 Beast 内部断言失败
            
            std::cout << "[WebSocket] RECOVER: 系统已完全重置" << std::endl;
            response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}};

        // ====== START_TELEMETRY: 启动数据采集 ======
        } else if (cmd == "START_TELEMETRY") {
            std::string data_folder;
            const fs::path default_root = "/mnt/guodi_dataset";
            std::string raw_folder = data.contains("data_folder") ? data["data_folder"].get<std::string>() : "";

            if (!raw_folder.empty()) {
                if (raw_folder.front() == '/') {
                    // 绝对路径: 直接使用
                    data_folder = raw_folder;
                    std::cout << "[WebSocket] START_TELEMETRY: 使用用户指定绝对路径=" << data_folder << std::endl;
                } else {
                    // 相对路径/文件夹名: 落在默认根目录下
                    fs::path episode_folder = default_root / raw_folder;
                    fs::create_directories(episode_folder);
                    data_folder = episode_folder.string();
                    std::cout << "[WebSocket] START_TELEMETRY: 使用默认根目录+子文件夹=" << data_folder << std::endl;
                }
            } else {
                // 未指定: 自动生成时间戳路径
                auto now = std::chrono::system_clock::now();
                auto in_time_t = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
                fs::path episode_folder = default_root / ss.str();
                // fs::path auto_data_folder = episode_folder / "data";
                // fs::create_directories(auto_data_folder);
                // data_folder = auto_data_folder.string();
                fs::create_directories(episode_folder);
                data_folder = episode_folder.string();
                std::cout << "[WebSocket] START_TELEMETRY: 自动生成路径=" << data_folder << std::endl;
            }

            // 关键修复：将局部变量 data_folder 赋值给 Session 的成员变量 data_folder_
            this->data_folder_ = data_folder; 
            this->is_data_saved_ = false;      // 新采集周期,重置保存标志

            processor_ = DoubleArmRGBDDataProcessor::get_instance();
            std::cout << "[WebSocket] START_TELEMETRY: 处理器实例已就绪,开始采集..." << std::endl;
            
            bool success = processor_->start_episode("", data_folder);
            if (success) {
                start_sampler_push(mid);
                start_stream_push(mid);
                std::cout << "[WebSocket] START_TELEMETRY: 采集启动成功,进度+帧推送已启动" << std::endl;
            } else {
                std::cout << "[WebSocket] START_TELEMETRY: 采集启动失败" << std::endl;
                this->data_folder_.clear(); // 启动失败则清空路径
            }
            response = {{"cmd", success ? "ACK" : "ERROR"}, {"mid", mid}, {"type", "EmptyPayload"}};
            if (!success) response["data"] = {{"code", 500}, {"message", "采集启动失败"}};

        // ====== STOP_TELEMETRY: 停止数据采集 ======
        } else if (cmd == "STOP_TELEMETRY") {
            std::cout << "[WebSocket] STOP_TELEMETRY: 停止采集" << std::endl;
            if (sampler_timer_) sampler_timer_->cancel();
            if (stream_timer_) stream_timer_->cancel();
            if (processor_) {
                processor_->end_episode();
                is_data_saved_ = true;  // 标记数据已保存完成
                std::cout << "[WebSocket] STOP_TELEMETRY: 数据落盘完成" << std::endl;
            }
            response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}};

        // ====== DATA_POSTPROCESS: 数据后处理+tar压缩 ======
        } else if (cmd == "DATA_POSTPROCESS") {
            try {
                // 对齐Python: 使用会话状态中保存的data_folder,而非从请求中读取
                if (!processor_) {
                    throw std::runtime_error("数据处理器未初始化");
                }
                if (!is_data_saved_) {
                    throw std::runtime_error("数据未保存完成");
                }
                if (data_folder_.empty()) {
                    throw std::runtime_error("无数据路径记录");
                }

                fs::path source_path(data_folder_);
                if (!fs::exists(source_path)) {
                    throw std::runtime_error("源路径不存在: " + data_folder_);
                }

                // 校验关键文件是否存在
                std::vector<std::string> required = {"record", "camera"};
                std::string missing;
                for (const auto& f : required) {
                    if (!fs::exists(source_path / f)) {
                        if (!missing.empty()) missing += ", ";
                        missing += f;
                    }
                }
                if (!missing.empty()) {
                    throw std::runtime_error("源路径缺少关键文件: " + missing);
                }

                // 确定压缩目标目录和文件名
                std::string target_compress = data.contains("target_compress_folder") ? data["target_compress_folder"].get<std::string>() : "";
                std::string tar_name = data.contains("tar_filename") ? data["tar_filename"].get<std::string>() : "";

                fs::path target_folder = target_compress.empty() ? source_path.parent_path() : fs::path(target_compress);
                fs::create_directories(target_folder);

                std::string source_folder_name = source_path.filename().string();
                if (tar_name.empty()) {
                    tar_name = source_folder_name + "_compressed.tar";
                }
                fs::path tar_file = target_folder / tar_name;

                // 删除旧压缩包(如果存在)
                if (fs::exists(tar_file)) {
                    fs::remove(tar_file);
                }

                // 执行tar压缩
                std::string tar_cmd = "tar -cf " + tar_file.string() + " -C " + source_path.parent_path().string() + " " + source_folder_name;
                std::cout << "[WebSocket] DATA_POSTPROCESS: 开始tar压缩: " << tar_cmd << std::endl;
                int res = std::system(tar_cmd.c_str());

                if (res != 0) {
                    throw std::runtime_error("tar压缩失败(返回码:" + std::to_string(res) + ")");
                }
                if (!fs::exists(tar_file) || fs::file_size(tar_file) == 0) {
                    if (fs::exists(tar_file)) fs::remove(tar_file);
                    throw std::runtime_error("生成的压缩包不存在或为空");
                }

                double size_mb = static_cast<double>(fs::file_size(tar_file)) / (1024 * 1024);
                std::cout << "[WebSocket] DATA_POSTPROCESS: 压缩完成, 大小=" << size_mb << "MB" << std::endl;
                response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "PostProgressPayload"}, {"data", {
                    {"progress", 1.0},
                    {"msg", "采集文件夹tar压缩完成"},
                    {"data_path", data_folder_},
                    {"tar_file_path", tar_file.string()},
                    {"tar_file_size_mb", size_mb}
                }}};
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] DATA_POSTPROCESS 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"message", e.what()}}}};
            }

        // ====== DISCARD_DATA: 丢弃采集数据 ======
        } else if (cmd == "DISCARD_DATA") {
            try {
                // 对齐Python: 使用会话状态中保存的data_folder,而非从请求中读取
                if (data_folder_.empty()) {
                    throw std::runtime_error("无数据路径记录，无法执行丢弃操作");
                }

                fs::path target_path(data_folder_);

                // 如果当前路径是采集子目录'data',删除外层Session目录
                if (target_path.filename() == "data") {
                    fs::path parent_path = target_path.parent_path();
                    std::cout << "[WebSocket] DISCARD_DATA: 检测到路径为data子目录，将删除外层Session目录: " << parent_path.string() << std::endl;
                    target_path = parent_path;
                }

                std::cout << "[WebSocket] DISCARD_DATA: 准备删除文件夹: " << target_path.string() << std::endl;

                if (fs::exists(target_path)) {
                    fs::remove_all(target_path);
                    std::cout << "[WebSocket] DISCARD_DATA: 文件夹及其内容已成功删除" << std::endl;
                } else {
                    std::cout << "[WebSocket] DISCARD_DATA: 目标路径不存在: " << target_path.string() << std::endl;
                }

                // 重置会话状态
                std::string deleted_path = data_folder_;
                data_folder_.clear();
                is_data_saved_ = false;

                response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}, {"data", {
                    {"msg", "数据丢弃成功"},
                    {"deleted_path", deleted_path}
                }}};
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] DISCARD_DATA 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"message", e.what()}}}};
            }

        } else {
            std::cout << "[WebSocket] 不支持的指令类型: " << cmd << std::endl;
            response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"message", "不支持的指令类型: " + cmd}}}};
        }
                
        std::cout << "[WebSocket] 指令处理完成 cmd=" << cmd << ", mid=" << mid 
                  << ", 结果=" << (response["cmd"] == "ACK" ? "成功" : "失败") << std::endl;
        queue_message(response.dump());
    }
};

// ==============================
// WebSocket 监听器: 接受新连接并创建会话
// ==============================
class Listener : public std::enable_shared_from_this<Listener> {
    net::io_context& ioc_;
    tcp::acceptor acceptor_;

public:
    Listener(net::io_context& ioc, tcp::endpoint endpoint)
        : ioc_(ioc), acceptor_(net::make_strand(ioc)) {
        beast::error_code ec;
        acceptor_.open(endpoint.protocol(), ec);
        acceptor_.set_option(net::socket_base::reuse_address(true), ec);
        acceptor_.bind(endpoint, ec);
        acceptor_.listen(net::socket_base::max_listen_connections, ec);
    }

    void run() { do_accept(); }

private:
    void do_accept() {
        acceptor_.async_accept(net::make_strand(ioc_), beast::bind_front_handler(&Listener::on_accept, shared_from_this()));
    }

    void on_accept(beast::error_code ec, tcp::socket socket) {
        if (!ec) std::make_shared<Session>(std::move(socket))->run();
        do_accept();
    }
};

/// 自动获取 eth 网卡(eth0/eth1/...)的 IPv4 地址，若未找到则返回 "127.0.0.1"
std::string get_eth_ip_address() {
    struct ifaddrs* ifap = nullptr;
    if (getifaddrs(&ifap) != 0) {
        return "127.0.0.1";
    }
    std::string result = "127.0.0.1";
    for (struct ifaddrs* ifa = ifap; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
        if (!(ifa->ifa_flags & IFF_UP)) continue;
        if (strncmp(ifa->ifa_name, "eth", 3) != 0) continue;  // eth0, eth1, ...
        const auto* sa = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
        char buf[INET_ADDRSTRLEN];
        if (inet_ntop(AF_INET, &sa->sin_addr, buf, sizeof(buf))) {
            result = buf;
            break;  // 使用第一个找到的 eth 接口
        }
    }
    freeifaddrs(ifap);
    return result;
}

} // namespace guodi

// ==============================
// 程序入口: 初始化ROS2并启动WebSocket服务
// ==============================
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    
    // 预初始化：使机械臂和相机订阅在程序启动时即生效
    guodi::DoubleArmRGBDDataProcessor::get_instance("Keenon_F1_DoubleArm", {});
    std::cout << "机械臂与相机订阅已预初始化(程序启动即订阅)" << std::endl;

    auto const address = net::ip::make_address("127.0.0.1");
    // auto const address = net::ip::make_address("172.16.9.71");

    // std::string bind_ip = guodi::get_eth_ip_address();
    // auto const address = net::ip::make_address(bind_ip);
    auto const port = static_cast<unsigned short>(10780);
    auto const threads = 1;

    net::io_context ioc{threads};
    std::make_shared<guodi::Listener>(ioc, tcp::endpoint{address, port})->run();
    std::cout << "WebSocket服务器启动成功,监听地址: ws://127.0.0.1:10780" << std::endl;
    // std::cout << "WebSocket服务器启动成功,监听地址: ws://172.16.9.71:10780" << std::endl;
    // std::cout << "WebSocket服务器启动成功,监听地址: ws://" << bind_ip << ":" << port << std::endl;

    ioc.run();
    
    rclcpp::shutdown();
    return 0;
}
