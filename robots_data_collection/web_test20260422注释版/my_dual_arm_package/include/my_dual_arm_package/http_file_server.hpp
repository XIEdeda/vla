#ifndef HTTP_FILE_SERVER_HPP
#define HTTP_FILE_SERVER_HPP

#include <string>
#include <thread>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <regex>

// cpp-httplib header-only library
#include "external/httplib.h"

namespace fs = std::filesystem;

class HttpFileServer {
private:
    httplib::Server server_;
    std::thread server_thread_;
    std::atomic<bool> running_{false};
    int port_;
    std::string base_path_;

public:
    HttpFileServer(int port = 18080, const std::string& base_path = "/mnt/keenon_baidu_dataset")
        : port_(port), base_path_(base_path) {
        
        // 设置GET路由处理文件下载
        server_.Get(R"(/files/(.*))", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                // 从URL匹配中获取文件路径
                std::string requested_file = req.matches[1].str();
                
                //构建完整文件路径，防止路径遍历攻击
                fs::path requested_path = fs::path(base_path_) / requested_file;
                fs::path canonical_requested;
                try {
                    // 使用weakly_canonical，即使路径不存在也不抛出异常
                    canonical_requested = fs::weakly_canonical(requested_path);
                } catch (...) {
                    res.status = 404;
                    res.set_content("Invalid path: " + requested_path.string(), "text/plain");
                    return;
                }
                                
                fs::path canonical_base;
                try {
                    canonical_base = fs::weakly_canonical(base_path_);
                } catch (...) {
                    res.status = 500;
                    res.set_content("Internal Server Error", "text/plain");
                    return;
                }
                
                // 检查请求的路径是否在基础路径下（防止路径遍历）
                std::string canonical_req_str = canonical_requested.string();
                std::string canonical_base_str = canonical_base.string();
                if (canonical_req_str.find(canonical_base_str) != 0) {
                    res.status = 403;
                    res.set_content("Forbidden: Path traversal detected. Requested: " + canonical_req_str + ", Base: " + canonical_base_str, "text/plain");
                    return;
                }
                
                std::string file_path = canonical_requested.string();
                
                //检查文件是否存在
                if (!fs::exists(file_path)) {
                    res.status = 404;
                    res.set_content("File not found: " + file_path, "text/plain");
                    std::cout << "[HTTP Server] File not found, returning 404" << std::endl;
                    return;
                }
                
                // 检查是否是普通文件（不是目录等）
                if (!fs::is_regular_file(file_path)) {
                    res.status = 400;
                    res.set_content("Not a regular file", "text/plain");
                    return;
                }
                
                // 设置内容类型为二进制流
                res.set_header("Content-Type", "application/octet-stream");
                res.set_header("Accept-Ranges", "bytes");
                
                // 处理Range请求（支持断点续传）
                auto range_header = req.get_header_value("Range");
                if (!range_header.empty()) {
                    handleRangeRequest(file_path, range_header, res);
                } else {
                    //✅ 正确：让 httplib 自己流式发送文件
                    res.set_file_content(file_path, "application/octet-stream");
                }
                
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(("Internal Server Error: " + std::string(e.what())).c_str(), "text/plain");
            }
        });
        
        // 根路径返回简单状态信息
        server_.Get("/", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("HTTP File Server is running", "text/plain");
        });
    }
    
    ~HttpFileServer() {
        stop();
    }
    
    void start() {
        if (!running_) {
            running_ = true;
            server_thread_ = std::thread([this]() {
                std::cout << "[HTTP Server] Starting server on port " << port_ << std::endl;
                server_.listen("0.0.0.0", port_);
            });
        }
    }
    
    void stop() {
        if (running_) {
            running_ = false;
            server_.stop();
            if (server_thread_.joinable()) {
                server_thread_.join();
            }
            std::cout << "[HTTP Server] Server stopped" << std::endl;
        }
    }
    
    int getPort() const {
        return port_;
    }
    
    bool isRunning() const {
        return server_.is_running();
    }
    
private:
    void handleRangeRequest(const std::string& file_path, const std::string& range_header, httplib::Response& res) {
        try {
            // 解析Range头部，格式如 "bytes=100-200" 或 "bytes=100-" 或 "bytes=-100"
            std::regex range_regex(R"(bytes=(\d*)-(\d*))");
            std::smatch match;
            
            if (std::regex_match(range_header, match, range_regex)) {
                std::string start_str = match[1].str();
                std::string end_str = match[2].str();
                
                std::ifstream file(file_path, std::ios::binary | std::ios::ate);
                if (!file.is_open()) {
                    res.status = 500;
                    res.set_content("Cannot open file", "text/plain");
                    return;
                }
                
                std::streamsize file_size = file.tellg();
                file.seekg(0, std::ios::beg);
                
                std::streamsize start = start_str.empty() ? 0 : std::stoll(start_str);
                std::streamsize end = end_str.empty() ? file_size - 1 : std::stoll(end_str);
                
                // 验证范围
                if (start >= file_size || end >= file_size || start > end) {
                    res.status = 416; // Range Not Satisfiable
                    res.set_header("Content-Range", "bytes */" + std::to_string(file_size));
                    return;
                }
                
                // 计算要发送的数据长度
                std::streamsize content_length = end - start + 1;
                
                // 创建缓冲区并跳转到开始位置
                file.seekg(start);
                
                // 读取指定范围的数据
                std::vector<char> buffer(content_length);
                file.read(buffer.data(), content_length);
                
                // 设置响应头
                res.status = 206; // Partial Content
                res.set_header("Content-Range", "bytes " + std::to_string(start) + "-" + 
                                             std::to_string(end) + "/" + std::to_string(file_size));
                res.set_header("Content-Length", std::to_string(content_length));
                res.set_content(buffer.data(), content_length, "application/octet-stream");
                
            } else {
                // 如果Range格式不正确，返回整个文件
                res.set_file_content(file_path, "application/octet-stream");
            }
        } catch (...) {
            // 如果Range处理出错，回退到完整文件传输
            res.set_file_content(file_path, "application/octet-stream");
        }
    }
};

#endif // HTTP_FILE_SERVER_HPP