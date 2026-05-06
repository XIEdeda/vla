#ifndef HTTP_FILE_SERVER_HPP
#define HTTP_FILE_SERVER_HPP

#include <string>
#include <thread>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <regex>

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
        

        server_.Get(R"(/files/(.*))", [this](const httplib::Request& req, httplib::Response& res) {
            try {

                std::string requested_file = req.matches[1].str();
                
                fs::path requested_path = fs::path(base_path_) / requested_file;
                fs::path canonical_requested;
                try {

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
                
                std::string canonical_req_str = canonical_requested.string();
                std::string canonical_base_str = canonical_base.string();
                if (canonical_req_str.find(canonical_base_str) != 0) {
                    res.status = 403;
                    res.set_content("Forbidden: Path traversal detected. Requested: " + canonical_req_str + ", Base: " + canonical_base_str, "text/plain");
                    return;
                }
                
                std::string file_path = canonical_requested.string();
                

                if (!fs::exists(file_path)) {
                    res.status = 404;
                    res.set_content("File not found: " + file_path, "text/plain");
                    std::cout << "[HTTP Server] File not found, returning 404" << std::endl;
                    return;
                }
                

                if (!fs::is_regular_file(file_path)) {
                    res.status = 400;
                    res.set_content("Not a regular file", "text/plain");
                    return;
                }
                
                
                res.set_header("Content-Type", "application/octet-stream");
                res.set_header("Accept-Ranges", "bytes");
                

                auto range_header = req.get_header_value("Range");
                if (!range_header.empty()) {
                    handleRangeRequest(file_path, range_header, res);
                } else {

                    res.set_file_content(file_path, "application/octet-stream");
                }
                
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(("Internal Server Error: " + std::string(e.what())).c_str(), "text/plain");
            }
        });
        

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
                

                if (start >= file_size || end >= file_size || start > end) {
                    res.status = 416; 
                    res.set_header("Content-Range", "bytes */" + std::to_string(file_size));
                    return;
                }
                
                std::streamsize content_length = end - start + 1;
                
                file.seekg(start);
                
                std::vector<char> buffer(content_length);
                file.read(buffer.data(), content_length);
                
                res.status = 206; 
                res.set_header("Content-Range", "bytes " + std::to_string(start) + "-" + 
                                             std::to_string(end) + "/" + std::to_string(file_size));
                res.set_header("Content-Length", std::to_string(content_length));
                res.set_content(buffer.data(), content_length, "application/octet-stream");
                
            } else {
                res.set_file_content(file_path, "application/octet-stream");
            }
        } catch (...) {
            res.set_file_content(file_path, "application/octet-stream");
        }
    }
};

#endif // HTTP_FILE_SERVER_HPP