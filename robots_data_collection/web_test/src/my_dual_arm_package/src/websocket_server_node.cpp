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

#define CPPHTTPLIB_OPENSSL_SUPPORT
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include "my_dual_arm_package/external/httplib.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#include "my_dual_arm_package/data_collector.hpp"
#include "my_dual_arm_package/http_file_server.hpp"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <openssl/sha.h>

namespace beast = boost::beast;         
namespace http = beast::http;           
namespace websocket = beast::websocket; 
namespace net = boost::asio;            
using tcp = boost::asio::ip::tcp;       

#include <boost/asio/steady_timer.hpp>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include <cstdlib>

#include <mutex>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>

#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>
#include <cctype>

namespace fs = std::filesystem;

extern std::string base64_encode(const unsigned char* bytes_to_encode, size_t in_len);

namespace my_dual_arm_package {

static int ws_parse_mid(const json& msg) {
    if (!msg.contains("mid") || msg["mid"].is_null()) return 0;
    const auto& m = msg["mid"];
    if (m.is_number_integer()) return m.get<int>();
    if (m.is_number_unsigned()) return static_cast<int>(m.get<unsigned>());
    if (m.is_number_float()) return static_cast<int>(m.get<double>());
    return 0;
}


static std::string ws_parse_cmd(const json& msg) {
    if (!msg.contains("cmd") || !msg["cmd"].is_string()) return "";
    return msg["cmd"].get<std::string>();
}

static int json_int_or_default(const json& obj, const char* key, int default_value) {
    if (!obj.contains(key) || obj[key].is_null()) return default_value;
    const auto& v = obj[key];
    if (v.is_number_integer()) return v.get<int>();
    if (v.is_number_unsigned()) return static_cast<int>(v.get<unsigned>());
    if (v.is_number_float()) return static_cast<int>(v.get<double>());
    return default_value;
}

static int active_connections = 0;

std::unique_ptr<HttpFileServer> g_http_file_server = nullptr;
std::mutex g_http_server_mutex;

static std::string g_samba_host = "172.16.9.153";
static std::string g_samba_share = "data";
static std::string g_samba_user = "xiededa";
static std::string g_samba_pass = "qlzn1234";

void load_samba_config() {
    std::ifstream f("/etc/orin_samba.conf");
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        while (!val.empty() && (val.back() == ' ' || val.back() == '\t' || val.back() == '\r' || val.back() == ',')) {
            val.pop_back();
        }
        while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) {
            val.erase(0, 1);
        }
        if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
            val = val.substr(1, val.size() - 2);
        }
        if (key == "SAMBA_HOST") g_samba_host = val;
        else if (key == "SAMBA_SHARE") g_samba_share = val;
        else if (key == "SAMBA_USER") g_samba_user = val;
        else if (key == "SAMBA_PASS") g_samba_pass = val;
    }
}

static constexpr const char* kRobotConfigPath = "/etc/robot_config.json";
static constexpr const char* kUrlReportPath = "/api/data/acquisition/robot/url";
static constexpr const char* kDefaultAesKeyAscii = "b1KjDadDasFpjUnS";
static constexpr const char* kDefaultAesIvAscii = "9PlzfiN2KVHIapQT";

static std::string sha256_hex_lower(const std::string& s) {
    unsigned char h[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(s.data()), s.size(), h);
    std::ostringstream o;
    o << std::hex << std::setfill('0');
    for (unsigned char c : h) {
        o << std::setw(2) << static_cast<unsigned>(c);
    }
    return o.str();
}

static bool hex_to_bytes_16(const std::string& hex, unsigned char out[16]) {
    if (hex.size() != 32) return false;
    for (size_t i = 0; i < 16; ++i) {
        char hs[3] = {hex[i * 2], hex[i * 2 + 1], 0};
        char* end = nullptr;
        long v = std::strtol(hs, &end, 16);
        if (end != hs + 2 || v < 0 || v > 255) return false;
        out[i] = static_cast<unsigned char>(v);
    }
    return true;
}

static bool is_hex32(const std::string& s) {
    if (s.size() != 32) return false;
    for (unsigned char c : s) {
        if (!std::isxdigit(c)) return false;
    }
    return true;
}

static bool load_aes_key_iv_from_config_strings(const std::string& ks, const std::string& vs,
        unsigned char key[16], unsigned char iv[16]) {
    if (is_hex32(ks) && is_hex32(vs)) {
        return hex_to_bytes_16(ks, key) && hex_to_bytes_16(vs, iv);
    }
    if (ks.size() == 16 && vs.size() == 16) {
        std::memcpy(key, ks.data(), 16);
        std::memcpy(iv, vs.data(), 16);
        return true;
    }
    return false;
}

static std::string aes128_cbc_encrypt_base64(const std::string& plaintext,
    const unsigned char key[16], const unsigned char iv[16]) {
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return "";
    std::vector<unsigned char> out(plaintext.size() + static_cast<size_t>(EVP_MAX_BLOCK_LENGTH));
    int len = 0;
    int ct_len = 0;
    if (EVP_EncryptInit_ex(ctx, EVP_aes_128_cbc(), nullptr, key, iv) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return "";
    }
    if (EVP_EncryptUpdate(ctx, out.data(), &len,
            reinterpret_cast<const unsigned char*>(plaintext.data()),
            static_cast<int>(plaintext.size())) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return "";
    }
    ct_len = len;
    if (EVP_EncryptFinal_ex(ctx, out.data() + len, &len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return "";
    }
    ct_len += len;
    EVP_CIPHER_CTX_free(ctx);
    return base64_encode(out.data(), static_cast<size_t>(ct_len));
}

static std::string derive_api_base_url(const std::string& url_field, const nlohmann::json& j) {
    if (j.contains("apiBaseUrl") && j["apiBaseUrl"].is_string()) {
        std::string b = j["apiBaseUrl"].get<std::string>();
        while (!b.empty() && b.back() == '/') b.pop_back();
        if (b.find("://") == std::string::npos) {
            b = "http://" + b;
        }
        return b;
    }
    std::string u = url_field;
    if (u.empty()) return "";
    const size_t scheme_pos = u.find("://");
    if (scheme_pos != std::string::npos) {
        const size_t host_start = scheme_pos + 3;
        size_t path_slash = u.find('/', host_start);
        std::string base = (path_slash == std::string::npos) ? u : u.substr(0, path_slash);
        return base;
    }
    size_t slash = u.find('/');
    std::string hostport = (slash == std::string::npos) ? u : u.substr(0, slash);
    if (hostport.empty()) return "";
    return "http://" + hostport;
}

void maybe_report_robot_url_from_config() {
    const fs::path cfg_path(kRobotConfigPath);
    if (!fs::exists(cfg_path)) {
        return;
    }
    std::ifstream ifs(cfg_path);
    if (!ifs) {
        std::cout << "[robot_config] 无法打开 " << kRobotConfigPath << "，跳过 URL 上报" << std::endl;
        return;
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(buffer.str());
    } catch (const std::exception& e) {
        std::cout << "[robot_config] JSON 解析失败，跳过 URL 上报: " << e.what() << std::endl;
        return;
    }
    if (!j.contains("robotSn") || !j.contains("tenantId")) {
        std::cout << "[robot_config] 缺少 robotSn/tenantId，跳过 URL 上报" << std::endl;
        return;
    }
    std::string addr_raw;
    if (j.contains("robotIp") && j["robotIp"].is_string()) {
        addr_raw = j["robotIp"].get<std::string>();
    } else if (j.contains("url") && j["url"].is_string()) {
        addr_raw = j["url"].get<std::string>();
    } else {
        std::cout << "[robot_config] 缺少 robotIp（或旧字段 url），跳过 URL 上报" << std::endl;
        return;
    }
    const std::string robot_sn = j["robotSn"].get<std::string>();
    const std::string addr_for_derive_api = addr_raw;

    nlohmann::json inner;
    inner["robotSn"] = robot_sn;
    inner["tenantId"] = j["tenantId"];
    inner["url"] = addr_raw;
    const std::string plain = inner.dump();

    unsigned char key[16], iv[16];
    std::string ks = kDefaultAesKeyAscii;
    std::string vs = kDefaultAesIvAscii;
    if (j.contains("aesKey") && j.contains("aesIv") &&
            j["aesKey"].is_string() && j["aesIv"].is_string()) {
        ks = j["aesKey"].get<std::string>();
        vs = j["aesIv"].get<std::string>();
    }
    if (!load_aes_key_iv_from_config_strings(ks, vs, key, iv)) {
        std::cout << "[robot_config] aesKey/aesIv 无效（须各 16 字符或各 32 位 hex），跳过 URL 上报" << std::endl;
        return;
    }

    const std::string param = aes128_cbc_encrypt_base64(plain, key, iv);
    if (param.empty()) {
        std::cout << "[robot_config] AES 加密失败，跳过 URL 上报" << std::endl;
        return;
    }

    const int64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const std::string sign_src = "keenonrobot|" + param + "|" + std::to_string(timestamp);
    const std::string signature = sha256_hex_lower(sign_src);

    nlohmann::json body;
    body["timestamp"] = timestamp;

    std::string payload_key = "params";
    if (j.contains("encryptField") && j["encryptField"].is_string()) {
        payload_key = j["encryptField"].get<std::string>();
    }
    body[payload_key] = param;
    body["signature"] = signature;

    const std::string api_base = derive_api_base_url(addr_for_derive_api, j);
    if (api_base.empty()) {
        std::cout << "[robot_config] 无法推导 api 地址，跳过 URL 上报" << std::endl;
        return;
    }
    if (api_base.rfind("http://", 0) != 0 && api_base.rfind("https://", 0) != 0) {
        std::cout << "[robot_config] api 地址需以 http:// 或 https:// 开头: " << api_base << std::endl;
        return;
    }

    httplib::Client cli(api_base);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(15, 0);
    if (api_base.rfind("https://", 0) == 0) {
        cli.enable_server_certificate_verification(false);
    }

    httplib::Result res = cli.Post(kUrlReportPath, body.dump(), "application/json");

    if (!res) {
        std::cout << "[robot_config] URL 上报请求失败: " << httplib::to_string(res.error()) << std::endl;
        return;
    }
    std::cout << "[robot_config] URL 上报 HTTP " << res->status << " 响应: " << res->body << std::endl;
}


std::string get_eth_ip_address();

static const std::string GUODI_DATASET_ROOT = "/mnt/data/keenon_baidu_dataset";
static std::mutex g_samba_upload_mutex;  

static std::mutex g_robot_status_mutex;
static std::string g_robot_status_agent_id;  
static std::shared_ptr<net::steady_timer> g_robot_status_timer;  


static std::mutex g_remote_control_mutex;
static pid_t g_remote_control_pid = -1;

static constexpr const char* kRemoteControlWorkdirDefault = "/home/peanut/sc_ros";
static constexpr const char* kRemoteControlRosCmd = "taskset -c 0,1 ros2 run sc_ros2 remote_control_data";

static void reap_remote_control_if_exited() {
    if (g_remote_control_pid <= 0) return;
    int status = 0;
    pid_t w = waitpid(g_remote_control_pid, &status, WNOHANG);
    if (w == g_remote_control_pid) {
        g_remote_control_pid = -1;
    }
}

static bool remote_control_start(const std::string& workdir, const std::string& shell_cmd,
        pid_t* out_pid, std::string* err_out) {
    std::lock_guard<std::mutex> lock(g_remote_control_mutex);
    reap_remote_control_if_exited();
    if (g_remote_control_pid > 0 && kill(g_remote_control_pid, 0) == 0) {
        if (err_out) {
            *err_out = "remote_control 已在运行 (pid=" + std::to_string(static_cast<long long>(g_remote_control_pid)) + ")";
        }
        return false;
    }
    g_remote_control_pid = -1;

    pid_t pid = fork();
    if (pid < 0) {
        if (err_out) *err_out = std::string("fork 失败: ") + strerror(errno);
        return false;
    }
    if (pid == 0) {
        if (setsid() < 0) _exit(127);
        if (!workdir.empty()) {
            if (chdir(workdir.c_str()) != 0) {
                _exit(126);
            }
        }
        execl("/bin/sh", "sh", "-c", shell_cmd.c_str(), (char*)nullptr);
        _exit(127);
    }

    usleep(100000);
    int st = 0;
    pid_t w = waitpid(pid, &st, WNOHANG);
    if (w == pid) {
        if (err_out) {
            if (WIFEXITED(st) && WEXITSTATUS(st) == 126) {
                *err_out = "无法进入工作目录: " + workdir;
            } else {
                *err_out = "进程立即退出（请检查 ROS 环境、包 sc_ros2 与 command）";
            }
        }
        return false;
    }
    g_remote_control_pid = pid;
    if (out_pid) *out_pid = pid;
    return true;
}

static std::string remote_control_stop() {
    std::lock_guard<std::mutex> lock(g_remote_control_mutex);
    reap_remote_control_if_exited();
    if (g_remote_control_pid <= 0) {
        return "未在运行";
    }
    pid_t p = g_remote_control_pid;
    errno = 0;
    if (kill(-p, SIGINT) != 0) {
        if (errno == ESRCH) {
            g_remote_control_pid = -1;
            return "进程已不存在";
        }
        return std::string("SIGINT 失败: ") + strerror(errno);
    }
    for (int i = 0; i < 100; ++i) {
        int status = 0;
        pid_t w = waitpid(p, &status, WNOHANG);
        if (w == p) {
            g_remote_control_pid = -1;
            return "已停止(SIGINT)";
        }
        usleep(100000);
    }
    kill(-p, SIGKILL);
    waitpid(p, nullptr, 0);
    g_remote_control_pid = -1;
    return "已强制结束(SIGKILL)";
}

static std::string get_iso8601_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&tt), "%Y-%m-%dT%H:%M:%S")
        << "." << std::setfill('0') << std::setw(3) << ms << "Z";
    return oss.str();
}

static std::string create_time_to_iso8601_utc(const std::string& create_time) {
    int y = 0, mo = 0, d = 0, h = 0, mi = 0, s = 0;
    if (sscanf(create_time.c_str(), "%d-%d-%d %d:%d:%d", &y, &mo, &d, &h, &mi, &s) >= 6) {
        std::ostringstream oss;
        oss << std::setfill('0') << std::setw(4) << y << "-"
            << std::setw(2) << mo << "-"
            << std::setw(2) << d << "T"
            << std::setw(2) << h << ":"
            << std::setw(2) << mi << ":"
            << std::setw(2) << s << ".000Z";
        return oss.str();
    }
    return "";
}

static std::string compute_end_time_iso8601(const std::string& start_time_iso8601, int duration_sec) {
    int y = 0, mo = 0, d = 0, h = 0, mi = 0, s = 0;

    if (sscanf(start_time_iso8601.c_str(), "%d-%d-%dT%d:%d:%d", &y, &mo, &d, &h, &mi, &s) >= 6) {
        std::tm tm = {};
        tm.tm_year = y - 1900;
        tm.tm_mon = mo - 1;
        tm.tm_mday = d;
        tm.tm_hour = h;
        tm.tm_min = mi;
        tm.tm_sec = s + duration_sec; 
        tm.tm_isdst = 0;

#if defined(__linux__) || defined(__GLIBC__)
        std::time_t t = timegm(&tm);
#else
        std::time_t t = std::mktime(&tm);
#endif
        if (t != static_cast<std::time_t>(-1)) {
            std::tm* utc_tm = std::gmtime(&t);
            std::ostringstream oss;
            oss << std::put_time(utc_tm, "%Y-%m-%dT%H:%M:%S") << ".000Z";
            return oss.str();
        }
    }
    return "";
}

static void compute_robot_status_fields(std::string& out_status, int& out_error_code, std::string& out_error_msg) {
    out_status = "running";
    out_error_code = 0;
    out_error_msg = "";
    try {
        auto proc = DoubleArmRGBDDataProcessor::get_current_instance();
        if (!proc) proc = DoubleArmRGBDDataProcessor::get_instance();
        if (!proc) return;
        std::string motor_status = proc->arm_interface ? proc->arm_interface->get_motor_error_status_effective() : "未知";
        std::string camera_status = (proc->camera_sub && proc->camera_sub->running)
            ? proc->camera_sub->get_camera_status() : "相机状态正常";

        const bool camera_ok = (camera_status == "相机状态正常");
        const bool motor_normal = (motor_status == "机械臂状态正常");
        const bool motor_must_stop = (motor_status.find("请停止采集") != std::string::npos);
        const bool motor_vwall_only =
            !motor_must_stop && motor_status.find("虚拟墙预警") == 0;  

        if (camera_ok && (motor_normal || motor_vwall_only)) return;

        out_status = "error";
        if (camera_ok) {
            out_error_code = 1;
            out_error_msg = motor_status;
        } else if (motor_normal) {
            out_error_code = 2;
            out_error_msg = camera_status;
        } else {
            out_error_code = 3;
            out_error_msg = camera_status + "；" + motor_status;
        }
    } catch (const std::exception&) {
    }
}

static void write_robot_status_to_samba(const std::string& agent_id) {
    try {
        if (agent_id.empty()) return;
        if (agent_id.find("..") != std::string::npos || agent_id.find('/') != std::string::npos) {
            std::cout << "[robot_status] 无效 agent_id，跳过写入" << std::endl;
            return;
        }
        std::string dyn_status;
        int dyn_error_code;
        std::string dyn_error_msg;
        compute_robot_status_fields(dyn_status, dyn_error_code, dyn_error_msg);
        json status = {
            {"timestamp", get_iso8601_timestamp()},
            {"status", dyn_status},
            {"errorCode", dyn_error_code},
            {"errorMsg", dyn_error_msg}
        };
        std::string content = status.dump();
        fs::path tmp_path = fs::path("/tmp") / ("robot_status_" + std::to_string(getpid()) + ".json");
        try {
            std::ofstream f(tmp_path);
            if (!f) {
                std::cout << "[robot_status] 无法创建临时文件: " << tmp_path << std::endl;
                return;
            }
            f << content;
            f.close();
        } catch (const std::exception& e) {
            std::cout << "[robot_status] 写入临时文件失败: " << e.what() << std::endl;
            return;
        }
        std::string smb_base = "smbclient \"//" + g_samba_host + "/" + g_samba_share
            + "\" -U " + g_samba_user + "%" + g_samba_pass;
        std::string smb_cmd = smb_base + " -c \"mkdir " + agent_id + " 2>/dev/null; cd " + agent_id
            + "; put \\\"" + tmp_path.string() + "\\\" robot_status.json\" >/dev/null 2>&1";
        int ret = std::system(smb_cmd.c_str());
        fs::remove(tmp_path);
        if (ret != 0) {
            std::cout << "[robot_status] smbclient 写入失败, agent_id=" << agent_id << std::endl;
        }
    } catch (const nlohmann::json::exception& e) {
        std::cout << "[robot_status] JSON 异常(不应导致进程退出): " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[robot_status] 写入异常(不应导致进程退出): " << e.what() << std::endl;
    }
}

static void stop_robot_status_timer() {
    std::lock_guard<std::mutex> lock(g_robot_status_mutex);
    if (g_robot_status_timer) {
        g_robot_status_timer->cancel();
        g_robot_status_timer.reset();
        g_robot_status_agent_id.clear();
        std::cout << "[WebSocket] 客户端断开: 已停止 robot_status.json 定时更新" << std::endl;
    }
}

template<typename Executor>
static void schedule_robot_status_update(const Executor& exec) {
    if (!g_robot_status_timer) {
        g_robot_status_timer = std::make_shared<net::steady_timer>(exec, std::chrono::seconds(10));
    }
    g_robot_status_timer->expires_after(std::chrono::seconds(10));
    g_robot_status_timer->async_wait([](const boost::system::error_code& ec) {
        if (ec) return;
        std::string aid;
        { std::lock_guard<std::mutex> lock(g_robot_status_mutex); aid = g_robot_status_agent_id; }
        if (!aid.empty()) {
            std::thread(write_robot_status_to_samba, aid).detach();
        }
        schedule_robot_status_update(g_robot_status_timer->get_executor());
    });
}

static constexpr int joint_push_fps = 100;          
static constexpr int img_quality = 70;              


class Session : public std::enable_shared_from_this<Session> {
    websocket::stream<beast::tcp_stream> ws_;
    beast::flat_buffer buffer_;
    std::shared_ptr<DoubleArmRGBDDataProcessor> processor_;  
    std::shared_ptr<net::steady_timer> sampler_timer_;       
    std::shared_ptr<net::steady_timer> stream_timer_;        
    std::deque<std::string> write_queue_;                   
    bool is_writing_ = false;                                
    int stream_push_count_ = 0;                              
    std::string data_folder_;                                
    bool is_data_saved_ = false;                             
    
    struct CameraStats {
        int success_count = 0;                               
        std::chrono::steady_clock::time_point start_time;    
        CameraStats() : start_time(std::chrono::steady_clock::now()) {}
    };
    std::map<std::string, CameraStats> camera_stats_;        
    const int stats_interval_ = 30;                          
    
    void cleanup_on_disconnect() {
        if (sampler_timer_) { sampler_timer_->cancel(); sampler_timer_.reset(); }
        if (stream_timer_) { stream_timer_->cancel(); stream_timer_.reset(); }
        stop_robot_status_timer();  
        if (processor_) {
            if (processor_->is_recording) {
                std::cout << "[WebSocket] 客户端断开: 停止采集并落盘" << std::endl;
                try {
                    processor_->end_episode();
                    is_data_saved_ = true;
                } catch (const std::exception& e) {
                    std::cout << "[WebSocket] 客户端断开: end_episode 异常(数据可能不完整): " << e.what() << std::endl;
                    is_data_saved_ = false;
                }

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

    void start_sampler_push(int mid) {
        sampler_timer_ = std::make_shared<net::steady_timer>(ws_.get_executor(), std::chrono::milliseconds(33));
        sampler_timer_->async_wait([this, mid, self = shared_from_this()](const boost::system::error_code& ec) {
            if (ec) return;

            try {
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
                    {"motor_status", proc->arm_interface ? proc->arm_interface->get_motor_error_status_effective() : "未知"},
                    {"camera_status", proc->camera_sub ? proc->camera_sub->get_camera_status() : "未知"},
                    {"push_fps", joint_push_fps}
                };

                queue_message(progress.dump());
            } catch (const nlohmann::json::exception& e) {
                std::cout << "[WebSocket] 采集进度推送 JSON 异常，已跳过本次进度: " << e.what() << std::endl;
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] 采集进度推送异常，已跳过本次进度: " << e.what() << std::endl;
            }
            start_sampler_push(mid);
        });
    }

    void start_stream_push(int mid) {
        stream_timer_ = std::make_shared<net::steady_timer>(ws_.get_executor(), std::chrono::milliseconds(33));
        stream_timer_->async_wait([this, mid, self = shared_from_this()](const boost::system::error_code& ec) {
            if (ec) return;

            try {
                auto proc = processor_;
                if (!proc || !proc->camera_sub) return;

                int frames_found = 0;
                std::vector<std::pair<std::string, json>> valid_frames;
                for (const auto& cam_name : proc->camera_sub->camera_names) {
                    json frame_data = proc->camera_sub->get_current_frame(cam_name, "color");
                    if (!frame_data.empty() && frame_data.contains("image_data")) {
                        valid_frames.emplace_back(cam_name, frame_data);
                        frames_found++;
                        camera_stats_[cam_name].success_count++;
                    }
                }
                for (const auto& [cam_name, frame_data] : valid_frames) {
                    json msg;
                    msg["cmd"] = "SEND_CAMERA_FRAME";
                    msg["mid"] = mid;
                    msg["type"] = "CameraFramePayload";
                    json msg_data;
                    msg_data["image"] = frame_data["image_data"];
                    msg_data["image_type"] = "BASE64_JPG";
                    msg_data["camera_name"] = cam_name;
                    msg_data["width"] = json_int_or_default(frame_data, "width", get_camera_param(cam_name, "color", "width"));
                    msg_data["height"] = json_int_or_default(frame_data, "height", get_camera_param(cam_name, "color", "height"));
                    msg_data["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
                    msg_data["source_fps"] = get_camera_param(cam_name, "color", "fps");
                    msg_data["push_fps"] = get_camera_param(cam_name, "color", "fps");
                    msg["data"] = msg_data;
                    queue_message(msg.dump());
                }

                stream_push_count_++;

                bool is_recording_mode = proc->is_recording;

                if (is_recording_mode) {

                } else {
                    if (stream_push_count_ % stats_interval_ == 0) {
                        auto current_time = std::chrono::steady_clock::now();
                        std::vector<std::string> fps_logs;

                        for (const auto& cam_name : proc->camera_sub->camera_names) {
                            auto& stats = camera_stats_[cam_name];
                            auto elapsed = std::chrono::duration<double>(current_time - stats.start_time).count();

                            double fps = elapsed > 0 ? stats.success_count / elapsed : 0.0;
                            char fps_str[10];
                            snprintf(fps_str, sizeof(fps_str), "%.1f", fps);
                            fps_logs.push_back(cam_name + ": " + std::string(fps_str));

                            stats.success_count = 0;
                            stats.start_time = current_time;
                        }

                        std::string fps_log_str;
                        for (size_t i = 0; i < fps_logs.size(); ++i) {
                            if (i > 0) fps_log_str += ", ";
                            fps_log_str += fps_logs[i];
                        }
                        std::cout << "[WebSocket] 预览模式相机实际推送FPS: " << fps_log_str << std::endl;
                    }
                }
            } catch (const nlohmann::json::exception& e) {
                std::cout << "[WebSocket] 相机帧推送 JSON 异常，已跳过本帧: " << e.what() << std::endl;
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] 相机帧推送异常，已跳过本帧: " << e.what() << std::endl;
            }

            start_stream_push(mid);
        });
    }

    void queue_message(std::string msg) {
        net::post(ws_.get_executor(), [this, self = shared_from_this(), msg = std::move(msg)]() mutable {
            write_queue_.push_back(std::move(msg));
            if (!is_writing_) {
                do_write();
            }
        });
    }
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

    void run() {
        ws_.read_message_max(1000 * 1024 * 1024);
        ws_.write_buffer_bytes(1000 * 1024 * 1024);
        ws_.async_accept(beast::bind_front_handler(&Session::on_accept, shared_from_this()));
    }
    void on_accept(beast::error_code ec) {
        if (ec) {
            std::cout << "[WebSocket] 接受连接失败: " << ec.message() << std::endl;
            return;
        }
        active_connections++;
        std::cout << "[WebSocket] 新客户端连接已建立,当前活跃客户端数:" << active_connections << std::endl;
        do_read();
    }

    void do_read() {
        buffer_.consume(buffer_.size());
        ws_.async_read(buffer_, beast::bind_front_handler(&Session::on_read, shared_from_this()));
    }

    void on_read(beast::error_code ec, std::size_t bytes_transferred) {
        boost::ignore_unused(bytes_transferred);
        if (ec) {
            active_connections = std::max(0, active_connections - 1);
            if (ec == websocket::error::closed || 
                ec == net::error::eof || 
                ec == net::error::connection_reset) {
                std::cout << "[WebSocket] 客户端连接已正常断开,当前活跃数:" << active_connections << std::endl;
            } else {
                std::cout << "[WebSocket] 连接读取出错 (" << ec.value() << "): " << ec.message() << std::endl;
            }
            cleanup_on_disconnect();
            return;
        }

        std::string message = beast::buffers_to_string(buffer_.data());
        buffer_.consume(buffer_.size());
        
        try {
            json msg = json::parse(message);
            std::string cmd = ws_parse_cmd(msg);
            if (cmd.empty()) {
                std::cout << "[WebSocket] 缺少 cmd 或 cmd 非字符串，已忽略该帧" << std::endl;
                do_read();
                return;
            }
            const int mid = ws_parse_mid(msg);
            json data = (msg.contains("data") && !msg["data"].is_null()) ? msg["data"] : json::object();

            std::cout << "[WebSocket] 收到客户端指令: cmd=" << cmd << ", mid=" << mid << std::endl;
            handle_command(cmd, mid, data);
        } catch (const nlohmann::json::exception& e) {
            std::cout << "[WebSocket] JSON 异常: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "[WebSocket] 指令解析/处理失败: " << e.what() << std::endl;
        }
        do_read();
    }

    void handle_command(const std::string& cmd, int mid, const json& data) {
        json response;
        if (cmd == "GET_INFO") {
            try {
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
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] GET_INFO 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "GET_AGENT_ID") {
            std::string agent_id = data.contains("agent_id") ? data["agent_id"].get<std::string>() : "";
            if (agent_id.empty()) {
                std::cout << "[WebSocket] GET_AGENT_ID: agent_id 为空" << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", "agent_id 不能为空"}}}};
            } else {
                if (agent_id.find("..") != std::string::npos || agent_id.find('/') != std::string::npos) {
                    response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", "agent_id 不能包含 .. 或 /"}}}};
                } else {
                    {
                        std::lock_guard<std::mutex> lock(g_robot_status_mutex);
                        g_robot_status_agent_id = agent_id;
                    }
                    write_robot_status_to_samba(agent_id);
                    schedule_robot_status_update(ws_.get_executor());
                    std::cout << "[WebSocket] GET_AGENT_ID: agent_id=" << agent_id
                              << ", 已启动 robot_status.json 每10秒更新至 samba:/data/" << agent_id << "/" << std::endl;
                    response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}, {"data", {{"agent_id", agent_id}}}};
                }
            }

        } else if (cmd == "GET_ROBOT_STATUS") {
            std::string motor_status = "未知";
            std::string camera_status = "未知";
            std::string status_str = "running";
            int error_code = 0;
            std::string error_msg = "";
            try {
                auto proc = DoubleArmRGBDDataProcessor::get_current_instance();
                if (!proc) proc = DoubleArmRGBDDataProcessor::get_instance(); 
                if (proc) {
                    motor_status = proc->arm_interface ? proc->arm_interface->get_motor_error_status_effective() : "未知";
                    camera_status = (proc->camera_sub && proc->camera_sub->running)
                        ? proc->camera_sub->get_camera_status() : "相机状态正常";
                }
                compute_robot_status_fields(status_str, error_code, error_msg);
            } catch (const std::exception&) {}
            response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "GetRobotStatusPayload"}, {"data", {
                {"motor_status", motor_status},
                {"camera_status", camera_status},
                {"status", status_str},
                {"errorCode", error_code},
                {"errorMsg", error_msg}
            }}};

        } else if (cmd == "START_PREVIEW") {
            std::vector<std::string> cameras;
            if (data.contains("camera_names") && data["camera_names"].is_array()) {
                cameras = data["camera_names"].get<std::vector<std::string>>();
            }
            std::cout << "[WebSocket] START_PREVIEW: 相机列表=[";
            for (size_t i = 0; i < cameras.size(); ++i) { std::cout << cameras[i]; if (i+1 < cameras.size()) std::cout << ","; }
            std::cout << "]" << std::endl;

            try {
                processor_ = DoubleArmRGBDDataProcessor::get_instance("Keenon_F1_DoubleArm", cameras);
                std::cout << "[WebSocket] START_PREVIEW: 处理器实例已就绪" << std::endl;

                processor_->camera_sub->start();
                std::cout << "[WebSocket] START_PREVIEW: 相机ROS2订阅已启动" << std::endl;

                start_stream_push(mid);
                std::cout << "[WebSocket] START_PREVIEW: 帧推送定时器已启动(30fps)" << std::endl;

                response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}};
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] START_PREVIEW 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "STOP_PREVIEW") {
            try {
                std::cout << "[WebSocket] STOP_PREVIEW: 停止预览" << std::endl;
                if (stream_timer_) stream_timer_->cancel();
                if (processor_ && processor_->camera_sub) {
                    processor_->camera_sub->is_preview_active = false;
                    std::cout << "[WebSocket] STOP_PREVIEW: 已关闭预览处理流" << std::endl;
                }
                response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}};
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] STOP_PREVIEW 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "RECOVER") {
            std::cout << "[WebSocket] RECOVER: 系统状态恢复中..." << std::endl;
            bool recover_close_ok = true;
            std::string recover_err;

            if (sampler_timer_) { sampler_timer_->cancel(); std::cout << "[WebSocket] RECOVER: 采集进度定时器已取消" << std::endl; }
            if (stream_timer_) { stream_timer_->cancel(); std::cout << "[WebSocket] RECOVER: 帧推送定时器已取消" << std::endl; }

            if (processor_) {
                try {
                    processor_->close();
                    std::cout << "[WebSocket] RECOVER: 处理器资源已释放" << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "[WebSocket] RECOVER: 处理器释放异常: " << e.what() << std::endl;
                    recover_close_ok = false;
                    recover_err = e.what();
                } catch (...) {
                    std::cout << "[WebSocket] RECOVER: 处理器释放未知异常" << std::endl;
                    recover_close_ok = false;
                    recover_err = "处理器释放未知异常";
                }
                processor_.reset();
            }
            DoubleArmRGBDDataProcessor::reset_instance();

            stream_push_count_ = 0;
            camera_stats_.clear();
            data_folder_.clear();
            is_data_saved_ = false;

            std::cout << "[WebSocket] RECOVER: 系统已完全重置" << std::endl;
            if (recover_close_ok) {
                response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}};
            } else {
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", recover_err}}}};
            }

        } else if (cmd == "START_TELEMETRY") {
            std::string data_folder;
            const fs::path default_root = "/mnt/data/keenon_baidu_dataset";
            std::string raw_folder = data.contains("data_folder") ? data["data_folder"].get<std::string>() : "";

            if (!raw_folder.empty()) {
                if (raw_folder.front() == '/') {
                    data_folder = raw_folder;
                    std::cout << "[WebSocket] START_TELEMETRY: 使用用户指定绝对路径=" << data_folder << std::endl;
                } else {
                    fs::path episode_folder = default_root / raw_folder;
                    fs::create_directories(episode_folder);
                    data_folder = episode_folder.string();
                    std::cout << "[WebSocket] START_TELEMETRY: 使用默认根目录+子文件夹=" << data_folder << std::endl;
                }
            } else {
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

            this->data_folder_ = data_folder; 
            this->is_data_saved_ = false;      

            processor_ = DoubleArmRGBDDataProcessor::get_instance();
            std::cout << "[WebSocket] START_TELEMETRY: 处理器实例已就绪,开始采集..." << std::endl;
            
            bool success = processor_->start_episode("", data_folder);
            if (success) {
                start_sampler_push(mid);
                start_stream_push(mid);
                std::cout << "[WebSocket] START_TELEMETRY: 采集启动成功,进度+帧推送已启动" << std::endl;
            } else {
                std::cout << "[WebSocket] START_TELEMETRY: 采集启动失败" << std::endl;
                this->data_folder_.clear();
            }
            response = {{"cmd", success ? "ACK" : "ERROR"}, {"mid", mid}, {"type", "EmptyPayload"}};
            if (!success) response["data"] = {{"code", 500}, {"message", "采集启动失败"}};

        } else if (cmd == "STOP_TELEMETRY") {
            try {
                std::cout << "[WebSocket] STOP_TELEMETRY: 停止采集" << std::endl;
                if (sampler_timer_) sampler_timer_->cancel();
                if (stream_timer_) stream_timer_->cancel();
                if (processor_) {
                    processor_->end_episode();
                    is_data_saved_ = true;  
                    std::cout << "[WebSocket] STOP_TELEMETRY: 数据落盘完成" << std::endl;
                }
                response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}};
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] STOP_TELEMETRY 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "STOP_AND_SUB") {
            std::cout << "[WebSocket] STOP_AND_SUB: 停止采集并上传到 Samba" << std::endl;
            try {
                if (sampler_timer_) sampler_timer_->cancel();
                if (stream_timer_) stream_timer_->cancel();
                if (processor_) {
                    processor_->end_episode();
                    is_data_saved_ = true;
                    std::cout << "[WebSocket] STOP_AND_SUB: 数据落盘完成" << std::endl;
                }

                if (data_folder_.empty()) {
                    throw std::runtime_error("无数据路径记录，无法上传");
                }
                fs::path source_path(data_folder_);
                fs::path canonical_root = fs::path(GUODI_DATASET_ROOT).lexically_normal();
                fs::path canonical_source = source_path.lexically_normal();
                std::string src_str = canonical_source.string();
                std::string root_str = canonical_root.string();
                if (src_str.find(root_str) != 0 || (src_str.size() <= root_str.size())) {
                    throw std::runtime_error("数据路径必须在 " + GUODI_DATASET_ROOT + " 下");
                }
                if (!fs::exists(source_path) || !fs::is_directory(source_path)) {
                    throw std::runtime_error("源路径不存在或不是目录: " + source_path.string());
                }

                std::string rel_path = src_str.substr(root_str.size());
                if (!rel_path.empty() && rel_path[0] == '/') rel_path.erase(0, 1);
                if (rel_path.empty()) {
                    throw std::runtime_error("无法解析 Samba 目标路径");
                }
                if (rel_path.find("..") != std::string::npos) {
                    throw std::runtime_error("路径不能包含 ..");
                }

                std::lock_guard<std::mutex> lock(g_samba_upload_mutex);
                std::string smb_base = "smbclient \"//" + g_samba_host + "/" + g_samba_share
                    + "\" -U " + g_samba_user + "%" + g_samba_pass;

                std::vector<std::string> parts;
                for (size_t i = 0; i < rel_path.size(); ) {
                    size_t j = rel_path.find('/', i);
                    if (j == std::string::npos) {
                        parts.push_back(rel_path.substr(i));
                        break;
                    }
                    parts.push_back(rel_path.substr(i, j - i));
                    i = j + 1;
                }
                std::string cd_chain;  
                for (const auto& p : parts) {
                    if (p.empty()) continue;
                    std::string mk_cmd = smb_base + " -c \"";
                    mk_cmd += (cd_chain.empty() ? "" : cd_chain + "; ") + "mkdir " + p;
                    mk_cmd += "\" 2>/dev/null";
                    std::system(mk_cmd.c_str());
                    cd_chain = cd_chain.empty() ? ("cd " + p) : (cd_chain + "; cd " + p);
                }

                std::string smb_cmd = smb_base + " -c \"";
                if (!cd_chain.empty()) {
                    smb_cmd += cd_chain + "; ";
                }
                smb_cmd += "lcd \\\"" + source_path.string() + "\\\"; recurse; prompt; mput *\" >/dev/null 2>&1";
                std::cout << "[WebSocket] STOP_AND_SUB: 上传中 " << source_path.string()
                          << " -> smb://" << g_samba_host << "/" << g_samba_share << "/" << rel_path << std::endl;
                int ret = std::system(smb_cmd.c_str());
                if (ret != 0) {
                    throw std::runtime_error("smbclient 上传失败");
                }

                std::string dest_path = "smb://" + g_samba_host + "/" + g_samba_share + "/" + rel_path;
                std::cout << "[WebSocket] STOP_AND_SUB: 上传完成 -> " << dest_path << std::endl;
                response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}, {"data", {
                    {"msg", "停止采集完成，数据已上传到 Samba"},
                    {"source_path", source_path.string()},
                    {"dest_path", dest_path},
                    {"rel_path", rel_path}
                }}};
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] STOP_AND_SUB 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "DATA_POSTPROCESS") {
            try {
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

                std::string target_compress = data.contains("target_compress_folder") ? data["target_compress_folder"].get<std::string>() : "";
                std::string tar_name = data.contains("tar_filename") ? data["tar_filename"].get<std::string>() : "";

                fs::path target_folder = target_compress.empty() ? source_path.parent_path() : fs::path(target_compress);
                fs::create_directories(target_folder);

                std::string source_folder_name = source_path.filename().string();
                if (tar_name.empty()) {
                    tar_name = source_folder_name + "_compressed.tar";
                }
                fs::path tar_file = target_folder / tar_name;
                if (fs::exists(tar_file)) {
                    fs::remove(tar_file);
                }

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
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "DISCARD_DATA") {
            try {
                if (data_folder_.empty()) {
                    throw std::runtime_error("无数据路径记录，无法执行丢弃操作");
                }

                fs::path target_path(data_folder_);

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

                std::string deleted_path = data_folder_;
                data_folder_.clear();
                is_data_saved_ = false;

                response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}, {"data", {
                    {"msg", "数据丢弃成功"},
                    {"deleted_path", deleted_path}
                }}};
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] DISCARD_DATA 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "DISCARD_BOTH_DATA") {
            try {
                if (data_folder_.empty()) {
                    throw std::runtime_error("无数据路径记录，无法执行丢弃操作");
                }

                fs::path target_path(data_folder_);
                if (target_path.filename() == "data") {
                    fs::path parent_path = target_path.parent_path();
                    std::cout << "[WebSocket] DISCARD_BOTH_DATA: 检测到路径为data子目录，将删除外层Session目录: " << parent_path.string() << std::endl;
                    target_path = parent_path;
                }

                std::cout << "[WebSocket] DISCARD_BOTH_DATA: 准备删除本机+Samba文件夹: " << target_path.string() << std::endl;
                if (fs::exists(target_path)) {
                    fs::remove_all(target_path);
                    std::cout << "[WebSocket] DISCARD_BOTH_DATA: 本机数据已删除" << std::endl;
                } else {
                    std::cout << "[WebSocket] DISCARD_BOTH_DATA: 本机路径不存在: " << target_path.string() << std::endl;
                }

                fs::path canonical_root = fs::path(GUODI_DATASET_ROOT).lexically_normal();
                fs::path canonical_target = target_path.lexically_normal();
                std::string src_str = canonical_target.string();
                std::string root_str = canonical_root.string();
                if (src_str.size() > root_str.size() && src_str.find(root_str) == 0 && src_str[root_str.size()] == '/') {
                    std::string rel_path = src_str.substr(root_str.size() + 1);
                    if (rel_path.find("..") == std::string::npos && !rel_path.empty()) {
                        std::lock_guard<std::mutex> lock(g_samba_upload_mutex);
                        
                        std::string smb_base = "smbclient \"//" + g_samba_host + "/" + g_samba_share
                            + "\" -U " + g_samba_user + "%" + g_samba_pass;

                        std::vector<std::string> parts;
                        for (size_t i = 0; i < rel_path.size(); ) {
                            size_t j = rel_path.find('/', i);
                            if (j == std::string::npos) {
                                parts.push_back(rel_path.substr(i));
                                break;
                            }
                            parts.push_back(rel_path.substr(i, j - i));
                            i = j + 1;
                        }
                        if (parts.size() >= 3) {
                            std::string agent_id = parts[0];
                            std::string task_id = parts[1];
                            std::string sub_number = parts[2];
                            if (agent_id.find("..") == std::string::npos && agent_id.find('/') == std::string::npos &&
                                task_id.find("..") == std::string::npos && task_id.find('/') == std::string::npos) {
                                std::string cd_chain = "cd " + agent_id + "; cd " + task_id + "; cd task_info";
                                fs::path tmp_get = fs::path("/tmp") / ("sub_task_discard_" + std::to_string(getpid()) + ".json");
                                std::string get_cmd = smb_base + " -c \"" + cd_chain + "; get sub_task.json \\\"" + tmp_get.string() + "\\\"\" 2>/dev/null";
                                int get_ret = std::system(get_cmd.c_str());
                                if (get_ret == 0 && fs::exists(tmp_get)) {
                                    try {
                                        std::ifstream f(tmp_get);
                                        json sub_task_array = json::parse(f);
                                        f.close();
                                        if (sub_task_array.is_array()) {
                                            auto it = std::remove_if(sub_task_array.begin(), sub_task_array.end(),
                                                [&sub_number](const json& e) {
                                                    return e.value("subNumber", "") == sub_number;
                                                });
                                            sub_task_array.erase(it, sub_task_array.end());
                                            fs::path tmp_put = fs::path("/tmp") / ("sub_task_discard_put_" + std::to_string(getpid()) + ".json");
                                            std::ofstream pf(tmp_put);
                                            pf << sub_task_array.dump(4);
                                            pf.close();
                                            std::string put_cmd = smb_base + " -c \"" + cd_chain + "; put \\\"" + tmp_put.string() + "\\\" sub_task.json\" >/dev/null 2>&1";
                                            int put_ret = std::system(put_cmd.c_str());
                                            fs::remove(tmp_put);
                                            if (put_ret == 0) {
                                                std::cout << "[WebSocket] DISCARD_BOTH_DATA: 已从 sub_task.json 移除 subNumber=" << sub_number << std::endl;
                                            }
                                        }
                                        fs::remove(tmp_get);
                                    } catch (...) {
                                        if (fs::exists(tmp_get)) fs::remove(tmp_get);
                                    }
                                }
                            }
                        }

                        std::cout << "[WebSocket] DISCARD_BOTH_DATA: 正在通过 smbclient 深度清理三级目录结构 " << rel_path << std::endl;
                        std::vector<std::string> cams = {"hand_left", "hand_right", "head"};
                        std::vector<std::string> types = {"color", "depth"};
                        for (const auto& cam : cams) {
                            for (const auto& type : types) {
                                std::string p = rel_path + "/camera/" + cam + "/" + type;
                                std::system((smb_base + " -c \"cd " + p + "; prompt; del *; cd ..; rmdir " + type + "\" 2>/dev/null").c_str());
                            }
                            std::system((smb_base + " -c \"cd " + rel_path + "/camera; rmdir " + cam + "\" 2>/dev/null").c_str());
                        }
                        std::system((smb_base + " -c \"cd " + rel_path + "; rmdir camera\" 2>/dev/null").c_str());
                        std::vector<std::string> param_subs = {"camera", "hardware"};
                        for (const auto& s : param_subs) {
                            std::string p = rel_path + "/parameters/" + s;
                            std::system((smb_base + " -c \"cd " + p + "; prompt; del *; cd ..; rmdir " + s + "\" 2>/dev/null").c_str());
                        }
                        std::system((smb_base + " -c \"cd " + rel_path + "; rmdir parameters\" 2>/dev/null").c_str());

                        std::string p_record = rel_path + "/record";
                        std::system((smb_base + " -c \"cd " + p_record + "; prompt; del *; cd ..; rmdir record\" 2>/dev/null").c_str());

                        std::string del_main = smb_base + " -c \"cd " + rel_path + "; prompt; del *\" 2>/dev/null";
                        std::system(del_main.c_str());
                        
                        size_t last_slash = rel_path.find_last_of('/');
                        std::string parent_path = (last_slash == std::string::npos) ? "" : rel_path.substr(0, last_slash);
                        std::string leaf_name = (last_slash == std::string::npos) ? rel_path : rel_path.substr(last_slash + 1);

                        std::string rmdir_main_cmd;
                        if (parent_path.empty()) {
                            rmdir_main_cmd = smb_base + " -c \"rmdir " + leaf_name + "\" 2>/dev/null";
                        } else {
                            rmdir_main_cmd = smb_base + " -c \"cd " + parent_path + "; rmdir " + leaf_name + "\" 2>/dev/null";
                        }
                        
                        int ret = std::system(rmdir_main_cmd.c_str());

                        if (ret == 0) {
                            std::cout << "[WebSocket] DISCARD_BOTH_DATA: Samba 数据删除成功" << std::endl;
                        } else {
                            std::cout << "[WebSocket] DISCARD_BOTH_DATA: Samba 数据删除失败或目录不存在" << std::endl;
                        }
                    }
                }

                std::string deleted_path = data_folder_;
                data_folder_.clear();
                is_data_saved_ = false;

                response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}, {"data", {
                    {"msg", "本机及Samba数据丢弃成功"},
                    {"deleted_path", deleted_path}
                }}};
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] DISCARD_BOTH_DATA 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "UPLOAD_TO_SAMBA") {
            try {
                std::lock_guard<std::mutex> lock(g_samba_upload_mutex);
                std::string agent_id = data.contains("agent_id") ? data["agent_id"].get<std::string>() : "";
                std::string task_id = data.contains("task_id") ? data["task_id"].get<std::string>() : "";
                if (agent_id.empty() || task_id.empty()) {
                    throw std::runtime_error("请提供 agent_id 和 task_id 参数");
                }
                auto reject_path_traversal = [](const std::string& s) {
                    return s.find("..") != std::string::npos || s.find('/') != std::string::npos;
                };
                if (reject_path_traversal(agent_id) || reject_path_traversal(task_id)) {
                    throw std::runtime_error("agent_id 和 task_id 不能包含 .. 或 /");
                }

                fs::path source_path = fs::path(GUODI_DATASET_ROOT) / agent_id / task_id;
                std::cout << "[WebSocket] UPLOAD_TO_SAMBA: agent_id=" << agent_id << ", task_id=" << task_id
                          << ", 路径=" << source_path.string() << std::endl;

                fs::path canonical_root = fs::path(GUODI_DATASET_ROOT).lexically_normal();
                fs::path canonical_source = source_path.lexically_normal();
                std::string src_str = canonical_source.string();
                std::string root_str = canonical_root.string();
                if (src_str.find(root_str) != 0 || (src_str.size() > root_str.size() && src_str[root_str.size()] != '/')) {
                    throw std::runtime_error("路径必须在 " + GUODI_DATASET_ROOT + " 下");
                }
                if (!fs::exists(source_path) || !fs::is_directory(source_path)) {
                    throw std::runtime_error("源路径不存在或不是目录: " + source_path.string());
                }

                std::string smb_base = "smbclient \"//" + g_samba_host + "/" + g_samba_share
                    + "\" -U " + g_samba_user + "%" + g_samba_pass;

                std::system((smb_base + " -c \"mkdir " + agent_id + "\" 2>/dev/null").c_str());
                std::system((smb_base + " -c \"cd " + agent_id + "; mkdir " + task_id + "\" 2>/dev/null").c_str());

                std::string smb_cmd = smb_base + " -c \"cd " + agent_id + "; cd " + task_id
                    + "; lcd \\\"" + source_path.string() + "\\\"; recurse; prompt; mput *\" >/dev/null 2>&1";
                std::cout << "[WebSocket] UPLOAD_TO_SAMBA: 上传中 " << source_path.string()
                          << " -> smb://" << g_samba_host << "/" << g_samba_share << "/" << agent_id << "/" << task_id << std::endl;
                int ret = std::system(smb_cmd.c_str());
                if (ret != 0) {
                    throw std::runtime_error("smbclient上传失败(sudo apt install smbclient)");
                }

                std::string dest_path = "smb://" + g_samba_host + "/" + g_samba_share + "/" + agent_id + "/" + task_id;
                std::cout << "[WebSocket] UPLOAD_TO_SAMBA: 上传完成 -> " << dest_path << std::endl;
                response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}, {"data", {
                    {"msg", "数据已上传到Samba共享目录"},
                    {"source_path", source_path.string()},
                    {"dest_path", dest_path},
                    {"agent_id", agent_id},
                    {"task_id", task_id}
                }}};
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] UPLOAD_TO_SAMBA 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "LIST_FILES") {
            std::string path = data.contains("path") ? data["path"].get<std::string>() : "/mnt/data/keenon_baidu_dataset";
            std::cout << "[WebSocket] LIST_FILES: 请求列出目录内容 path=" << path << std::endl;
            json items = json::array();
            try {
                if (fs::exists(path) && fs::is_directory(path)) {
                    size_t file_count = 0, dir_count = 0;
                    for (const auto& entry : fs::directory_iterator(path)) {
                        auto f_time = fs::last_write_time(entry);
                        auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(f_time - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
                        std::time_t tt = std::chrono::system_clock::to_time_t(sctp);
                        
                        bool is_dir = entry.is_directory();
                        if (is_dir) dir_count++; else file_count++;
                        
                        items.push_back({
                            {"name", entry.path().filename().string()},
                            {"type", is_dir ? "directory" : "file"},
                            {"size_bytes", entry.is_regular_file() ? fs::file_size(entry) : 0},
                            {"mtime", tt}
                        });
                    }
                    std::cout << "[WebSocket] LIST_FILES: 成功列出 " << items.size() << " 个项目 (文件:" << file_count 
                              << ", 目录:" << dir_count << ")" << std::endl;
                    response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "ListFilesPayload"}, {"data", {{"items", items}, {"path", path}}}};
                } else {
                    std::cout << "[WebSocket] LIST_FILES: 路径不存在或不是目录: " << path << std::endl;
                    response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", "路径不存在或不是目录"}}}};
                }
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] LIST_FILES: 列表获取失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "DELETE_ITEM") {
            std::string path = data.at("path").get<std::string>();
            std::cout << "[WebSocket] DELETE_ITEM: 准备删除项目 path=" << path << std::endl;
            try {
                if (!fs::exists(path)) {
                    std::cout << "[WebSocket] DELETE_ITEM: 项目不存在: " << path << std::endl;
                    response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", "项目不存在"}}}};
                } else {
                    if (fs::is_directory(path)) {
                        fs::remove_all(path);
                        std::cout << "[WebSocket] DELETE_ITEM: 目录删除成功: " << path << std::endl;
                    } else {
                        fs::remove(path);
                        std::cout << "[WebSocket] DELETE_ITEM: 文件删除成功: " << path << std::endl;
                    }
                    response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}, {"data", {
                        {"msg", "项目删除成功"},
                        {"path", path},
                        {"type", fs::is_directory(path) ? "directory" : "file"}
                    }}};
                }
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] DELETE_ITEM: 删除失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "CLEAR_DIRECTORY") {
            std::string path = data.at("path").get<std::string>();
            std::cout << "[WebSocket] CLEAR_DIRECTORY: 准备清空目录 path=" << path << std::endl;
            try {
                if (!fs::exists(path)) {
                    std::cout << "[WebSocket] CLEAR_DIRECTORY: 目录不存在: " << path << std::endl;
                    response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", "目录不存在"}}}};
                } else if (!fs::is_directory(path)) {
                    std::cout << "[WebSocket] CLEAR_DIRECTORY: 路径不是目录: " << path << std::endl;
                    response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", "路径不是目录"}}}};
                } else {
                    size_t file_count = 0, dir_count = 0;
                    for (const auto& entry : fs::directory_iterator(path)) {
                        if (entry.is_directory()) {
                            fs::remove_all(entry.path());
                            dir_count++;
                        } else {
                            fs::remove(entry.path());
                            file_count++;
                        }
                    }
                    std::cout << "[WebSocket] CLEAR_DIRECTORY: 清空完成 - 删除文件:" << file_count << "个, 目录:" << dir_count << "个" << std::endl;
                    response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}, {"data", {
                        {"msg", "目录内容清空成功"},
                        {"path", path},
                        {"cleared_files", static_cast<int>(file_count)},
                        {"cleared_dirs", static_cast<int>(dir_count)}
                    }}};
                }
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] CLEAR_DIRECTORY: 清空失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "DOWNLOAD_FILE") {
            std::string file_path = data.contains("path") ? data["path"].get<std::string>() : "";
            std::cout << "[WebSocket] DOWNLOAD_FILE: 请求下载文件 path=" << file_path << std::endl;
            
            try {
                if (file_path.empty()) {
                    std::cout << "[WebSocket] DOWNLOAD_FILE: 文件路径为空" << std::endl;
                    response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", "文件路径不能为空"}}}};
                } else if (!fs::exists(file_path)) {
                    std::cout << "[WebSocket] DOWNLOAD_FILE: 文件不存在: " << file_path << std::endl;
                    response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", "文件不存在: " + file_path}}}};
                } else if (!fs::is_regular_file(file_path)) {
                    std::cout << "[WebSocket] DOWNLOAD_FILE: 路径不是文件: " << file_path << std::endl;
                    response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", "指定路径不是文件: " + file_path}}}};
                } else {
                    size_t file_size = fs::file_size(file_path);

                    std::cout << "[WebSocket] DOWNLOAD_FILE:使用HTTP服务器提供下载，文件大小=" 
                              << (file_size / 1024 / 1024) << "MB" << std::endl;
                    {
                        std::lock_guard<std::mutex> lock(my_dual_arm_package::g_http_server_mutex);
                        if (!my_dual_arm_package::g_http_file_server) {
                            std::string base_path = "/";  
                            my_dual_arm_package::g_http_file_server = std::make_unique<HttpFileServer>(18080, base_path);
                            my_dual_arm_package::g_http_file_server->start();
                            std::cout << "[WebSocket] HTTP文件服务器已启动，基础路径: " << base_path << std::endl;
                        }
                    }

                    fs::path file_path_obj(file_path);
                    std::string url_path = file_path_obj.relative_path().string();
                    if (!url_path.empty() && url_path[0] == '/') {
                        url_path = url_path.substr(1);
                    }
                    std::string server_address = my_dual_arm_package::get_eth_ip_address();
                    std::string download_url = "http://" + server_address + ":18080/files/" + url_path;
                    response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "DownloadFilePayload"}, {"data", {
                        {"file_name", file_path_obj.filename().string()},
                        {"download_url", download_url},
                        {"file_size", file_size},
                        {"size_mb", static_cast<double>(file_size) / (1024 * 1024)}
                    }}};
                                    
                    std::cout << "[WebSocket] DOWNLOAD_FILE:准备HTTP下载链接 file_name=" 
                              << file_path_obj.filename().string() << ", size=" << file_size << " bytes" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] DOWNLOAD_FILE: 下载失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "VERIFY_DATA") {
            try {
                std::string data_folder = data.contains("data_folder") ? data["data_folder"].get<std::string>() : "";
                if (data_folder.empty()) {
                    data_folder = data_folder_;
                    std::cout << "[WebSocket] VERIFY_DATA: 请求中未指定路径，使用会话路径=" << data_folder << std::endl;
                } else {
                    std::cout << "[WebSocket] VERIFY_DATA: 使用请求指定路径=" << data_folder << std::endl;
                }
                
                double folder_size_mb = 0;
                if (!data_folder.empty() && fs::exists(data_folder)) {
                    size_t total_size = 0;
                    try {
                        for (const auto& entry : fs::recursive_directory_iterator(data_folder)) {
                            if (entry.is_regular_file()) total_size += entry.file_size();
                        }
                        folder_size_mb = static_cast<double>(total_size) / (1024 * 1024);
                    } catch (const std::exception& e) {
                        std::cout << "[WebSocket] VERIFY_DATA: 遍历目录计算大小时出错: " << e.what() << std::endl;
                    }
                } else {
                    std::cout << "[WebSocket] VERIFY_DATA: 路径不存在或为空: " << data_folder << std::endl;
                }
                
                response = {{"cmd", "ACK"}};
                response["mid"] = mid;
                response["type"] = "VerifyDataPayload";
                json verify_data;
                verify_data["data_folder"] = data_folder;
                verify_data["folder_size_mb"] = folder_size_mb;

                json cam_frames = processor_ ? json(processor_->camera_frames_count) : json::object();
                json arm_frames = processor_ ? json(processor_->joint_frames_count) : json::object();

                fs::path meta_path = fs::path(data_folder) / "meta_info.json";
                if (fs::exists(meta_path)) {
                    try {
                        std::ifstream f(meta_path);
                        json meta = json::parse(f);
                        if (meta.contains("camera_frame_details")) {
                            cam_frames = meta["camera_frame_details"];
                        }
                        if (meta.contains("arm_frame_details")) {
                            arm_frames["left"] = meta["arm_frame_details"].value("left_arm", 0);
                            arm_frames["right"] = meta["arm_frame_details"].value("right_arm", 0);
                        }
                        std::cout << "[WebSocket] VERIFY_DATA: 从 meta_info.json 读取到精确统计数值" << std::endl;
                    } catch (const std::exception& e) {
                        std::cout << "[WebSocket] VERIFY_DATA: 读取 meta_info.json 失败: " << e.what() << std::endl;
                    }
                }

                verify_data["camera_frames"] = cam_frames;
                verify_data["arm_frames"] = arm_frames;
                response["data"] = verify_data;
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] VERIFY_DATA 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "WRITE_SUB_TASK_JSON") {
            try {
                std::string meta_base = data_folder_;
                if (meta_base.empty()) {
                    throw std::runtime_error("无采集数据路径，请先执行 START_TELEMETRY 开始采集");
                }

                fs::path canonical_root = fs::path(GUODI_DATASET_ROOT).lexically_normal();
                fs::path canonical_base = fs::path(meta_base).lexically_normal();
                std::string base_str = canonical_base.string();
                std::string root_str = canonical_root.string();
                std::string agent_id, task_id, sub_number;
                if (base_str.size() > root_str.size() && base_str.compare(0, root_str.size(), root_str) == 0) {
                    std::string rel = base_str.substr(root_str.size());
                    if (!rel.empty() && rel[0] == '/') rel.erase(0, 1);
                    std::vector<std::string> parts;
                    for (size_t i = 0; i < rel.size(); ) {
                        size_t j = rel.find('/', i);
                        if (j == std::string::npos) {
                            parts.push_back(rel.substr(i));
                            break;
                        }
                        parts.push_back(rel.substr(i, j - i));
                        i = j + 1;
                    }
                    if (parts.size() >= 3) {
                        agent_id = parts[0];
                        task_id = parts[1];
                        sub_number = parts[2];  
                    } else if (parts.size() >= 2) {
                        agent_id = parts[0];
                        task_id = parts[1];
                        sub_number = task_id;   
                    }
                }
                if (agent_id.empty() || task_id.empty()) {
                    throw std::runtime_error("无法从 data_folder 解析 agent_id/task_id: " + meta_base);
                }
                if (agent_id.find("..") != std::string::npos || agent_id.find('/') != std::string::npos ||
                    task_id.find("..") != std::string::npos || task_id.find('/') != std::string::npos) {
                    throw std::runtime_error("解析到的 agent_id 或 task_id 无效");
                }

                fs::path meta_path = fs::path(meta_base) / "meta_info.json";
                if (!fs::exists(meta_path)) {
                    throw std::runtime_error("meta_info.json 不存在: " + meta_path.string());
                }

                std::ifstream meta_f(meta_path);
                json meta = json::parse(meta_f);
                int duration_ms = 0;
                if (meta.contains("durationInMs") && meta["durationInMs"].is_number()) {
                    duration_ms = meta["durationInMs"].get<int>();
                }
                int duration_sec = duration_ms / 1000;
                std::string create_time = meta.value("create_time", "");
                if (create_time.empty()) {
                    throw std::runtime_error("meta_info.json 缺少 create_time");
                }

                std::string start_time_iso = create_time_to_iso8601_utc(create_time);
                if (start_time_iso.empty()) {
                    throw std::runtime_error("无法解析 create_time 格式: " + create_time);
                }
                std::string end_time_iso = compute_end_time_iso8601(start_time_iso, duration_sec);

                double file_size_mb = 0;
                if (data.contains("file_size_mb") && data["file_size_mb"].is_number()) {
                    file_size_mb = data["file_size_mb"].get<double>();
                } else if (fs::exists(meta_base)) {
                    size_t total = 0;
                    for (const auto& entry : fs::recursive_directory_iterator(meta_base)) {
                        if (entry.is_regular_file()) total += entry.file_size();
                    }
                    file_size_mb = static_cast<double>(total) / (1024 * 1024);
                }
                char file_size_str[32];
                if (file_size_mb >= 1024.0) {
                    snprintf(file_size_str, sizeof(file_size_str), "%.1fG", file_size_mb / 1024.0);
                } else {
                    snprintf(file_size_str, sizeof(file_size_str), "%.1fMB", file_size_mb);
                }

                json new_entry = {
                    {"subNumber", sub_number},
                    {"duration", duration_sec},
                    {"startTime", start_time_iso},
                    {"endTime", end_time_iso},
                    {"fileSize", std::string(file_size_str)}
                };

                json sub_task_array = json::array();
                std::string smb_base = "smbclient \"//" + g_samba_host + "/" + g_samba_share
                    + "\" -U " + g_samba_user + "%" + g_samba_pass;
                std::string cd_chain = "cd " + agent_id + "; cd " + task_id + "; cd task_info";
                std::string task_info_path = agent_id + "/" + task_id + "/task_info";
                fs::path tmp_get = fs::path("/tmp") / ("sub_task_get_" + std::to_string(getpid()) + ".json");
                {
                    std::lock_guard<std::mutex> lock(g_samba_upload_mutex);
                    std::string mk_cmd = smb_base + " -c \"mkdir " + agent_id + "\" 2>/dev/null";
                    std::system(mk_cmd.c_str());
                    mk_cmd = smb_base + " -c \"cd " + agent_id + "; mkdir " + task_id + "\" 2>/dev/null";
                    std::system(mk_cmd.c_str());
                    mk_cmd = smb_base + " -c \"cd " + agent_id + "; cd " + task_id + "; mkdir task_info\" 2>/dev/null";
                    std::system(mk_cmd.c_str());

                    std::string get_cmd = smb_base + " -c \"" + cd_chain + "; get sub_task.json \\\"" + tmp_get.string() + "\\\"\" 2>/dev/null";
                    int get_ret = std::system(get_cmd.c_str());
                    if (get_ret == 0 && fs::exists(tmp_get)) {
                        try {
                            std::ifstream f(tmp_get);
                            sub_task_array = json::parse(f);
                            if (!sub_task_array.is_array()) sub_task_array = json::array();
                        } catch (...) {
                            sub_task_array = json::array();
                        }
                        fs::remove(tmp_get);
                    }
                }

                bool replaced = false;
                for (size_t i = 0; i < sub_task_array.size(); ++i) {
                    if (sub_task_array[i].value("subNumber", "") == sub_number) {
                        sub_task_array[i] = new_entry;
                        replaced = true;
                        break;
                    }
                }
                if (!replaced) {
                    sub_task_array.push_back(new_entry);
                }

                fs::path tmp_put = fs::path("/tmp") / ("sub_task_put_" + std::to_string(getpid()) + ".json");
                {
                    std::ofstream f(tmp_put);
                    f << sub_task_array.dump(4);
                    f.close();
                }
                {
                    std::lock_guard<std::mutex> lock(g_samba_upload_mutex);
                    std::string put_cmd = smb_base + " -c \"" + cd_chain + "; put \\\"" + tmp_put.string() + "\\\" sub_task.json\"";
                    int put_ret = std::system(put_cmd.c_str());
                    fs::remove(tmp_put);
                    if (put_ret != 0) {
                        throw std::runtime_error("smbclient 写入 sub_task.json 失败");
                    }
                }

                std::string dest_path = "smb://" + g_samba_host + "/" + g_samba_share + "/" + task_info_path + "/sub_task.json";
                std::cout << "[WebSocket] WRITE_SUB_TASK_JSON: 写入成功 -> " << dest_path << std::endl;
                response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}, {"data", {
                    {"msg", "sub_task.json 写入成功"},
                    {"dest_path", dest_path},
                    {"agent_id", agent_id},
                    {"task_id", task_id}
                }}};
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] WRITE_SUB_TASK_JSON 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "RECOLLECT_TASK") {
            try {
                std::string agent_id = data.contains("agent_id") ? data["agent_id"].get<std::string>() : "";
                std::string task_id = data.contains("task_id") ? data["task_id"].get<std::string>() : "";
                if (agent_id.empty() || task_id.empty()) {
                    throw std::runtime_error("请提供 agent_id 和 task_id 参数");
                }
                auto reject_path_traversal = [](const std::string& s) {
                    return s.find("..") != std::string::npos || s.find('/') != std::string::npos;
                };
                if (reject_path_traversal(agent_id) || reject_path_traversal(task_id)) {
                    throw std::runtime_error("agent_id 和 task_id 不能包含 .. 或 /");
                }

                fs::path task_path = fs::path(GUODI_DATASET_ROOT) / agent_id / task_id;
                fs::path canonical_root = fs::path(GUODI_DATASET_ROOT).lexically_normal();
                fs::path canonical_task = task_path.lexically_normal();
                std::string task_str = canonical_task.string();
                std::string root_str = canonical_root.string();
                if (task_str.find(root_str) != 0 || (task_str.size() > root_str.size() && task_str[root_str.size()] != '/')) {
                    throw std::runtime_error("路径必须在 " + GUODI_DATASET_ROOT + " 下");
                }

                std::string rel_path_base = agent_id + "/" + task_id;
                std::vector<std::string> data_subdirs;
                if (fs::exists(task_path) && fs::is_directory(task_path)) {
                    for (const auto& entry : fs::directory_iterator(task_path)) {
                        if (entry.is_directory()) {
                            std::string name = entry.path().filename().string();
                            if (name != "task_info") {
                                data_subdirs.push_back(name);
                            }
                        }
                    }
                }

                std::string smb_base = "smbclient \"//" + g_samba_host + "/" + g_samba_share
                    + "\" -U " + g_samba_user + "%" + g_samba_pass;

                {
                    std::lock_guard<std::mutex> lock(g_samba_upload_mutex);

                    for (const std::string& sub : data_subdirs) {
                        std::string rel_path = rel_path_base + "/" + sub;
                        std::vector<std::string> cams = {"hand_left", "hand_right", "head"};
                        std::vector<std::string> types = {"color", "depth"};
                        for (const auto& cam : cams) {
                            for (const auto& type : types) {
                                std::string p = rel_path + "/camera/" + cam + "/" + type;
                                std::system((smb_base + " -c \"cd " + p + "; prompt; del *; cd ..; rmdir " + type + "\" 2>/dev/null").c_str());
                            }
                            std::system((smb_base + " -c \"cd " + rel_path + "/camera; rmdir " + cam + "\" 2>/dev/null").c_str());
                        }
                        std::system((smb_base + " -c \"cd " + rel_path + "; rmdir camera\" 2>/dev/null").c_str());
                        std::vector<std::string> param_subs = {"camera", "hardware"};
                        for (const auto& s : param_subs) {
                            std::string p = rel_path + "/parameters/" + s;
                            std::system((smb_base + " -c \"cd " + p + "; prompt; del *; cd ..; rmdir " + s + "\" 2>/dev/null").c_str());
                        }
                        std::system((smb_base + " -c \"cd " + rel_path + "; rmdir parameters\" 2>/dev/null").c_str());
                        std::string p_record = rel_path + "/record";
                        std::system((smb_base + " -c \"cd " + p_record + "; prompt; del *; cd ..; rmdir record\" 2>/dev/null").c_str());
                        std::system((smb_base + " -c \"cd " + rel_path + "; prompt; del *\" 2>/dev/null").c_str());
                        size_t last_slash = rel_path.find_last_of('/');
                        std::string parent_path = (last_slash == std::string::npos) ? "" : rel_path.substr(0, last_slash);
                        std::string leaf_name = (last_slash == std::string::npos) ? rel_path : rel_path.substr(last_slash + 1);
                        std::string rmdir_cmd = parent_path.empty()
                            ? smb_base + " -c \"rmdir " + leaf_name + "\" 2>/dev/null"
                            : smb_base + " -c \"cd " + parent_path + "; rmdir " + leaf_name + "\" 2>/dev/null";
                        std::system(rmdir_cmd.c_str());
                    }

                    std::string cd_chain = "cd " + agent_id + "; cd " + task_id + "; cd task_info";
                    std::system((smb_base + " -c \"mkdir " + agent_id + "\" 2>/dev/null").c_str());
                    std::system((smb_base + " -c \"cd " + agent_id + "; mkdir " + task_id + "\" 2>/dev/null").c_str());
                    std::system((smb_base + " -c \"cd " + agent_id + "; cd " + task_id + "; mkdir task_info\" 2>/dev/null").c_str());
                    fs::path tmp_put = fs::path("/tmp") / ("sub_task_recollect_" + std::to_string(getpid()) + ".json");
                    { std::ofstream f(tmp_put); f << "[]"; f.close(); }
                    std::string put_cmd = smb_base + " -c \"" + cd_chain + "; put \\\"" + tmp_put.string() + "\\\" sub_task.json\" 2>/dev/null";
                    std::system(put_cmd.c_str());
                    fs::remove(tmp_put);
                }

                for (const std::string& sub : data_subdirs) {
                    fs::path sub_path = task_path / sub;
                    if (fs::exists(sub_path)) {
                        fs::remove_all(sub_path);
                    }
                }

                std::string task_prefix = task_path.string() + "/";
                if (!data_folder_.empty() && (data_folder_ == task_path.string() || data_folder_.find(task_prefix) == 0)) {
                    data_folder_.clear();
                    is_data_saved_ = false;
                }

                std::cout << "[WebSocket] RECOLLECT_TASK: 数据目录已清空，sub_task.json 已重置, agent_id=" << agent_id << ", task_id=" << task_id << std::endl;
                response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}, {"data", {
                    {"msg", "重新采集：数据目录已清空，sub_task.json 已重置"},
                    {"agent_id", agent_id},
                    {"task_id", task_id}
                }}};
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] RECOLLECT_TASK 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "CANCEL_COLLECTION") {
            try {
                std::string agent_id = data.contains("agent_id") ? data["agent_id"].get<std::string>() : "";
                std::string task_id = data.contains("task_id") ? data["task_id"].get<std::string>() : "";
                if (agent_id.empty() || task_id.empty()) {
                    throw std::runtime_error("请提供 agent_id 和 task_id 参数");
                }
                auto reject_path_traversal = [](const std::string& s) {
                    return s.find("..") != std::string::npos || s.find('/') != std::string::npos;
                };
                if (reject_path_traversal(agent_id) || reject_path_traversal(task_id)) {
                    throw std::runtime_error("agent_id 和 task_id 不能包含 .. 或 /");
                }

                fs::path task_path = fs::path(GUODI_DATASET_ROOT) / agent_id / task_id;
                fs::path canonical_root = fs::path(GUODI_DATASET_ROOT).lexically_normal();
                fs::path canonical_task = task_path.lexically_normal();
                std::string task_str = canonical_task.string();
                std::string root_str = canonical_root.string();
                if (task_str.find(root_str) != 0 || (task_str.size() > root_str.size() && task_str[root_str.size()] != '/')) {
                    throw std::runtime_error("路径必须在 " + GUODI_DATASET_ROOT + " 下");
                }

                std::string rel_path_base = agent_id + "/" + task_id;
                std::vector<std::string> all_subdirs;
                
                if (fs::exists(task_path) && fs::is_directory(task_path)) {
                    for (const auto& entry : fs::directory_iterator(task_path)) {
                        if (entry.is_directory()) {
                            all_subdirs.push_back(entry.path().filename().string());
                        }
                    }
                }

                std::set<std::string> subdir_set(all_subdirs.begin(), all_subdirs.end());
                subdir_set.insert("task_info");

                std::string smb_base = "smbclient \"//" + g_samba_host + "/" + g_samba_share
                    + "\" -U " + g_samba_user + "%" + g_samba_pass;

                {
                    std::lock_guard<std::mutex> lock(g_samba_upload_mutex);

                    for (const std::string& sub : subdir_set) {
                        std::string rel_path = rel_path_base + "/" + sub;
                        if (sub == "task_info") {
                            std::system((smb_base + " -c \"cd " + rel_path + "; prompt; del *; cd ..; rmdir task_info\" 2>/dev/null").c_str());
                        } else {
                            std::vector<std::string> cams = {"hand_left", "hand_right", "head"};
                            std::vector<std::string> types = {"color", "depth"};
                            for (const auto& cam : cams) {
                                for (const auto& type : types) {
                                    std::string p = rel_path + "/camera/" + cam + "/" + type;
                                    std::system((smb_base + " -c \"cd " + p + "; prompt; del *; cd ..; rmdir " + type + "\" 2>/dev/null").c_str());
                                }
                                std::system((smb_base + " -c \"cd " + rel_path + "/camera; rmdir " + cam + "\" 2>/dev/null").c_str());
                            }
                            std::system((smb_base + " -c \"cd " + rel_path + "; rmdir camera\" 2>/dev/null").c_str());
                            
                            std::vector<std::string> param_subs = {"camera", "hardware"};
                            for (const auto& s : param_subs) {
                                std::string p = rel_path + "/parameters/" + s;
                                std::system((smb_base + " -c \"cd " + p + "; prompt; del *; cd ..; rmdir " + s + "\" 2>/dev/null").c_str());
                            }
                            std::system((smb_base + " -c \"cd " + rel_path + "; rmdir parameters\" 2>/dev/null").c_str());
                            
                            std::string p_record = rel_path + "/record";
                            std::system((smb_base + " -c \"cd " + p_record + "; prompt; del *; cd ..; rmdir record\" 2>/dev/null").c_str());

                            std::system((smb_base + " -c \"cd " + rel_path + "; prompt; del *\" 2>/dev/null").c_str());
                            std::system((smb_base + " -c \"cd " + rel_path_base + "; rmdir " + sub + "\" 2>/dev/null").c_str());
                        }
                    }
                    std::system((smb_base + " -c \"cd " + rel_path_base + "; prompt; del *\" 2>/dev/null").c_str());
                    std::system((smb_base + " -c \"cd " + agent_id + "; rmdir " + task_id + "\" 2>/dev/null").c_str());
                    std::cout << "[WebSocket] CANCEL_COLLECTION: Samba 已尝试删除整个 task 目录: " << rel_path_base << std::endl;
                }

                if (fs::exists(task_path)) {
                    fs::remove_all(task_path);
                    std::cout << "[WebSocket] CANCEL_COLLECTION: 本机 task 目录已删除" << std::endl;
                }

                std::string task_prefix = task_path.string() + "/";
                if (!data_folder_.empty() && (data_folder_ == task_path.string() || data_folder_.find(task_prefix) == 0)) {
                    data_folder_.clear();
                    is_data_saved_ = false;
                }

                std::cout << "[WebSocket] CANCEL_COLLECTION: 任务已取消，agent_id=" << agent_id << ", task_id=" << task_id << std::endl;
                response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}, {"data", {
                    {"msg", "取消采集：本机及PC端 task 目录已删除"},
                    {"agent_id", agent_id},
                    {"task_id", task_id}
                }}};
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] CANCEL_COLLECTION 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "START_ARM_CONTROL") {
            try {
                std::string shell_cmd = kRemoteControlRosCmd;
                if (data.contains("command") && data["command"].is_string()) {
                    shell_cmd = data["command"].get<std::string>();
                }
                std::string wd = kRemoteControlWorkdirDefault;
                if (data.contains("workdir") && data["workdir"].is_string()) {
                    wd = data["workdir"].get<std::string>();
                }
                pid_t child = -1;
                std::string err;
                std::cout << "[WebSocket] START_ARM_CONTROL: workdir=\"" << wd << "\", shell=\"" << shell_cmd << "\"" << std::endl;
                if (!remote_control_start(wd, shell_cmd, &child, &err)) {
                    response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", err}}}};
                } else {
                    response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}, {"data", {
                        {"pid", child},
                        {"msg", "remote_control_data 已启动"}
                    }}};
                }
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] START_ARM_CONTROL 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }

        } else if (cmd == "STOP_ARM_CONTROL") {
            try {
                std::string msg = remote_control_stop();
                std::cout << "[WebSocket] STOP_ARM_CONTROL: " << msg << std::endl;
                response = {{"cmd", "ACK"}, {"mid", mid}, {"type", "EmptyPayload"}, {"data", {{"msg", msg}}}};
            } catch (const std::exception& e) {
                std::cout << "[WebSocket] STOP_ARM_CONTROL 失败: " << e.what() << std::endl;
                response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500}, {"message", e.what()}}}};
            }
        } else {
            std::cout << "[WebSocket] 不支持的指令类型: " << cmd << std::endl;
            response = {{"cmd", "ERROR"}, {"mid", mid}, {"data", {{"code", 500},{"message", "不支持的指令类型: " + cmd}}}};
        }
                
        {
            const std::string rc = response.value("cmd", std::string());
            std::cout << "[WebSocket] 指令处理完成 cmd=" << cmd << ", mid=" << mid
                      << ", 结果=" << (rc == "ACK" ? "成功" : "失败") << std::endl;
        }
        queue_message(response.dump());
    }
};

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
        if (!ec) {
            beast::error_code ep_ec;
            auto peer = socket.remote_endpoint(ep_ec);
            if (!ep_ec) {
                std::cout << "[WebSocket] 新连接 客户端地址: "
                          << peer.address().to_string() << ":" << peer.port() << std::endl;
            } else {
                std::cout << "[WebSocket] 新连接 无法获取客户端地址: " << ep_ec.message() << std::endl;
            }
            std::make_shared<Session>(std::move(socket))->run();
        }
        do_accept();
    }
};

std::string get_eth_ip_address() {
    struct ifaddrs* ifap = nullptr;
    if (getifaddrs(&ifap) != 0) {
        return "127.0.0.1";
    }
    std::string result = "127.0.0.1";
    for (struct ifaddrs* ifa = ifap; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
        if (!(ifa->ifa_flags & IFF_UP)) continue;
        if (strncmp(ifa->ifa_name, "eth", 3) != 0) continue;  
        const auto* sa = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
        char buf[INET_ADDRSTRLEN];
        if (inet_ntop(AF_INET, &sa->sin_addr, buf, sizeof(buf))) {
            result = buf;
            break;  
        }
    }
    freeifaddrs(ifap);
    return result;
}

} // namespace my_dual_arm_package


int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    my_dual_arm_package::load_samba_config();
    my_dual_arm_package::maybe_report_robot_url_from_config();

    my_dual_arm_package::DoubleArmRGBDDataProcessor::get_instance("Keenon_F1_DoubleArm", {});
    std::cout << "机械臂与相机订阅已预初始化(程序启动即订阅)" << std::endl;

    // auto const address = net::ip::make_address("127.0.0.1");
    // auto const address = net::ip::make_address("172.16.9.71");

    std::string bind_ip = my_dual_arm_package::get_eth_ip_address();
    auto const address = net::ip::make_address(bind_ip);
    auto const port = static_cast<unsigned short>(10780);
    auto const threads = 1;

    net::io_context ioc{threads};
    std::make_shared<my_dual_arm_package::Listener>(ioc, tcp::endpoint{address, port})->run();
    // std::cout << "WebSocket服务器启动成功,监听地址: ws://127.0.0.1:10780" << std::endl;
    // std::cout << "WebSocket服务器启动成功,监听地址: ws://172.16.9.71:10780" << std::endl;
    std::cout << "WebSocket服务器启动成功,监听地址: ws://" << bind_ip << ":" << port << std::endl;

    ioc.run();
    {
        std::lock_guard<std::mutex> lock(my_dual_arm_package::g_http_server_mutex);
        if (my_dual_arm_package::g_http_file_server) {
            my_dual_arm_package::g_http_file_server->stop();
            my_dual_arm_package::g_http_file_server.reset();
            std::cout << "HTTP文件服务器已关闭" << std::endl;
        }
    }
    
    rclcpp::shutdown();
    return 0;
}
