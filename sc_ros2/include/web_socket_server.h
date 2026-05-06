#ifndef WEB_SOCKET_SERVER_H
#define WEB_SOCKET_SERVER_H

#include <string>
#include <mutex>

namespace web_socket_server {
    extern std::mutex receive_A_mutex; // 声明为extern
    void start_server(unsigned short port, std::string& receive_A);
}

#endif // WEB_SOCKET_SERVER_H
