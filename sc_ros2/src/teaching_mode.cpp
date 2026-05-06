#include "qlrobotarm_canfd.h"
#include <unistd.h>
#include <iostream>
using namespace Eigen;
using namespace std;

std::vector<double> joint_angles_r = {0,0,0,0,0,0,0};
std::vector<double> joint_angles_l = {0,0,0,0,0,0,0};

std::vector<double> send_joint_r = {0,0,0,0,0,0,0};
std::vector<double> send_joint_l = {0,0,0,0,0,0,0};

int main(int argc, char* argv[]) {
    robot_init(1);
    bool running = true;
    setZeroCurrent_r();
    setZeroCurrent_l();

    cout << "程序运行中... 按下回车键退出" << endl;
  
    while(running) {
        joint_angles_r = getJointAngle_r();
        joint_angles_l = getJointAngle_l();

        cout << "当前右臂关节角度 " << joint_angles_r[0] << " , " << joint_angles_r[1] << " , " 
             << joint_angles_r[2] << " , " << joint_angles_r[3] << " , " << joint_angles_r[4] 
             << " , " << joint_angles_r[5] << " , " << joint_angles_r[6] << endl;
        cout << "当前左臂关节角度 " << joint_angles_l[0] << " , " << joint_angles_l[1] << " , " 
             << joint_angles_l[2] << " , " << joint_angles_l[3] << " , " << joint_angles_l[4] 
             << " , " << joint_angles_l[5] << " , " << joint_angles_l[6] << endl;

        // 检查标准输入是否有数据
        struct timeval tv = {0, 0};
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO, &fds);
        
        // 非阻塞检查
        int ret = select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv);
        if (ret > 0 && FD_ISSET(STDIN_FILENO, &fds)) {
            char c;
            read(STDIN_FILENO, &c, 1);
            if (c == '\n') {
                running = false;
                cout << "退出示教模式" << endl;
                for(int i = 0; i < 7 ; i ++ )
                {
                send_joint_r[i] = joint_angles_r[i]/180.0*M_PI;
                send_joint_l[i] = joint_angles_l[i]/180.0*M_PI;

                }
                moveJ_r(send_joint_r);
                moveJ_l(send_joint_l);

            }
        }
    }

    robot_end();
    return EXIT_SUCCESS;
}