#include "qlrobotarm_canfd.h"
#include <unistd.h>
#include <iostream>
using namespace Eigen;
using namespace std;


int main(int argc, char* argv[]) {
    robot_init(0);

    //2，3，4电机同时回零
    setJointAngle(2,0);
    setJointAngle(3,0);
    setJointAngle(4,0);

    sleep(2);
    robot_end();
    return EXIT_SUCCESS;
}