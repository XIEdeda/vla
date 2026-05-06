#include "qlrobotarm_canfd.h"
#include <unistd.h>
#include <iostream>
using namespace Eigen;
using namespace std;


int main(int argc, char* argv[]) {
    robot_init(0);

    //30,31电机同时回零
    setJointAngle(30,0);
    setJointAngle(31,-18);

    sleep(2);
    robot_end();
    return EXIT_SUCCESS;
}
