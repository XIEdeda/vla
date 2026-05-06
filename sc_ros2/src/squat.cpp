#include "qlrobotarm_canfd.h"
#include <unistd.h>
#include <iostream>
using namespace Eigen;
using namespace std;


int main(int argc, char* argv[]) {
    robot_init(0);

    //先让2，3，4下蹲到差不多位置
    
    setJointAngle(2,45);
    setJointAngle(3,90);
    setJointAngle(4,45);
    
    // sleep(3);

    //然后再让4关节转动调节位置
    //setJointAngle(4,45);
    // setJointAngle(16,0);

    sleep(2);
    robot_end();
    return EXIT_SUCCESS;
}
