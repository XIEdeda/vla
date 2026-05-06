#include "qlrobotarm_canfd.h"
#include <unistd.h>
using namespace Eigen;
using namespace std;

int main(int argc, char* argv[]) {

    robot_init(1);
    // clamp_init() ;

    std::cout << "Clamp Control Program" << std::endl;
    std::cout << "1: Clamp 1" << std::endl;
    std::cout << "2: Unclamp 1" << std::endl;
    std::cout << "3: Clamp 2" << std::endl;
    std::cout << "4: Unclamp 2" << std::endl;
    std::cout << "Enter your choice: ";
 
    int choice;
    std::cin >> choice;

    if (choice == 1) {
         controlRightClamp(0.0f,1.0f);
        }
    else if (choice == 2) {
         controlRightClamp(100.0f,1.0f);
    }  
        else if (choice == 3) {
         controlLeftClamp(0.0f, 0.5f);
    }
        else if (choice == 4) {
         controlLeftClamp(100.0f,0.5f);
    }

	sleep(1);
    double rightdata;
    double leftdata;
    //while(1)
    //{
    rightdata = getclampposition_r();
    leftdata = getclampposition_l();
    cout<<" 右边夹爪开度： "<< rightdata<< " 左边夹爪开度： "<< leftdata<<endl;
    //}
    // sleep(1);
    robot_end();
    return EXIT_SUCCESS;
}
