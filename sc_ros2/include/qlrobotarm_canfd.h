#ifndef QLROBOTARM_CANFD_H
#define QLROBOTARM_CANFD_H

/**
 * @file    qlrobotarm.h
 * @author  邵 超
 * @date    2026.3.5
 * @version 1.4.0
 * 
 */


#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>
#include <atomic>

using namespace Eigen;
using namespace std;

#define QL_VERSION 140

enum ArmSide
{
  r_arm = 1, // 右臂
  l_arm = 2  // 左臂
};
extern std::atomic<bool> PROGRAM_IS_DOWN;

//1.1单关节角度控制
void setJointAngle(int id, double degree);
//1.2单关节角度控制阻塞模式(新版未测)
void setJointAngle_block(int id, double degree);
//2单关节速度控制(新版未测)
void setJointSpeed(int id, double speed);
//3单关节电流控制
void setJointCurrent(int id, double current);
//4获取单关节角度
double getJointAngle(int id);
//5获取单关节速度(新版未测)
double getJointspeed(int id);
//6获取单关节电流
double getJointCurrent(int id);
//7右臂零电流模式
void setZeroCurrent_r();
//8左臂零电流模式
void setZeroCurrent_l();
//9.1右臂角度控制
int moveJ_r(const std::vector<double>& target_angle, double speed_factor = 1.0);
//9.2右臂角度控制阻塞模式
int moveJ_r_block(const std::vector<double>& target_angle, double speed_factor = 1.0);
//10.1左臂角度控制
int moveJ_l(const std::vector<double>& target_angle, double speed_factor = 1.0);
//10.2左臂角度控制阻塞模式
int moveJ_l_block(const std::vector<double>& target_angle, double speed_factor = 1.0);
//11.1右臂位姿控制
int moveJtoPose_r(const std::vector<double>& target_pose_vector, const double& armangle, double speed_factor = 1.0);
//11.2右臂位姿控制阻塞模式
int moveJtoPose_r_block(const std::vector<double>& target_pose_vector, const double& armangle, double speed_factor = 1.0);
//12.1左臂位姿控制
int moveJtoPose_l(const std::vector<double>& target_pose_vector, const double& armangle, double speed_factor = 1.0);
//12.2右臂位姿控制阻塞模式
int moveJtoPose_l_block(const std::vector<double>& target_pose_vector, const double& armangle, double speed_factor = 1.0);
//13.1右臂直线运动(新版未测)
int moveL_r(const std::vector<double>& target_pose_vector, const double& armangle, double tcpv);
//13.2右臂直线运动(测试，左右臂同步的点到点的直线运动)(新版未测)
int moveL_demo_r(const std::vector<double>& target_pose_vector, const std::vector<double>& current_pose_vector, const double& armangle, int steps);
//13.3右臂直线运动(测试，右臂点到点的直线运动)(新版未测)
int moveL_demo1_r(const std::vector<double>& target_pose_vector, const std::vector<double>& current_pose_vector, const double& armangle, int steps);
//14.1左臂直线运动(新版未测)
int moveL_l(const std::vector<double>& target_pose_vector, const double& armangle, double tcpv);
//14.2左臂直线运动(测试，左右臂同步的点到点的直线运动)(新版未测)
int moveL_demo_l(const std::vector<double>& target_pose_vector, const std::vector<double>& current_pose_vector, const double& armangle, int steps);
//14.3左臂直线运动(测试，左臂点到点的直线运动)(新版未测)
int moveL_demo1_l(const std::vector<double>& target_pose_vector, const std::vector<double>& current_pose_vector, const double& armangle, int steps);
//15右臂速度控制(新版未测)
int speedJ_r(const std::vector<double>& target_speed, const std::vector<double>& acceleration, double time);
//16左臂速度控制(新版未测)
int speedJ_l(const std::vector<double>& target_speed, const std::vector<double>& acceleration, double time);
//17.1右臂遥操作绝对位姿(新版未测)
int teleoperation_right(const std::vector<double>& joint_angles, double v, bool enable_);
//17.2右臂遥操作增量式
int teleoperation_incre_right(const std::vector<double>& joint_angles, double v, bool enable_);
//18.1左臂遥操作绝对位姿(新版未测)
int teleoperation_left(const std::vector<double>& joint_angles, double v, bool enable_);
//18.2左臂遥操作增量式
int teleoperation_incre_left(const std::vector<double>& joint_angles, double v, bool enable_);
//19获取右臂关节角度
std::vector<double> getJointAngle_r();
//20获取左臂关节角度
std::vector<double> getJointAngle_l();
//21获取右臂关节速度
std::vector<double> getJointSpeed_r();
//22获取左臂关节速度
std::vector<double> getJointSpeed_l();
//23获取右臂关节电流
std::vector<double> getJointCurrent_r();
//24获取左臂关节电流
std::vector<double> getJointCurrent_l();
//25获取右臂末端位姿
std::vector<double> getRobotPos_r();
//26获取右臂工具坐标系位姿(新版未测)
std::vector<double> getTcpPos_r();
//27获取左臂末端位姿
std::vector<double> getRobotPos_l();
//28获取左臂工具坐标系位姿(新版未测)
std::vector<double> getTcpPos_l();
//29获取DH参数
void getDH();
//30.1获取正运动学
Eigen::Matrix4d forward(const std::vector<double>& q_input_array);
//30.2获取正运动学tcp(新版未测)
Eigen::Matrix4d forward_tcp(const std::vector<double>& q_input_array);
//31获取雅可比矩阵
Eigen::MatrixXd getJacobiMatrix(const std::vector<double>& q_input_array);
//求取当前可操作度
double calculateManipulability(const std::vector<double>& q_input_array) ;
//32获取雅可比矩阵奇异值
double JacobianSingularValues(const std::vector<double>& q_input_array);
//33计算雅可比矩阵的伪逆
Eigen::MatrixXd calculateJacobianPseudoInverse(const std::vector<double>& q_input_array);
//34.1右臂逆运动学
std::vector<double> inverse_r(const std::vector<double>& target_pose_vector, const double& armangle);
//34.2右臂逆运动学带初始位置用于遥操作计算(新版未测)
std::vector<double> inverse_remote_r(const std::vector<double>& target_pose_vector, const std::vector<double>& initial_pose);
//34.3右臂逆运动学输入目标姿态矩阵和臂角用于增量式遥操作计算
std::vector<double> inverse_remote_t_r(const std::vector<double>& target_pose_vector_t, Eigen::Matrix3d target_q_m_r, const double& armangle);
//35.1左臂逆运动学
std::vector<double> inverse_l(const std::vector<double>& target_pose_vector, const double& armangle);
//35.2右臂逆运动学带初始位置用于遥操作计算(新版未测)
std::vector<double> inverse_remote_l(const std::vector<double>& target_pose_vector_t, Eigen::Matrix3d target_q_m_r, const double& armangle);
//35.3右臂逆运动学输入目标姿态矩阵和臂角用于增量式遥操作计算
std::vector<double> inverse_remote_t_l(const std::vector<double>& target_pose_vector_t, Eigen::Matrix3d target_q_m_r, const double& armangle);
//36机械臂初始化(1：仅右臂；2：仅左臂；3左右臂)
void robot_init(int arm_mode);
//37机器人控制结束
void robot_end();


//以下为测试用例
std::vector<double> getJointAngle_w();
int moveJ_w(const std::vector<double>& joint_angle);
int moveJ_wtest(const std::vector<double>& joint_angle);
int teleoperation_right_test(const std::vector<double>& joint_angles, double v, bool enable_);
int teleoperation_left_test(const std::vector<double>& joint_angles, double v, bool enable_);
double calculate_fused_score(const std::vector<double>& q);
std::vector<double> adaptive_inverse_r(const std::vector<double>& target_pose_vector);

void clamp_init() ;
void controlRightClamp(float pos_mm, float torque_nm);
void controlLeftClamp(float pos_mm, float torque_nm);
double getclampposition_r();
double getclampposition_l();
int parseMotorErrorCode(uint32_t error_code);
double queryJointerror(int id) ;
uint32_t getJointerror(int id );

double getJointhwvversion(int id) ;
void stopRightArm();
void stopLeftArm();

int get_motor_id(uint32_t id);
bool isMotorPowerOn(int id);


#endif // QLROBOTARM_CANFD_H