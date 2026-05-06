#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
# 导入自定义消息：原有ArmData + 新增MotorErrors
from topic_pub.msg import ArmData
from topic_pub.msg import MotorErrors
import numpy as np
from scipy.spatial.transform import Rotation as R
import time


class MockArmsPublisher(Node):
    def __init__(self):
        super().__init__("mock_arms_publisher")
        
        # ---------------------- 原有：ArmData发布器（100Hz，左右臂各一个） ----------------------
        self.right_publisher_ = self.create_publisher(ArmData, "right_data", 10)
        self.left_publisher_ = self.create_publisher(ArmData, "left_data", 10)
        
        # ---------------------- 修正：MotorErrors发布器（10Hz，仅单发布器，单话题） ----------------------
        # 取消左右臂区分，仅一个发布器，话题名简洁通用，队列大小与原有保持一致
        self.motor_error_pub_ = self.create_publisher(MotorErrors, "motor_errors_topic", 10)
        
        # ---------------------- 原有：100Hz定时器（ArmData发布） ----------------------
        self.frequency = 100
        self.timer = self.create_timer(1.0 / self.frequency, self.publish_data)
        
        # ---------------------- 保留：10Hz独立定时器（MotorErrors发布，解耦频率） ----------------------
        self.motor_error_freq = 10
        self.motor_error_timer = self.create_timer(1.0 / self.motor_error_freq, self.publish_motor_errors)
        
        # ---------------------- 周期控制核心变量（原有，无修改） ----------------------
        self.start_time = time.time()          # 节点启动时间（基准）
        self.cycle_duration = 5.0              # 周期时长（5s）
        self.ramp_duration = 0.5               # 原夹爪2线性变化时长（已不再使用，保留避免报错）
        self.last_cycle_idx = -1               # 上一个周期序号（用于检测周期切换）
        self.current_gripper1 = 1.0            # 夹爪1初始状态（1）
        
        # 初始化臂参数
        self.init_right_arm_params()
        self.init_left_arm_params()
        
        # 启动日志（修改：更新夹爪2逻辑说明）
        self.get_logger().info(
            "模拟双臂数据发布器已启动【多线程修正版】，100Hz发布right_data/left_data（每侧16条数据）\n"
            "10Hz单话题发布电机错误消息：motor_errors，持续发送14个0的int32[]错误码（带ROS2标准时间戳）\n"
            "夹爪规则更新：\n"
            "- 夹爪1：保持原有周期切换逻辑（0/1瞬间切换）\n"
            "- 夹爪2：不再线性变化，改为每次发布时生成0~69之间的随机浮点数（带小数点）"
        )

    # ---------------------- 初始化函数（原有，无修改） ----------------------
    def init_right_arm_params(self):
        self.right_joint_ranges = [
            (-1.57, 1.57), (-1.57, 0.5), (-1.57, 1.57), (-1.57, 1.57),
            (-1.57, 1.57), (-1.57, 1.57), (-0.785, 0.785)
        ]
        self.right_position_ranges = [(0.3, 0.8), (-0.5, 0.0), (0.5, 1.2)]
        self.right_rpy_ranges = [(-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4)]
        
        self.right_prev_joints = [np.mean(range_) for range_ in self.right_joint_ranges]
        self.right_prev_gripper1 = 1.0
        self.right_prev_gripper2 = 1.0
        self.right_prev_position = [np.mean(range_) for range_ in self.right_position_ranges]
        self.right_prev_rpy = [np.mean(range_) for range_ in self.right_rpy_ranges]

    def init_left_arm_params(self):
        self.left_joint_ranges = [
            (-1.57, 1.57), (-0.5, 1.57), (-1.57, 1.57), (-1.57, 1.57),
            (-1.57, 1.57), (-1.57, 1.57), (-0.785, 0.785)
        ]
        self.left_position_ranges = [(0.3, 0.8), (0.0, 0.5), (0.5, 1.2)]
        self.left_rpy_ranges = [(-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4)]
        
        self.left_prev_joints = [np.mean(range_) for range_ in self.left_joint_ranges]
        self.left_prev_gripper1 = 1.0
        self.left_prev_gripper2 = 1.0
        self.left_prev_position = [np.mean(range_) for range_ in self.left_position_ranges]
        self.left_prev_rpy = [np.mean(range_) for range_ in self.left_rpy_ranges]

    # ---------------------- 平滑生成函数（原有，无修改） ----------------------
    def generate_smooth_data(self, prev_value, range_, max_step=0.05):
        min_val, max_val = range_
        step = np.random.uniform(-max_step, max_step)
        new_value = prev_value + step
        return np.clip(new_value, min_val, max_val)

    # ---------------------- 生成臂数据函数（原有，无修改） ----------------------
    def generate_arm_data(self, prev_joints, joint_ranges, 
                          prev_gripper1, prev_gripper2,
                          prev_position, position_ranges, 
                          prev_rpy, rpy_ranges,
                          target_gripper1, target_gripper2):
        # 1. 关节角度（7个）
        joints = []
        for i in range(7):
            joint = self.generate_smooth_data(prev_joints[i], joint_ranges[i], max_step=0.03)
            joints.append(joint)
        
        # 2. 夹爪数据（外部传入目标值）
        gripper1 = target_gripper1
        gripper2 = target_gripper2
        
        # 3. 末端位置xyz（3个）
        position = []
        for i in range(3):
            pos = self.generate_smooth_data(prev_position[i], position_ranges[i], max_step=0.01)
            position.append(pos)
        
        # 4. 末端姿态：欧拉角→四元数（4个，x,y,z,w）
        rpy = []
        for i in range(3):
            r = self.generate_smooth_data(prev_rpy[i], rpy_ranges[i], max_step=0.02)
            rpy.append(r)
        quaternion = R.from_euler('xyz', rpy).as_quat().tolist()
        
        # 拼接16位数据：7关节 + 2夹爪 + 3位置 + 4四元数
        arm_data = joints + [gripper1, gripper2] + position + quaternion
        return arm_data, joints, gripper1, gripper2, position, rpy

    # ---------------------- 原有：ArmData发布回调（100Hz，核心修改处） ----------------------
    def publish_data(self):
        current_time = time.time()
        elapsed = current_time - self.start_time
        cycle_idx = int(elapsed // self.cycle_duration)
        cycle_elapsed = elapsed % self.cycle_duration

        # 夹爪1 瞬间切换逻辑（保留原有，无修改）
        if cycle_idx != self.last_cycle_idx:
            self.current_gripper1 = 0.0 if self.current_gripper1 == 1.0 else 1.0
            self.last_cycle_idx = cycle_idx

        # ---------------------- 核心修改：夹爪2改为0~69随机浮点数 ----------------------
        # 替换原有线性变化逻辑，生成0到69之间的随机浮点数（带小数点）
        target_gripper2 = np.random.uniform(0.0, 69.0)
        # 可选：保留1位小数（如需控制精度，取消下面注释即可）
        # target_gripper2 = round(target_gripper2, 1)

        # 生成左右臂数据（逻辑不变，仅传入新的target_gripper2）
        right_data, right_joints, right_g1, right_g2, right_pos, right_rpy = self.generate_arm_data(
            self.right_prev_joints, self.right_joint_ranges,
            self.right_prev_gripper1, self.right_prev_gripper2,
            self.right_prev_position, self.right_position_ranges,
            self.right_prev_rpy, self.right_rpy_ranges,
            self.current_gripper1, target_gripper2
        )
        left_data, left_joints, left_g1, left_g2, left_pos, left_rpy = self.generate_arm_data(
            self.left_prev_joints, self.left_joint_ranges,
            self.left_prev_gripper1, self.left_prev_gripper2,
            self.left_prev_position, self.left_position_ranges,
            self.left_prev_rpy, self.left_rpy_ranges,
            self.current_gripper1, target_gripper2
        )

        # 更新历史值
        self.right_prev_joints = right_joints
        self.right_prev_gripper1 = right_g1
        self.right_prev_gripper2 = right_g2
        self.right_prev_position = right_pos
        self.right_prev_rpy = right_rpy
        self.left_prev_joints = left_joints
        self.left_prev_gripper1 = left_g1
        self.left_prev_gripper2 = left_g2
        self.left_prev_position = left_pos
        self.left_prev_rpy = left_rpy

        # 发布消息（带ROS2标准时间戳）
        right_msg = ArmData()
        right_msg.header.stamp = self.get_clock().now().to_msg()
        right_msg.data = right_data
        self.right_publisher_.publish(right_msg)

        left_msg = ArmData()
        left_msg.header.stamp = self.get_clock().now().to_msg()
        left_msg.data = left_data
        self.left_publisher_.publish(left_msg)

    # ---------------------- 修正：MotorErrors发布回调（10Hz，单话题单消息） ----------------------
    def publish_motor_errors(self):
        """
        单话题发布MotorErrors消息，仅1个实例，含ROS2标准时间戳+14个0的int32[]错误码
        与左右臂无关，独立10Hz发布
        """
        # 构建单个MotorErrors消息实例
        motor_error_msg = MotorErrors()
        # 设置ROS2标准时间戳（与ArmData消息时间戳规范一致）
        motor_error_msg.header.stamp = self.get_clock().now().to_msg()
        # 赋值14个0的int32数组，直接适配msg的int32[]类型
        motor_error_msg.error_codes = [0] * 14
        # 单发布器发布单消息
        self.motor_error_pub_.publish(motor_error_msg)


# ---------------------- 核心：多线程执行器（原有，无修改，保证频率不阻塞） ----------------------
def main(args=None):
    rclpy.init(args=args)
    node = MockArmsPublisher()
    # ROS2原生多线程执行器，4线程处理100Hz+10Hz并行回调，避免阻塞
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        # 优雅销毁资源，避免泄漏
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
