#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from topic_pub.msg import ArmData  # 导入自定义的ArmData消息
import numpy as np
from scipy.spatial.transform import Rotation as R
import time


class MockArmsPublisher(Node):
    def __init__(self):
        super().__init__("mock_arms_publisher")
        
        # 发布者
        self.right_publisher_ = self.create_publisher(ArmData, "right_data", 10)
        self.left_publisher_ = self.create_publisher(ArmData, "left_data", 10)
        
        self.frequency = 100
        self.timer = self.create_timer(1.0 / self.frequency, self.publish_data)
        
        # ---------------------- 周期控制核心变量 ----------------------
        self.start_time = time.time()          # 节点启动时间（基准）
        self.cycle_duration = 5.0              # 周期时长（5s）
        self.ramp_duration = 0.5               # 夹爪2线性变化时长（0.5s）
        self.last_cycle_idx = -1               # 上一个周期序号（用于检测周期切换）
        self.current_gripper1 = 1.0            # 夹爪1初始状态（1）
        
        # 初始化臂参数
        self.init_right_arm_params()
        self.init_left_arm_params()
        
        self.get_logger().info(
            "模拟双臂数据发布器已启动(含时间戳)，110Hz发布right_data/left_data（每侧16条数据）\n"
            "夹爪严格同步规则：\n"
            "- 0s（第1周期开始）：夹爪1 瞬间从1→0 | 夹爪2 前0.5s从1线性降0.4，后4.5s保持0.4\n"
            "- 5s（第2周期开始）：夹爪1 瞬间从0→1 | 夹爪2 前0.5s从0.4线性升1，后4.5s保持1\n"
            "- 10s（第3周期开始）：夹爪1 瞬间从1→0 | 夹爪2 重复1→0.4变化\n"
            "- 以此类推，周期完全同步"
        )

    # ---------------------- 初始化函数 ----------------------
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

    # ---------------------- 平滑生成函数（保持不变） ----------------------
    def generate_smooth_data(self, prev_value, range_, max_step=0.05):
        min_val, max_val = range_
        step = np.random.uniform(-max_step, max_step)
        new_value = prev_value + step
        return np.clip(new_value, min_val, max_val)

    # ---------------------- 生成臂数据函数（修改：欧拉角→四元数） ----------------------
    def generate_arm_data(self, prev_joints, joint_ranges, 
                          prev_gripper1, prev_gripper2,
                          prev_position, position_ranges, 
                          prev_rpy, rpy_ranges,
                          target_gripper1, target_gripper2):
        # 1. 关节角度（保持不变）
        joints = []
        for i in range(7):
            joint = self.generate_smooth_data(prev_joints[i], joint_ranges[i], max_step=0.03)
            joints.append(joint)
        
        # 2. 夹爪数据（外部传入目标值）
        gripper1 = target_gripper1
        gripper2 = target_gripper2
        
        # 3. 末端位置xyz
        position = []
        for i in range(3):
            pos = self.generate_smooth_data(prev_position[i], position_ranges[i], max_step=0.01)
            position.append(pos)
        
        # 4. 末端姿态：生成平滑欧拉角 → 转换为四元数（x,y,z,w顺序）
        rpy = []
        for i in range(3):
            r = self.generate_smooth_data(prev_rpy[i], rpy_ranges[i], max_step=0.02)
            rpy.append(r)
        # 欧拉角（roll,pitch,yaw）转换为四元数
        quaternion = R.from_euler('xyz', rpy).as_quat().tolist()  # 返回[x,y,z,w]列表
        
        # 拼接16位数据：7关节 + 2夹爪 + 3位置 + 4四元数
        arm_data = joints + [gripper1, gripper2] + position + quaternion
        return arm_data, joints, gripper1, gripper2, position, rpy

    # ---------------------- 核心修改：发布逻辑（夹爪1瞬间切换） ----------------------
    def publish_data(self):
        current_time = time.time()
        elapsed = current_time - self.start_time  # 总运行时长
        cycle_idx = int(elapsed // self.cycle_duration)  # 当前周期序号（0,1,2...）
        cycle_elapsed = elapsed % self.cycle_duration    # 当前周期内已过时长（0~5s）

        # ---------------------- 关键：夹爪1 瞬间切换逻辑 ----------------------
        if cycle_idx != self.last_cycle_idx:
            # 进入新周期 → 瞬间切换夹爪1状态
            self.current_gripper1 = 0.0 if self.current_gripper1 == 1.0 else 1.0
            self.last_cycle_idx = cycle_idx  # 更新上一周期序号
            # self.get_logger().warn(
            #     f"=== 周期切换 === 第{cycle_idx+1}个周期开始 | 夹爪1 瞬间切换为：{self.current_gripper1:.0f}"
            # )

        # ---------------------- 夹爪2 同步变化逻辑 ----------------------
        if cycle_idx % 2 == 0:
            # 偶数周期（0-5s、10-15s...）：夹爪2 1→0.4
            if cycle_elapsed <= self.ramp_duration:
                ratio = cycle_elapsed / self.ramp_duration
                target_gripper2 = 1.0 - ratio * 0.6  # 前0.5s线性降0.4
            else:
                target_gripper2 = 0.4  # 后4.5s保持0.4
        else:
            # 奇数周期（5-10s、15-20s...）：夹爪2 0.4→1
            if cycle_elapsed <= self.ramp_duration:
                ratio = cycle_elapsed / self.ramp_duration
                target_gripper2 = 0.4 + ratio * 0.6  # 前0.5s线性升1
            else:
                target_gripper2 = 1.0  # 后4.5s保持1

        # ---------------------- 生成并发布数据 ----------------------
        # 右臂数据
        right_data, right_joints, right_g1, right_g2, right_pos, right_rpy = self.generate_arm_data(
            self.right_prev_joints, self.right_joint_ranges,
            self.right_prev_gripper1, self.right_prev_gripper2,
            self.right_prev_position, self.right_position_ranges,
            self.right_prev_rpy, self.right_rpy_ranges,
            self.current_gripper1, target_gripper2  # 传入夹爪目标值
        )

        # 左臂数据（和右臂完全同步）
        left_data, left_joints, left_g1, left_g2, left_pos, left_rpy = self.generate_arm_data(
            self.left_prev_joints, self.left_joint_ranges,
            self.left_prev_gripper1, self.left_prev_gripper2,
            self.left_prev_position, self.left_position_ranges,
            self.left_prev_rpy, self.left_rpy_ranges,
            self.current_gripper1, target_gripper2  # 传入夹爪目标值
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

        # 发布右臂消息
        right_msg = ArmData()
        right_msg.header.stamp = self.get_clock().now().to_msg()
        right_msg.data = right_data
        self.right_publisher_.publish(right_msg)

        # 发布左臂消息
        left_msg = ArmData()
        left_msg.header.stamp = self.get_clock().now().to_msg()
        left_msg.data = left_data
        self.left_publisher_.publish(left_msg)

        # 调试输出（每0.5s打印一次夹爪状态）
        # if int(cycle_elapsed * 10) % 5 == 0:  # 0.5s打印一次
        #     self.get_logger().debug(
        #         f"周期{cycle_idx} | 已过{cycle_elapsed:.1f}s | 夹爪1={self.current_gripper1:.0f} | 夹爪2={target_gripper2:.2f}"
        #     )


def main(args=None):
    rclpy.init(args=args)
    node = MockArmsPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
