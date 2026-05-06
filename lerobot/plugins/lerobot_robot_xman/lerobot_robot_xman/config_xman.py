from dataclasses import dataclass

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("xman")
@dataclass(kw_only=True)
class XmanConfig(RobotConfig):
    """
    Minimal config for Xman custom humanoid plugin.

    Replace/add fields to match your transport stack (ROS2 topic, SDK IP, etc.).
    """

    # ROS topics/services
    head_topic: str = "/camera3/camera_head/color/image_raw"
    left_topic: str = "/camera1/camera_left/color/image_rect_raw"
    right_topic: str = "/camera2/camera_right/color/image_rect_raw"
    arm_topic: str = "/dual_data"
    arm_service: str = "/MulSetJointAngles"

    # Observation/action dimensions
    state_dim: int = 16
    action_dim: int = 16

    # Expected image shapes used by observation_features
    head_image_height: int = 720
    head_image_width: int = 1280
    wrist_image_height: int = 480
    wrist_image_width: int = 640

    command_timeout_s: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be > 0, got {self.state_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {self.action_dim}")
        if self.command_timeout_s <= 0:
            raise ValueError(f"command_timeout_s must be > 0, got {self.command_timeout_s}")
        if self.head_image_height <= 0 or self.head_image_width <= 0:
            raise ValueError("head image shape must be positive")
        if self.wrist_image_height <= 0 or self.wrist_image_width <= 0:
            raise ValueError("wrist image shape must be positive")
