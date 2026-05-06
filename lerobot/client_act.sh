source ~/sc_ros/install/setup.bash

python -m lerobot.async_inference.robot_client \
  --robot.type=xman \
  --robot.state_dim=16 \
  --robot.action_dim=16 \
  --robot.head_topic=/camera3/camera_head/color/image_raw \
  --robot.left_topic=/camera1/camera_left/color/image_rect_raw \
  --robot.right_topic=/camera2/camera_right/color/image_rect_raw \
  --robot.arm_topic=/dual_data \
  --robot.arm_service=/MulSetJointAngles \
  --server_address=172.16.10.14:8090 \
  --policy_type=act \
  --pretrained_name_or_path=/mnt/datas/finetune_model/act_coffee_motion_downsample_0323/checkpoints/last/pretrained_model \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=10 \
  --fps=10 \
  --task="Make coffee"