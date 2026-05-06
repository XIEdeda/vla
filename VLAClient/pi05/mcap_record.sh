source /opt/ros/humble/setup.bash
ros2 bag record -s mcap -o bag_dual_arm_$(date +%Y%m%d_%H%M%S) /set_dual_arm /get_dual_arm