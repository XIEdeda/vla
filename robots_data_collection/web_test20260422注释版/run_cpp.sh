#!/bin/bash
# =============================================
# 双臂机器人数据采集系统 - C++版启动脚本
# 启动3个相机 + C++ WebSocket服务器节点(内含数据处理器)
# =============================================

set -e

# 检查ROS2环境
if [ -z "$ROS_DISTRO" ]; then
    echo "[错误] 未检测到ROS2环境,请先source setup.bash"
    echo "  例: source /opt/ros/humble/setup.bash"
    echo "       source /mnt/web_test/install/setup.bash"
    exit 1
fi

echo "=========================================="
echo " 双臂数据采集系统 (C++ 版)"
echo " ROS2版本: $ROS_DISTRO"
echo "=========================================="

# 3cameras.sh 在 run_cpp.sh 所在目录的上级目录
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export CAMERA_SN_SCRIPT="${SCRIPT_DIR}/3cameras.sh"

# 启动3个相机
echo "===== Starting 3 cameras ====="
"${CAMERA_SN_SCRIPT}" &
CAMERA_PID=$!

echo "===== Waiting 2s for cameras initialization ====="
sleep 2

# 按北京日期创建日志目录，将 WebSocket 节点输出记录到文件
LOG_BASE="/mnt/keenon_baidu_log"
LOG_DATE=$(TZ=Asia/Shanghai date +%Y-%m-%d)
LOG_DIR="${LOG_BASE}/${LOG_DATE}"
mkdir -p "$LOG_DIR"
chmod 755 "$LOG_DIR"
echo "===== 日志将保存到: ${LOG_DIR}/websocket_server.log (北京时间) ====="

# 是否保存深度图( true / false ),设为false可节省磁盘和IO
SAVE_DEPTH="${SAVE_DEPTH:-true}"
# 是否使用进程内 rosbag2 mcap 录制( true / false )
# 设为true时跳过图片/TXT落盘，直接将原始 topic 写入 mcap bag（零 DDS 延迟，不丢帧）
USE_MCAP="${USE_MCAP:-false}"
echo "===== Starting websocket_server_node (C++) [save_depth=$SAVE_DEPTH] [use_mcap=$USE_MCAP] ====="
# 每行日志前加北京时间戳；若未安装 ts 则无时间戳（可运行: sudo apt install moreutils）
if command -v ts &>/dev/null; then
  ( ROS_AFFINITY_CPU=1 ros2 run my_dual_arm_package websocket_server_node --ros-args -p save_depth:=$SAVE_DEPTH -p use_mcap:=$USE_MCAP 2>&1 | TZ=Asia/Shanghai ts '[%Y-%m-%d %H:%M:%S]' | tee -a "${LOG_DIR}/websocket_server.log" ) &
else
  echo "[提示] 未安装 ts，日志无时间戳。安装: sudo apt install moreutils"
  ( ROS_AFFINITY_CPU=1 ros2 run my_dual_arm_package websocket_server_node --ros-args -p save_depth:=$SAVE_DEPTH -p use_mcap:=$USE_MCAP 2>&1 | tee -a "${LOG_DIR}/websocket_server.log" ) &
fi
WS_PID=$!

echo "===== All components started ====="
echo "Camera PID: $CAMERA_PID"
echo "WebSocket PID: $WS_PID"
echo "Press Ctrl+C to stop all processes"

# 递归杀死进程树的函数
kill_tree() {
    local PID=$1
    for child in $(pgrep -P $PID 2>/dev/null); do
        kill_tree $child
    done
    if [ -n "$PID" ] && kill -0 $PID 2>/dev/null; then
        kill -9 $PID 2>/dev/null
    fi
}

# 捕获Ctrl+C/终止信号,优雅关闭
cleanup() {
    set +e  # 关闭exit-on-error,防止wait非零返回导致cleanup中途退出
    echo ""
    echo "===== Stopping all processes ====="

    if [ -n "$WS_PID" ] && kill -0 $WS_PID 2>/dev/null; then
        kill_tree $WS_PID
        wait $WS_PID 2>/dev/null
        echo "WebSocket node (PID: $WS_PID) and children stopped"
    fi

    if [ -n "$CAMERA_PID" ] && kill -0 $CAMERA_PID 2>/dev/null; then
        kill_tree $CAMERA_PID
        wait $CAMERA_PID 2>/dev/null
        echo "Camera script (PID: $CAMERA_PID) and children stopped"
    fi

    echo "===== All processes stopped ====="
    exit 0
}
trap cleanup SIGINT SIGTERM

# 等待所有后台进程
wait

