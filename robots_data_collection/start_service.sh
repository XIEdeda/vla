#!/bin/bash

set -e

HTTP_DIR="/home/peanut/mlt/dist"
CPP_DIR="/home/peanut/mlt/web_test"

HTTP_CMD="http-server -p 8080"
CPP_CMD="bash run_cpp.sh"

cleanup() {
    echo ""
    echo "================================"
    echo "收到退出信号，正在关闭所有服务..."
    echo "================================"

    kill $HTTP_PID 2>/dev/null || true
    kill $CPP_PID 2>/dev/null || true

    wait

    echo "所有服务已停止"
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "================================"
echo "启动系统服务"
echo "================================"

# ---------------- HTTP SERVER ----------------
echo ""
echo "[1/2] 检查端口 8080..."

if lsof -i:8080 > /dev/null 2>&1; then
    echo "端口 8080 已被占用，正在关闭旧进程..."
    lsof -ti:8080 | xargs kill -9
    sleep 1
fi

echo "启动 http-server..."

cd "$HTTP_DIR"

$HTTP_CMD &
HTTP_PID=$!

echo "HTTP PID: $HTTP_PID"

sleep 3

# ---------------- CPP BACKEND ----------------
echo ""
echo "[2/2] 启动 run_cpp.sh..."

cd "$CPP_DIR"

$CPP_CMD &
CPP_PID=$!

echo "CPP PID: $CPP_PID"

echo ""
echo "================================"
echo "全部服务启动完成"
echo "================================"
echo ""
echo "HTTP : $HTTP_PID"
echo "CPP  : $CPP_PID"
echo ""
echo "按 Ctrl+C 可关闭全部服务"
echo ""

wait
