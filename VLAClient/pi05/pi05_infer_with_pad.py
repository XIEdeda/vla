#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pad 入口：复用 pi05_infer.py 的 Pi05InferNode，并增加 websocket 控制。

websocket 入参示例（send_example.json）:
{
  "action_type": "makeCoffee",
  "msg_id": "1111",
  "data": {
    "cmd": "coffee1",
    "prompt": "Make coffee"
  }
}

行为：
- 收到合法消息后等价于按下键盘 's'（self._key_state = "s"）。
- data.prompt 非空时更新推理 prompt。
- 记录 data.cmd（咖啡类型），后续可用于分支策略。
"""

import asyncio
import json
import threading
from typing import Any, Dict, Optional

import rclpy
import websockets

from pi05_infer import Pi05InferNode


class Pi05InferWithPadNode(Pi05InferNode):
    def __init__(self):
        super().__init__()

        self.declare_parameter("pad_ws_host", "0.0.0.0")
        self.declare_parameter("pad_ws_port", 8765)
        self.declare_parameter("pad_ws_action_type", "makeCoffee")

        self._pad_ws_host = str(self.get_parameter("pad_ws_host").value)
        self._pad_ws_port = int(self.get_parameter("pad_ws_port").value)
        self._pad_action_type = str(self.get_parameter("pad_ws_action_type").value)

        self._control_state_lock = threading.Lock()
        self._coffee_type: str = ""
        self._last_pad_msg_id: str = ""

        self._pad_loop: Optional[asyncio.AbstractEventLoop] = None
        self._pad_server = None
        self._pad_thread = threading.Thread(target=self._run_pad_ws_server, daemon=True)
        self._pad_thread.start()
        self.get_logger().info(
            f"Pad websocket listening on ws://{self._pad_ws_host}:{self._pad_ws_port}"
        )

    def _apply_pad_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        action_type = payload.get("action_type", "")
        msg_id = str(payload.get("msg_id", ""))
        data = payload.get("data") or {}
        cmd = str(data.get("cmd", "")).strip()
        prompt = str(data.get("prompt", "")).strip()

        resp = {
            "action_type": action_type,
            "data": {
                "cmd": cmd,
                "prompt": prompt,
            },
            "code": 0,
            "msg": "",
            "msg_id": msg_id,
        }

        if action_type != self._pad_action_type:
            resp["code"] = 1
            resp["msg"] = (
                f"unsupported action_type: {action_type}, expected {self._pad_action_type}"
            )
            return resp

        with self._control_state_lock:
            self._key_state = "s"
            self._last_pad_msg_id = msg_id
            if cmd:
                self._coffee_type = cmd
            if prompt:
                if prompt != self._prompt:
                    self.get_logger().info(f"Prompt updated by pad websocket: {self._prompt}")
                    self._prompt = prompt

        self.get_logger().info(
            f"Pad start signal accepted: msg_id={msg_id}, cmd={self._coffee_type or '-'}"
        )
        return resp

    async def _pad_ws_handler(self, websocket):
        async for raw in websocket:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                err = {
                    "action_type": "",
                    "data": {},
                    "code": 1,
                    "msg": "invalid json",
                    "msg_id": "",
                }
                await websocket.send(json.dumps(err, ensure_ascii=False))
                continue

            try:
                resp = self._apply_pad_command(payload)
            except Exception as exc:  # 防止控制线程因异常中断
                resp = {
                    "action_type": str(payload.get("action_type", "")),
                    "data": payload.get("data") or {},
                    "code": 1,
                    "msg": f"internal error: {exc}",
                    "msg_id": str(payload.get("msg_id", "")),
                }
                self.get_logger().error(f"Pad command handling failed: {exc}")

            await websocket.send(json.dumps(resp, ensure_ascii=False))

    async def _pad_ws_main(self):
        self._pad_server = await websockets.serve(
            self._pad_ws_handler, self._pad_ws_host, self._pad_ws_port
        )
        await self._pad_server.wait_closed()

    def _run_pad_ws_server(self):
        self._pad_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._pad_loop)
        try:
            self._pad_loop.run_until_complete(self._pad_ws_main())
        except Exception as exc:
            self.get_logger().error(f"Pad websocket server stopped unexpectedly: {exc}")
        finally:
            self._pad_loop.close()

    def destroy_node(self):
        if self._pad_loop is not None:
            if self._pad_server is not None:
                self._pad_loop.call_soon_threadsafe(self._pad_server.close)
            self._pad_loop.call_soon_threadsafe(self._pad_loop.stop)
        if self._pad_thread.is_alive():
            self._pad_thread.join(timeout=1.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = Pi05InferWithPadNode()
    try:
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
