#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple websocket client for testing pi05_infer_with_pad.py."""

import argparse
import asyncio
import json
import time
import uuid

import websockets


def _build_payload(cmd: str, prompt: str, action_type: str, msg_id: str):
    return {
        "action_type": action_type,
        "msg_id": msg_id,
        "data": {
            "cmd": cmd,
            "prompt": prompt,
        },
    }


async def _send_once(uri: str, payload: dict):
    async with websockets.connect(uri) as ws:
        req = json.dumps(payload, ensure_ascii=False)
        print(f"[send] {req}")
        await ws.send(req)
        resp = await ws.recv()
        print(f"[recv] {resp}")


def main():
    parser = argparse.ArgumentParser(description="Send test websocket command to pad node.")
    parser.add_argument("--host", default="127.0.0.1", help="pad websocket host")
    parser.add_argument("--port", type=int, default=8765, help="pad websocket port")
    parser.add_argument("--action-type", default="makeCoffee", help="action_type")
    parser.add_argument("--cmd", default="coffee1", help="coffee cmd, e.g. coffee1")
    parser.add_argument("--prompt", default="Make coffee", help="language prompt")
    parser.add_argument(
        "--msg-id",
        default="",
        help="optional msg_id; empty means auto-generate",
    )
    args = parser.parse_args()

    msg_id = args.msg_id.strip() or f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
    payload = _build_payload(
        cmd=args.cmd,
        prompt=args.prompt,
        action_type=args.action_type,
        msg_id=msg_id,
    )
    uri = f"ws://{args.host}:{args.port}"
    asyncio.run(_send_once(uri, payload))


if __name__ == "__main__":
    main()
