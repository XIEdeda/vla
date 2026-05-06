import numpy as np
import logging
import time
from PIL import Image
import io
from openpi_client import websocket_client_policy


def get_fake_obs():
    # 1. 正常伪造
    h, w = 224, 224
    img = np.random.randint(0, 256, (3, h, w), dtype=np.uint8)
    state = np.random.randn(16).astype(np.float32)

    obs = {
        "observation/head_image": np.array(img, copy=True),
        "observation/left_wrist_image": np.array(img, copy=True),
        "observation/right_wrist_image": np.array(img, copy=True),
        "observation/state": np.array(state, copy=True),
        "prompt": "pour wine",
    }
    return obs


def test_expert_client():
    logging.basicConfig(level=logging.INFO)

    try:
        policy = websocket_client_policy.WebsocketClientPolicy(
            host="172.16.10.14", port=5556
        )
        logging.info("成功连接")

        # observation = {
        #     "state": np.zeros(16).astype(np.float32),
        #     # "image": np.zeros((224, 224, 3)).astype(np.uint8)
        # }
        observation = get_fake_obs()

        print("\n正在请求专家数据...")
        # WebsocketClientPolicy.infer 会阻塞直到收到服务器返回的 action
        for i in range(20):
            tic = time.time()
            result = policy.infer(observation)

            print(i, time.time() - tic)

    except Exception as e:
        logging.error(f"运行失败: {e}")


if __name__ == "__main__":
    test_expert_client()
