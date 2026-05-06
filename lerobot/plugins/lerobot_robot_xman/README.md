# Xman Robot Plugin Template

This folder is a starter plugin package for connecting Xman to LeRobot.

## 1) Install plugin in editable mode

From repository root:

```bash
pip install -e plugins/lerobot_robot_xman
```

## 2) Verify config type is discoverable

Any LeRobot script that accepts `--robot.*` can now use:

```bash
--robot.type=xman
```

## 3) Minimal usage example

```bash
python -m lerobot.async_inference.robot_client \
  --robot.type=xman \
  --robot.endpoint=192.168.1.10:9000 \
  --robot.state_dim=16 \
  --robot.action_dim=16 \
  --server_address=127.0.0.1:8080 \
  --policy_type=act \
  --pretrained_name_or_path=/mnt/datas/finetune_model/act_coffee_pre_static_MA_0324/checkpoints/last/pretrained_model \
  --policy_device=cuda \
  --actions_per_chunk=10 \
  --fps=10
```

## Files you should modify

- `lerobot_robot_xman/xman.py`
  - `_connect_transport()`
  - `_disconnect_transport()`
  - `_read_state()`
  - `_write_action()`

- `lerobot_robot_xman/config_xman.py`
  - Add your own transport and controller parameters.

## Notes

- Align observation and action key order with your training dataset feature order.
- If your robot has cameras, pass them in `--robot.cameras` using LeRobot camera configs.
- Keep the plugin package name starting with `lerobot_robot_` for auto-discovery compatibility.
