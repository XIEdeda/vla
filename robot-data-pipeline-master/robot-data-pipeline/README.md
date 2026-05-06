# convert_to_lerobotv3.py

```bash
conda activate lerobotv3

python3 convert_to_lerobotv3.py \
  --repo-id coffee_motion_downsample_0403 \
  --input-root /mnt/datas/vla_datasets/coffee_src_data/0317_cyw_coffee_01 /mnt/datas/vla_datasets/coffee_src_data/0318_cyw_coffee_01 /mnt/datas/vla_datasets/coffee_src_data/0318_cyw_coffee_02 /mnt/datas/vla_datasets/coffee_src_data/0319_cyw_coffee_01 /mnt/datas/vla_datasets/coffee_src_data/0320_cyw_coffee_01 /mnt/datas/vla_datasets/coffee_src_data/0326_cyw_coffee_02 /mnt/datas/vla_datasets/coffee_src_data/0331_cyw_coffee_01 /mnt/datas/vla_datasets/coffee_src_data/0401_cyw_coffee_01 /mnt/datas/vla_datasets/coffee_src_data/0403_cyw_coffee_01 \
  --dst-dir /mnt/datas/vla_datasets/coffee_src_data/ \
  --task "Make coffee" \
  --storage-mode video \
  --source-fps 30 \
  --fps 10 \
  --downsample-mode adaptive_static \
  --static-fps 2 \
  --static-motion-score-threshold 0.03 \
  --static-transition-guard-frames 30 \
  --static-debug-plot-first-episode \
  --image-writer-processes 2 \
  --image-writer-threads 4 \
  --streaming-encoding \
  --encoder-queue-maxsize 30 \
  --encoder-threads 4

# 增量式追加：先复制历史 LeRobot 数据集到新的 repo 目录，再把新原始数据追加进去
python3 convert_to_lerobotv3.py \
  --repo-id coffee_motion_downsample_0407 \
  --input-root /mnt/datas/vla_datasets/coffee_src_data/0407_cyw_coffee_01 \
  --dst-dir /mnt/datas/vla_datasets/coffee_src_data/ \
  --incremental-from-dataset /mnt/datas/vla_datasets/coffee_src_data/coffee_motion_downsample_0403 \
  --task "Make coffee" \
  --storage-mode video \
  --source-fps 30 \
  --fps 10 \
  --downsample-mode adaptive_static \
  --static-fps 2 \
  --static-motion-score-threshold 0.03 \
  --static-transition-guard-frames 30 \
  --static-debug-plot-first-episode \
  --image-writer-processes 2 \
  --image-writer-threads 4 \
  --streaming-encoding \
  --encoder-queue-maxsize 30 \
  --encoder-threads 4
```

## 增量模式说明

- 不传 `--incremental-from-dataset` 时，行为和之前一致：从零创建一个新的 LeRobot dataset。
- 传入 `--incremental-from-dataset` 时，会先把该历史数据集完整复制到新的输出目录 `--dst-dir/--repo-id`，然后只把这次 `--input-root` 对应的新原始数据追加到复制出来的新数据集里。
- 这样不会直接修改历史数据集目录；历史数据集始终保持不变。
- 增量追加前会校验关键兼容项：`--fps`、`--robot-type`、相机 shape、相机存储类型是否与历史数据集一致。不一致会直接报错，避免生成混合格式数据集。
- 复制过来的 `summary_max_diffs.csv` 会和这次新追加 episode 的 summary 结果合并，不会被覆盖丢失。

## 参数说明

- `--repo-id`：输出数据集名称
- `--input-root`：输入原始数据根目录，可传多个
- `--dst-dir`：输出目录根路径
- `--incremental-from-dataset`：可选，输入一个已有 LeRobot 数据集目录；脚本会先复制它，再在副本上增量追加
- `--task`：写入每帧的任务文本
- `--source-fps`：原始数据采样帧率
- `--fps`：输出数据目标帧率
- `--robot-type`：写入数据集的机器人类型
- `--storage-mode`：图像保存方式，`image` 或 `video`
- `--action-gripper-mode`：夹爪动作写法，`open_value` 或 `bin`
- `--max-image-diff-ms`：多相机时间对齐允许的最大误差
- `--min-episode-images`：单个 episode 至少需要的图像数
- `--skip-head-frames`：每个 episode 开头跳过的帧数
- `--skip-tail-frames`：每个 episode 结尾跳过的帧数
- `--image-decode-cache-size`：图像解码缓存大小
- `--overwrite`：若输出目录已存在则先删除重建
- `--downsample-mode`：降采样方式，`timestamp`、`stride` 或 `adaptive_static`
- `--static-fps`：静止片段使用的降采样帧率
- `--static-min-duration-sec`：判定为静止片段的最短持续时间
- `--static-motion-score-threshold`：静止判定的运动分数阈值
- `--static-motion-scale-quantile`：运动归一化时使用的分位数
- `--static-motion-scale-floor`：运动归一化的最小尺度
- `--static-transition-guard-frames`：静止转运动附近保留全帧率的保护帧数
- `--static-debug-plot-first-episode`：为每个 source 的首个保留 episode 生成静止判定调试图
- `--debug`：输出详细日志并写调试 CSV
- `--image-writer-processes`：异步写图进程数
- `--image-writer-threads`：异步写图线程数
- `--streaming-encoding`：在写入过程中实时编码视频
- `--encoder-queue-maxsize`：实时编码队列最大长度
- `--encoder-threads`：视频编码线程数
- `--max-episodes-per-source`：每个输入源最多处理的 episode 数
- `--episode-sample-mode`：限制 episode 数时的采样方式，`first` 或 `uniform`
- `--num-workers`：并行准备 episode 的进程数
- `--max-inflight-episodes`：并行模式下同时在途的 episode 数上限
- `--worker-batch-size`：每批交给进程池处理的 episode 数
- `--profile-timing`：打印每个 episode 各阶段耗时
- `--log-level`：`--debug` 模式下的日志级别