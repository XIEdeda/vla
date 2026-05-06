#!/usr/bin/env python3

# 使用示例
# conda activate lerobotv3
# python convert_to_lerobotv3.py --repo-id baomihua_lerobotv3_test --input-root /mnt/datas/vla_datasets/fold_clothes/baomihua_0306_5_1 --dst-dir /mnt/datas/vla_datasets/popcorn/ --task "Prepare popcorn" --storage-mode video


import argparse
import bisect
import concurrent.futures
import contextlib
import dataclasses
import functools
import glob
import inspect
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


MOTORS = [
	"left_joint0",
	"left_joint1",
	"left_joint2",
	"left_joint3",
	"left_joint4",
	"left_joint5",
	"left_joint6",
	"left_gripper_pose",
	"right_joint0",
	"right_joint1",
	"right_joint2",
	"right_joint3",
	"right_joint4",
	"right_joint5",
	"right_joint6",
	"right_gripper_pose",
]

CAMERA_KEYS = [
	"observation.images.head_image",
	"observation.images.left_wrist_image",
	"observation.images.right_wrist_image",
]

CAMERA_IMAGE_DIRS = {
	"observation.images.head_image": "head_image_images",
	"observation.images.left_wrist_image": "left_wrist_image_images",
	"observation.images.right_wrist_image": "right_wrist_image_images",
}

# Joint-only indices in observation.state (exclude gripper pose at 7 and 15).
STATE_JOINT_INDICES = list(range(0, 7)) + list(range(8, 15))
LEFT_GRIPPER_STATE_INDEX = 7


@dataclasses.dataclass(frozen=True)
class ConverterConfig:
	fps: int = 30
	source_fps: float = 30.0
	downsample_mode: Literal["timestamp", "stride", "adaptive_static"] = "timestamp"
	robot_type: str = "Keenon_arm"
	storage_mode: Literal["image", "video"] = "video"
	action_gripper_mode: Literal["open_value", "bin"] = "open_value"
	max_image_diff_ms: float = 40.0
	min_episode_images: int = 100
	skip_head_frames: int = 30
	skip_tail_frames: int = 5
	image_decode_cache_size: int = 256
	overwrite: bool = False
	debug: bool = False
	num_workers: int = 1
	max_inflight_episodes: int = 4
	worker_batch_size: int = 100
	profile_timing: bool = False
	image_writer_processes: int = 0
	image_writer_threads: int = 0
	streaming_encoding: bool = False
	encoder_queue_maxsize: int = 30
	encoder_threads: int | None = None
	max_episodes_per_source: int | None = None
	episode_sample_mode: Literal["first", "uniform"] = "first"
	static_fps: float = 3.0
	static_min_duration_sec: float = 2.5
	static_motion_score_threshold: float = 0.03
	static_motion_scale_quantile: float = 0.95
	static_motion_scale_floor: float = 1e-4
	static_transition_guard_frames: int = 2
	static_debug_plot_first_episode: bool = False
	enable_contact_rich_upsample: bool = False
	contact_rich_fps: float | None = None
	contact_rich_close_threshold: float = 12.0
	contact_rich_min_duration_sec: float = 0.5
	contact_rich_max_ranges: int = 2
	contact_rich_require_before_longest_static: bool = True
	contact_rich_debug_all_rejections: bool = True


def build_features(storage_mode: Literal["image", "video"], camera_shapes: dict[str, tuple[int, int]]) -> dict:
	features = {
		"observation.state": {
			"dtype": "float32",
			"shape": (len(MOTORS),),
			"names": [MOTORS],
		},
		"action": {
			"dtype": "float32",
			"shape": (len(MOTORS),),
			"names": [MOTORS],
		},
	}

	for cam_key in CAMERA_KEYS:
		height, width = camera_shapes[cam_key]
		features[cam_key] = {
			"dtype": storage_mode,
			"shape": (3, height, width),
			"names": ["channels", "height", "width"],
		}

	return features


def infer_camera_shapes(input_roots: list[Path]) -> dict[str, tuple[int, int]]:
	"""Infer (height, width) for each camera key from the first readable image in input roots."""
	shapes: dict[str, tuple[int, int]] = {}

	for root in input_roots:
		temp_images_dir = root / "temp_images"
		if not temp_images_dir.exists():
			continue

		episodes = sorted([d for d in temp_images_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
		for ep_dir in episodes:
			for cam_key in CAMERA_KEYS:
				if cam_key in shapes:
					continue
				cam_dir = ep_dir / CAMERA_IMAGE_DIRS[cam_key]
				if not cam_dir.exists():
					continue

				image_paths = sorted(glob.glob(str(cam_dir / "*.jpg")))
				if not image_paths:
					continue

				img = cv2.imread(image_paths[0])
				if img is None:
					continue

				height, width = img.shape[:2]
				shapes[cam_key] = (height, width)

			if len(shapes) == len(CAMERA_KEYS):
				return shapes

	missing = [k for k in CAMERA_KEYS if k not in shapes]
	raise FileNotFoundError(f"Failed to infer camera shapes for keys: {missing}")


def ns_to_sec_nsec_str(ts_ns: int) -> str:
	sec = ts_ns // 1_000_000_000
	nsec = ts_ns % 1_000_000_000
	return f"{sec}.{nsec:09d}"


def parse_image_filename(path: str) -> tuple[int, int]:
	"""Parse '<frame_id>_<timestamp_ns>.jpg'."""
	base = os.path.basename(path)
	stem = base.rsplit(".", 1)[0]
	frame_id_str, ts_ns_str = stem.split("_", 1)
	return int(frame_id_str), int(ts_ns_str)


def index_images(folder: Path) -> list[dict]:
	rows = []
	for path in glob.glob(str(folder / "*.jpg")):
		frame_id, ts_ns = parse_image_filename(path)
		rows.append(
			{
				"frame_id": frame_id,
				"ts_ns": ts_ns,
				"ts_str": ns_to_sec_nsec_str(ts_ns),
				"path": path,
			}
		)

	rows.sort(key=lambda x: x["ts_ns"])
	return rows


def build_image_loader(cache_size: int):
	@functools.lru_cache(maxsize=cache_size)
	def load_image(path: str) -> np.ndarray:
		img = cv2.imread(path)
		if img is None:
			raise FileNotFoundError(f"Failed to read image: {path}")
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	return load_image


def find_nearest_image(
	img_list: list[dict], ts_list: list[int], target_ts_ns: int, max_diff_ns: int | None
) -> dict | None:
	if not img_list:
		return None

	idx = bisect.bisect_left(ts_list, target_ts_ns)

	candidates = []
	if idx < len(img_list):
		candidates.append(img_list[idx])
	if idx > 0:
		candidates.append(img_list[idx - 1])

	if not candidates:
		return None

	best = min(candidates, key=lambda x: abs(x["ts_ns"] - target_ts_ns))
	if max_diff_ns is not None and abs(best["ts_ns"] - target_ts_ns) > max_diff_ns:
		return None
	return best


def load_arm_txt(path: Path) -> list[dict]:
	rows = []
	if not path.exists():
		return rows

	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue

			ts_ns_str, payload = line.split(" ", 1)
			ts_ns = int(ts_ns_str)
			values = [float(x) for x in payload.split(",")]
			rows.append(
				{
					"ts_ns": ts_ns,
					"ts_str": ns_to_sec_nsec_str(ts_ns),
					"data": values,
				}
			)

	rows.sort(key=lambda x: x["ts_ns"])
	return rows


def find_next_arm_timestamp(arm_list: list[dict], ts_list: list[int], target_ts_ns: int) -> dict | None:
	if not arm_list:
		return None

	idx = bisect.bisect_right(ts_list, target_ts_ns)
	if idx >= len(arm_list):
		return None
	return arm_list[idx]


def max_and_count(values: list[float], atol: float = 10.0) -> tuple[float, int]:
	if not values:
		return float("nan"), 0

	arr = np.asarray(values, dtype=np.float64)
	arr = arr[~np.isnan(arr)]
	if arr.size == 0:
		return float("nan"), 0

	max_val = arr.max()
	count = int(np.sum(np.abs(arr - max_val) <= atol))
	return float(max_val), count


def map_gripper_action_value(
	bin_signal: float,
	open_value_signal: float,
	action_gripper_mode: Literal["open_value", "bin"],
) -> float:
	"""Map gripper action from raw arm fields: [7]=bin, [8]=open value (0-70)."""
	if action_gripper_mode == "open_value":
		return float(open_value_signal) / 100.0
	if action_gripper_mode == "bin":
		return float(bin_signal)
	raise ValueError(f"Unsupported action gripper mode: {action_gripper_mode}")


def build_observation_state(left_arm_data: list[float], right_arm_data: list[float]) -> np.ndarray:
	"""Build observation.state from joints and normalized gripper pose."""
	return np.array(
		left_arm_data[:7]
		+ [left_arm_data[8] / 100.0]
		+ right_arm_data[:7]
		+ [right_arm_data[8] / 100.0],
		dtype=np.float32,
	)


def build_action_source(
	left_arm_data: list[float],
	right_arm_data: list[float],
	action_gripper_mode: Literal["open_value", "bin"],
) -> np.ndarray:
	"""Build action source from commanded joints + gripper command."""
	left_gripper = map_gripper_action_value(
		bin_signal=left_arm_data[7],
		open_value_signal=left_arm_data[8],
		action_gripper_mode=action_gripper_mode,
	)
	right_gripper = map_gripper_action_value(
		bin_signal=right_arm_data[7],
		open_value_signal=right_arm_data[8],
		action_gripper_mode=action_gripper_mode,
	)
	return np.array(
		left_arm_data[:7] + [left_gripper] + right_arm_data[:7] + [right_gripper],
		dtype=np.float32,
	)


def build_dual_arm_vectors(
	left_arm_data: list[float],
	right_arm_data: list[float],
	action_gripper_mode: Literal["open_value", "bin"],
) -> tuple[np.ndarray, np.ndarray]:
	if len(left_arm_data) < 9 or len(right_arm_data) < 9:
		raise ValueError("Arm data length is insufficient, expected at least 9 values per arm.")

	observation_state = build_observation_state(left_arm_data, right_arm_data)
	action = build_action_source(left_arm_data, right_arm_data, action_gripper_mode)

	return observation_state, action


def select_step_action(aligned: list[dict], index: int) -> np.ndarray:
	"""Action target uses next frame's action_source; last frame falls back to current."""
	next_index = min(index + 1, len(aligned) - 1)
	return aligned[next_index]["action_source"]


def align_episode(
	root_path: Path,
	episode_name: str,
	cfg: ConverterConfig,
) -> tuple[list[dict], dict]:
	img_root = root_path / "temp_images" / episode_name
	arm_root = root_path / "arm_data" / episode_name

	head_imgs = index_images(img_root / "head_image_images")
	left_imgs = index_images(img_root / "left_wrist_image_images")
	right_imgs = index_images(img_root / "right_wrist_image_images")
	left_img_ts = [x["ts_ns"] for x in left_imgs]
	right_img_ts = [x["ts_ns"] for x in right_imgs]

	if min(len(head_imgs), len(left_imgs), len(right_imgs)) < cfg.min_episode_images:
		return [], {"skip_reason": "not_enough_images"}

	left_arm = load_arm_txt(arm_root / "left_data.txt")
	right_arm = load_arm_txt(arm_root / "right_data.txt")
	left_arm_ts = [x["ts_ns"] for x in left_arm]
	right_arm_ts = [x["ts_ns"] for x in right_arm]
	if not left_arm or not right_arm:
		return [], {"skip_reason": "missing_arm_data"}

	max_diff_ns = int(cfg.max_image_diff_ms * 1e6)

	if len(head_imgs) <= cfg.skip_head_frames + cfg.skip_tail_frames:
		return [], {"skip_reason": "head_sequence_too_short"}

	candidates = head_imgs[cfg.skip_head_frames : len(head_imgs) - cfg.skip_tail_frames]
	aligned = []
	for h in candidates:
		head_ts = h["ts_ns"]
		l = find_nearest_image(left_imgs, left_img_ts, head_ts, max_diff_ns)
		r = find_nearest_image(right_imgs, right_img_ts, head_ts, max_diff_ns)
		if l is None or r is None:
			continue

		ts_max_img = max(h["ts_ns"], l["ts_ns"], r["ts_ns"])
		left_match = find_next_arm_timestamp(left_arm, left_arm_ts, ts_max_img)
		right_match = find_next_arm_timestamp(right_arm, right_arm_ts, ts_max_img)
		if left_match is None or right_match is None:
			continue

		observation_state, action_source = build_dual_arm_vectors(
			left_match["data"], right_match["data"], cfg.action_gripper_mode
		)

		aligned.append(
			{
				"head_img_path": h["path"],
				"left_img_path": l["path"],
				"right_img_path": r["path"],
				"head_ts": h["ts_str"],
				"left_ts": l["ts_str"],
				"right_ts": r["ts_str"],
				"head_ts_ns": h["ts_ns"],
				"left_ts_ns": l["ts_ns"],
				"right_ts_ns": r["ts_ns"],
				"left_arm_ts": left_match["ts_str"],
				"right_arm_ts": right_match["ts_str"],
				"observation_state": observation_state,
				"action_source": action_source,
			}
		)

	return aligned, {"skip_reason": None}


def write_debug_csv(debug_csv_path: Path, rows: list[dict]) -> None:
	debug_csv_path.parent.mkdir(parents=True, exist_ok=True)
	df = pd.DataFrame(rows)
	df.to_csv(debug_csv_path, index=False)


@contextlib.contextmanager
def suppress_output(enabled: bool):
	if not enabled:
		yield
		return

	sys.stdout.flush()
	sys.stderr.flush()
	stdout_fd = os.dup(1)
	stderr_fd = os.dup(2)

	with open(os.devnull, "w", encoding="utf-8") as devnull:
		try:
			os.dup2(devnull.fileno(), 1)
			os.dup2(devnull.fileno(), 2)
			yield
		finally:
			sys.stdout.flush()
			sys.stderr.flush()
			os.dup2(stdout_fd, 1)
			os.dup2(stderr_fd, 2)
			os.close(stdout_fd)
			os.close(stderr_fd)

def downsample_by_timestamp(aligned: list[dict], target_fps: int) -> tuple[list[dict], np.ndarray]:
	interval_ns = int(round(1e9 / float(target_fps)))
	keep_mask = np.zeros(len(aligned), dtype=bool)
	keep_mask[0] = True
	next_keep_ts = aligned[0]["head_ts_ns"] + interval_ns
	for i, item in enumerate(aligned[1:], start=1):
		ts = item["head_ts_ns"]
		if ts >= next_keep_ts:
			keep_mask[i] = True
			while ts >= next_keep_ts:
				next_keep_ts += interval_ns
	if not keep_mask[-1]:
		keep_mask[-1] = True
	return [item for item, keep in zip(aligned, keep_mask) if keep], keep_mask


def downsample_by_stride(aligned: list[dict], source_fps: float, target_fps: int) -> tuple[list[dict], np.ndarray]:
	stride = max(1, int(round(source_fps / float(target_fps))))
	keep_mask = np.zeros(len(aligned), dtype=bool)
	keep_mask[::stride] = True
	if not keep_mask[-1]:
		keep_mask[-1] = True
	return [item for item, keep in zip(aligned, keep_mask) if keep], keep_mask


def _run_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
	ranges = []
	start = None
	for i, v in enumerate(mask):
		if v and start is None:
			start = i
		elif not v and start is not None:
			ranges.append((start, i - 1))
			start = None
	if start is not None:
		ranges.append((start, len(mask) - 1))
	return ranges


def _mask_from_ranges(length: int, ranges: list[tuple[int, int]]) -> np.ndarray:
	mask = np.zeros(length, dtype=bool)
	for start, end in ranges:
		mask[start : end + 1] = True
	return mask


def _median_frame_interval_ns(ts: np.ndarray) -> int:
	if ts.size <= 1:
		return 1
	return max(int(np.median(np.diff(ts))), 1)


def _min_points_for_duration(ts: np.ndarray, min_duration_sec: float) -> int:
	return max(2, int(round(min_duration_sec * 1e9 / _median_frame_interval_ns(ts))))


def _range_duration_ns(ts: np.ndarray, start: int, end: int) -> int:
	if end < start:
		return 0
	if start == end:
		return _median_frame_interval_ns(ts)
	return int(ts[end] - ts[start] + _median_frame_interval_ns(ts))


def _select_longest_ranges(
	ranges: list[tuple[int, int]], ts: np.ndarray, max_ranges: int
) -> list[tuple[int, int]]:
	if max_ranges <= 0 or not ranges:
		return []
	selected = sorted(
		ranges,
		key=lambda item: (_range_duration_ns(ts, item[0], item[1]), item[0]),
		reverse=True,
	)[:max_ranges]
	return sorted(selected, key=lambda item: item[0])


def _longest_range(mask: np.ndarray, ts: np.ndarray) -> tuple[int, int] | None:
	ranges = _run_ranges(mask)
	if not ranges:
		return None
	return max(ranges, key=lambda item: (_range_duration_ns(ts, item[0], item[1]), -item[0]))


def classify_static_frames(
	aligned: list[dict],
	min_duration_sec: float,
	motion_score_threshold: float,
	motion_scale_quantile: float,
	motion_scale_floor: float,
	transition_guard_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	n = len(aligned)
	if n == 0:
		return np.array([]), np.array([]), np.array([])

	states = np.stack([item["observation_state"] for item in aligned], axis=0)
	ts = np.asarray([item["head_ts_ns"] for item in aligned], dtype=np.int64)
	if n == 1:
		return np.array([0.0], dtype=np.float64), np.zeros(1, dtype=bool), np.zeros(1, dtype=bool)

	joint_states = states[:, STATE_JOINT_INDICES]
	delta = np.abs(np.diff(joint_states, axis=0))
	scale = np.quantile(delta, motion_scale_quantile, axis=0)
	scale = np.maximum(scale, motion_scale_floor)
	normalized_delta = delta / scale
	motion_score = np.zeros(n, dtype=np.float64)
	motion_score[1:] = normalized_delta.max(axis=1)

	base_static = motion_score < motion_score_threshold
	min_points = _min_points_for_duration(ts, min_duration_sec)
	long_static = np.zeros(n, dtype=bool)
	for start, end in _run_ranges(base_static):
		if end - start + 1 >= min_points:
			long_static[start : end + 1] = True

	# Keep higher fps near static->motion boundaries by shrinking only the static tail.
	guarded_static = np.zeros(n, dtype=bool)
	for start, end in _run_ranges(long_static):
		s = start
		e = end - transition_guard_frames
		if s <= e:
			guarded_static[s : e + 1] = True

	return motion_score, long_static, guarded_static


def classify_contact_rich_frames(
	aligned: list[dict],
	close_threshold: float,
	min_duration_sec: float,
	max_ranges: int,
	require_before_longest_static: bool,
	longest_static_range: tuple[int, int] | None,
	debug_all_rejections: bool,
) -> dict:
	n = len(aligned)
	if n == 0:
		return {
			"left_gripper_pose": np.array([], dtype=np.float64),
			"closed_mask": np.array([], dtype=bool),
			"long_closed_mask": np.array([], dtype=bool),
			"candidate_mask": np.array([], dtype=bool),
			"selected_mask": np.array([], dtype=bool),
			"rejected_after_static_mask": np.array([], dtype=bool),
			"rejected_by_rank_mask": np.array([], dtype=bool),
			"selected_ranges": [],
			"long_closed_ranges": [],
			"reasons": [],
			"should_save_debug_plot": False,
			"close_threshold_norm": close_threshold / 100.0,
		}

	ts = np.asarray([item["head_ts_ns"] for item in aligned], dtype=np.int64)
	left_gripper_pose = np.asarray(
		[float(item["observation_state"][LEFT_GRIPPER_STATE_INDEX]) for item in aligned], dtype=np.float64
	)
	close_threshold_norm = close_threshold / 100.0
	closed_mask = left_gripper_pose <= close_threshold_norm
	min_points = _min_points_for_duration(ts, min_duration_sec)
	long_closed_ranges = [
		(start, end) for start, end in _run_ranges(closed_mask) if (end - start + 1) >= min_points
	]
	candidate_ranges = list(long_closed_ranges)
	rejected_after_static_ranges: list[tuple[int, int]] = []
	if require_before_longest_static and longest_static_range is not None:
		longest_static_start = longest_static_range[0]
		candidate_ranges = []
		for start, end in long_closed_ranges:
			if end < longest_static_start:
				candidate_ranges.append((start, end))
			else:
				rejected_after_static_ranges.append((start, end))

	selected_ranges = _select_longest_ranges(candidate_ranges, ts, max_ranges)
	selected_range_set = set(selected_ranges)
	rejected_by_rank_ranges = [item for item in candidate_ranges if item not in selected_range_set]

	reasons = []
	if np.any(closed_mask) and not long_closed_ranges:
		reasons.append("closed_runs_shorter_than_min_duration")
	if rejected_after_static_ranges:
		reasons.append("contact_after_longest_static")
	if rejected_by_rank_ranges:
		reasons.append("kept_only_longest_contact_runs")
	if np.any(closed_mask) and not selected_ranges:
		reasons.append("no_contact_run_selected")

	return {
		"left_gripper_pose": left_gripper_pose,
		"closed_mask": closed_mask,
		"long_closed_mask": _mask_from_ranges(n, long_closed_ranges),
		"candidate_mask": _mask_from_ranges(n, candidate_ranges),
		"selected_mask": _mask_from_ranges(n, selected_ranges),
		"rejected_after_static_mask": _mask_from_ranges(n, rejected_after_static_ranges),
		"rejected_by_rank_mask": _mask_from_ranges(n, rejected_by_rank_ranges),
		"selected_ranges": selected_ranges,
		"long_closed_ranges": long_closed_ranges,
		"reasons": reasons,
		"should_save_debug_plot": bool(debug_all_rejections and reasons),
		"close_threshold_norm": close_threshold_norm,
	}


def downsample_adaptive_static(aligned: list[dict], cfg: ConverterConfig) -> tuple[list[dict], dict]:
	if not aligned:
		return aligned, {}

	timestamps_ns = np.asarray([item["head_ts_ns"] for item in aligned], dtype=np.int64)
	motion_score, long_static_mask, static_mask = classify_static_frames(
		aligned=aligned,
		min_duration_sec=cfg.static_min_duration_sec,
		motion_score_threshold=cfg.static_motion_score_threshold,
		motion_scale_quantile=cfg.static_motion_scale_quantile,
		motion_scale_floor=cfg.static_motion_scale_floor,
		transition_guard_frames=cfg.static_transition_guard_frames,
	)
	transition_guard_mask = long_static_mask & (~static_mask)
	longest_static_range = _longest_range(long_static_mask, timestamps_ns)
	longest_static_mask = _mask_from_ranges(
		len(aligned), [] if longest_static_range is None else [longest_static_range]
	)

	contact_meta = {
		"selected_mask": np.zeros(len(aligned), dtype=bool),
		"candidate_mask": np.zeros(len(aligned), dtype=bool),
		"closed_mask": np.zeros(len(aligned), dtype=bool),
		"long_closed_mask": np.zeros(len(aligned), dtype=bool),
		"rejected_after_static_mask": np.zeros(len(aligned), dtype=bool),
		"rejected_by_rank_mask": np.zeros(len(aligned), dtype=bool),
		"left_gripper_pose": np.zeros(len(aligned), dtype=np.float64),
		"selected_ranges": [],
		"long_closed_ranges": [],
		"reasons": [],
		"should_save_debug_plot": False,
		"close_threshold_norm": cfg.contact_rich_close_threshold / 100.0,
	}
	contact_mask = contact_meta["selected_mask"]
	contact_rich_fps = float(cfg.source_fps if cfg.contact_rich_fps is None else cfg.contact_rich_fps)
	if cfg.enable_contact_rich_upsample:
		contact_meta = classify_contact_rich_frames(
			aligned=aligned,
			close_threshold=cfg.contact_rich_close_threshold,
			min_duration_sec=cfg.contact_rich_min_duration_sec,
			max_ranges=cfg.contact_rich_max_ranges,
			require_before_longest_static=cfg.contact_rich_require_before_longest_static,
			longest_static_range=longest_static_range,
			debug_all_rejections=cfg.contact_rich_debug_all_rejections,
		)
		contact_mask = contact_meta["selected_mask"]

	sampling_fps = np.full(len(aligned), float(cfg.fps), dtype=np.float64)
	sampling_fps[static_mask] = float(cfg.static_fps)
	if cfg.enable_contact_rich_upsample:
		sampling_fps[contact_mask] = np.maximum(sampling_fps[contact_mask], contact_rich_fps)

	keep_mask = np.zeros(len(aligned), dtype=bool)
	keep_mask[0] = True
	last_keep_ts = aligned[0]["head_ts_ns"]
	for i in range(1, len(aligned)):
		is_transition = sampling_fps[i] != sampling_fps[i - 1]
		if is_transition:
			keep_mask[i] = True
			last_keep_ts = aligned[i]["head_ts_ns"]
			continue

		fps = float(sampling_fps[i])
		interval_ns = int(round(1e9 / fps))
		if aligned[i]["head_ts_ns"] - last_keep_ts >= interval_ns:
			keep_mask[i] = True
			last_keep_ts = aligned[i]["head_ts_ns"]

	if not keep_mask[-1]:
		keep_mask[-1] = True

	out = [item for item, keep in zip(aligned, keep_mask) if keep]
	meta = {
		"mode": "adaptive_static",
		"motion_score": motion_score,
		"long_static_mask": long_static_mask,
		"static_mask": static_mask,
		"longest_static_mask": longest_static_mask,
		"transition_guard_mask": transition_guard_mask,
		"keep_mask": keep_mask,
		"sampling_fps": sampling_fps,
		"timestamps_ns": timestamps_ns,
		"static_frames_raw": int(np.sum(static_mask)),
		"static_frames_kept": int(np.sum(keep_mask & static_mask)),
		"moving_frames_raw": int(np.sum(~static_mask)),
		"moving_frames_kept": int(np.sum(keep_mask & (~static_mask))),
		"transition_guard_frames_raw": int(np.sum(transition_guard_mask)),
		"transition_guard_frames_kept": int(np.sum(keep_mask & transition_guard_mask)),
		"contact_closed_frames_raw": int(np.sum(contact_meta["closed_mask"])),
		"contact_frames_raw": int(np.sum(contact_mask)),
		"contact_frames_kept": int(np.sum(keep_mask & contact_mask)),
		"contact_candidate_frames_raw": int(np.sum(contact_meta["candidate_mask"])),
		"contact_reasons": contact_meta["reasons"],
		"contact_should_save_debug_plot": contact_meta["should_save_debug_plot"],
		"contact_debug": contact_meta,
	}
	return out, meta


def downsample_aligned_frames(aligned: list[dict], cfg: ConverterConfig) -> tuple[list[dict], dict]:
	"""Downsample aligned frames according to configured mode."""
	if not aligned:
		return aligned, {"mode": cfg.downsample_mode}

	if cfg.fps <= 0:
		raise ValueError(f"target fps must be > 0, got {cfg.fps}")

	if cfg.source_fps <= 0:
		raise ValueError(f"source fps must be > 0, got {cfg.source_fps}")

	if cfg.downsample_mode == "adaptive_static":
		if cfg.static_fps <= 0:
			raise ValueError(f"static fps must be > 0, got {cfg.static_fps}")
		if cfg.static_fps > cfg.fps:
			raise ValueError(f"static fps must be <= target fps, got static={cfg.static_fps}, target={cfg.fps}")
		if cfg.enable_contact_rich_upsample:
			contact_rich_fps = cfg.source_fps if cfg.contact_rich_fps is None else cfg.contact_rich_fps
			if contact_rich_fps < cfg.fps:
				raise ValueError(
					f"contact-rich fps must be >= target fps, got contact={contact_rich_fps}, target={cfg.fps}"
				)
			if contact_rich_fps > cfg.source_fps:
				raise ValueError(
					f"contact-rich fps must be <= source fps, got contact={contact_rich_fps}, source={cfg.source_fps}"
				)
		return downsample_adaptive_static(aligned, cfg)

	if cfg.fps >= cfg.source_fps:
		keep_mask = np.ones(len(aligned), dtype=bool)
		return aligned, {"mode": cfg.downsample_mode, "keep_mask": keep_mask}

	if cfg.downsample_mode == "timestamp":
		out, keep_mask = downsample_by_timestamp(aligned, cfg.fps)
		return out, {"mode": "timestamp", "keep_mask": keep_mask}

	if cfg.downsample_mode == "stride":
		out, keep_mask = downsample_by_stride(aligned, cfg.source_fps, cfg.fps)
		return out, {"mode": "stride", "keep_mask": keep_mask}

	raise ValueError(f"Unsupported downsample mode: {cfg.downsample_mode}")


def write_static_debug_plot(debug_plot_path: Path, downsample_meta: dict, score_threshold: float) -> bool:
	if not downsample_meta or downsample_meta.get("mode") != "adaptive_static":
		return False

	try:
		import matplotlib.pyplot as plt
	except ImportError:
		logging.warning("matplotlib not installed, skip static debug plot: %s", debug_plot_path)
		return False

	ts = downsample_meta["timestamps_ns"]
	t0 = ts[0]
	time_sec = (ts - t0) / 1e9
	motion_score = downsample_meta["motion_score"]
	static_mask = downsample_meta["static_mask"].astype(np.int32)
	long_static_mask = downsample_meta["long_static_mask"].astype(np.int32)
	longest_static_mask = downsample_meta.get("longest_static_mask")
	if longest_static_mask is not None:
		longest_static_mask = longest_static_mask.astype(np.int32)
	transition_guard_mask = downsample_meta.get("transition_guard_mask")
	if transition_guard_mask is not None:
		transition_guard_mask = transition_guard_mask.astype(np.int32)
	keep_mask = downsample_meta["keep_mask"].astype(np.int32)
	sampling_fps = downsample_meta.get("sampling_fps")
	contact_debug = downsample_meta.get("contact_debug", {})
	contact_closed_mask = contact_debug.get("closed_mask")
	contact_long_mask = contact_debug.get("long_closed_mask")
	contact_candidate_mask = contact_debug.get("candidate_mask")
	contact_selected_mask = contact_debug.get("selected_mask")
	contact_rejected_after_static_mask = contact_debug.get("rejected_after_static_mask")
	contact_rejected_by_rank_mask = contact_debug.get("rejected_by_rank_mask")
	left_gripper_pose = contact_debug.get("left_gripper_pose")
	close_threshold_norm = contact_debug.get("close_threshold_norm")

	debug_plot_path.parent.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

	axes[0].plot(time_sec, motion_score, color="C0", linewidth=1.0, label="motion_score")
	axes[0].axhline(score_threshold, color="C3", linestyle="--", linewidth=1.0, label="static_threshold")
	axes[0].set_ylabel("score")
	axes[0].legend(loc="upper right")
	axes[0].grid(alpha=0.3)

	axes[1].step(time_sec, long_static_mask, where="post", color="C5", linewidth=1.0, label="long_static_mask")
	axes[1].step(time_sec, static_mask, where="post", color="C2", linewidth=1.2, label="static_mask")
	if longest_static_mask is not None:
		axes[1].step(
			time_sec,
			longest_static_mask,
			where="post",
			color="C1",
			linewidth=1.0,
			linestyle=":",
			label="longest_static_mask",
		)
	if transition_guard_mask is not None:
		axes[1].step(
			time_sec,
			transition_guard_mask,
			where="post",
			color="C3",
			linewidth=1.0,
			linestyle="--",
			label="transition_guard_mask",
		)
	axes[1].set_ylabel("static")
	axes[1].set_yticks([0, 1])
	axes[1].legend(loc="upper right")
	axes[1].grid(alpha=0.3)

	if left_gripper_pose is not None and len(left_gripper_pose) == len(time_sec):
		axes[2].plot(time_sec, left_gripper_pose, color="C4", linewidth=1.0, label="left_gripper_pose")
		if close_threshold_norm is not None:
			axes[2].axhline(
				close_threshold_norm,
				color="C3",
				linestyle="--",
				linewidth=1.0,
				label="contact_close_threshold",
			)
	axes[2].set_ylabel("gripper")
	axes[2].legend(loc="upper right")
	axes[2].grid(alpha=0.3)

	if contact_closed_mask is not None:
		axes[3].step(time_sec, contact_closed_mask.astype(np.int32), where="post", color="0.6", linewidth=1.0, label="closed_mask")
	if contact_long_mask is not None:
		axes[3].step(time_sec, contact_long_mask.astype(np.int32), where="post", color="C4", linewidth=1.0, label="long_closed_mask")
	if contact_candidate_mask is not None:
		axes[3].step(time_sec, contact_candidate_mask.astype(np.int32), where="post", color="C0", linewidth=1.0, linestyle="--", label="candidate_mask")
	if contact_selected_mask is not None:
		axes[3].step(time_sec, contact_selected_mask.astype(np.int32), where="post", color="C2", linewidth=1.2, label="selected_contact_mask")
	if contact_rejected_after_static_mask is not None:
		axes[3].step(
			time_sec,
			contact_rejected_after_static_mask.astype(np.int32),
			where="post",
			color="C3",
			linewidth=1.0,
			linestyle=":",
			label="rejected_after_static",
		)
	if contact_rejected_by_rank_mask is not None:
		axes[3].step(
			time_sec,
			contact_rejected_by_rank_mask.astype(np.int32),
			where="post",
			color="C1",
			linewidth=1.0,
			linestyle=":",
			label="rejected_by_rank",
		)
	axes[3].set_ylabel("contact")
	axes[3].set_yticks([0, 1])
	axes[3].legend(loc="upper right")
	axes[3].grid(alpha=0.3)

	if sampling_fps is not None:
		axes[4].plot(time_sec, sampling_fps, color="C1", linewidth=1.0, label="sampling_fps")
	keep_ax = axes[4].twinx()
	keep_ax.step(time_sec, keep_mask, where="post", color="C6", linewidth=1.0, label="keep_mask")
	keep_ax.set_yticks([0, 1])
	keep_ax.set_ylabel("keep")
	h1, l1 = axes[4].get_legend_handles_labels()
	h2, l2 = keep_ax.get_legend_handles_labels()
	axes[4].legend(h1 + h2, l1 + l2, loc="upper right")
	axes[4].set_ylabel("fps")
	axes[4].set_xlabel("time (s)")
	axes[4].grid(alpha=0.3)

	fig.suptitle("Adaptive Static + Contact-Rich Debug", fontsize=12)
	plt.tight_layout(rect=[0, 0.02, 1, 0.95])
	plt.savefig(debug_plot_path, dpi=140, bbox_inches="tight")
	plt.close(fig)
	return True


def chunked(items: list[str], chunk_size: int) -> list[list[str]]:
	if chunk_size <= 0:
		raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
	return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def select_episodes(
	episodes: list[str],
	max_episodes_per_source: int | None,
	sample_mode: Literal["first", "uniform"],
) -> list[str]:
	if max_episodes_per_source is None or max_episodes_per_source >= len(episodes):
		return episodes

	if max_episodes_per_source <= 0:
		return []

	if sample_mode == "first":
		return episodes[:max_episodes_per_source]

	if sample_mode == "uniform":
		indices = np.linspace(0, len(episodes) - 1, num=max_episodes_per_source, dtype=int)
		indices = np.unique(indices)
		return [episodes[idx] for idx in indices]

	raise ValueError(f"Unsupported episode sample mode: {sample_mode}")


def prepare_episode(source_root: Path, ep_name: str, cfg: ConverterConfig) -> dict:
	t_prepare_start = time.perf_counter()
	t_align_start = time.perf_counter()
	aligned, meta = align_episode(source_root, ep_name, cfg)
	align_s = time.perf_counter() - t_align_start
	if not aligned:
		return {
			"episode": ep_name,
			"skip_reason": meta["skip_reason"],
			"aligned": [],
			"raw_frame_count": 0,
			"timing": {
				"align_s": align_s,
				"downsample_s": 0.0,
				"prepare_s": time.perf_counter() - t_prepare_start,
			},
		}

	raw_frame_count = len(aligned)
	t_downsample_start = time.perf_counter()
	aligned, downsample_meta = downsample_aligned_frames(aligned=aligned, cfg=cfg)
	downsample_s = time.perf_counter() - t_downsample_start
	prepare_s = time.perf_counter() - t_prepare_start
	if not aligned:
		return {
			"episode": ep_name,
			"skip_reason": "empty_after_downsampling",
			"aligned": [],
			"raw_frame_count": raw_frame_count,
			"downsample_meta": downsample_meta,
			"timing": {
				"align_s": align_s,
				"downsample_s": downsample_s,
				"prepare_s": prepare_s,
			},
		}

	return {
		"episode": ep_name,
		"skip_reason": None,
		"aligned": aligned,
		"raw_frame_count": raw_frame_count,
		"downsample_meta": downsample_meta,
		"timing": {
			"align_s": align_s,
			"downsample_s": downsample_s,
			"prepare_s": prepare_s,
		},
	}


def prepare_episode_worker(source_root_str: str, ep_name: str, cfg_dict: dict) -> dict:
	cfg = ConverterConfig(**cfg_dict)
	return prepare_episode(Path(source_root_str), ep_name, cfg)


def convert_one_source(
	dataset: LeRobotDataset,
	source_root: Path,
	task: str,
	cfg: ConverterConfig,
	debug_root: Path,
	source_name: str,
) -> list[dict]:
	summary_stats = []
	load_image = build_image_loader(cfg.image_decode_cache_size)

	temp_images_dir = source_root / "temp_images"
	arm_data_dir = source_root / "arm_data"
	if not temp_images_dir.exists() or not arm_data_dir.exists():
		logging.warning("Skip source %s: temp_images or arm_data missing", source_root)
		return summary_stats

	episodes = sorted([d.name for d in temp_images_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])

	episodes = select_episodes(
		episodes=episodes,
		max_episodes_per_source=cfg.max_episodes_per_source,
		sample_mode=cfg.episode_sample_mode,
	)
	if cfg.max_episodes_per_source is not None:
		logging.warning(
			"Episode sampling enabled for %s: selected %d episodes (mode=%s, max=%d)",
			source_name,
			len(episodes),
			cfg.episode_sample_mode,
			cfg.max_episodes_per_source,
		)
	static_plot_written = False

	def write_prepared_episode(prepared: dict) -> None:
		ep_total_start = time.perf_counter()
		ep_name = prepared["episode"]
		skip_reason = prepared["skip_reason"]
		prepare_timing = prepared.get("timing", {})
		align_s = float(prepare_timing.get("align_s", 0.0))
		downsample_s = float(prepare_timing.get("downsample_s", 0.0))
		prepare_s = float(prepare_timing.get("prepare_s", 0.0))
		if skip_reason is not None:
			logging.info("Skip %s/%s: %s", source_name, ep_name, skip_reason)
			if cfg.profile_timing:
				tqdm.tqdm.write(
					(
						f"[timing] {source_name}/{ep_name} skip={skip_reason} "
						f"prepare={prepare_s:.3f}s (align={align_s:.3f}s, downsample={downsample_s:.3f}s)"
					)
				)
			return

		raw_frame_count = prepared["raw_frame_count"]
		aligned = prepared["aligned"]
		if cfg.debug:
			logging.info(
				"Downsample %s/%s: raw=%d -> kept=%d (source_fps=%.3f, target_fps=%d)",
				source_name,
				ep_name,
				raw_frame_count,
				len(aligned),
				cfg.source_fps,
				cfg.fps,
			)

		rows = []
		cam_align_diffs = []
		head_diffs = []
		left_diffs = []
		right_diffs = []
		downsample_meta = prepared.get("downsample_meta", {})
		load_image_s = 0.0
		add_frame_api_s = 0.0
		t_add_frame_start = time.perf_counter()
		t_save_start = None
		t_save_end = None
		with suppress_output(enabled=not cfg.debug):
			for i in range(len(aligned)):
				current = aligned[i]
				action = select_step_action(aligned, i)

				t_load_start = time.perf_counter()
				frame = {
					"task": task,
					"observation.images.head_image": load_image(current["head_img_path"]),
					"observation.images.left_wrist_image": load_image(current["left_img_path"]),
					"observation.images.right_wrist_image": load_image(current["right_img_path"]),
					"observation.state": current["observation_state"],
					"action": action,
				}
				load_image_s += time.perf_counter() - t_load_start

				t_add_api_start = time.perf_counter()
				dataset.add_frame(frame)
				add_frame_api_s += time.perf_counter() - t_add_api_start

				cam_ts = [current["head_ts_ns"], current["left_ts_ns"], current["right_ts_ns"]]
				cam_align_max_diff_ms = (max(cam_ts) - min(cam_ts)) / 1e6
				cam_align_diffs.append(cam_align_max_diff_ms)

				if i > 0:
					prev = aligned[i - 1]
					head_dt_ms = (current["head_ts_ns"] - prev["head_ts_ns"]) / 1e6
					left_dt_ms = (current["left_ts_ns"] - prev["left_ts_ns"]) / 1e6
					right_dt_ms = (current["right_ts_ns"] - prev["right_ts_ns"]) / 1e6
				else:
					head_dt_ms = np.nan
					left_dt_ms = np.nan
					right_dt_ms = np.nan

				head_diffs.append(head_dt_ms)
				left_diffs.append(left_dt_ms)
				right_diffs.append(right_dt_ms)

				rows.append(
					{
						"head_image_ts": current["head_ts"],
						"left_image_ts": current["left_ts"],
						"right_image_ts": current["right_ts"],
						"left_arm_ts": current["left_arm_ts"],
						"right_arm_ts": current["right_arm_ts"],
						"is_last_frame": i == len(aligned) - 1,
						"cam_align_max_diff_ms": cam_align_max_diff_ms,
						"head_dt_ms": head_dt_ms,
						"left_dt_ms": left_dt_ms,
						"right_dt_ms": right_dt_ms,
						"raw_aligned_frames": raw_frame_count,
						"kept_frames": len(aligned),
						"source_fps": cfg.source_fps,
						"target_fps": cfg.fps,
					}
				)

			t_save_start = time.perf_counter()
			dataset.save_episode()
			t_save_end = time.perf_counter()

		add_frame_s = (t_save_start - t_add_frame_start) if t_save_start is not None else 0.0
		save_episode_s = (t_save_end - t_save_start) if t_save_start is not None and t_save_end is not None else 0.0

		t_stats_start = time.perf_counter()
		if cfg.debug:
			write_debug_csv(debug_root / source_name / f"{ep_name}.csv", rows)
		head_max, head_cnt = max_and_count(head_diffs)
		left_max, left_cnt = max_and_count(left_diffs)
		right_max, right_cnt = max_and_count(right_diffs)

		summary_stats.append(
			{
				"source": source_name,
				"episode": ep_name,
				"cam_align_max_diff_ms": max(cam_align_diffs) if cam_align_diffs else float("nan"),
				"head_max_dt_ms": head_max,
				"head_max_dt_cnt": head_cnt,
				"left_max_dt_ms": left_max,
				"left_max_dt_cnt": left_cnt,
				"right_max_dt_ms": right_max,
				"right_max_dt_cnt": right_cnt,
				"num_frames": len(aligned),
				"raw_aligned_frames": raw_frame_count,
				"source_fps": cfg.source_fps,
				"target_fps": cfg.fps,
				"downsample_mode": downsample_meta.get("mode", cfg.downsample_mode),
				"static_frames_raw": downsample_meta.get("static_frames_raw", float("nan")),
				"static_frames_kept": downsample_meta.get("static_frames_kept", float("nan")),
				"moving_frames_raw": downsample_meta.get("moving_frames_raw", float("nan")),
				"moving_frames_kept": downsample_meta.get("moving_frames_kept", float("nan")),
				"contact_closed_frames_raw": downsample_meta.get("contact_closed_frames_raw", float("nan")),
				"contact_candidate_frames_raw": downsample_meta.get("contact_candidate_frames_raw", float("nan")),
				"contact_frames_raw": downsample_meta.get("contact_frames_raw", float("nan")),
				"contact_frames_kept": downsample_meta.get("contact_frames_kept", float("nan")),
				"contact_reasons": "|".join(downsample_meta.get("contact_reasons", [])),
			}
		)
		stats_s = time.perf_counter() - t_stats_start
		ep_total_s = time.perf_counter() - ep_total_start

		if cfg.profile_timing:
			add_frame_other_s = max(add_frame_s - load_image_s - add_frame_api_s, 0.0)
			tqdm.tqdm.write(
				(
					f"[timing] {source_name}/{ep_name} "
					f"prepare={prepare_s:.3f}s (align={align_s:.3f}s, downsample={downsample_s:.3f}s) "
					f"add_frame={add_frame_s:.3f}s "
					f"(load_image={load_image_s:.3f}s, add_frame_api={add_frame_api_s:.3f}s, other={add_frame_other_s:.3f}s) "
					f"save_episode={save_episode_s:.3f}s "
					f"stats={stats_s:.3f}s total={ep_total_s:.3f}s frames={len(aligned)}"
				)
			)

		nonlocal static_plot_written
		should_save_debug_plot = False
		if cfg.static_debug_plot_first_episode and not static_plot_written:
			should_save_debug_plot = True
		if downsample_meta.get("contact_should_save_debug_plot", False):
			should_save_debug_plot = True
		if should_save_debug_plot:
			debug_plot_path = debug_root / source_name / f"{ep_name}_static_debug.png"
			plot_saved = write_static_debug_plot(
				debug_plot_path=debug_plot_path,
				downsample_meta=downsample_meta,
				score_threshold=cfg.static_motion_score_threshold,
			)
			if plot_saved:
				logging.warning("Saved static debug plot: %s", debug_plot_path)
				if cfg.static_debug_plot_first_episode and not static_plot_written:
					static_plot_written = True

	if cfg.num_workers <= 1:
		for ep_name in tqdm.tqdm(episodes, desc=f"{source_name}"):
			prepared = prepare_episode(source_root, ep_name, cfg)
			write_prepared_episode(prepared)
		return summary_stats

	cfg_dict = dataclasses.asdict(cfg)
	batches = chunked(episodes, cfg.worker_batch_size)
	pbar = tqdm.tqdm(total=len(episodes), desc=f"{source_name}")
	try:
		for batch_idx, batch_eps in enumerate(batches):
			if cfg.debug:
				logging.info(
					"Parallel batch %s/%s for %s: episodes=%d, workers=%d, max_inflight=%d",
					batch_idx + 1,
					len(batches),
					source_name,
					len(batch_eps),
					cfg.num_workers,
					cfg.max_inflight_episodes,
				)

			with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.num_workers) as executor:
				episode_iter = iter(batch_eps)
				inflight: dict[concurrent.futures.Future, str] = {}

				while len(inflight) < cfg.max_inflight_episodes:
					try:
						ep_name = next(episode_iter)
					except StopIteration:
						break
					future = executor.submit(prepare_episode_worker, str(source_root), ep_name, cfg_dict)
					inflight[future] = ep_name

				while inflight:
					done, _ = concurrent.futures.wait(
						inflight.keys(), return_when=concurrent.futures.FIRST_COMPLETED
					)
					for fut in done:
						_ = inflight.pop(fut)
						prepared = fut.result()
						write_prepared_episode(prepared)
						pbar.update(1)

					while len(inflight) < cfg.max_inflight_episodes:
						try:
							ep_name = next(episode_iter)
						except StopIteration:
							break
						future = executor.submit(prepare_episode_worker, str(source_root), ep_name, cfg_dict)
						inflight[future] = ep_name
	finally:
		pbar.close()

	return summary_stats


def dedup_paths(paths: list[Path]) -> list[Path]:
	out = []
	seen = set()
	for p in paths:
		rp = str(p.resolve())
		if rp in seen:
			continue
		seen.add(rp)
		out.append(Path(rp))
	return out


def extract_camera_shapes_from_features(features: dict) -> dict[str, tuple[int, int]]:
	shapes: dict[str, tuple[int, int]] = {}
	for cam_key in CAMERA_KEYS:
		if cam_key not in features:
			raise ValueError(f"Existing dataset is missing camera feature: {cam_key}")
		shape = tuple(features[cam_key]["shape"])
		if len(shape) != 3:
			raise ValueError(f"Unexpected shape for {cam_key}: {shape}")
		channels, height, width = shape
		if channels != 3:
			raise ValueError(f"Unexpected channel count for {cam_key}: {channels}")
		shapes[cam_key] = (height, width)
	return shapes


def extract_storage_mode_from_features(features: dict) -> Literal["image", "video"]:
	dtypes = {features[cam_key]["dtype"] for cam_key in CAMERA_KEYS}
	if len(dtypes) != 1:
		raise ValueError(f"Existing dataset uses inconsistent camera storage dtypes: {sorted(dtypes)}")
	storage_mode = dtypes.pop()
	if storage_mode not in {"image", "video"}:
		raise ValueError(f"Unsupported camera storage dtype in existing dataset: {storage_mode}")
	return storage_mode


def load_existing_summary(summary_csv_path: Path) -> pd.DataFrame:
	if not summary_csv_path.exists():
		return pd.DataFrame()
	return pd.read_csv(summary_csv_path)


def iter_dataset_files(root_dir: Path):
	for dirpath, _, filenames in os.walk(root_dir):
		for filename in filenames:
			yield Path(dirpath) / filename


def copytree_with_progress(source_dir: Path, target_dir: Path) -> None:
	total_bytes = 0
	for path in iter_dataset_files(source_dir):
		if path.is_file():
			total_bytes += path.stat().st_size

	if total_bytes <= 0:
		shutil.copytree(source_dir, target_dir)
		return

	with tqdm.tqdm(
		total=total_bytes,
		desc="Cloning dataset",
		unit="B",
		unit_scale=True,
		unit_divisor=1024,
	) as progress_bar:
		def copy_with_progress(src: str, dst: str) -> str:
			copied_path = shutil.copy2(src, dst)
			progress_bar.update(os.path.getsize(src))
			return copied_path

		shutil.copytree(source_dir, target_dir, copy_function=copy_with_progress)


def clone_existing_dataset(source_dir: Path, target_dir: Path, overwrite: bool) -> pd.DataFrame:
	if not source_dir.exists() or not source_dir.is_dir():
		raise FileNotFoundError(f"Incremental source dataset directory does not exist: {source_dir}")
	if not (source_dir / "meta" / "info.json").exists():
		raise FileNotFoundError(
			f"Incremental source dataset does not look like a LeRobot dataset: {source_dir}"
		)

	source_resolved = source_dir.resolve()
	target_resolved = target_dir.resolve()
	if source_resolved == target_resolved:
		raise ValueError(
			"Incremental source dataset and output dataset path are the same. "
			"Choose a different --repo-id or --dst-dir so history is not modified in place."
		)

	if target_dir.exists():
		if not overwrite:
			raise FileExistsError(
				f"Output dataset path already exists: {target_dir}. Use --overwrite to recreate it from the incremental base."
			)
		shutil.rmtree(target_dir)

	logging.warning("Cloning existing dataset: %s -> %s", source_dir, target_dir)
	copytree_with_progress(source_dir, target_dir)
	return load_existing_summary(target_dir / "summary_max_diffs.csv")


def validate_incremental_dataset_compatibility(
	dataset: LeRobotDataset,
	cfg: ConverterConfig,
	new_camera_shapes: dict[str, tuple[int, int]],
) -> None:
	existing_shapes = extract_camera_shapes_from_features(dataset.features)
	if existing_shapes != new_camera_shapes:
		raise ValueError(
			"New raw data camera shapes do not match the existing dataset. "
			f"existing={existing_shapes}, new={new_camera_shapes}"
		)

	existing_storage_mode = extract_storage_mode_from_features(dataset.features)
	if existing_storage_mode != cfg.storage_mode:
		raise ValueError(
			"--storage-mode does not match the existing dataset. "
			f"existing={existing_storage_mode}, requested={cfg.storage_mode}"
		)

	if dataset.fps != cfg.fps:
		raise ValueError(
			f"--fps does not match the existing dataset. existing={dataset.fps}, requested={cfg.fps}"
		)

	if dataset.meta.robot_type != cfg.robot_type:
		raise ValueError(
			"--robot-type does not match the existing dataset. "
			f"existing={dataset.meta.robot_type}, requested={cfg.robot_type}"
		)

	expected_vector_shape = (len(MOTORS),)
	for key in ["observation.state", "action"]:
		shape = tuple(dataset.features[key]["shape"])
		if shape != expected_vector_shape:
			raise ValueError(
				f"Existing dataset feature shape mismatch for {key}: {shape}, expected {expected_vector_shape}"
			)


def open_incremental_dataset(repo_id: str, root: Path, cfg: ConverterConfig) -> LeRobotDataset:
	init_signature = inspect.signature(LeRobotDataset.__init__)
	init_kwargs = {
		"repo_id": repo_id,
		"root": root,
	}
	optional_init_kwargs = {
		"image_writer_processes": cfg.image_writer_processes,
		"image_writer_threads": cfg.image_writer_threads,
		"streaming_encoding": cfg.streaming_encoding,
		"encoder_queue_maxsize": cfg.encoder_queue_maxsize,
		"encoder_threads": cfg.encoder_threads,
	}
	passed_init_keys = set()
	for key, value in optional_init_kwargs.items():
		if key in init_signature.parameters:
			init_kwargs[key] = value
			passed_init_keys.add(key)

	dataset = LeRobotDataset(**init_kwargs)
	if (
		("image_writer_processes" not in passed_init_keys)
		and ("image_writer_threads" not in passed_init_keys)
		and hasattr(dataset, "start_image_writer")
		and (cfg.image_writer_processes or cfg.image_writer_threads)
	):
		dataset.start_image_writer(cfg.image_writer_processes, cfg.image_writer_threads)
	return dataset


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Convert raw Keenon data to LeRobotDataset v3 format.")
	parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo id, e.g. user/keenon-demo")
	parser.add_argument(
		"--input-root",
		type=Path,
		nargs="+",
		action="append",
		required=True,
		help=(
			"Raw data root(s) containing temp_images/ and arm_data/. "
			"Supports '--input-root a b c' or repeated '--input-root a --input-root b'."
		),
	)
	parser.add_argument("--dst-dir", type=Path, required=True, help="Output directory root")
	parser.add_argument(
		"--incremental-from-dataset",
		type=Path,
		default=None,
		help=(
			"Optional existing LeRobot dataset directory. If set, the script first clones that dataset into "
			"the new output path (--dst-dir/--repo-id), then appends newly converted episodes to the clone."
		),
	)
	parser.add_argument("--task", type=str, required=True, help="Task string written into each frame")
	parser.add_argument("--source-fps", type=float, default=30.0, help="Original raw data capture FPS, e.g. 30")
	parser.add_argument("--fps", type=int, default=30, help="Target LeRobot dataset FPS after downsampling")
	parser.add_argument("--robot-type", type=str, default="Keenon_arm")
	parser.add_argument("--storage-mode", choices=["image", "video"], default="video")
	parser.add_argument("--action-gripper-mode", choices=["open_value", "bin"], default="open_value")
	parser.add_argument("--max-image-diff-ms", type=float, default=40.0)
	parser.add_argument("--min-episode-images", type=int, default=100)
	parser.add_argument("--skip-head-frames", type=int, default=30)
	parser.add_argument("--skip-tail-frames", type=int, default=5)
	parser.add_argument("--image-decode-cache-size", type=int, default=256)
	parser.add_argument("--overwrite", action="store_true", help="Delete existing output dataset folder")
	parser.add_argument(
		"--downsample-mode",
		choices=["timestamp", "stride", "adaptive_static"],
		default="timestamp",
		help=(
			"Downsample mode: 'timestamp' is robust to jitter, 'stride' keeps every Nth frame, "
			"'adaptive_static' keeps normal fps on motion but lowers long static regions"
		),
	)
	parser.add_argument(
		"--static-fps",
		type=float,
		default=3.0,
		help="FPS used inside long static regions when --downsample-mode adaptive_static",
	)
	parser.add_argument(
		"--static-min-duration-sec",
		type=float,
		default=2.5,
		help="Minimum continuous static duration to trigger static downsample",
	)
	parser.add_argument(
		"--static-motion-score-threshold",
		type=float,
		default=0.03,
		help="Motion score threshold below which frames are considered static",
	)
	parser.add_argument(
		"--static-motion-scale-quantile",
		type=float,
		default=0.95,
		help="Quantile used to normalize joint delta per episode (jitter-robust)",
	)
	parser.add_argument(
		"--static-motion-scale-floor",
		type=float,
		default=1e-4,
		help="Lower bound for normalization scale to avoid zero-division and over-sensitivity",
	)
	parser.add_argument(
		"--static-transition-guard-frames",
		type=int,
		default=30,
		help=(
			"Keep full-fps before static->motion transitions by shrinking only static segment tails"
		),
	)
	parser.add_argument(
		"--static-debug-plot-first-episode",
		action="store_true",
		help="Write one temporary debug plot for the first kept episode in each source",
	)
	parser.add_argument(
		"--enable-contact-rich-upsample",
		action="store_true",
		help="Keep higher FPS inside selected contact-rich intervals when using --downsample-mode adaptive_static.",
	)
	parser.add_argument(
		"--contact-rich-fps",
		type=float,
		default=None,
		help="Sampling FPS used inside selected contact-rich intervals. Defaults to --source-fps.",
	)
	parser.add_argument(
		"--contact-rich-close-threshold",
		type=float,
		default=12.0,
		help="Raw left gripper open-value threshold (0-70) used to detect contact-rich closure.",
	)
	parser.add_argument(
		"--contact-rich-min-duration-sec",
		type=float,
		default=0.5,
		help="Minimum continuous closure duration to qualify as a contact-rich interval.",
	)
	parser.add_argument(
		"--contact-rich-max-ranges",
		type=int,
		default=2,
		help="Maximum number of longest contact-rich intervals kept per episode.",
	)
	parser.add_argument(
		"--contact-rich-require-before-longest-static",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Require selected contact-rich intervals to end before the longest static interval.",
	)
	parser.add_argument(
		"--contact-rich-debug-all-rejections",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Write debug plots for episodes where contact-rich candidates are rejected by the selection rules.",
	)
	parser.add_argument("--debug", action="store_true", help="Enable verbose logs and write per-episode debug CSV files")
	parser.add_argument(
		"--image-writer-processes",
		type=int,
		default=0,
		help="Number of processes for async image writer used by LeRobotDataset.",
	)
	parser.add_argument(
		"--image-writer-threads",
		type=int,
		default=0,
		help="Number of threads for async image writer used by LeRobotDataset.",
	)
	parser.add_argument(
		"--streaming-encoding",
		action="store_true",
		help="Enable real-time streaming video encoding during add_frame to reduce save_episode cost.",
	)
	parser.add_argument(
		"--encoder-queue-maxsize",
		type=int,
		default=30,
		help="Max buffered frames per camera when --streaming-encoding is enabled.",
	)
	parser.add_argument(
		"--encoder-threads",
		type=int,
		default=None,
		help="Encoder thread count for video encoding. None lets codec decide.",
	)
	parser.add_argument(
		"--max-episodes-per-source",
		type=int,
		default=None,
		help="Optionally cap the number of episodes processed from each source root.",
	)
	parser.add_argument(
		"--episode-sample-mode",
		choices=["first", "uniform"],
		default="first",
		help="How to pick episodes when --max-episodes-per-source is set.",
	)
	parser.add_argument(
		"--num-workers",
		type=int,
		default=1,
		help="Number of worker processes for episode preparation. 1 disables parallelism.",
	)
	parser.add_argument(
		"--max-inflight-episodes",
		type=int,
		default=4,
		help="Upper bound of queued/running episode tasks in parallel mode.",
	)
	parser.add_argument(
		"--worker-batch-size",
		type=int,
		default=100,
		help="Episodes per worker-pool batch before pool recycle (helps memory stability).",
	)
	parser.add_argument(
		"--profile-timing",
		action="store_true",
		help="Print per-episode stage timing (prepare/add_frame/save_episode/stats/total).",
	)
	parser.add_argument(
		"--log-level",
		type=str,
		default="INFO",
		help="Logging level used only when --debug is enabled",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.debug:
		log_level = getattr(logging, args.log_level.upper(), logging.INFO)
	else:
		# Default to warning/error only so terminal stays clean and only tqdm progress is visible.
		log_level = logging.WARNING
	logging.basicConfig(level=log_level)

	cfg = ConverterConfig(
		fps=args.fps,
		source_fps=args.source_fps,
		downsample_mode=args.downsample_mode,
		robot_type=args.robot_type,
		storage_mode=args.storage_mode,
		action_gripper_mode=args.action_gripper_mode,
		max_image_diff_ms=args.max_image_diff_ms,
		min_episode_images=args.min_episode_images,
		skip_head_frames=args.skip_head_frames,
		skip_tail_frames=args.skip_tail_frames,
		image_decode_cache_size=args.image_decode_cache_size,
		overwrite=args.overwrite,
		debug=args.debug,
		num_workers=args.num_workers,
		max_inflight_episodes=args.max_inflight_episodes,
		worker_batch_size=args.worker_batch_size,
		profile_timing=args.profile_timing,
		image_writer_processes=args.image_writer_processes,
		image_writer_threads=args.image_writer_threads,
		streaming_encoding=args.streaming_encoding,
		encoder_queue_maxsize=args.encoder_queue_maxsize,
		encoder_threads=args.encoder_threads,
		max_episodes_per_source=args.max_episodes_per_source,
		episode_sample_mode=args.episode_sample_mode,
		static_fps=args.static_fps,
		static_min_duration_sec=args.static_min_duration_sec,
		static_motion_score_threshold=args.static_motion_score_threshold,
		static_motion_scale_quantile=args.static_motion_scale_quantile,
		static_motion_scale_floor=args.static_motion_scale_floor,
		static_transition_guard_frames=args.static_transition_guard_frames,
		static_debug_plot_first_episode=args.static_debug_plot_first_episode,
		enable_contact_rich_upsample=args.enable_contact_rich_upsample,
		contact_rich_fps=args.contact_rich_fps,
		contact_rich_close_threshold=args.contact_rich_close_threshold,
		contact_rich_min_duration_sec=args.contact_rich_min_duration_sec,
		contact_rich_max_ranges=args.contact_rich_max_ranges,
		contact_rich_require_before_longest_static=args.contact_rich_require_before_longest_static,
		contact_rich_debug_all_rejections=args.contact_rich_debug_all_rejections,
	)

	if cfg.num_workers <= 0:
		raise ValueError(f"--num-workers must be > 0, got {cfg.num_workers}")
	if cfg.max_inflight_episodes <= 0:
		raise ValueError(f"--max-inflight-episodes must be > 0, got {cfg.max_inflight_episodes}")
	if cfg.worker_batch_size <= 0:
		raise ValueError(f"--worker-batch-size must be > 0, got {cfg.worker_batch_size}")
	if cfg.image_writer_processes < 0:
		raise ValueError(
			f"--image-writer-processes must be >= 0, got {cfg.image_writer_processes}"
		)
	if cfg.image_writer_threads < 0:
		raise ValueError(
			f"--image-writer-threads must be >= 0, got {cfg.image_writer_threads}"
		)
	if cfg.encoder_queue_maxsize <= 0:
		raise ValueError(
			f"--encoder-queue-maxsize must be > 0, got {cfg.encoder_queue_maxsize}"
		)
	if cfg.encoder_threads is not None and cfg.encoder_threads <= 0:
		raise ValueError(f"--encoder-threads must be > 0 when set, got {cfg.encoder_threads}")
	if cfg.max_episodes_per_source is not None and cfg.max_episodes_per_source <= 0:
		raise ValueError(
			f"--max-episodes-per-source must be > 0 when set, got {cfg.max_episodes_per_source}"
		)
	if cfg.static_fps <= 0:
		raise ValueError(f"--static-fps must be > 0, got {cfg.static_fps}")
	if cfg.static_fps > cfg.fps:
		raise ValueError(f"--static-fps must be <= --fps, got static={cfg.static_fps}, fps={cfg.fps}")
	if cfg.static_min_duration_sec <= 0:
		raise ValueError(f"--static-min-duration-sec must be > 0, got {cfg.static_min_duration_sec}")
	if not 0 < cfg.static_motion_scale_quantile <= 1:
		raise ValueError(
			f"--static-motion-scale-quantile must be in (0, 1], got {cfg.static_motion_scale_quantile}"
		)
	if cfg.static_motion_scale_floor <= 0:
		raise ValueError(
			f"--static-motion-scale-floor must be > 0, got {cfg.static_motion_scale_floor}"
		)
	if cfg.static_transition_guard_frames < 0:
		raise ValueError(
			f"--static-transition-guard-frames must be >= 0, got {cfg.static_transition_guard_frames}"
		)
	if cfg.enable_contact_rich_upsample and cfg.downsample_mode != "adaptive_static":
		raise ValueError("--enable-contact-rich-upsample currently requires --downsample-mode adaptive_static")
	if cfg.contact_rich_close_threshold < 0 or cfg.contact_rich_close_threshold > 70:
		raise ValueError(
			f"--contact-rich-close-threshold must be in [0, 70], got {cfg.contact_rich_close_threshold}"
		)
	if cfg.contact_rich_min_duration_sec <= 0:
		raise ValueError(
			f"--contact-rich-min-duration-sec must be > 0, got {cfg.contact_rich_min_duration_sec}"
		)
	if cfg.contact_rich_max_ranges <= 0:
		raise ValueError(f"--contact-rich-max-ranges must be > 0, got {cfg.contact_rich_max_ranges}")
	if cfg.enable_contact_rich_upsample:
		contact_rich_fps = cfg.source_fps if cfg.contact_rich_fps is None else cfg.contact_rich_fps
		if contact_rich_fps > cfg.source_fps:
			raise ValueError(
				f"--contact-rich-fps must be <= --source-fps, got contact={contact_rich_fps}, source={cfg.source_fps}"
			)
		if contact_rich_fps < cfg.fps:
			raise ValueError(
				f"--contact-rich-fps must be >= --fps, got contact={contact_rich_fps}, fps={cfg.fps}"
			)
	if cfg.fps > int(round(cfg.source_fps)):
		logging.warning(
			"target fps (%s) is higher than source fps (%s). No upsampling is performed; frames will be kept as-is.",
			cfg.fps,
			cfg.source_fps,
		)

	input_root_groups: list[list[Path]] = args.input_root
	input_roots = dedup_paths([p for group in input_root_groups for p in group])
	camera_shapes = infer_camera_shapes(input_roots)
	logging.warning("Detected camera shapes: %s", camera_shapes)
	output_root = args.dst_dir / args.repo_id

	existing_summary_df = pd.DataFrame()
	if args.incremental_from_dataset is not None:
		existing_summary_df = clone_existing_dataset(
			source_dir=args.incremental_from_dataset,
			target_dir=output_root,
			overwrite=cfg.overwrite,
		)
		dataset = open_incremental_dataset(args.repo_id, output_root, cfg)
		validate_incremental_dataset_compatibility(dataset, cfg, camera_shapes)
	else:
		if output_root.exists():
			if not cfg.overwrite:
				raise FileExistsError(
					f"Output dataset path already exists: {output_root}. Use --overwrite to delete it first."
				)
			shutil.rmtree(output_root)

		dataset = LeRobotDataset.create(
			repo_id=args.repo_id,
			root=output_root,
			fps=cfg.fps,
			robot_type=cfg.robot_type,
			features=build_features(cfg.storage_mode, camera_shapes),
			use_videos=cfg.storage_mode == "video",
			image_writer_processes=cfg.image_writer_processes,
			image_writer_threads=cfg.image_writer_threads,
			streaming_encoding=cfg.streaming_encoding,
			encoder_queue_maxsize=cfg.encoder_queue_maxsize,
			encoder_threads=cfg.encoder_threads,
		)

	all_summary_stats = []
	try:
		for i, source_root in enumerate(input_roots):
			source_name = f"src{i:02d}_{source_root.name}"
			logging.info("Processing source: %s (%s)", source_name, source_root)
			summary = convert_one_source(
				dataset=dataset,
				source_root=source_root,
				task=args.task,
				cfg=cfg,
				debug_root=output_root / "debug",
				source_name=source_name,
			)
			all_summary_stats.extend(summary)
	finally:
		with suppress_output(enabled=not cfg.debug):
			dataset.finalize()

	summary_csv_path = output_root / "summary_max_diffs.csv"
	new_summary_df = pd.DataFrame(all_summary_stats)
	if not existing_summary_df.empty:
		summary_df = pd.concat([existing_summary_df, new_summary_df], ignore_index=True)
	else:
		summary_df = new_summary_df
	summary_df.to_csv(summary_csv_path, index=False)
	logging.info("Saved summary CSV: %s", summary_csv_path)
	logging.info("Done. total episodes written: %s", dataset.meta.total_episodes)


if __name__ == "__main__":
	main()
