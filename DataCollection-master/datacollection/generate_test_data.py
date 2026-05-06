#!/usr/bin/env python3
"""Generate mock data to verify all error detection paths in main.py.

Each episode directory triggers exactly one specific error condition.
Episode 0 is a normal case that should pass all checks.
"""

from pathlib import Path
import shutil, zipfile, os
import numpy as np
from PIL import Image

OUTPUT_ZIP = Path("test_data.zip")
WORK = Path("/tmp/test_data_gen")
TASK = "test_task"

BASE = 1_000_000_000_000   # base timestamp (ns)
CDT  = 33_333_333          # camera interval ~30fps
JDT  = 10_000_000          # joint interval 100Hz
NF   = 30                  # camera frames per stream
NJ   = 100                 # joint rows

CAMS = ["head", "hand_left", "hand_right"]


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def mkep(eid):
    ep = WORK / TASK / str(eid)
    for c in CAMS:
        for s in ("color", "depth"):
            (ep / "camera" / c / s).mkdir(parents=True, exist_ok=True)
    (ep / "record").mkdir(parents=True, exist_ok=True)
    return ep


def save_color(p, w=640, h=480, rgb=(128, 100, 80)):
    Image.new("RGB", (w, h), rgb).save(str(p), "JPEG")


def save_depth(p, w=640, h=480, v=5000):
    Image.fromarray(np.full((h, w), v, np.uint16)).save(str(p), "PNG")


def cam_ts(n=NF):
    return [BASE + i * CDT for i in range(n)]


def joint_ts(n=NJ, dt=JDT, t0=BASE):
    return [t0 + i * dt for i in range(n)]


def jv_default():
    """16-dim: 7 joint angles + action(1) + width(60mm) + 7 zeros"""
    return [j * 0.1 for j in range(7)] + [1.0, 60.0] + [0.0] * 7


def write_cam(ep, ts=None, *, overrides=None):
    """
    Write camera frames for all cameras and streams.
    overrides: dict mapping (cam, stream) to:
      - "skip"    → don't write frames
      - "corrupt" → write garbage bytes
      - dict      → kwargs for save_color / save_depth
      - list      → custom timestamp list
    """
    if ts is None:
        ts = cam_ts()
    if overrides is None:
        overrides = {}
    for c in CAMS:
        for s in ("color", "depth"):
            d = ep / "camera" / c / s
            sfx = "jpg" if s == "color" else "png"
            ov = overrides.get((c, s))
            if ov == "skip":
                continue
            corrupt = (ov == "corrupt")
            if isinstance(ov, list):
                cur_ts, kw = ov, {}
            elif isinstance(ov, dict):
                cur_ts, kw = ts, ov
            else:
                cur_ts, kw = ts, {}
            for t in cur_ts:
                fp = d / f"{t}.{sfx}"
                if corrupt:
                    fp.write_bytes(b"\x00\x01bad_image_data")
                elif s == "color":
                    save_color(fp, **kw)
                else:
                    save_depth(fp, **kw)


def write_joints(ep, ts=None, *, lv=None, rv=None):
    if ts is None:
        ts = joint_ts()
    if lv is None:
        lv = [jv_default() for _ in ts]
    if rv is None:
        rv = [jv_default() for _ in ts]
    for side, vals in [("left", lv), ("right", rv)]:
        with open(ep / "record" / f"{side}_data.txt", "w") as f:
            for t, v in zip(ts, vals):
                f.write(f"{t} {','.join(str(x) for x in v)}\n")


def write_joint_raw(ep, left_text, right_text):
    (ep / "record" / "left_data.txt").write_text(left_text)
    (ep / "record" / "right_data.txt").write_text(right_text)


def right_normal_text():
    ts = joint_ts()
    lines = []
    for t in ts:
        v = jv_default()
        lines.append(f"{t} {','.join(str(x) for x in v)}")
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════
#  Test case generators
# ═══════════════════════════════════════════════════════════════

def gen_all_cases():
    expected = {}

    # ── 0: Normal (pass) ──
    ep = mkep(0)
    write_cam(ep)
    write_joints(ep)
    expected[0] = ("pass", "")

    # ── 1: color head 相机无图像帧 ──
    ep = mkep(1)
    write_cam(ep, overrides={("head", "color"): "skip"})
    write_joints(ep)
    expected[1] = ("failed", "color head 相机无图像帧")

    # ── 2: color 图像无法解码 ──
    ep = mkep(2)
    write_cam(ep, overrides={("head", "color"): "corrupt"})
    write_joints(ep)
    expected[2] = ("failed", "color 图像无法解码")

    # ── 3: color 全黑 ──
    ep = mkep(3)
    write_cam(ep, overrides={("head", "color"): {"rgb": (0, 0, 0)}})
    write_joints(ep)
    expected[3] = ("failed", "color 空图像")

    # ── 4: color 全白 ──
    ep = mkep(4)
    write_cam(ep, overrides={("head", "color"): {"rgb": (255, 255, 255)}})
    write_joints(ep)
    expected[4] = ("failed", "color 空图像")

    # ── 5: color 分辨率错误 ──
    ep = mkep(5)
    write_cam(ep, overrides={("head", "color"): {"w": 320, "h": 240}})
    write_joints(ep)
    expected[5] = ("failed", "color 图像分辨率错误")

    # ── 6: color 相机帧数不一致 (head=30, hand_left=20) ──
    ep = mkep(6)
    ts20 = cam_ts(20)
    write_cam(ep, overrides={
        ("hand_left", "color"): ts20,
        ("hand_left", "depth"): ts20,
    })
    write_joints(ep)
    expected[6] = ("failed", "color 相机帧数不一致")

    # ── 7: color 时间帧不连续 (all gaps = 50ms > 35ms) ──
    ep = mkep(7)
    big_ts = [BASE + i * 50_000_000 for i in range(NF)]
    write_cam(ep, ts=big_ts)
    j_n = int((big_ts[-1] - BASE) / JDT) + 1
    write_joints(ep, ts=joint_ts(n=j_n))
    expected[7] = ("failed", "color 相机时间帧不连续")

    # ── 8: color/depth 帧数不一致 ──
    #  color: all 30, depth: all 25 → cross-check triggers
    ep = mkep(8)
    ts25 = cam_ts(25)
    write_cam(ep, overrides={
        ("head",       "depth"): ts25,
        ("hand_left",  "depth"): ts25,
        ("hand_right", "depth"): ts25,
    })
    write_joints(ep)
    expected[8] = ("failed", "color/depth 帧数不一致")

    # ── 9: depth 全黑 (value <= 100) ──
    ep = mkep(9)
    write_cam(ep, overrides={("head", "depth"): {"v": 50}})
    write_joints(ep)
    expected[9] = ("failed", "depth 空图像")

    # ── 10: 关节状态文件不存在 ──
    ep = mkep(10)
    write_cam(ep)
    expected[10] = ("failed", "关节状态文件不存在")

    # ── 11: 格式错误（缺少空格分隔） ──
    ep = mkep(11)
    write_cam(ep)
    write_joint_raw(ep,
        left_text="no_space_here\n",
        right_text=right_normal_text())
    expected[11] = ("failed", "格式错误")

    # ── 12: 时间戳非法 ──
    ep = mkep(12)
    write_cam(ep)
    v_str = ",".join(str(x) for x in jv_default())
    write_joint_raw(ep,
        left_text=f"abc {v_str}\n",
        right_text=right_normal_text())
    expected[12] = ("failed", "时间戳非法")

    # ── 13: 维度错误 (10 instead of 16) ──
    ep = mkep(13)
    write_cam(ep)
    ts_j = joint_ts()
    bad_lines = "\n".join(f"{t} {','.join(['0.0'] * 10)}" for t in ts_j) + "\n"
    write_joint_raw(ep,
        left_text=bad_lines,
        right_text=right_normal_text())
    expected[13] = ("failed", "维度错误")

    # ── 14: 非法数值 ──
    ep = mkep(14)
    write_cam(ep)
    ts_j = joint_ts()
    bad_vals = "0.1,0.2,abc,0.4,0.5,0.6,0.7,1.0,60.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"
    write_joint_raw(ep,
        left_text=f"{ts_j[0]} {bad_vals}\n",
        right_text=right_normal_text())
    expected[14] = ("failed", "非法数值")

    # ── 15: 关节时间帧不连续 (alternating 20ms / 10ms) ──
    ep = mkep(15)
    write_cam(ep)
    ts_bad = []
    t = BASE
    for i in range(NJ):
        ts_bad.append(t)
        t += 20_000_000 if i % 2 == 0 else JDT
    write_joints(ep, ts=ts_bad)
    expected[15] = ("failed", "关节时间帧不连续")

    # ── 16: 关节状态文件为空 ──
    ep = mkep(16)
    write_cam(ep)
    write_joint_raw(ep, left_text="", right_text="")
    expected[16] = ("failed", "关节状态文件为空")

    # ── 17: 机械臂数据不在相机时间窗内 ──
    ep = mkep(17)
    write_cam(ep)
    far_start = BASE - 200_000_000_000
    write_joints(ep, ts=joint_ts(t0=far_start))
    expected[17] = ("failed", "机械臂数据不在相机时间窗内")

    # ── 18: 图像-机械臂数据量比例超阈值 ──
    #  joint at 14ms → ratio = 30/70 ≈ 0.43 > 5/14 ≈ 0.36
    ep = mkep(18)
    c_ts = cam_ts()
    write_cam(ep, ts=c_ts)
    t_end = c_ts[-1]
    n_j = int((t_end - BASE) / 14_000_000) + 1
    write_joints(ep, ts=joint_ts(n=n_j, dt=14_000_000))
    expected[18] = ("failed", "图像-机械臂数据量比例超阈值")

    # ── 19: 左右机械臂比例不一致 ──
    #  left: 200 rows (~5ms), right: 100 rows (~10ms)
    #  ratio_left ≈ 0.15, ratio_right ≈ 0.30, diff = 0.15 > 0.035
    ep = mkep(19)
    c_ts = cam_ts()
    write_cam(ep, ts=c_ts)
    t_end = c_ts[-1]
    duration = t_end - BASE
    n_left, n_right = 200, 100
    dt_left = duration // (n_left - 1)
    dt_right = duration // (n_right - 1)
    ts_left = [BASE + i * dt_left for i in range(n_left)]
    ts_right = [BASE + i * dt_right for i in range(n_right)]
    lv = [jv_default() for _ in range(n_left)]
    rv = [jv_default() for _ in range(n_right)]
    with open(ep / "record" / "left_data.txt", "w") as f:
        for t, v in zip(ts_left, lv):
            f.write(f"{t} {','.join(str(x) for x in v)}\n")
    with open(ep / "record" / "right_data.txt", "w") as f:
        for t, v in zip(ts_right, rv):
            f.write(f"{t} {','.join(str(x) for x in v)}\n")
    expected[19] = ("failed", "左右机械臂比例不一致")

    # ── 20: 夹爪动作值非法 (action = 0.5) ──
    ep = mkep(20)
    write_cam(ep)
    lv = [jv_default() for _ in range(NJ)]
    lv[50][7] = 0.5
    write_joints(ep, lv=lv)
    expected[20] = ("failed", "夹爪动作值非法")

    # ── 21: 夹爪开度越界 (width = 80 > 70) ──
    ep = mkep(21)
    write_cam(ep)
    lv = [jv_default() for _ in range(NJ)]
    lv[50][8] = 80.0
    write_joints(ep, lv=lv)
    expected[21] = ("failed", "夹爪开度越界")

    # ── 22: 夹爪动作-开度不同步 (action=0 but width increases) ──
    ep = mkep(22)
    write_cam(ep)
    lv = [jv_default() for _ in range(NJ)]
    lv[50][7] = 0.0; lv[50][8] = 10.0
    lv[51][7] = 0.0; lv[51][8] = 50.0   # prev+5=15 < 50 → triggers
    write_joints(ep, lv=lv)
    expected[22] = ("failed", "夹爪动作-开度不同步")

    # ── 23: 夹爪动作-开度不同步 (action=1 but width decreases) ──
    ep = mkep(23)
    write_cam(ep)
    lv = [jv_default() for _ in range(NJ)]
    lv[50][7] = 1.0; lv[50][8] = 50.0
    lv[51][7] = 1.0; lv[51][8] = 10.0   # prev-1=49 > 10 → triggers
    write_joints(ep, lv=lv)
    expected[23] = ("failed", "夹爪动作-开度不同步")

    return expected


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / TASK / "task_info").mkdir(parents=True)

    print("Generating test episodes ...")
    expected = gen_all_cases()

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    print("Creating zip ...")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(WORK):
            for f in files:
                fp = Path(root) / f
                zf.write(fp, fp.relative_to(WORK))

    shutil.rmtree(WORK)
    size_mb = OUTPUT_ZIP.stat().st_size / 1024 / 1024
    print(f"\nCreated {OUTPUT_ZIP} ({size_mb:.1f} MB) with {len(expected)} test episodes\n")

    print(f"{'ID':>4}  {'Expected':>8}  Error keyword")
    print("-" * 60)
    for eid in sorted(expected):
        result, err = expected[eid]
        print(f"{eid:>4}  {result:>8}  {err}")
