"""
插值方法对比可视化：PCHIP vs CubicSpline vs np.interp (线性)
对应机器人关节轨迹上采样场景（模拟 pi05_infer.py 中 action_mode 5/6 的插值）
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import PchipInterpolator, CubicSpline

matplotlib.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 测试数据 ────────────────────────────────────────────────────────────────
# 模拟一段典型关节轨迹的稀疏关键帧（16 Hz 推理，10x 上采样至 ~160 Hz）
INTERPOLATION_SCALE = 10

# 场景 A：平滑曲线（正常抓取动作）
np.random.seed(42)
src_steps_A = 8
t_src_A = np.linspace(0.0, 1.0, src_steps_A)
vals_A = np.array([0.0, 0.3, 0.8, 1.2, 1.0, 0.6, 0.2, 0.0])  # 单关节示例

# 场景 B：含急变（抓手开合 or 关节急跳）
src_steps_B = 8
t_src_B = np.linspace(0.0, 1.0, src_steps_B)
vals_B = np.array([0.0, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.0])  # 阶跃式抓手

# 场景 C：多维关节（展示整体轨迹误差）
src_steps_C = 10
t_src_C = np.linspace(0.0, 1.0, src_steps_C)
joints_C = np.stack(
    [
        np.sin(np.linspace(0, np.pi, src_steps_C)),
        np.cos(np.linspace(0, np.pi, src_steps_C)) * 0.5,
        np.linspace(0.4, -0.4, src_steps_C),
        np.sin(np.linspace(0, 2 * np.pi, src_steps_C)) * 0.3,
    ],
    axis=1,
)  # (10, 4)


def interp_all(t_src, vals, scale):
    target_steps = int(scale * len(t_src))
    t_dst = np.linspace(0.0, 1.0, target_steps)

    # PCHIP
    pchip_vals = PchipInterpolator(t_src, vals)(t_dst)

    # Cubic Spline (natural boundary)
    cs_vals = CubicSpline(t_src, vals, bc_type="natural")(t_dst)

    # np.interp (线性)
    linear_vals = np.interp(t_dst, t_src, vals)

    return t_dst, pchip_vals, cs_vals, linear_vals


# ── 绘图 ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle(
    "插值方法对比：PCHIP  vs  CubicSpline  vs  np.interp（线性）\n"
    "模拟机器人关节轨迹上采样（pi05_infer.py action_mode 5/6）",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

COLORS = {
    "pchip": "#E74C3C",
    "spline": "#3498DB",
    "linear": "#2ECC71",
    "keyframe": "#2C3E50",
}

# ── 子图 1：场景 A 插值曲线 ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
t_dst_A, pchip_A, cs_A, lin_A = interp_all(t_src_A, vals_A, INTERPOLATION_SCALE)

ax1.plot(t_dst_A, pchip_A, color=COLORS["pchip"], lw=2, label="PCHIP（形状保持）")
ax1.plot(t_dst_A, cs_A, color=COLORS["spline"], lw=2, ls="--", label="CubicSpline（自然边界）")
ax1.plot(t_dst_A, lin_A, color=COLORS["linear"], lw=2, ls=":", label="np.interp（线性）")
ax1.scatter(t_src_A, vals_A, color=COLORS["keyframe"], zorder=5, s=60, label="原始关键帧")
ax1.set_title("场景 A：平滑曲线（正常抓取动作，8 帧 → 80 帧）", fontsize=11)
ax1.set_xlabel("归一化时间")
ax1.set_ylabel("关节角度（rad）")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3)

# 标注过冲区域
ax1.axhline(max(vals_A), color="gray", lw=0.8, ls="-.", alpha=0.6)
ax1.axhline(min(vals_A), color="gray", lw=0.8, ls="-.", alpha=0.6)
ax1.annotate(
    "CubicSpline 可能过冲\n超过关键帧极值",
    xy=(0.38, cs_A.max()),
    xytext=(0.42, cs_A.max() + 0.05),
    fontsize=8,
    color=COLORS["spline"],
    arrowprops=dict(arrowstyle="->", color=COLORS["spline"]),
)

# ── 子图 2：场景 B 急变曲线（抓手开合）─────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
t_dst_B, pchip_B, cs_B, lin_B = interp_all(t_src_B, vals_B, INTERPOLATION_SCALE)

ax2.plot(t_dst_B, pchip_B, color=COLORS["pchip"], lw=2, label="PCHIP")
ax2.plot(t_dst_B, cs_B, color=COLORS["spline"], lw=2, ls="--", label="CubicSpline")
ax2.plot(t_dst_B, lin_B, color=COLORS["linear"], lw=2, ls=":", label="np.interp")
ax2.scatter(t_src_B, vals_B, color=COLORS["keyframe"], zorder=5, s=60)
ax2.set_title("场景 B：急变（抓手开合阶跃）", fontsize=11)
ax2.set_xlabel("归一化时间")
ax2.set_ylabel("夹爪开度（归一化）")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.3, 1.3)
ax2.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
ax2.axhline(1, color="gray", lw=0.5, ls="--", alpha=0.5)

# ── 子图 3：速度对比（一阶导数）────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
dt = t_dst_A[1] - t_dst_A[0]
vel_pchip = np.gradient(pchip_A, dt)
vel_cs = np.gradient(cs_A, dt)
vel_lin = np.gradient(lin_A, dt)

ax3.plot(t_dst_A, vel_pchip, color=COLORS["pchip"], lw=2, label="PCHIP 速度")
ax3.plot(t_dst_A, vel_cs, color=COLORS["spline"], lw=2, ls="--", label="CubicSpline 速度")
ax3.plot(t_dst_A, vel_lin, color=COLORS["linear"], lw=2, ls=":", label="np.interp 速度")
# 标记关键帧位置
for t in t_src_A:
    ax3.axvline(t, color="gray", lw=0.5, ls="--", alpha=0.4)
ax3.set_title("场景 A：速度剖面（一阶导数）\n线性插值在关键帧处有速度突变", fontsize=11)
ax3.set_xlabel("归一化时间")
ax3.set_ylabel("角速度（rad/s）")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ── 子图 4：多维关节 RMS 误差热图 ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])

# 以更密的参考点作为"ground truth"进行对比
t_ref = np.linspace(0.0, 1.0, 1000)

# 用高密度 PCHIP 作为参考基准（接近真实轨迹意图）
pchip_ref_list, cs_ref_list, lin_ref_list = [], [], []
for d in range(joints_C.shape[1]):
    pchip_ref = PchipInterpolator(t_src_C, joints_C[:, d])(t_ref)
    cs_ref = CubicSpline(t_src_C, joints_C[:, d], bc_type="natural")(t_ref)
    lin_ref = np.interp(t_ref, t_src_C, joints_C[:, d])
    pchip_ref_list.append(pchip_ref)
    cs_ref_list.append(cs_ref)
    lin_ref_list.append(lin_ref)

pchip_arr = np.stack(pchip_ref_list, axis=1)
cs_arr = np.stack(cs_ref_list, axis=1)
lin_arr = np.stack(lin_ref_list, axis=1)

# 以 PCHIP 为基准，计算 CubicSpline 和 np.interp 偏差
err_cs = np.sqrt(np.mean((cs_arr - pchip_arr) ** 2, axis=0))
err_lin = np.sqrt(np.mean((lin_arr - pchip_arr) ** 2, axis=0))

joint_names = ["Joint-1\n(sin)", "Joint-2\n(cos)", "Joint-3\n(线性)", "Joint-4\n(sin2x)"]
x = np.arange(len(joint_names))
width = 0.35
bars1 = ax4.bar(x - width / 2, err_cs, width, color=COLORS["spline"], label="CubicSpline vs PCHIP")
bars2 = ax4.bar(x + width / 2, err_lin, width, color=COLORS["linear"], label="np.interp vs PCHIP")
ax4.set_title("多维关节：各方法相对 PCHIP 的 RMS 误差", fontsize=11)
ax4.set_ylabel("RMS 误差（rad）")
ax4.set_xticks(x)
ax4.set_xticklabels(joint_names, fontsize=9)
ax4.legend(fontsize=9)
ax4.grid(True, axis="y", alpha=0.3)
for bar in bars1:
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
             f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7)
for bar in bars2:
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
             f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7)

# ── 子图 5：特性总结对比表 ──────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis("off")

table_data = [
    ["特性", "PCHIP", "CubicSpline", "np.interp"],
    ["连续阶数", "C¹（一阶连续）", "C²（二阶连续）", "C⁰（仅连续）"],
    ["是否过冲", "O 不过冲", "! 可能过冲", "O 不过冲"],
    ["速度平滑", "O 平滑", "O 更平滑", "X 有突变"],
    ["急变适应", "O 好", "~ 一般", "O 好"],
    ["计算量", "中等", "中等", "极低"],
    ["适用场景", "抓取/关节轨迹", "平滑曲线拟合", "简单快速插值"],
    ["pi05 使用", "mode 5/6 ✅", "未使用", "未使用"],
]

colors_table = []
for i, row in enumerate(table_data):
    if i == 0:
        colors_table.append(["#2C3E50"] * 4)
    else:
        base = ["#f5f5f5", "#FADBD8", "#D6EAF8", "#D5F5E3"]
        colors_table.append(base)

tbl = ax5.table(
    cellText=table_data,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.6)
# 表头加粗
for j in range(4):
    tbl[0, j].set_facecolor("#2C3E50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
# PCHIP 列高亮
for i in range(1, len(table_data)):
    tbl[i, 0].set_facecolor("#ECF0F1")
    tbl[i, 1].set_facecolor("#FADBD8")
    tbl[i, 2].set_facecolor("#D6EAF8")
    tbl[i, 3].set_facecolor("#D5F5E3")

ax5.set_title("方法特性对比", fontsize=11, pad=12)

plt.savefig("interp_comparison.png", dpi=150, bbox_inches="tight")
print("图像已保存至 interp_comparison.png")
plt.show()
