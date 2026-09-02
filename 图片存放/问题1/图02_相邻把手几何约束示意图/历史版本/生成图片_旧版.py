"""绘制图2：相邻把手几何约束示意图。

运行：python draw_fig2.py
输出：当前“历史版本”目录中的旧版 PNG
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc


# 中文与数学符号显示设置（Windows 常见字体按顺序回退）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"


def polar_point(b: float, theta: float) -> np.ndarray:
    """返回阿基米德螺线 r=bθ 在角度 theta 处的直角坐标。"""
    radius = b * theta
    return np.array([radius * np.cos(theta), radius * np.sin(theta)])


def solve_outer_theta(b: float, theta_i: float, target_distance: float) -> float:
    """求螺线外侧第一个与 P_i 相距 target_distance 的点。"""
    p_i = polar_point(b, theta_i)

    def residual(theta: float) -> float:
        return np.linalg.norm(polar_point(b, theta) - p_i) - target_distance

    samples = np.linspace(theta_i + 1e-8, theta_i + 2 * np.pi, 4000)
    values = np.array([residual(value) for value in samples])
    crossings = np.flatnonzero(values[:-1] * values[1:] <= 0)
    if crossings.size == 0:
        raise ValueError("在一圈范围内未找到满足定长约束的外侧把手点")

    lower = samples[crossings[0]]
    upper = samples[crossings[0] + 1]
    for _ in range(70):
        middle = (lower + upper) / 2
        if residual(lower) * residual(middle) <= 0:
            upper = middle
        else:
            lower = middle
    return (lower + upper) / 2


def main() -> None:
    # 题目给定螺距为 0.55 m，故阿基米德螺线参数 b=p/(2π)
    pitch = 0.55
    b = pitch / (2 * np.pi)

    # 以一节龙身板凳为例，其两个把手中心的固定直线距离为 1.65 m
    target_distance = 1.65
    theta_i = 20.0
    theta_next = solve_outer_theta(b, theta_i, target_distance)

    theta = np.linspace(0, theta_next + 0.8, 1800)
    radius = b * theta
    x, y = radius * np.cos(theta), radius * np.sin(theta)

    p_i = polar_point(b, theta_i)
    p_next = polar_point(b, theta_next)
    segment_length = np.linalg.norm(p_next - p_i)

    fig, ax = plt.subplots(figsize=(8.2, 6.4), constrained_layout=True)
    ax.plot(x, y, color="#4573B4", linewidth=2.2, label="阿基米德螺线  $r=b\\theta$")

    # 原点至两把手的极径，以及相邻把手间固定长度约束
    ax.plot([0, p_i[0]], [0, p_i[1]], color="#777777", linewidth=1.2, zorder=1)
    ax.plot([0, p_next[0]], [0, p_next[1]], color="#777777", linewidth=1.2, zorder=1)
    ax.plot(
        [p_i[0], p_next[0]],
        [p_i[1], p_next[1]],
        color="#D83A2E",
        linewidth=3.5,
        zorder=3,
    )
    ax.scatter(
        [p_i[0], p_next[0]], [p_i[1], p_next[1]], s=70, color="#D83A2E", zorder=4
    )
    ax.scatter(0, 0, s=28, color="#333333", zorder=4)

    # 极角差 Δθ 的弧线标注
    arc_radius = 0.32
    angle_i = np.degrees(theta_i % (2 * np.pi))
    angle_next = angle_i + np.degrees(theta_next - theta_i)
    arc = Arc(
        (0, 0), 2 * arc_radius, 2 * arc_radius,
        theta1=angle_i, theta2=angle_next, color="#555555", linewidth=1.4,
    )
    ax.add_patch(arc)
    theta_mid = (theta_i + theta_next) / 2
    ax.text(
        0.44 * np.cos(theta_mid), 0.44 * np.sin(theta_mid),
        r"$\Delta\theta_i$", fontsize=13, ha="center", va="center",
    )

    # 点、极径和长度的标注
    ax.annotate(
        r"$P_i$", p_i, xytext=p_i + np.array([0.08, -0.16]), textcoords="data",
        ha="center", va="center", fontsize=14,
    )
    ax.annotate(
        r"$P_{i+1}$", p_next, xytext=p_next + np.array([-0.02, -0.17]), textcoords="data",
        ha="center", va="center", fontsize=14,
    )
    ax.annotate(
        "极点 O", (0, 0), xytext=(10, -4), textcoords="offset points",
        ha="left", va="center", fontsize=11,
    )
    midpoint = (p_i + p_next) / 2
    ax.annotate(
        rf"$d_i=|P_iP_{{i+1}}|={segment_length:.2f}\,\mathrm{{m}}$",
        midpoint, xytext=(0, -32), textcoords="offset points", ha="center",
        color="#B5241C", fontsize=13,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("图2  相邻把手几何约束示意图", fontsize=16, pad=30)
    ax.text(
        0.5, 1.012,
        "$d_0=2.86\,\mathrm{m}$（龙头）；"
        "$d_i=1.65\,\mathrm{m}$（龙身、龙尾，$i\geq1$）",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5,
    )
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="lower right", frameon=False)

    output_path = Path(__file__).resolve().parent / "图02_相邻把手几何约束示意图_旧版.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"已生成：{output_path}")


if __name__ == "__main__":
    main()
