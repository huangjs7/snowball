"""绘制 2024 CUMCM A 题“板凳龙”问题 1 的前三幅论文插图。

输出格式：PNG（300 dpi）、SVG、PDF。
"""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
import numpy as np


# 题目参数
PITCH = 0.55
B = PITCH / (2 * np.pi)
THETA_INITIAL = 32 * np.pi
R_INITIAL = B * THETA_INITIAL

# 用户指定色板
DARK_RED = "#d73221"
DARK_ORANGE = "#f79015"
DARK_BLUE = "#4573b4"
DARK_GREEN = "#457635"
DARK_PURPLE = "#4E3282"
LIGHT_RED = "#f57c6e"
LIGHT_ORANGE = "#f2b56e"
LIGHT_YELLOW = "#fbe79e"
LIGHT_GREEN = "#84c3b7"
LIGHT_CYAN = "#88d7da"
LIGHT_BLUE = "#71b8ed"
LIGHT_PURPLE = "#b8aeea"
LIGHT_PINK = "#f2a8da"
INK = "#262626"
GRID = "#d8d8d8"

OUTPUT_DIR = Path(__file__).resolve().parent / "figures_problem1"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "axes.linewidth": 0.8,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def spiral_xy(theta: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    r = B * theta
    return r * np.cos(theta), r * np.sin(theta)


def spiral_derivative(theta: float) -> np.ndarray:
    return B * np.array(
        [np.cos(theta) - theta * np.sin(theta), np.sin(theta) + theta * np.cos(theta)]
    )


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def add_axis_arrows(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    arrow = dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=10)
    ax.annotate("", xy=(xlim[1], 0), xytext=(xlim[0], 0), arrowprops=arrow, zorder=0)
    ax.annotate("", xy=(0, ylim[1]), xytext=(0, ylim[0]), arrowprops=arrow, zorder=0)
    ax.text(xlim[1], -0.035 * (ylim[1] - ylim[0]), "$x$", ha="right", va="top")
    ax.text(0.025 * (xlim[1] - xlim[0]), ylim[1], "$y$", ha="left", va="top")


def add_curve_arrow(
    ax: plt.Axes,
    theta_start: float,
    theta_end: float,
    color: str,
    lw: float = 2.2,
    scale: float = 13,
) -> None:
    x0, y0 = spiral_xy(theta_start)
    x1, y1 = spiral_xy(theta_end)
    arrow = FancyArrowPatch(
        (float(x0), float(y0)),
        (float(x1), float(y1)),
        arrowstyle="-|>",
        mutation_scale=scale,
        lw=lw,
        color=color,
        connectionstyle="arc3,rad=-0.08",
        zorder=6,
    )
    ax.add_patch(arrow)


def add_polar_arc(
    ax: plt.Axes,
    radius: float,
    theta1: float,
    theta2: float,
    color: str,
    label: str,
    label_radius: float | None = None,
    lw: float = 1.3,
) -> None:
    deg1, deg2 = np.degrees([theta1, theta2])
    if deg2 < deg1:
        deg2 += 360
    ax.add_patch(
        Arc((0, 0), 2 * radius, 2 * radius, theta1=deg1, theta2=deg2, lw=lw, color=color)
    )
    mid = 0.5 * (theta1 + theta2)
    rr = radius * 1.10 if label_radius is None else label_radius
    if label:
        ax.text(rr * np.cos(mid), rr * np.sin(mid), label, color=color, ha="center", va="center")


def solve_outer_theta(theta_i: float, chord_length: float) -> float:
    """求满足固定弦长的最近外侧根 theta_{i+1}>theta_i。"""

    p_i = np.array(spiral_xy(theta_i), dtype=float)

    def residual(theta: float) -> float:
        p = np.array(spiral_xy(theta), dtype=float)
        return float(np.linalg.norm(p - p_i) - chord_length)

    samples = theta_i + np.linspace(1e-6, 2 * np.pi, 4001)
    values = np.array([residual(value) for value in samples])
    crossings = np.flatnonzero(values >= 0)
    if crossings.size == 0:
        raise RuntimeError("未在给定区间找到外侧根")

    hi_index = int(crossings[0])
    lo = theta_i if hi_index == 0 else float(samples[hi_index - 1])
    hi = float(samples[hi_index])
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if residual(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def figure_1_parameter_definition() -> None:
    fig, (ax_overview, ax_local) = plt.subplots(
        1, 2, figsize=(12.6, 5.6), gridspec_kw={"width_ratios": [1.12, 0.88]}
    )

    # (a) 16 圈完整螺线与初始盘入方向
    theta = np.linspace(0.03, THETA_INITIAL, 6000)
    x, y = spiral_xy(theta)
    ax_overview.plot(x, y, color=LIGHT_BLUE, lw=1.15, zorder=1)
    outer_theta = np.linspace(30.6 * np.pi, THETA_INITIAL, 500)
    xo, yo = spiral_xy(outer_theta)
    ax_overview.plot(xo, yo, color=DARK_BLUE, lw=1.8, zorder=2)

    ax_overview.scatter([0], [0], s=28, color=DARK_PURPLE, zorder=7)
    ax_overview.text(0.22, 0.22, "$O$", color=DARK_PURPLE, weight="bold")
    ax_overview.scatter([R_INITIAL], [0], s=58, color=DARK_RED, edgecolor="white", lw=0.7, zorder=8)
    ax_overview.annotate(
        "$A=P_0(0)=(8.8,0)$\n$\\theta_0(0)=32\\pi$",
        xy=(R_INITIAL, 0),
        xytext=(4.5, 2.0),
        arrowprops=dict(arrowstyle="->", color=DARK_RED, lw=1.1),
        bbox=dict(boxstyle="round,pad=0.25", facecolor=LIGHT_YELLOW, edgecolor="none", alpha=0.88),
        color=INK,
        ha="left",
        va="bottom",
    )
    add_curve_arrow(ax_overview, THETA_INITIAL - 0.08, THETA_INITIAL - 0.70, DARK_ORANGE)
    ax_overview.text(
        5.0,
        -6.5,
        "顺时针盘入\n($\\theta$ 减小)",
        color=DARK_ORANGE,
        weight="bold",
        ha="center",
        va="center",
    )

    overview_limit = 9.55
    add_axis_arrows(ax_overview, (-overview_limit, overview_limit), (-overview_limit, overview_limit))
    ax_overview.set_xlim(-overview_limit, overview_limit)
    ax_overview.set_ylim(-overview_limit, overview_limit)
    ax_overview.set_aspect("equal")
    ax_overview.set_xticks([])
    ax_overview.set_yticks([])
    ax_overview.set_title("(a) 16 圈螺线与龙头初始位置", loc="left", pad=8)
    for spine in ax_overview.spines.values():
        spine.set_visible(False)

    # (b) 参数定义局部图
    theta_local = np.linspace(0.02, 3.55, 700)
    xl, yl = spiral_xy(theta_local)
    ax_local.plot(xl, yl, color=LIGHT_GREEN, lw=2.1, zorder=2)
    theta_p = 2.34
    px, py = map(float, spiral_xy(theta_p))
    ax_local.plot([0, px], [0, py], color=DARK_PURPLE, lw=2.0, zorder=4)
    ax_local.add_patch(
        FancyArrowPatch(
            (0, 0), (px, py), arrowstyle="-|>", mutation_scale=12, color=DARK_PURPLE, lw=1.7, zorder=5
        )
    )
    ax_local.scatter([0], [0], s=30, color=DARK_PURPLE, zorder=6)
    ax_local.text(0.012, -0.025, "$O$", color=DARK_PURPLE, weight="bold")
    ax_local.scatter([px], [py], s=60, color=DARK_RED, edgecolor="white", lw=0.7, zorder=7)
    ax_local.annotate(
        "$P(\\theta)$",
        xy=(px, py),
        xytext=(px - 0.085, py + 0.07),
        arrowprops=dict(arrowstyle="->", color=DARK_RED, lw=1.0),
        color=DARK_RED,
        weight="bold",
    )
    ax_local.text(0.53 * px - 0.03, 0.53 * py + 0.025, "$r=b\\theta$", color=DARK_PURPLE, rotation=-43)
    add_polar_arc(ax_local, 0.075, 0, theta_p, DARK_ORANGE, "$\\theta$", label_radius=0.098)
    ax_local.text(
        0.02,
        0.95,
        "$r=b\\theta,\\quad b=\\dfrac{0.55}{2\\pi}$\n"
        "$x=b\\theta\\cos\\theta,\\quad y=b\\theta\\sin\\theta$",
        transform=ax_local.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=LIGHT_PINK, edgecolor="none", alpha=0.72),
        color=INK,
    )
    local_limit = 0.34
    add_axis_arrows(ax_local, (-local_limit, local_limit), (-local_limit, local_limit))
    ax_local.set_xlim(-local_limit, local_limit)
    ax_local.set_ylim(-local_limit, local_limit)
    ax_local.set_aspect("equal")
    ax_local.set_xticks([])
    ax_local.set_yticks([])
    ax_local.set_title("(b) 极坐标参数的局部定义", loc="left", pad=8)
    for spine in ax_local.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.025, right=0.985, top=0.92, bottom=0.04, wspace=0.12)
    save_figure(fig, "fig1_archimedean_spiral_definition")


def representative_geometry() -> tuple[float, float, np.ndarray, np.ndarray]:
    theta_i = 15.0
    theta_next = solve_outer_theta(theta_i, chord_length=1.65)
    p_i = np.array(spiral_xy(theta_i), dtype=float)
    p_next = np.array(spiral_xy(theta_next), dtype=float)
    return theta_i, theta_next, p_i, p_next


def figure_2_chord_constraint() -> None:
    theta_i, theta_next, p_i, p_next = representative_geometry()
    fig, ax = plt.subplots(figsize=(7.7, 6.6))

    theta = np.linspace(0.02, theta_next + 0.75, 2600)
    x, y = spiral_xy(theta)
    ax.plot(x, y, color=LIGHT_GREEN, lw=1.55, zorder=1)

    theta_segment = np.linspace(theta_i, theta_next, 350)
    xs, ys = spiral_xy(theta_segment)
    ax.plot(xs, ys, color=DARK_ORANGE, lw=2.0, ls=(0, (4, 3)), zorder=3)

    # 极径与固定弦长
    ax.plot([0, p_i[0]], [0, p_i[1]], color=DARK_BLUE, lw=1.55, zorder=2)
    ax.plot([0, p_next[0]], [0, p_next[1]], color=DARK_GREEN, lw=1.55, zorder=2)
    ax.plot([p_i[0], p_next[0]], [p_i[1], p_next[1]], color=DARK_PURPLE, lw=3.0, zorder=5)

    ax.scatter([0], [0], s=32, color=DARK_PURPLE, zorder=7)
    ax.text(0.06, 0.04, "$O$", color=DARK_PURPLE, weight="bold")
    ax.scatter([p_i[0]], [p_i[1]], s=70, color=DARK_BLUE, edgecolor="white", lw=0.8, zorder=8)
    ax.scatter([p_next[0]], [p_next[1]], s=70, color=DARK_GREEN, edgecolor="white", lw=0.8, zorder=8)
    ax.annotate(
        "$P_i$",
        xy=p_i,
        xytext=(p_i[0] + 0.12, p_i[1] + 0.14),
        arrowprops=dict(arrowstyle="->", color=DARK_BLUE, lw=1.0),
        color=DARK_BLUE,
        weight="bold",
    )
    ax.annotate(
        "$P_{i+1}$",
        xy=p_next,
        xytext=(p_next[0] - 0.42, p_next[1] - 0.19),
        arrowprops=dict(arrowstyle="->", color=DARK_GREEN, lw=1.0),
        color=DARK_GREEN,
        weight="bold",
    )

    mid_chord = 0.5 * (p_i + p_next)
    chord_angle = np.degrees(np.arctan2(*(p_next - p_i)[::-1]))
    ax.text(
        mid_chord[0] + 0.10,
        mid_chord[1],
        "$|P_iP_{i+1}|=L_i$",
        color=DARK_PURPLE,
        rotation=chord_angle,
        rotation_mode="anchor",
        ha="left",
        va="bottom",
        weight="bold",
    )
    ax.text(0.48 * p_i[0] - 0.03, 0.48 * p_i[1] + 0.06, "$r_i=b\\theta_i$", color=DARK_BLUE)
    ax.text(
        0.53 * p_next[0] + 0.03,
        0.53 * p_next[1] - 0.04,
        "$r_{i+1}=b\\theta_{i+1}$",
        color=DARK_GREEN,
        ha="right",
    )

    # 角度均按等效极角展示；三层圆弧避免符号混淆
    angle_i = theta_i % (2 * np.pi)
    angle_next = theta_next % (2 * np.pi)
    if angle_next < angle_i:
        angle_next += 2 * np.pi
    add_polar_arc(ax, 0.32, 0, angle_i, DARK_BLUE, "$\\theta_i$", label_radius=0.39)
    add_polar_arc(ax, 0.49, 0, angle_next, DARK_GREEN, "$\\theta_{i+1}$", label_radius=0.57)
    add_polar_arc(
        ax,
        0.74,
        angle_i,
        angle_next,
        DARK_ORANGE,
        "",
        label_radius=0.88,
        lw=1.8,
    )
    ax.text(
        -1.10,
        0.16,
        "$\\Delta\\theta_i=\\theta_{i+1}-\\theta_i$",
        color=DARK_ORANGE,
        ha="center",
        va="center",
    )

    ax.text(
        0.97,
        0.06,
        "$L_0=2.86\\,\\mathrm{m}$\n$L_i=1.65\\,\\mathrm{m}\quad(i\\geq1)$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=LIGHT_YELLOW, edgecolor="none", alpha=0.88),
        color=INK,
    )
    ax.annotate(
        "螺线弧（非杆长）",
        xy=(xs[len(xs) // 2], ys[len(ys) // 2]),
        xytext=(0.67, 0.88),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=DARK_ORANGE, lw=1.0),
        color=DARK_ORANGE,
        ha="center",
    )

    limit = 1.72
    add_axis_arrows(ax, (-limit, limit), (-limit, limit))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    save_figure(fig, "fig2_adjacent_handle_chord_constraint")


def _closest_line_angle(line_angle: float, target_angle: float) -> float:
    candidates = [line_angle + k * np.pi for k in range(-2, 3)]
    return min(candidates, key=lambda value: abs(value - target_angle))


def add_local_angle(
    ax: plt.Axes,
    center: np.ndarray,
    angle_a: float,
    angle_b: float,
    radius: float,
    label: str,
    color: str,
) -> None:
    angle_b = _closest_line_angle(angle_b, angle_a)
    low, high = sorted([angle_a, angle_b])
    ax.add_patch(
        Arc(
            tuple(center),
            2 * radius,
            2 * radius,
            theta1=np.degrees(low),
            theta2=np.degrees(high),
            color=color,
            lw=1.7,
            zorder=8,
        )
    )
    mid = 0.5 * (low + high)
    pos = center + 1.25 * radius * np.array([np.cos(mid), np.sin(mid)])
    ax.text(pos[0], pos[1], label, color=color, ha="center", va="center", weight="bold", zorder=9)


def figure_3_velocity_transfer() -> None:
    theta_i, theta_next, p_i, p_next = representative_geometry()
    fig, ax = plt.subplots(figsize=(8.4, 6.2))

    theta = np.linspace(theta_i - 0.95, theta_next + 0.95, 900)
    x, y = spiral_xy(theta)
    ax.plot(x, y, color=LIGHT_CYAN, lw=2.2, zorder=1)

    delta = p_next - p_i
    delta_unit = unit(delta)
    rod_angle = math.atan2(delta_unit[1], delta_unit[0])
    ax.add_patch(
        FancyArrowPatch(
            tuple(p_i),
            tuple(p_next),
            arrowstyle="-|>",
            mutation_scale=15,
            color=DARK_PURPLE,
            lw=3.0,
            zorder=5,
        )
    )
    mid = 0.5 * (p_i + p_next)
    ax.text(
        mid[0] + 0.10,
        mid[1] - 0.03,
        "$\\mathbf{\\Delta}_i=\\mathbf{P}_{i+1}-\\mathbf{P}_i$",
        color=DARK_PURPLE,
        rotation=np.degrees(rod_angle),
        rotation_mode="anchor",
        ha="left",
        va="top",
        weight="bold",
    )

    velocity_scale = 0.95
    tangent_span = 1.15
    colors = [DARK_BLUE, DARK_GREEN]
    labels = ["$P_i$", "$P_{i+1}$"]
    velocity_labels = ["$\\mathbf{v}_i$", "$\\mathbf{v}_{i+1}$"]
    angle_labels = ["$\\alpha_i$", "$\\alpha_{i+1}$"]
    centers = [p_i, p_next]
    theta_values = [theta_i, theta_next]

    for idx, (point, theta_value, color) in enumerate(zip(centers, theta_values, colors)):
        tangent = unit(spiral_derivative(theta_value))
        velocity_dir = -tangent  # 顺时针盘入时 theta 减小
        ax.plot(
            [point[0] - tangent_span * tangent[0], point[0] + tangent_span * tangent[0]],
            [point[1] - tangent_span * tangent[1], point[1] + tangent_span * tangent[1]],
            color=color,
            lw=1.25,
            ls=(0, (4, 3)),
            alpha=0.78,
            zorder=2,
        )
        velocity_end = point + velocity_scale * velocity_dir
        ax.add_patch(
            FancyArrowPatch(
                tuple(point),
                tuple(velocity_end),
                arrowstyle="-|>",
                mutation_scale=14,
                color=color,
                lw=2.4,
                zorder=7,
            )
        )
        ax.scatter([point[0]], [point[1]], s=72, color=color, edgecolor="white", lw=0.8, zorder=8)
        node_offset = np.array([-0.23, 0.10]) if idx == 0 else np.array([-0.25, 0.10])
        node_label_pos = point + node_offset
        ax.text(node_label_pos[0], node_label_pos[1], labels[idx], color=color, weight="bold")
        ax.text(
            velocity_end[0] + (0.04 if idx == 0 else -0.16),
            velocity_end[1] + 0.05,
            velocity_labels[idx],
            color=color,
            weight="bold",
        )
        velocity_angle = math.atan2(velocity_dir[1], velocity_dir[0])
        add_local_angle(ax, point, velocity_angle, rod_angle, 0.23, angle_labels[idx], DARK_ORANGE)

    ax.text(
        0.03,
        0.06,
        "$v_i\\cos\\alpha_i=v_{i+1}\\cos\\alpha_{i+1}$\n"
        "$\\mathbf{\\Delta}_i\\cdot(\\mathbf{v}_{i+1}-\\mathbf{v}_i)=0$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.38", facecolor=LIGHT_YELLOW, edgecolor="none", alpha=0.88),
        color=INK,
    )
    ax.text(
        0.97,
        0.94,
        "虚线：螺线切向方向",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=DARK_GREEN,
    )

    all_points = np.vstack([np.column_stack([x, y]), p_i, p_next])
    x_center, y_center = np.mean([p_i, p_next], axis=0)
    ax.set_xlim(x_center - 1.75, x_center + 1.75)
    ax.set_ylim(y_center - 1.55, y_center + 1.55)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.98, bottom=0.03)
    save_figure(fig, "fig3_adjacent_handle_velocity_transfer")


def main() -> None:
    configure_style()
    figure_1_parameter_definition()
    figure_2_chord_constraint()
    figure_3_velocity_transfer()
    print(f"已生成图 1～3：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
