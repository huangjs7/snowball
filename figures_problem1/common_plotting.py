"""绘制 2024 CUMCM A 题“板凳龙”问题 1 的前三幅论文插图。

输出格式：PNG（300 dpi）、SVG、PDF。
"""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon
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

OUTPUT_DIR = Path(__file__).resolve().parent


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
        "$A=P_0(0)$\n$\\theta_0(0)=32\\pi,\\quad r_0(0)=8.8\\,\\mathrm{m}$",
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
        "龙头顺时针盘入",
        color=DARK_ORANGE,
        weight="bold",
        ha="center",
        va="center",
    )
    ax_overview.text(
        5.0,
        -7.45,
        "盘入方向：$\\theta\\downarrow$",
        color=DARK_RED,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=LIGHT_RED, edgecolor="none", alpha=0.50),
    )

    overview_limit = 9.55
    add_axis_arrows(ax_overview, (-overview_limit, overview_limit), (-overview_limit, overview_limit))
    ax_overview.set_xlim(-overview_limit, overview_limit)
    ax_overview.set_ylim(-overview_limit, overview_limit)
    ax_overview.set_aspect("equal")
    ax_overview.set_xticks([])
    ax_overview.set_yticks([])
    ax_overview.set_title("(a) 完整螺线、初始点与盘入方向", loc="left", pad=8)
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
    arc_start = theta_p - 0.18
    ax_local.add_patch(
        FancyArrowPatch(
            (0.075 * np.cos(arc_start), 0.075 * np.sin(arc_start)),
            (0.075 * np.cos(theta_p), 0.075 * np.sin(theta_p)),
            arrowstyle="-|>",
            mutation_scale=10,
            color=DARK_ORANGE,
            lw=1.3,
            connectionstyle="arc3,rad=0.08",
            zorder=6,
        )
    )
    ax_local.text(
        0.02,
        0.95,
        "$r=b\\theta,\\qquad b=\\dfrac{0.55}{2\\pi}$",
        transform=ax_local.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=LIGHT_PINK, edgecolor="none", alpha=0.72),
        color=INK,
    )
    ax_local.text(
        0.50,
        0.08,
        "$\\theta$：从正 $x$ 轴逆时针量到 $OP$ 的极角",
        transform=ax_local.transAxes,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.30", facecolor=LIGHT_YELLOW, edgecolor="none", alpha=0.82),
        color=INK,
    )
    local_limit = 0.34
    add_axis_arrows(ax_local, (-local_limit, local_limit), (-local_limit, local_limit))
    ax_local.set_xlim(-local_limit, local_limit)
    ax_local.set_ylim(-local_limit, local_limit)
    ax_local.set_aspect("equal")
    ax_local.set_xticks([])
    ax_local.set_yticks([])
    ax_local.set_title("(b) 极角与极径的符号定义", loc="left", pad=8)
    for spine in ax_local.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.025, right=0.985, top=0.92, bottom=0.04, wspace=0.12)
    save_figure(fig, "fig1_archimedean_spiral_parameters")


def _bench_corners(p_start: np.ndarray, p_end: np.ndarray, total_length: float) -> np.ndarray:
    """按真实长、宽生成板凳俯视矩形，孔中心位于短边内侧 0.275 m。"""

    direction = unit(p_end - p_start)
    normal = np.array([-direction[1], direction[0]])
    center = 0.5 * (p_start + p_end)
    half_length = 0.5 * total_length
    half_width = 0.15
    return np.array(
        [
            center - half_length * direction - half_width * normal,
            center + half_length * direction - half_width * normal,
            center + half_length * direction + half_width * normal,
            center - half_length * direction + half_width * normal,
        ]
    )


def _to_chord_coordinates(points: np.ndarray, p_start: np.ndarray, p_end: np.ndarray) -> np.ndarray:
    """将点变换到以直线弦为横轴的局部坐标系。"""

    direction = unit(p_end - p_start)
    normal = np.array([-direction[1], direction[0]])
    relative = np.asarray(points) - p_start
    return np.column_stack([relative @ direction, relative @ normal])


def _dimension_arrow(
    ax: plt.Axes,
    x_start: float,
    x_end: float,
    y: float,
    label: str,
    color: str,
    label_above: bool = True,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x_start, y),
            (x_end, y),
            arrowstyle="<->",
            mutation_scale=11,
            color=color,
            lw=1.35,
            zorder=8,
        )
    )
    offset = 0.055 if label_above else -0.055
    ax.text(
        0.5 * (x_start + x_end),
        y + offset,
        label,
        color=color,
        ha="center",
        va="bottom" if label_above else "top",
        weight="bold",
    )


def _figure_1_bench_scale_legacy() -> None:
    """真实尺度总览与直线弦长约束的局部放大图。"""

    fig, (ax_overview, ax_local) = plt.subplots(
        1,
        2,
        figsize=(13.4, 6.2),
        gridspec_kw={"width_ratios": [1.08, 0.92]},
    )

    # ---------- (a) 8.8 m 圆域内按真实比例布置多节板凳 ----------
    theta_spiral = np.linspace(0.03, THETA_INITIAL, 6500)
    spiral_x, spiral_y = spiral_xy(theta_spiral)
    ax_overview.plot(spiral_x, spiral_y, color=LIGHT_CYAN, lw=0.82, alpha=0.48, zorder=1)
    ax_overview.add_patch(
        Circle((0, 0), R_INITIAL, fill=False, edgecolor=DARK_BLUE, lw=2.0, zorder=2)
    )

    # P0 位于圆域内侧；节点按题目编号向外递推，保证多节实体均落在 8.8 m 圆域内。
    theta_nodes = [THETA_INITIAL - 3.20]
    handle_lengths = [2.86, 1.65, 1.65, 1.65, 1.65]
    for handle_length in handle_lengths:
        theta_nodes.append(solve_outer_theta(theta_nodes[-1], handle_length))
    node_points = np.array([spiral_xy(value) for value in theta_nodes], dtype=float)
    total_lengths = [3.41, 2.20, 2.20, 2.20, 2.20]

    selected_index = 2
    selected_midpoint = None
    head_midpoint = None
    body_midpoint = None
    for index, total_length in enumerate(total_lengths):
        p_start, p_end = node_points[index], node_points[index + 1]
        is_head = index == 0
        edge_color = DARK_RED if is_head else DARK_BLUE
        face_color = LIGHT_RED if is_head else LIGHT_BLUE
        corners = _bench_corners(p_start, p_end, total_length)
        ax_overview.add_patch(
            Polygon(
                corners,
                closed=True,
                facecolor=face_color,
                edgecolor=edge_color,
                lw=1.45,
                alpha=0.82,
                zorder=5,
            )
        )
        ax_overview.plot(
            [p_start[0], p_end[0]],
            [p_start[1], p_end[1]],
            color=DARK_PURPLE,
            lw=1.35,
            zorder=6,
        )
        ax_overview.scatter(
            [p_start[0], p_end[0]],
            [p_start[1], p_end[1]],
            s=18,
            color=DARK_PURPLE,
            edgecolor="white",
            lw=0.45,
            zorder=7,
        )

        midpoint = 0.5 * (p_start + p_end)
        if is_head:
            head_midpoint = midpoint
        elif index == 1:
            body_midpoint = midpoint
        if index == selected_index:
            selected_midpoint = midpoint
            ax_overview.add_patch(
                Circle(
                    tuple(midpoint),
                    1.35,
                    fill=False,
                    edgecolor=DARK_ORANGE,
                    lw=1.6,
                    ls=(0, (4, 3)),
                    zorder=9,
                )
            )

    ax_overview.scatter([0], [0], s=28, color=DARK_PURPLE, zorder=9)
    ax_overview.text(0.22, 0.20, "$O$", color=DARK_PURPLE, weight="bold")

    radius_angle = 2.38
    radius_end = R_INITIAL * np.array([np.cos(radius_angle), np.sin(radius_angle)])
    ax_overview.add_patch(
        FancyArrowPatch(
            (0, 0),
            tuple(radius_end),
            arrowstyle="-|>",
            mutation_scale=12,
            color=DARK_GREEN,
            lw=1.65,
            zorder=4,
        )
    )
    radius_label = 0.53 * radius_end
    ax_overview.text(
        radius_label[0] - 0.25,
        radius_label[1] + 0.20,
        "$R=8.80\\,\\mathrm{m}=880\\,\\mathrm{cm}$",
        color=DARK_GREEN,
        rotation=np.degrees(radius_angle) - 180,
        rotation_mode="anchor",
        ha="center",
        va="bottom",
        weight="bold",
    )
    if head_midpoint is not None:
        ax_overview.annotate(
            "龙头板凳：$3.41\\,\\mathrm{m}\\times0.30\\,\\mathrm{m}$",
            xy=head_midpoint,
            xytext=(2.2, -6.3),
            arrowprops=dict(arrowstyle="->", color=DARK_RED, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.25", facecolor=LIGHT_RED, edgecolor="none", alpha=0.55),
            color=DARK_RED,
            ha="left",
            va="center",
            weight="bold",
            zorder=10,
        )
    if body_midpoint is not None:
        ax_overview.annotate(
            "龙身板凳：$2.20\\,\\mathrm{m}\\times0.30\\,\\mathrm{m}$",
            xy=body_midpoint,
            xytext=(2.2, -5.0),
            arrowprops=dict(arrowstyle="->", color=DARK_BLUE, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.25", facecolor=LIGHT_BLUE, edgecolor="none", alpha=0.55),
            color=DARK_BLUE,
            ha="left",
            va="center",
            weight="bold",
            zorder=10,
        )
    if selected_midpoint is not None:
        ax_overview.annotate(
            "局部放大",
            xy=selected_midpoint,
            xytext=(-5.8, -3.6),
            arrowprops=dict(arrowstyle="->", color=DARK_ORANGE, lw=1.25),
            color=DARK_ORANGE,
            ha="right",
            va="center",
            weight="bold",
        )

    overview_limit = 9.35
    ax_overview.set_xlim(-overview_limit, overview_limit)
    ax_overview.set_ylim(-overview_limit, overview_limit)
    ax_overview.set_aspect("equal")
    ax_overview.set_xticks([])
    ax_overview.set_yticks([])
    ax_overview.set_title("(a) $R=8.80$ m（$880$ cm）圆域内的板凳实体比例", loc="left", pad=8)
    for spine in ax_overview.spines.values():
        spine.set_visible(False)

    # ---------- (b) 选取一节龙身板凳，按同一几何关系局部放大 ----------
    theta_a = theta_nodes[selected_index]
    theta_b = theta_nodes[selected_index + 1]
    p_a = node_points[selected_index]
    p_b = node_points[selected_index + 1]
    handle_distance = float(np.linalg.norm(p_b - p_a))

    theta_context = np.linspace(theta_a - 0.10, theta_b + 0.10, 450)
    context_global = np.column_stack(spiral_xy(theta_context))
    context_local = _to_chord_coordinates(context_global, p_a, p_b)
    theta_arc = np.linspace(theta_a, theta_b, 260)
    arc_global = np.column_stack(spiral_xy(theta_arc))
    arc_local = _to_chord_coordinates(arc_global, p_a, p_b)

    body_left = -0.275
    body_right = handle_distance + 0.275
    body_polygon = np.array(
        [
            [body_left, -0.15],
            [body_right, -0.15],
            [body_right, 0.15],
            [body_left, 0.15],
        ]
    )
    ax_local.add_patch(
        Polygon(
            body_polygon,
            closed=True,
            facecolor=LIGHT_BLUE,
            edgecolor=DARK_BLUE,
            lw=1.6,
            alpha=0.72,
            zorder=1,
        )
    )
    ax_local.plot(context_local[:, 0], context_local[:, 1], color=LIGHT_GREEN, lw=2.2, zorder=2)
    ax_local.plot(
        arc_local[:, 0],
        arc_local[:, 1],
        color=DARK_ORANGE,
        lw=2.5,
        ls=(0, (4, 3)),
        zorder=5,
    )
    ax_local.plot([0, handle_distance], [0, 0], color=DARK_PURPLE, lw=3.0, zorder=6)
    for x_value, label in ((0, "$P_i$"), (handle_distance, "$P_{i+1}$")):
        ax_local.add_patch(
            Circle(
                (x_value, 0),
                0.035,
                facecolor=DARK_PURPLE,
                edgecolor="white",
                lw=0.6,
                zorder=8,
            )
        )
        ax_local.text(x_value, -0.205, label, color=DARK_PURPLE, ha="center", va="top", weight="bold")

    _dimension_arrow(
        ax_local,
        0,
        handle_distance,
        0.31,
        "$L_i=|P_iP_{i+1}|=1.65\\,\\mathrm{m}$（直线弦长）",
        DARK_PURPLE,
        label_above=True,
    )
    _dimension_arrow(
        ax_local,
        body_left,
        body_right,
        -0.34,
        "龙身板凳实长 $2.20\\,\\mathrm{m}$",
        DARK_BLUE,
        label_above=False,
    )

    arc_mid = arc_local[len(arc_local) // 2]
    ax_local.annotate(
        "$s_i$：两把手间螺线弧",
        xy=arc_mid,
        xytext=(0.55 * handle_distance, 0.66),
        arrowprops=dict(arrowstyle="->", color=DARK_ORANGE, lw=1.2),
        color=DARK_ORANGE,
        ha="center",
        va="center",
        weight="bold",
    )
    ax_local.text(
        0.5 * handle_distance,
        -0.56,
        "$L_i\\neq s_i$",
        color=DARK_RED,
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=LIGHT_RED, edgecolor="none", alpha=0.45),
    )
    ax_local.text(
        0.5 * handle_distance,
        -0.88,
        "龙头：$3.41-2\\times0.275=2.86\\,\\mathrm{m}$\n"
        "龙身：$2.20-2\\times0.275=1.65\\,\\mathrm{m}$",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=LIGHT_YELLOW, edgecolor="none", alpha=0.88),
        color=INK,
    )

    ax_local.set_xlim(-0.48, handle_distance + 0.48)
    ax_local.set_ylim(-0.98, 0.82)
    ax_local.set_aspect("equal")
    ax_local.set_xticks([])
    ax_local.set_yticks([])
    ax_local.set_title("(b) 局部放大：固定的是直线弦长，而非螺线弧长", loc="left", pad=8)
    for spine in ax_local.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.05, wspace=0.08)
    save_figure(fig, "fig1_bench_scale_and_chord_constraint")


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
