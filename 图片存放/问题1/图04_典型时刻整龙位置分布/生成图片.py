"""生成图 4：六个典型时刻的整条板凳龙位置分布。"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from openpyxl import load_workbook


HERE = Path(__file__).resolve().parent
FIGURES_ROOT = HERE.parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(FIGURES_ROOT))

import common_plotting as plotting  # noqa: E402


TIMES = (0, 60, 120, 180, 240, 300)
NODE_COUNT = 224
OUTPUT_STEM = "图04_典型时刻整龙位置分布"


def locate_result_file(explicit_path: Path | None) -> Path:
    """优先使用命令行路径，其次寻找桌面上的填充结果文件。"""

    candidates = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.extend(
        [
            HERE / "result1.xlsx",
            PROJECT_ROOT / "提交" / "result1.xlsx",
            PROJECT_ROOT / "建模计算" / "问题1" / "输出" / "result1.xlsx",
            Path.home() / "Desktop" / "result1.xlsx",
        ]
    )
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved.is_file():
            return resolved
    checked = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"未找到填充后的 result1.xlsx，已检查：\n{checked}")


def parse_time_columns(ws) -> dict[int, int]:
    columns: dict[int, int] = {}
    for column in range(2, ws.max_column + 1):
        header = ws.cell(row=1, column=column).value
        match = re.fullmatch(r"\s*(-?\d+)\s*s\s*", str(header))
        if match:
            columns[int(match.group(1))] = column
    return columns


def load_snapshots(path: Path) -> dict[int, np.ndarray]:
    workbook = load_workbook(path, read_only=False, data_only=True)
    if "位置" not in workbook.sheetnames:
        raise ValueError("result1.xlsx 缺少“位置”工作表")
    ws = workbook["位置"]
    columns = parse_time_columns(ws)
    missing = [time for time in TIMES if time not in columns]
    if missing:
        raise ValueError(f"位置表缺少典型时刻列：{missing}")

    snapshots: dict[int, np.ndarray] = {}
    for time in TIMES:
        column = columns[time]
        points = np.empty((NODE_COUNT, 2), dtype=float)
        for index in range(NODE_COUNT):
            x_value = ws.cell(row=2 + 2 * index, column=column).value
            y_value = ws.cell(row=3 + 2 * index, column=column).value
            if x_value is None or y_value is None:
                raise ValueError(f"t={time} s 时第 {index} 个把手坐标为空")
            points[index] = float(x_value), float(y_value)
        if not np.isfinite(points).all():
            raise ValueError(f"t={time} s 的位置数据含非有限值")
        snapshots[time] = points
    workbook.close()
    return snapshots


def validate_chord_constraints(snapshots: dict[int, np.ndarray]) -> float:
    expected = np.array([2.86] + [1.65] * (NODE_COUNT - 2), dtype=float)
    max_error = 0.0
    for points in snapshots.values():
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        max_error = max(max_error, float(np.max(np.abs(distances - expected))))
    if max_error > 1e-5:
        raise ValueError(f"相邻把手定长约束最大误差 {max_error:.3e} m，超过容差")
    return max_error


def draw_figure(snapshots: dict[int, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.7), sharex=True, sharey=True)
    panel_tags = "abcdef"
    time_box_colors = [
        plotting.LIGHT_BLUE,
        plotting.LIGHT_CYAN,
        plotting.LIGHT_GREEN,
        plotting.LIGHT_PURPLE,
        plotting.LIGHT_ORANGE,
        plotting.LIGHT_RED,
    ]
    axis_limit = 12.45
    ticks = np.arange(-10, 11, 5)

    for panel_index, (ax, time, box_color) in enumerate(
        zip(axes.flat, TIMES, time_box_colors)
    ):
        points = snapshots[time]

        ax.axhline(0, color=plotting.GRID, lw=0.8, zorder=0)
        ax.axvline(0, color=plotting.GRID, lw=0.8, zorder=0)
        ax.grid(color=plotting.GRID, lw=0.45, alpha=0.38, zorder=0)

        ax.plot(
            points[:, 0],
            points[:, 1],
            color=plotting.DARK_BLUE,
            lw=1.55,
            zorder=2,
        )
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=5.0,
            color=plotting.LIGHT_BLUE,
            edgecolors="none",
            alpha=0.92,
            zorder=3,
        )
        ax.scatter(
            points[0, 0],
            points[0, 1],
            s=34,
            color=plotting.DARK_RED,
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        ax.scatter(
            points[-1, 0],
            points[-1, 1],
            s=30,
            color=plotting.DARK_PURPLE,
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        ax.scatter(0, 0, s=13, color=plotting.INK, zorder=4)

        ax.text(
            0.035,
            0.955,
            rf"({panel_tags[panel_index]})  $t={time}\,\mathrm{{s}}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=plotting.INK,
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.24",
                facecolor=box_color,
                edgecolor="none",
                alpha=0.56,
            ),
            zorder=8,
        )

        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(length=2.5, width=0.6, color="#777777")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b7b7b7")
        ax.spines["bottom"].set_color("#b7b7b7")
        ax.spines["left"].set_linewidth(0.65)
        ax.spines["bottom"].set_linewidth(0.65)

    legend_items = [
        Line2D([0], [0], color=plotting.DARK_BLUE, lw=1.8, label="相邻把手中心连线"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=plotting.DARK_RED,
            markeredgecolor="white",
            markersize=6.3,
            label="龙头前把手",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=plotting.DARK_PURPLE,
            markeredgecolor="white",
            markersize=6.0,
            label="龙尾后把手",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=plotting.INK,
            markersize=4.0,
            label="螺线中心 $O$",
        ),
    ]
    fig.supxlabel(r"$x\,/\,\mathrm{m}$", y=0.062)
    fig.supylabel(r"$y\,/\,\mathrm{m}$", x=0.035)
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
        handlelength=2.4,
        columnspacing=1.7,
    )
    fig.subplots_adjust(left=0.075, right=0.992, top=0.985, bottom=0.135, wspace=0.10, hspace=0.16)
    plotting.save_figure(fig, OUTPUT_STEM)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="填充后的 result1.xlsx 路径")
    args = parser.parse_args()

    plotting.OUTPUT_DIR = HERE
    plotting.configure_style()
    source = locate_result_file(args.input)
    snapshots = load_snapshots(source)
    max_error = validate_chord_constraints(snapshots)
    draw_figure(snapshots)
    print(f"图 4 已生成：{HERE}")
    print(f"数据源：{source}")
    print(f"节点数：{NODE_COUNT}；典型时刻：{TIMES}")
    print(f"定长约束最大误差：{max_error:.3e} m")


if __name__ == "__main__":
    main()
