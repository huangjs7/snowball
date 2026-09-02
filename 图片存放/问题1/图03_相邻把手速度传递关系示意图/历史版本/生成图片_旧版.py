"""绘制图3：相邻把手速度传递关系示意图。

此文件仅用于保留迁移前的兼容入口；推荐运行上一级目录的“生成图片.py”。
"""
from pathlib import Path
import shutil
import sys


HERE = Path(__file__).resolve().parent
FIGURE_SOURCE_DIR = HERE.parent
QUESTION_DIR = HERE.parents[1]

sys.path.insert(0, str(QUESTION_DIR))
import common_plotting as plotting  # noqa: E402


def main() -> None:
    """生成标准图3，并复制一份采用论文中文文件名的 PNG。"""
    plotting.OUTPUT_DIR = FIGURE_SOURCE_DIR
    plotting.configure_style()
    plotting.figure_3_velocity_transfer()

    source = FIGURE_SOURCE_DIR / "图03_相邻把手速度传递关系示意图.png"
    target = HERE / "图03_相邻把手速度传递关系示意图_旧版副本.png"
    shutil.copy2(source, target)
    print(f"已生成：{target}")


if __name__ == "__main__":
    main()
