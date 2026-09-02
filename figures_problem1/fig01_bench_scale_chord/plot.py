"""单独生成图 1：板凳真实尺度与直线弦长约束。"""

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
FIGURES_ROOT = HERE.parent
sys.path.insert(0, str(FIGURES_ROOT))

import common_plotting as plotting  # noqa: E402


def main() -> None:
    plotting.OUTPUT_DIR = HERE
    plotting.configure_style()
    plotting.figure_1_parameter_definition()
    print(f"图 1 已生成：{HERE}")


if __name__ == "__main__":
    main()
