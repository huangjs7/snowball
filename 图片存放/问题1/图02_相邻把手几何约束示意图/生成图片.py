"""单独生成图 2：相邻把手几何约束。"""

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
FIGURES_ROOT = HERE.parent
sys.path.insert(0, str(FIGURES_ROOT))

import common_plotting as plotting  # noqa: E402


def main() -> None:
    plotting.OUTPUT_DIR = HERE
    plotting.configure_style()
    plotting.figure_2_chord_constraint()
    print(f"图 2 已生成：{HERE}")


if __name__ == "__main__":
    main()
