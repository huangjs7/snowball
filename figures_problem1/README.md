# 问题一论文插图

每幅图单独存放，目录内包含独立运行入口、PNG、SVG、PDF 和说明文件。

| 图号 | 目录 | 独立入口 |
|---|---|---|
| 图 1 | `fig01_bench_scale_chord` | `fig01_bench_scale_chord/plot.py` |
| 图 2 | `fig02_handle_geometry` | `fig02_handle_geometry/plot.py` |
| 图 3 | `fig03_velocity_transfer` | `fig03_velocity_transfer/plot.py` |

`common_plotting.py` 保存统一配色、数学参数和公共几何函数，各图的
`plot.py` 只生成本目录对应的一幅图。
