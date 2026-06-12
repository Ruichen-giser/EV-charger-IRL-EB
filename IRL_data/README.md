# IRL_data — 数据处理 pipeline

运行（在仓库根目录下）：

```bash
cd IRL_data
python main.py
```

## 三步总览

```text
┌─────────────────────────────────────────────────────────────┐
│ 步骤 1  导入原始 GIS/EVCS，构建 2 km 网格特征（按县内部执行）   │
│   geo          县界、事件落在县内                             │
│   gis_align    栅格 WorldPOP/GDP + 矢量 POI/公路/加油站 → 1km │
│   grid_ops     1→2km 聚合、县内归一化、补缺格                 │
├─────────────────────────────────────────────────────────────┤
│ 步骤 2  按 county 切分                                        │
│   county_prepare   多县并行；每县 → events + grids 表         │
├─────────────────────────────────────────────────────────────┤
│ 步骤 3  存储，供 IRL_BC / IRL_IQ 读取                         │
│   main           prepared_irl_dataset.pkl                     │
│   grid_tensor    grid_tensors/*_grid_features.npz            │
└─────────────────────────────────────────────────────────────┘
```

## 文件职责

| 文件 | 步骤 | 作用 |
|------|------|------|
| `main.py` | 入口 / 3 | 配置、多县批处理、写 pickle 与 npz |
| `county_prepare.py` | 2 | 单县编排；`load_evcs_events` |
| `geo.py` | 1a | 县界 GeoJSON、空间筛选 |
| `gis_align.py` | 1b | 栅格/矢量对齐；`build_county_grid_1km` |
| `grid_ops.py` | 1c | `GridSystem`、2 km 聚合、归一化、裁剪 |
| `grid_tensor.py` | 3 | 稀疏表 → `(H,W,5)` + `expert_actions` |
| `feature_statistics.py` | — | 网格/建站格特征分布柱状图 |
| `paths.py` | — | 输出路径 |
| `schema.py` | — | 版本号、五通道列名 |

## 县清单（默认）

默认从 `data/us_mdp_ge5_state_county_list.xlsx` 读取 MDP 轨迹长度 ≥ 5 的 state-county（约 1149 县），
逐县裁剪 GIS/建站轨迹并写入 `grid_tensors/`。在 `main.py` 中设 `use_mdp_ge5_county_list = False` 可恢复单州 + `county_names` 模式。

## 输出

- `outputs/prepared_data/prepared_irl_dataset.pkl`
- `outputs/prepared_data/grid_tensors/<州>__<县>_grid_features.npz`（多州同名县不冲突）

仅重导出 npz：`python grid_tensor.py`

## 特征分布统计

```bash
python feature_statistics.py
```

读取 `prepared_irl_dataset.pkl`，输出至 `outputs/statistics/`。

## → IRL_BC / IRL_IQ

`grid_tensors/*.npz` 作为 `--grid-npz-dir` 的默认输入。
