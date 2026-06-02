# IRL_BC

行为克隆（BC）+ SimpleGridCNN。独立包，仅依赖 `IRL_data` 生成的网格 npz。

## 数据

默认读取 `outputs/prepared_data/grid_tensors/` 下的：

- `Los_Angeles_grid_features.npz`（或其它县 `<县名>_grid_features.npz`）

首次运行会自动生成 `*_grid_cropped.npz`。

## MDP 模式（`--mdp-mode`，默认 **legacy**）

| 模式 | 专家轨迹 | 动作掩膜 |
|------|----------|----------|
| **legacy** | 每格 first_visit 去重 | `valid_mask & ~stations` |
| **repeat** | 完整 OpenDate 序列 | 仅 `valid_mask` |

## 安装与运行

```bash
pip install -r requirements.txt
python main.py
python main.py --mdp-mode repeat
python main.py --county Los_Angeles
```

默认输出：`outputs/bc_output/los_angeles/`

训练结束保存权重：`IRL_BC_YYYYMMDD_seed{N}.pt`

评估：`expert roll-in`（teacher forcing）与 `policy rollout`（closed-loop）；指标含 match、top10、MRR、Jaccard、Hausdorff、dist_km。

## 目录说明

| 路径 | 作用 |
|------|------|
| `main.py` | 入口 |
| `data_prep.py` | 裁剪网格 npz |
| `envs/` | 建站 MDP + 多通道观测包装 |
| `obs_channels.py` | 观测通道定义 |
| `expert_data.py` | 专家轨迹收集 |
| `bc_learn/` | BC 训练与评估 |
| `models/` | SimpleGridCNN |
