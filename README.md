# EV-charger-IRL-EB

Multi-county IRL training with **CountyLocationEmbed** (revised from [EV-charger-IRL](https://github.com/Ruichen-giser/EV-charger-IRL)).

默认：**MDP≥5 全量县（1149）** 联合 IQ-Learn，共享 SimpleGridCNN + state/county embedding。

## 与 EV-charger-IRL 的区别

| | EV-charger-IRL | EV-charger-IRL-EB（本仓库） |
|--|----------------|----------------------------|
| 训练范围 | 单县 | **MDP≥5 多县联合（默认 1149）** |
| Q 网络 | 每县独立 | **共享一个 Q** |
| County 条件 | 无 | **state_embed + meta MLP + residual（α=0）** |
| IQ 模式 | online/offline | **默认 offline（仅专家轨迹）** |
| 入口 | `IRL_IQ/main.py --county X` | `IRL_IQ/main.py` |

## 仓库结构

```text
EV-charger-IRL-EB/
├── data/us_mdp_ge5_state_county_list.xlsx
├── IRL_data/          # 数据准备（MDP≥5 县 → grid_tensors）
├── IRL_IQ/            # ★ 联合 IQ-Learn
│   ├── main.py
│   ├── county_meta.py
│   ├── state_county.py
│   └── iq_learn/
│       ├── grid_align.py      # 画布 padding / 动作对齐
│       ├── expert_data.py     # 单县收集 + 多县 pool
│       ├── train_joint.py     # 联合训练循环
│       └── ...
├── IRL_BC/            # 单县 BC baseline
└── outputs/
    ├── prepared_data/grid_tensors/
    └── iq_output/joint_mdp_ge5/
```

## Quick start

### 1. 数据

将原始 GIS 数据放入 `data/`（见 `data/README.md`）。默认人口栅格为 **WorldPOP 2020**，POI 为 **2024** geoparquet。

```bash
cd IRL_data
pip install -r requirements.txt
python main.py   # 默认 MDP≥5 Excel 全量县 → grid_tensors
```

### 2. 联合训练

```bash
cd IRL_IQ
pip install -r requirements.txt
python main.py   # 默认 offline IQ + 已有 npz 的县
```

常用参数：

```bash
python main.py --dev-8-counties              # 8 县 CA 调试
python main.py --iq-loss-mode online         # 专家+策略 rollout 混合
python main.py --eval-final-max-counties 0   # 最终评估全量 rollout
python main.py --train-steps 50000 --device cuda
```

### 3. 输出

- `outputs/iq_output/joint_mdp_ge5/iq_learn_shared.pt` — 最终共享 Q
- `outputs/iq_output/joint_mdp_ge5/iq_learn_shared_best.pt` — 验证最优
- `outputs/iq_output/joint_mdp_ge5/iq_learn_summary.json` — 含 `per_county_final`

## 训练流程

```text
MDP≥5 县 cropped npz
  → 各县 collect 专家 (s,a,s')
  → pool：pad 到 max(H×W)，打 state_id / county_id / county_meta（geojson 三维社会经济）
  → offline IQ：仅专家 buffer 采样（默认）
  → SimpleGridCNN：obs + location embed 广播 concat → Conv → 1×1 Q head
  → IQ-Learn loss → 共享 Q 更新
  → 评估：默认随机 32 县 policy rollout（可设 0 为全量）
```

## 开发 8 县

`python main.py --dev-8-counties`

Siskiyou, Modoc, Shasta, Lassen, Los Angeles, Sacramento, San Diego, Kern

（定义于 `IRL_IQ/cnn_config.py` 的 `DEV_STATE_COUNTIES_8`）
