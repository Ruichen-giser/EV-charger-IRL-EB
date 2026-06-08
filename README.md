# EV-charger-IRL-EB

**E**ight-county **B**undled training fork：8 县联合 IQ-Learn，共享 SimpleGridCNN + county embedding。

与 [EV-charger-IRL](https://github.com/)（单县 BC/IQ）分离维护，不破坏原仓库。

## 与 EV-charger-IRL 的区别

| | EV-charger-IRL | EV-charger-IRL-EB（本仓库） |
|--|----------------|----------------------------|
| 训练范围 | 单县 | **8 县联合** |
| Q 网络 | 每县独立 | **共享一个 Q** |
| County 条件 | 无 | **Embedding(16) + 输入层 concat** |
| Q head | 1×1 spatial conv | 同左（无 GAP） |
| 入口 | `IRL_IQ/main.py --county X` | `IRL_IQ/main.py`（默认 8 县） |

## 仓库结构

```text
EV-charger-IRL-EB/
├── IRL_data/          # 数据准备（与原版相同，8 县 county_names）
├── IRL_IQ/            # ★ 联合训练（已改写）
│   ├── main.py
│   └── iq_learn/
│       ├── grid_align.py      # 画布 padding / 动作对齐
│       ├── expert_data.py     # 单县收集 + 多县 pool
│       ├── train_joint.py     # 联合训练循环
│       └── ...
└── outputs/
    └── iq_output/joint_8counties/
```

`IRL_BC/` 仍保留（单县 BC baseline），联合 IQ 请只用 `IRL_IQ/`。

## Quick start

### 1. 数据

```bash
cd IRL_data
python main.py   # 生成 8 县 grid_tensors
```

### 2. 联合训练

```bash
cd IRL_IQ
pip install -r requirements.txt
python main.py
```

可选参数：

```bash
python main.py --train-steps 50000 --eval-every 200 --county-embed-dim 16 --device cuda
python main.py --counties Siskiyou,Modoc,Shasta,Lassen   # 自定义县列表
```

### 3. 输出

- `outputs/iq_output/joint_8counties/iq_learn_shared.pt` — 最终共享 Q
- `outputs/iq_output/joint_8counties/iq_learn_shared_best.pt` — 验证最优
- `outputs/iq_output/joint_8counties/iq_learn_summary.json` — 含各县 `per_county_final`

## 训练流程（逻辑）

```text
8 县 cropped npz
  → 各县 collect 专家 (s,a,s')
  → pool：pad 到 max(H×W)，打 county_id
  → StratifiedReplayBuffer（专家 + 8 县随机 policy rollout）
  → SimpleGridCNN：obs + Embedding 广播 concat → Conv → 1×1 Q head
  → IQ-Learn loss → 共享 Q 更新
```

## 默认 8 县

Siskiyou, Modoc, Shasta, Lassen, Los_Angeles, Sacramento, San_Diego, Kern

（在 `IRL_IQ/cnn_config.py` 的 `TRAINING_COUNTIES` 中修改）
