# IRL_IQ-EB — 8 县联合 IQ-Learn

SimpleGridCNN + county embedding（输入层 concat）+ 1×1 spatial Q head。

## 运行

```bash
pip install -r requirements.txt
python main.py
```

默认读取 `outputs/prepared_data/grid_tensors/` 下 8 县 npz，输出到 `outputs/iq_output/joint_8counties/`。

## 核心模块

| 文件 | 作用 |
|------|------|
| `main.py` | CLI 入口 |
| `iq_learn/train_joint.py` | 联合训练主循环 |
| `iq_learn/expert_data.py` | 单县收集 + `merge_county_expert_batches`（pool） |
| `iq_learn/grid_align.py` | 统一画布 padding、动作 index 对齐 |
| `iq_learn/policy_rollout.py` | 8 县随机 rollout → 策略 buffer |
| `iq_learn/evaluate.py` | 画布对齐下的逐县评估 |
| `models/simple_grid_cnn.py` | Conv encoder + embedding concat + 1×1 Q |

## 配置

`cnn_config.py`：`TRAINING_COUNTIES`、`COUNTY_EMBED_DIM=16`
