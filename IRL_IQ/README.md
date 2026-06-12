# IRL_IQ-EB — MDP≥5 全量县联合 IQ-Learn

SimpleGridCNN + county embedding（输入层 concat）+ 1×1 spatial Q head。

## 运行

```bash
pip install -r requirements.txt
python main.py
```

默认从 `data/us_mdp_ge5_state_county_list.xlsx` 加载 MDP≥5 县清单（1149），
仅训练 `grid_tensors/` 中已有 npz 的县；输出到 `outputs/iq_output/joint_mdp_ge5/`。

8 县调试：`python main.py --dev-8-counties`

默认 `iq_loss_mode=offline`（仅专家轨迹）；评估默认随机抽 32 县做 policy rollout。
全量最终评估：`python main.py --eval-final-max-counties 0`

## 核心模块

| 文件 | 作用 |
|------|------|
| `main.py` | CLI 入口 |
| `iq_learn/train_joint.py` | 联合训练主循环 |
| `iq_learn/expert_data.py` | 单县收集 + `merge_county_expert_batches`（pool） |
| `iq_learn/grid_align.py` | 统一画布 padding、动作 index 对齐 |
| `iq_learn/policy_rollout.py` | 多县惰性 rollout 池 → 策略 buffer |
| `iq_learn/evaluate.py` | 画布对齐下的逐县评估 |
| `models/simple_grid_cnn.py` | Conv encoder + embedding concat + 1×1 Q |

## 配置

`cnn_config.py`：`DEFAULT_IQ_LOSS_MODE=offline`、`DEFAULT_EVAL_MAX_COUNTIES=32`、`DEFAULT_EVAL_FINAL_MAX_COUNTIES=32`
