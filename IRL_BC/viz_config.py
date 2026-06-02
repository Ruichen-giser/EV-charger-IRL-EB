"""可视化开关（大规模实验默认配置）。

- 网格特征分布图：仅 scripts/visualize_expert_first_step.py 手动运行，训练流程不调用。
- 评估指标曲线：训练结束后生成 bc_training_metrics.png。
"""

# 训练 / main.py 流程中不绘制网格特征分布（勿在 data_prep / train 中开启）
ENABLE_GRID_FEATURE_VIZ = False

# 训练结束后绘制评估指标曲线（expert roll-in + policy rollout）
PLOT_EVAL_METRICS_AT_END = True
