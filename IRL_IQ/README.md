# IRL_IQ

IQ-Learn + SimpleGridCNN（网络与 IRL_BC 一致）。独立包，仅依赖 `IRL_data` 生成的网格 npz。

## 数据

同 IRL_BC：默认读取 `outputs/prepared_data/grid_tensors/<县名>_grid_features.npz`。

## MDP 模式（默认 **legacy**，与 IRL_BC 相同）

```bash
python main.py
python main.py --mdp-mode repeat
```

## 安装与运行

```bash
pip install -r requirements.txt
python main.py
```

默认输出：`outputs/iq_output/los_angeles/`

训练结束保存权重：`IRL_IQ_YYYYMMDD_seed{N}.pt`

评估：`expert roll-in`（teacher forcing）与 `policy rollout`（closed-loop）；环境 `max_steps` 与专家轨迹步数一致（`expert_action_sequence` 长度）。指标含 match、top10、MRR、Jaccard、Hausdorff、Chamfer、LCSS。

## 目录说明

| 路径 | 作用 |
|------|------|
| `main.py` | 入口 |
| `envs/` | 建站 MDP + 多通道观测包装 |
| `iq_learn/` | IQ-Learn 训练 |
| `models/` | SimpleGridCNN |
