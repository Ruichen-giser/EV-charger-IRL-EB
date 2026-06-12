"""SimpleGridCNN 行为克隆（BC）：单县专家均匀采样训练。"""
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bc_learn.evaluate import evaluate_bc_all
from bc_learn.metrics import eval_metrics_for_log, format_eval_log
from bc_learn.grid_cnn_bc import GridCNNBCAgent
from bc_learn.plotting import plot_bc_metrics
from cnn_config import DEFAULT_OBS_CHANNEL_NAMES
from expert_data import collect_expert_transitions
from mdp_config import current_mdp_mode
from paths import policy_checkpoint_path
from obs_channels import ObsChannelConfig
from viz_config import PLOT_EVAL_METRICS_AT_END



@dataclass
class BCTrainConfig:
    grid_npz_path: str
    output_dir: str
    obs_channel_names: tuple[str, ...] = DEFAULT_OBS_CHANNEL_NAMES
    lr: float = 1e-4
    batch_size: int = 64
    train_steps: int = 2000
    eval_every: int = 100
    seed: int = 0
    device: str = "auto"
    dropout: float = 0.0
    verbose: int = 1


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_bc(cfg: BCTrainConfig) -> dict[str, Any]:
    device = _resolve_device(cfg.device)
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    channel_cfg = ObsChannelConfig(names=tuple(cfg.obs_channel_names))
    raw_batch, layout = collect_expert_transitions(cfg.grid_npz_path, channel_cfg=channel_cfg)
    n_demo = len(raw_batch)

    agent = GridCNNBCAgent(
        grid_h=int(layout.H),
        grid_w=int(layout.W),
        in_channels=int(channel_cfg.n_channels),
        n_actions=int(layout.n_actions),
        device=device,
        lr=float(cfg.lr),
        dropout=float(cfg.dropout),
    )

    batch_size = min(int(cfg.batch_size), n_demo)
    metrics_log: list[dict[str, Any]] = []

    if cfg.verbose:
        print(
            f"[BC] {layout.state_name}/{layout.county_name} 专家转移 {n_demo} 条，"
            f"grid {layout.H}×{layout.W}×{channel_cfg.n_channels}，动作数 {layout.n_actions}，"
            f"SimpleGridCNN（3×Conv3×3 + 1×1 head，dropout={cfg.dropout}，均匀采样），"
            f"通道 {channel_cfg.names}",
            flush=True,
        )

    for step in range(1, int(cfg.train_steps) + 1):
        batch = raw_batch.sample(batch_size, agent.device, rng)
        train_stat = agent.train_step(batch)

        if step % int(cfg.eval_every) != 0:
            continue

        eval_idx = rng.integers(0, n_demo, size=batch_size)
        eval_batch = {
            "obs": torch.as_tensor(raw_batch.obs[eval_idx], dtype=torch.float32, device=agent.device).permute(
                0, 3, 1, 2
            ),
            "actions": torch.as_tensor(raw_batch.actions[eval_idx], dtype=torch.long, device=agent.device),
            "mask": torch.as_tensor(raw_batch.mask[eval_idx], dtype=torch.bool, device=agent.device),
        }
        batch_stat = agent.eval_batch(eval_batch)
        # 环境评估：expert_rollin（teacher forcing）+ policy_rollout（closed-loop）
        ev = evaluate_bc_all(agent, layout, channel_cfg)
        row = {
            "step": step,
            "loss": batch_stat["loss"],
            "top10_accuracy": batch_stat["top10_accuracy"],
            "policy_entropy": batch_stat["policy_entropy"],
            "train_loss": train_stat["loss"],
            **eval_metrics_for_log(ev),
        }
        metrics_log.append(row)

        print(
            f"step {step}: loss={row['loss']:.3f}, top10_batch={row['top10_accuracy']:.3f} | "
            f"{format_eval_log(row)}",
            flush=True,
        )

    ch_tag = "+".join(channel_cfg.names[:2]) + ("…" if len(channel_cfg.names) > 2 else "")
    metrics_png = out / "bc_training_metrics.png"
    if PLOT_EVAL_METRICS_AT_END:
        # 训练结束后一次性绘制评估曲线（不在 eval_every 循环内绘图）
        plot_bc_metrics(
            metrics_log,
            metrics_png,
            title=f"{layout.state_name}/{layout.county_name} SimpleGridCNN BC ({ch_tag})",
        )

    final_ev = evaluate_bc_all(agent, layout, channel_cfg)
    print(f"[BC] final eval | {format_eval_log(final_ev)}", flush=True)
    train_end = date.today()
    ckpt_path = policy_checkpoint_path(out, "BC", train_end=train_end, seed=int(cfg.seed))
    agent.save(str(ckpt_path))

    summary = {
        "state_name": layout.state_name,
        "county_name": layout.county_name,
        "location_key": f"{layout.state_name}/{layout.county_name}",
        "seed": int(cfg.seed),
        "train_end_date": train_end.isoformat(),
        "policy_checkpoint": ckpt_path.name,
        "mdp_mode": current_mdp_mode(),
        "method": "SimpleGridCNN BC（3×Conv3×3 + 1×1 head，均匀采样）",
        "network": "SimpleGridCNN",
        "dropout": float(cfg.dropout),
        "grid_h": layout.H,
        "grid_w": layout.W,
        "n_obs_channels": channel_cfg.n_channels,
        "n_actions": layout.n_actions,
        "n_expert_transitions": n_demo,
        "obs_channels": list(channel_cfg.names),
        "train_steps": int(cfg.train_steps),
        "eval_every": int(cfg.eval_every),
        "metrics_log": metrics_log,
        "final_eval": final_ev,
        "mean_expert_match_rate": final_ev["expert_rollin_expert_greedy_match_rate"],
        "top10_accuracy": final_ev["expert_rollin_top10_accuracy"],
        "mean_reciprocal_rank": final_ev["expert_rollin_mean_reciprocal_rank"],
        "mean_distance_km": final_ev["expert_rollin_mean_distance_km"],
        "final_eval_expert_rollin": {
            k: final_ev[k] for k in final_ev if str(k).startswith("expert_rollin_")
        },
        "final_eval_policy_rollout": {
            k: final_ev[k] for k in final_ev if str(k).startswith("policy_rollout_")
        },
        "metrics_plot": str(metrics_png) if PLOT_EVAL_METRICS_AT_END else "",
        "policy_path": str(ckpt_path),
    }

    with (out / "bc_summary.json").open("w", encoding="utf-8") as f:
        json.dump({**summary, "config": asdict(cfg)}, f, indent=2, ensure_ascii=False)
    with (out / "bc_metrics_log.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2, ensure_ascii=False)

    return summary


def run_bc(cfg: BCTrainConfig) -> dict[str, Any]:
    return train_bc(cfg)
