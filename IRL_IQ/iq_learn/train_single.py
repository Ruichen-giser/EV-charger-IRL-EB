"""单县 IQ-Learn：SimpleGridCNN（与 BC 同架构）、完整专家建站序列、分层 Replay。"""
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cnn_config import CNN_DROPOUT, DEFAULT_COUNTY, DEFAULT_OBS_CHANNEL_NAMES
from mdp_config import current_mdp_mode
from iq_learn.discrete_soft_q import DiscreteSoftQAgent
from iq_learn.evaluate import evaluate_iq_all
from iq_learn.metrics import eval_metrics_for_log, format_eval_log
from iq_learn.expert_data import collect_expert_transitions
from iq_learn.plotting import plot_iq_metrics
from iq_learn.policy_rollout import PolicyRolloutCollector, warm_start_policy_buffer
from iq_learn.replay_buffer import StratifiedReplayBuffer
from obs_channels import ObsChannelConfig
from paths import policy_checkpoint_path
from viz_config import PLOT_EVAL_METRICS_AT_END


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class SingleCountyTrainConfig:
    grid_npz_paths: list[str]
    output_dir: str
    county: str = DEFAULT_COUNTY
    gamma: float = 0.99
    alpha: float = 0.01
    alpha_reg: float = 0.5
    use_chi: bool = True
    lr: float = 5e-5
    batch_size: int = 64
    train_steps: int = 20_000
    eval_every: int = 100
    target_update_interval: int = 15
    seed: int = 0
    device: str = "auto"
    dropout: float = CNN_DROPOUT
    obs_channel_names: tuple[str, ...] = DEFAULT_OBS_CHANNEL_NAMES
    use_stratified_buffer: bool = True
    expert_batch_fraction: float = 0.5
    policy_buffer_capacity: int = 10_000
    policy_warmup_transitions: int = 100
    verbose: int = 1


def train_single_county(cfg: SingleCountyTrainConfig) -> tuple[DiscreteSoftQAgent, dict[str, Any]]:
    device = _resolve_device(cfg.device)
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if len(cfg.grid_npz_paths) != 1:
        raise ValueError("train_single_county 仅支持单县 npz")

    channel_cfg = ObsChannelConfig(names=tuple(cfg.obs_channel_names))
    npz_path = str(cfg.grid_npz_paths[0])

    raw_batch, layout = collect_expert_transitions(npz_path, channel_cfg=channel_cfg)

    agent = DiscreteSoftQAgent(
        obs_dim=int(layout.H * layout.W * channel_cfg.n_channels),
        n_actions=int(layout.n_actions),
        device=device,
        lr=float(cfg.lr),
        gamma=float(cfg.gamma),
        alpha=float(cfg.alpha),
        alpha_reg=float(cfg.alpha_reg),
        use_chi=bool(cfg.use_chi),
        target_update_interval=int(cfg.target_update_interval),
        grid_h=int(layout.H),
        grid_w=int(layout.W),
        in_channels=int(channel_cfg.n_channels),
        dropout=float(cfg.dropout),
    )

    buffer = StratifiedReplayBuffer(
        capacity_expert=50_000,
        capacity_policy=int(cfg.policy_buffer_capacity),
        expert_fraction=float(cfg.expert_batch_fraction),
        spatial=True,
    )
    buffer.add_expert_from_batch(
        raw_batch.obs,
        raw_batch.next_obs,
        raw_batch.actions,
        raw_batch.done,
        raw_batch.mask,
        raw_batch.next_mask,
    )

    rollout: PolicyRolloutCollector | None = None
    if cfg.use_stratified_buffer:
        n_warm = warm_start_policy_buffer(
            agent, buffer, npz_path, int(cfg.policy_warmup_transitions), rng, channel_cfg=channel_cfg
        )
        if cfg.verbose:
            print(f"[IQ] 策略 buffer 预填充 {n_warm} 条", flush=True)
        rollout = PolicyRolloutCollector(npz_path, spatial=True, channel_cfg=channel_cfg)
        rollout.reset()

    eval_every = max(1, int(cfg.eval_every))
    metrics_log: list[dict[str, Any]] = []
    county_tag = layout.county_name.replace(" ", "_").lower()
    metrics_png = out / f"{county_tag}_iq_training_metrics.png"

    if cfg.verbose:
        print(
            f"[IQ] {layout.county_name} 专家转移 {len(raw_batch)} 条，"
            f"grid {layout.H}×{layout.W}×{channel_cfg.n_channels}，"
            f"SimpleGridCNN（3×Conv3×3 + 1×1 head，dropout={cfg.dropout}），"
            f"通道 {channel_cfg.names}",
            flush=True,
        )

    for step in range(1, int(cfg.train_steps) + 1):
        if cfg.use_stratified_buffer and rollout is not None:
            rollout.collect_one_step(agent, buffer, rng, stochastic=True)

        batch = buffer.sample(int(cfg.batch_size), agent.device, rng)
        m = agent.train_step(batch)

        if step % eval_every != 0:
            continue

        # 环境评估：expert_rollin（teacher forcing）+ policy_rollout（closed-loop）
        ev = evaluate_iq_all(agent, layout, channel_cfg)
        row = {
            "step": step,
            "Q_mean": m["Q_mean"],
            "Q_std": m["Q_std"],
            "Q_max": m["Q_max"],
            "policy_entropy": m["policy_entropy"],
            "loss": m["loss"],
            **eval_metrics_for_log(ev),
        }
        metrics_log.append(row)

        print(
            f"step {step}: Q_mean={m['Q_mean']:.3f}, loss={m['loss']:.3f} | "
            f"{format_eval_log(row)}",
            flush=True,
        )

    if rollout is not None:
        rollout.close()

    ch_tag = "+".join(channel_cfg.names[:2]) + ("…" if len(channel_cfg.names) > 2 else "")
    if PLOT_EVAL_METRICS_AT_END:
        # 训练结束后一次性绘制评估曲线（不在 eval_every 循环内绘图）
        plot_iq_metrics(
            metrics_log,
            metrics_png,
            title=f"{layout.county_name} IQ-Learn SimpleGridCNN ({ch_tag})",
        )

    final_ev = evaluate_iq_all(agent, layout, channel_cfg)
    print(f"[IQ] final eval | {format_eval_log(final_ev)}", flush=True)
    summary = {
        "county_name": layout.county_name,
        "seed": int(cfg.seed),
        "mdp_mode": current_mdp_mode(),
        "method": "IQ-Learn SimpleGridCNN（与 BC 同架构）+ stratified replay",
        "network": "SimpleGridCNN",
        "dropout": float(cfg.dropout),
        "n_expert_transitions": len(raw_batch),
        "obs_channels": list(channel_cfg.names),
        "train_steps_ran": int(cfg.train_steps),
        "eval_every": eval_every,
        "replay_buffer_size": len(buffer),
        "expert_buffer_size": len(buffer._expert),
        "policy_buffer_size": len(buffer._policy),
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
    }
    return agent, summary


def run_single_county(cfg: SingleCountyTrainConfig) -> dict[str, Any]:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if cfg.verbose:
        print(
            f"[IQ] 单县 {cfg.grid_npz_paths[0]} "
            f"lr={cfg.lr} steps={cfg.train_steps} eval_every={cfg.eval_every} "
            f"channels={cfg.obs_channel_names}",
            flush=True,
        )
    agent, train_summary = train_single_county(cfg)
    train_end = date.today()
    ckpt_path = policy_checkpoint_path(out, "IQ", train_end=train_end, seed=int(cfg.seed))
    agent.save(str(ckpt_path))
    summary = {
        "config": asdict(cfg),
        "train_end_date": train_end.isoformat(),
        "seed": int(cfg.seed),
        "policy_checkpoint": ckpt_path.name,
        "policy_path": str(ckpt_path),
        **train_summary,
    }
    with (out / "iq_learn_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (out / "metrics_log.json").open("w", encoding="utf-8") as f:
        json.dump(train_summary.get("metrics_log", []), f, indent=2, ensure_ascii=False)
    return summary
