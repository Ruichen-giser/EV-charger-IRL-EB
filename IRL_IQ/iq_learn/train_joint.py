"""8 县联合 IQ-Learn：共享 SimpleGridCNN + county embedding。"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cnn_config import (
    CNN_DROPOUT,
    COUNTY_EMBED_DIM,
    DEFAULT_OBS_CHANNEL_NAMES,
    TRAINING_COUNTIES,
)
from iq_learn.discrete_soft_q import DiscreteSoftQAgent
from iq_learn.evaluate import evaluate_joint_all
from iq_learn.expert_data import build_merged_expert_dataset
from iq_learn.plotting import plot_iq_metrics
from iq_learn.policy_rollout import JointPolicyRolloutPool, warm_start_joint_policy_buffer
from iq_learn.replay_buffer import StratifiedReplayBuffer
from mdp_config import current_mdp_mode
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
class JointTrainConfig:
    grid_npz_paths: list[str]
    output_dir: str
    counties: tuple[str, ...] = TRAINING_COUNTIES
    gamma: float = 0.99
    alpha: float = 0.01
    alpha_reg: float = 0.5
    use_chi: bool = True
    iq_loss_mode: str = "online"
    lr: float = 5e-5
    batch_size: int = 64
    train_steps: int = 50_000
    eval_every: int = 200
    target_update_interval: int = 15
    seed: int = 0
    device: str = "auto"
    dropout: float = CNN_DROPOUT
    obs_channel_names: tuple[str, ...] = DEFAULT_OBS_CHANNEL_NAMES
    county_embed_dim: int = COUNTY_EMBED_DIM
    use_stratified_buffer: bool = True
    expert_batch_fraction: float = 0.5
    policy_buffer_capacity: int = 20_000
    policy_warmup_transitions: int = 200
    verbose: int = 1


def train_joint(cfg: JointTrainConfig) -> tuple[DiscreteSoftQAgent, dict[str, Any]]:
    device = _resolve_device(cfg.device)
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if cfg.iq_loss_mode == "online" and not cfg.use_stratified_buffer:
        raise ValueError("iq_loss_mode=online 需要 use_stratified_buffer=True")

    channel_cfg = ObsChannelConfig(names=tuple(cfg.obs_channel_names))
    merged = build_merged_expert_dataset(cfg.grid_npz_paths, channel_cfg=channel_cfg)
    canvas = merged.canvas
    counties = merged.counties
    county_names = [c.county_name for c in counties]

    agent = DiscreteSoftQAgent(
        obs_dim=int(canvas.max_h * canvas.max_w * channel_cfg.n_channels),
        n_actions=int(canvas.n_actions),
        device=device,
        lr=float(cfg.lr),
        gamma=float(cfg.gamma),
        alpha=float(cfg.alpha),
        alpha_reg=float(cfg.alpha_reg),
        use_chi=bool(cfg.use_chi),
        iq_loss_mode=str(cfg.iq_loss_mode),
        target_update_interval=int(cfg.target_update_interval),
        grid_h=int(canvas.max_h),
        grid_w=int(canvas.max_w),
        in_channels=int(channel_cfg.n_channels),
        n_counties=len(counties),
        county_embed_dim=int(cfg.county_embed_dim),
        dropout=float(cfg.dropout),
        county_names=county_names,
    )

    buffer = StratifiedReplayBuffer(
        capacity_expert=100_000,
        capacity_policy=int(cfg.policy_buffer_capacity),
        expert_fraction=float(cfg.expert_batch_fraction),
        spatial=True,
    )
    buffer.add_expert_from_merged(
        merged.obs,
        merged.next_obs,
        merged.actions,
        merged.done,
        merged.mask,
        merged.next_mask,
        merged.county_ids,
    )

    rollout_pool: JointPolicyRolloutPool | None = None
    if cfg.use_stratified_buffer:
        rollout_pool = JointPolicyRolloutPool(counties, canvas, channel_cfg=channel_cfg)
        n_warm = warm_start_joint_policy_buffer(
            agent, buffer, rollout_pool, int(cfg.policy_warmup_transitions), rng
        )
        if cfg.verbose:
            print(f"[EB-IQ] 策略 buffer 预填充 {n_warm} 条（8 县随机 rollout）", flush=True)

    eval_every = max(1, int(cfg.eval_every))
    metrics_log: list[dict[str, Any]] = []
    metrics_png = out / "joint_iq_training_metrics.png"

    best_dist = float("inf")
    best_match = 0.0
    best_state: dict[str, Any] | None = None

    if cfg.verbose:
        print(
            f"[EB-IQ] 联合训练 {len(counties)} 县，画布 {canvas.max_h}×{canvas.max_w}，"
            f"专家转移 {len(merged)} 条，obs 通道 {channel_cfg.n_channels}，"
            f"county_embed_dim={cfg.county_embed_dim}，IQ={cfg.iq_loss_mode}",
            flush=True,
        )
        for c in counties:
            print(f"  id={c.county_id} {c.county_name} grid {c.H}×{c.W} npz={Path(c.grid_npz).name}", flush=True)

    for step in range(1, int(cfg.train_steps) + 1):
        if rollout_pool is not None:
            rollout_pool.collect_one_step(agent, buffer, rng, stochastic=True)

        batch = buffer.sample(int(cfg.batch_size), agent.device, rng)
        m = agent.train_step(batch)

        if step % eval_every != 0:
            continue

        per_county, agg = evaluate_joint_all(agent, counties, canvas, channel_cfg)
        row: dict[str, Any] = {
            "step": step,
            "mean_expert_match_rate": agg["mean_expert_match_rate"],
            "mean_distance_km": agg["mean_distance_km"],
            "Q_mean": m["Q_mean"],
            "Q_std": m["Q_std"],
            "loss": m["loss"],
            "per_county": [
                {
                    "county_name": p["county_name"],
                    "expert_rollin_expert_greedy_match_rate": p["expert_rollin_expert_greedy_match_rate"],
                    "expert_rollin_mean_distance_km": p["expert_rollin_mean_distance_km"],
                }
                for p in per_county
            ],
        }
        # 跨县平均，供 plot_iq_metrics 绘制
        for prefix in ("expert_rollin_", "policy_rollout_"):
            keys = {k[len(prefix) :] for k in per_county[0] if k.startswith(prefix)}
            for short in keys:
                full = f"{prefix}{short}"
                vals = [float(p.get(full, 0.0)) for p in per_county]
                row[full] = float(np.mean(vals)) if vals else 0.0
        metrics_log.append(row)

        if cfg.verbose:
            print(
                f"step {step}: match={agg['mean_expert_match_rate']:.3f} "
                f"dist_km={agg['mean_distance_km']:.2f} loss={m['loss']:.3f}",
                flush=True,
            )

        if agg["mean_distance_km"] < best_dist:
            best_dist = float(agg["mean_distance_km"])
            best_match = float(agg["mean_expert_match_rate"])
            best_state = copy.deepcopy(agent.q_net.state_dict())
            agent.save(str(out / "iq_learn_shared_best.pt"))

    if rollout_pool is not None:
        rollout_pool.close()

    if best_state is not None:
        agent.q_net.load_state_dict(best_state)
        agent.target_net.load_state_dict(best_state)

    per_final, agg_final = evaluate_joint_all(agent, counties, canvas, channel_cfg)
    if cfg.verbose:
        print(
            f"[EB-IQ] final match={agg_final['mean_expert_match_rate']:.3f} "
            f"dist_km={agg_final['mean_distance_km']:.2f}",
            flush=True,
        )

    if PLOT_EVAL_METRICS_AT_END and metrics_log:
        plot_iq_metrics(
            metrics_log,
            metrics_png,
            title=f"Joint IQ-Learn ({len(counties)} counties, embed={cfg.county_embed_dim})",
        )

    summary: dict[str, Any] = {
        "training_mode": "joint_8counties_shared_q",
        "counties": county_names,
        "county_embed_dim": int(cfg.county_embed_dim),
        "canvas": {"max_h": canvas.max_h, "max_w": canvas.max_w, "n_actions": canvas.n_actions},
        "merged_meta": merged.meta,
        "seed": seed,
        "mdp_mode": current_mdp_mode(),
        "method": "IQ-Learn joint SimpleGridCNN + county embedding",
        "network": "SimpleGridCNN+CountyEmbed",
        "n_expert_transitions_pooled": len(merged),
        "train_steps_ran": int(cfg.train_steps),
        "eval_every": eval_every,
        "best_mean_expert_match_rate": best_match,
        "best_mean_distance_km": best_dist,
        "final_mean_expert_match_rate": agg_final["mean_expert_match_rate"],
        "final_mean_distance_km": agg_final["mean_distance_km"],
        "per_county_final": per_final,
        "metrics_log": metrics_log,
        "metrics_plot": str(metrics_png) if PLOT_EVAL_METRICS_AT_END else "",
    }
    return agent, summary


def run_joint_training(cfg: JointTrainConfig) -> dict[str, Any]:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if cfg.verbose:
        print(
            f"[EB-IQ] 启动联合训练：{len(cfg.grid_npz_paths)} 县，"
            f"steps={cfg.train_steps} eval_every={cfg.eval_every}",
            flush=True,
        )

    agent, train_summary = train_joint(cfg)
    train_end = date.today()
    ckpt_path = policy_checkpoint_path(out, "IQ", train_end=train_end, seed=int(cfg.seed))
    agent.save(str(ckpt_path))
    agent.save(str(out / "iq_learn_shared.pt"))

    summary = {
        "config": asdict(cfg),
        "train_end_date": train_end.isoformat(),
        "policy_checkpoint": ckpt_path.name,
        "policy_path": str(ckpt_path),
        **train_summary,
    }
    with (out / "iq_learn_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (out / "metrics_log.json").open("w", encoding="utf-8") as f:
        json.dump(train_summary.get("metrics_log", []), f, indent=2, ensure_ascii=False)
    return summary
