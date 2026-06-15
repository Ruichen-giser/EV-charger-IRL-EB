"""多县联合 IQ-Learn：共享 SimpleGridCNN + county embedding（默认 MDP≥5 全量）。"""
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
    COUNTY_META_DIM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_FINAL_MAX_COUNTIES,
    DEFAULT_EVAL_MAX_COUNTIES,
    DEFAULT_EVAL_EVERY,
    DEFAULT_IQ_LOSS_MODE,
    DEFAULT_LR,
    DEFAULT_OBS_CHANNEL_NAMES,
    DEFAULT_ROLLOUT_MAX_SESSIONS,
    DEFAULT_TRAIN_STEPS,
    DEFAULT_TARGET_UPDATE_INTERVAL,
    EMBED_DROPOUT,
    META_MLP_HIDDEN,
    N_MAX_COUNTY_RESIDUAL,
    N_US_STATES,
    RESIDUAL_ALPHA,
)
from county_meta import US_STATE_NAMES, state_name_to_id
from state_county import StateCountyPair, build_global_residual_index, build_location_vocab
from iq_learn.discrete_soft_q import DiscreteSoftQAgent
from iq_learn.evaluate import evaluate_joint_all
from iq_learn.metrics import aggregate_joint_eval_metrics, build_joint_final_eval, format_eval_log
from iq_learn.expert_data import build_merged_expert_dataset, concat_training_batches
from iq_learn.embedding_export import export_location_embeddings
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
    experiment: str = "baseline"
    state_counties: tuple[StateCountyPair, ...] = ()
    mdp_ge5_xlsx: str = ""
    gamma: float = 0.99
    alpha: float = 0.01
    alpha_reg: float = 0.5
    use_chi: bool = True
    iq_loss_mode: str = DEFAULT_IQ_LOSS_MODE
    lr: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    train_steps: int = DEFAULT_TRAIN_STEPS
    eval_every: int = DEFAULT_EVAL_EVERY
    target_update_interval: int = DEFAULT_TARGET_UPDATE_INTERVAL
    seed: int = 0
    device: str = "auto"
    dropout: float = CNN_DROPOUT
    obs_channel_names: tuple[str, ...] = DEFAULT_OBS_CHANNEL_NAMES
    embed_dim: int = COUNTY_EMBED_DIM
    meta_dim: int = COUNTY_META_DIM
    meta_hidden: int = META_MLP_HIDDEN
    n_residual: int = N_MAX_COUNTY_RESIDUAL
    residual_alpha: float = RESIDUAL_ALPHA
    embed_dropout: float = EMBED_DROPOUT
    use_stratified_buffer: bool = True
    expert_batch_fraction: float = 1.0
    policy_buffer_capacity: int = 50_000
    policy_warmup_transitions: int = 200
    eval_max_counties: int = DEFAULT_EVAL_MAX_COUNTIES
    eval_final_max_counties: int = DEFAULT_EVAL_FINAL_MAX_COUNTIES
    rollout_max_sessions: int = DEFAULT_ROLLOUT_MAX_SESSIONS
    verbose: int = 1


def _sample_train_batch(
    *,
    merged,
    buffer: StratifiedReplayBuffer | None,
    batch_size: int,
    expert_fraction: float,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    bs = int(batch_size)
    if buffer is None:
        return merged.sample(bs, device, rng)

    n_pol_avail = len(buffer._policy)
    if n_pol_avail <= 0:
        return merged.sample(bs, device, rng)

    n_exp = max(1, int(round(bs * float(expert_fraction))))
    n_exp = min(n_exp, len(merged), bs)
    n_pol = bs - n_exp
    if n_pol > n_pol_avail:
        n_pol = n_pol_avail
        n_exp = bs - n_pol

    expert_batch = merged.sample(n_exp, device, rng)
    if n_pol <= 0:
        return expert_batch
    policy_batch = buffer.sample_policy_only(n_pol, device, rng)
    return concat_training_batches(expert_batch, policy_batch)


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

    if len(cfg.grid_npz_paths) != len(cfg.state_counties):
        raise ValueError(
            f"grid_npz_paths ({len(cfg.grid_npz_paths)}) 与 state_counties "
            f"({len(cfg.state_counties)}) 数量不一致"
        )

    channel_cfg = ObsChannelConfig(names=tuple(cfg.obs_channel_names))
    pairs = list(cfg.state_counties)
    pair_to_loc_id, _, _, location_labels = build_location_vocab(pairs)
    state_ids_for_paths = [state_name_to_id(p.state_name) for p in pairs]
    if str(cfg.mdp_ge5_xlsx).strip():
        global_idx = build_global_residual_index(cfg.mdp_ge5_xlsx)
        county_ids_for_paths = [
            global_idx.get((p.state_name, p.county_name), pair_to_loc_id[(p.state_name, p.county_name)])
            for p in pairs
        ]
    else:
        county_ids_for_paths = [pair_to_loc_id[(p.state_name, p.county_name)] for p in pairs]
    merged = build_merged_expert_dataset(
        cfg.grid_npz_paths,
        channel_cfg=channel_cfg,
        state_ids=state_ids_for_paths,
        county_ids=county_ids_for_paths,
    )
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
        embed_dim=int(cfg.embed_dim),
        meta_dim=int(cfg.meta_dim),
        meta_hidden=int(cfg.meta_hidden),
        n_states=N_US_STATES,
        n_residual=int(cfg.n_residual),
        residual_alpha=float(cfg.residual_alpha),
        embed_dropout=float(cfg.embed_dropout),
        dropout=float(cfg.dropout),
        county_names=county_names,
        state_names=list(US_STATE_NAMES),
        location_labels=location_labels,
    )

    n_expert = len(merged)
    use_policy_rollout = str(cfg.iq_loss_mode).strip().lower() == "online"
    buffer: StratifiedReplayBuffer | None = None
    if use_policy_rollout:
        buffer = StratifiedReplayBuffer(
            capacity_expert=0,
            capacity_policy=int(cfg.policy_buffer_capacity),
            expert_fraction=float(cfg.expert_batch_fraction),
            spatial=True,
        )
    rollout_pool: JointPolicyRolloutPool | None = None
    if use_policy_rollout:
        if not cfg.use_stratified_buffer:
            raise ValueError("iq_loss_mode=online 需要 use_stratified_buffer=True")
        rollout_pool = JointPolicyRolloutPool(
            counties,
            canvas,
            channel_cfg=channel_cfg,
            max_active_sessions=int(cfg.rollout_max_sessions),
        )
        n_warm = warm_start_joint_policy_buffer(
            agent, buffer, rollout_pool, int(cfg.policy_warmup_transitions), rng
        )
        if cfg.verbose:
            print(
                f"[EB-IQ] 策略 buffer 预填充 {n_warm} 条（{len(counties)} 县随机 rollout，"
                f"活跃 session≤{cfg.rollout_max_sessions}）",
                flush=True,
            )
    elif cfg.verbose:
        print("[EB-IQ] iq_loss_mode=offline：仅专家轨迹训练，跳过策略 rollout 池", flush=True)

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
            f"experiment={cfg.experiment} embed_dim={cfg.embed_dim} meta_dim={cfg.meta_dim} "
            f"residual_alpha={cfg.residual_alpha}，IQ={cfg.iq_loss_mode}",
            flush=True,
        )
        canvas_cells = int(canvas.max_h) * int(canvas.max_w)
        if canvas_cells > 10_000 and int(cfg.batch_size) > 32:
            print(
                f"[EB-IQ] 提示: 大画布 ({canvas.max_h}×{canvas.max_w}) batch_size={cfg.batch_size} "
                f"可能超出 12GB GPU 显存，OOM 时请降至 8~16",
                flush=True,
            )
        print(
            f"[EB-IQ] 专家数据 lazy 存储（采样时 pad），避免 ~60GB 全画布缓存",
            flush=True,
        )
        preview = counties if len(counties) <= 8 else counties[:3]
        for c in preview:
            print(
                f"  loc={c.county_id} state={c.state_id} {c.state_name}/{c.county_name} "
                f"grid {c.H}×{c.W} npz={Path(c.grid_npz).name}",
                flush=True,
            )
        if len(counties) > len(preview):
            print(f"  ... 共 {len(counties)} 县", flush=True)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    for step in range(1, int(cfg.train_steps) + 1):
        if rollout_pool is not None:
            rollout_pool.collect_one_step(agent, buffer, rng, stochastic=True)

        batch = _sample_train_batch(
            merged=merged,
            buffer=buffer,
            batch_size=int(cfg.batch_size),
            expert_fraction=float(cfg.expert_batch_fraction),
            device=agent.device,
            rng=rng,
        )
        m = agent.train_step(batch)

        if step % eval_every != 0:
            continue

        eval_cap = int(cfg.eval_max_counties)
        per_county, agg = evaluate_joint_all(
            agent,
            counties,
            canvas,
            channel_cfg,
            max_counties=eval_cap if eval_cap > 0 else None,
            rng=rng,
        )
        # 环境评估：expert_rollin + policy_rollout（与 EV-charger-IRL metrics_log 键对齐）
        row: dict[str, Any] = {
            "step": step,
            "Q_mean": m["Q_mean"],
            "Q_std": m["Q_std"],
            "Q_max": m["Q_max"],
            "policy_entropy": m["policy_entropy"],
            "loss": m["loss"],
            **aggregate_joint_eval_metrics(per_county),
        }
        metrics_log.append(row)

        if cfg.verbose:
            print(
                f"step {step}: Q_mean={m['Q_mean']:.3f}, loss={m['loss']:.3f} | "
                f"{format_eval_log(row)}",
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

    final_cap = int(cfg.eval_final_max_counties)
    per_final, _agg_final = evaluate_joint_all(
        agent,
        counties,
        canvas,
        channel_cfg,
        max_counties=final_cap if final_cap > 0 else None,
        rng=rng,
    )
    final_ev = build_joint_final_eval(per_final)
    if cfg.verbose:
        n_eval = len(per_final)
        scope = f"{n_eval}/{len(counties)} 县"
        if final_cap <= 0 or n_eval >= len(counties):
            scope = f"全量 {len(counties)} 县"
        print(f"[EB-IQ] final eval ({scope}) | {format_eval_log(final_ev)}", flush=True)

    if PLOT_EVAL_METRICS_AT_END and metrics_log:
        plot_iq_metrics(
            metrics_log,
            metrics_png,
            title=f"Joint IQ-Learn ({len(counties)} counties, embed={cfg.embed_dim})",
        )

    summary: dict[str, Any] = {
        "experiment": str(cfg.experiment),
        "training_mode": (
            "joint_mdp_ge5_expert_only"
            if str(cfg.iq_loss_mode).strip().lower() == "offline"
            else "joint_mdp_ge5_shared_q"
        ),
        "iq_loss_mode": str(cfg.iq_loss_mode),
        "eval_max_counties": int(cfg.eval_max_counties),
        "eval_final_max_counties": int(cfg.eval_final_max_counties),
        "n_counties_evaluated_final": len(per_final),
        "state_counties": [p.key for p in pairs],
        "counties": county_names,
        "state_names": list(US_STATE_NAMES),
        "location_labels": location_labels,
        "embed_dim": int(cfg.embed_dim),
        "meta_dim": int(cfg.meta_dim),
        "n_residual": int(cfg.n_residual),
        "residual_alpha": float(cfg.residual_alpha),
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
        "best_policy_checkpoint": str(out / "iq_learn_shared_best.pt"),
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
        "per_county_final": per_final,
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

    embed_export_dir = out / "embedding_export"
    embed_paths = export_location_embeddings(str(ckpt_path), embed_export_dir)
    if cfg.verbose:
        print(f"[EB-IQ] embedding 导出 → {embed_export_dir}", flush=True)

    summary = {
        "config": asdict(cfg),
        "train_end_date": train_end.isoformat(),
        "policy_checkpoint": ckpt_path.name,
        "policy_path": str(ckpt_path),
        "embedding_export": embed_paths,
        **train_summary,
    }
    with (out / "iq_learn_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (out / "metrics_log.json").open("w", encoding="utf-8") as f:
        json.dump(train_summary.get("metrics_log", []), f, indent=2, ensure_ascii=False)
    return summary
