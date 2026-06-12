"""
IRL_IQ-EB 入口：多县联合 IQ-Learn + SimpleGridCNN + state/county embedding。

默认：MDP≥5 Excel 全量县（1149）联合训练；--dev-8-counties 为 8 县调试模式。
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from paths import (  # noqa: E402
    DEFAULT_GRID_NPZ_DIR,
    DEFAULT_IQ_OUTPUT_DIR,
    DEFAULT_MDP_GE5_XLSX,
    ensure_grid_npz_dir,
)
from mdp_config import apply_mdp_mode, current_mdp_mode  # noqa: E402
from cnn_config import (  # noqa: E402
    CNN_DROPOUT,
    COUNTY_EMBED_DIM,
    COUNTY_META_DIM,
    DEFAULT_EVAL_FINAL_MAX_COUNTIES,
    DEFAULT_EVAL_MAX_COUNTIES,
    DEFAULT_IQ_LOSS_MODE,
    DEFAULT_OBS_CHANNEL_NAMES,
    DEFAULT_ROLLOUT_MAX_SESSIONS,
    DEV_STATE_COUNTIES_8,
    EMBED_DROPOUT,
    META_MLP_HIDDEN,
    N_MAX_COUNTY_RESIDUAL,
    RESIDUAL_ALPHA,
)
from data_prep import prepare_state_counties_npz  # noqa: E402
from iq_learn.train_joint import JointTrainConfig, run_joint_training  # noqa: E402
from state_county import (  # noqa: E402
    StateCountyPair,
    filter_pairs_with_grid_npz,
    load_default_training_counties,
    load_mdp_ge5_county_list,
    parse_state_county_list,
)


@dataclass
class MainConfig:
    state_counties: tuple[StateCountyPair, ...] = ()
    mdp_ge5_xlsx: str = ""
    mdp_mode: str = "legacy"
    grid_npz_dir: str = field(default_factory=lambda: str(DEFAULT_GRID_NPZ_DIR))
    output_dir: str = field(default_factory=lambda: str(DEFAULT_IQ_OUTPUT_DIR))
    gamma: float = 0.99
    alpha: float = 0.01
    alpha_reg: float = 0.5
    use_chi: bool = True
    iq_loss_mode: str = DEFAULT_IQ_LOSS_MODE
    lr: float = 5e-5
    batch_size: int = 64
    train_steps: int = 50_000
    eval_every: int = 200
    eval_max_counties: int = DEFAULT_EVAL_MAX_COUNTIES
    eval_final_max_counties: int = DEFAULT_EVAL_FINAL_MAX_COUNTIES
    rollout_max_sessions: int = DEFAULT_ROLLOUT_MAX_SESSIONS
    target_update_interval: int = 15
    seed: int = 0
    device: str = "auto"
    dropout: float = CNN_DROPOUT
    embed_dim: int = COUNTY_EMBED_DIM
    meta_dim: int = COUNTY_META_DIM
    meta_hidden: int = META_MLP_HIDDEN
    n_residual: int = N_MAX_COUNTY_RESIDUAL
    residual_alpha: float = RESIDUAL_ALPHA
    embed_dropout: float = EMBED_DROPOUT
    obs_channel_names: tuple[str, ...] = DEFAULT_OBS_CHANNEL_NAMES
    expert_batch_fraction: float = 1.0
    policy_buffer_capacity: int = 50_000
    policy_warmup_transitions: int = 200
    verbose: int = 1


def run_main(cfg: MainConfig) -> dict:
    apply_mdp_mode(cfg.mdp_mode)
    grid_dir = Path(cfg.grid_npz_dir)
    ensure_grid_npz_dir(grid_dir)

    if current_mdp_mode() == "legacy":
        print("[EB-IQ] mdp-mode=legacy：每格仅建一次；mask = valid & ~stations", flush=True)
    else:
        print("[EB-IQ] mdp-mode=repeat：完整专家序列", flush=True)

    pairs = list(cfg.state_counties)
    if not pairs:
        raise ValueError("state_counties 为空，请检查县清单配置")

    npz_paths = prepare_state_counties_npz(grid_dir, pairs, log_prefix="EB-IQ")
    out = Path(cfg.output_dir)
    mdp_xlsx = str(cfg.mdp_ge5_xlsx).strip() or str(DEFAULT_MDP_GE5_XLSX)

    summary = run_joint_training(
        JointTrainConfig(
            grid_npz_paths=[str(p) for p in npz_paths],
            output_dir=str(out),
            state_counties=tuple(pairs),
            mdp_ge5_xlsx=mdp_xlsx,
            gamma=float(cfg.gamma),
            alpha=float(cfg.alpha),
            alpha_reg=float(cfg.alpha_reg),
            use_chi=bool(cfg.use_chi),
            iq_loss_mode=str(cfg.iq_loss_mode),
            lr=float(cfg.lr),
            batch_size=int(cfg.batch_size),
            train_steps=int(cfg.train_steps),
            eval_every=int(cfg.eval_every),
            eval_max_counties=int(cfg.eval_max_counties),
            eval_final_max_counties=int(cfg.eval_final_max_counties),
            rollout_max_sessions=int(cfg.rollout_max_sessions),
            target_update_interval=int(cfg.target_update_interval),
            seed=int(cfg.seed),
            device=str(cfg.device),
            dropout=float(cfg.dropout),
            embed_dim=int(cfg.embed_dim),
            meta_dim=int(cfg.meta_dim),
            meta_hidden=int(cfg.meta_hidden),
            n_residual=int(cfg.n_residual),
            residual_alpha=float(cfg.residual_alpha),
            embed_dropout=float(cfg.embed_dropout),
            obs_channel_names=tuple(cfg.obs_channel_names),
            expert_batch_fraction=float(cfg.expert_batch_fraction),
            policy_buffer_capacity=int(cfg.policy_buffer_capacity),
            policy_warmup_transitions=int(cfg.policy_warmup_transitions),
            verbose=int(cfg.verbose),
        )
    )
    print(
        f"[EB-IQ] 联合训练完成 match={summary['mean_expert_match_rate']:.3f} "
        f"dist_km={summary['mean_distance_km']:.2f} "
        f"ckpt={summary.get('policy_checkpoint', '')} → {out}",
        flush=True,
    )
    return summary


def _resolve_state_counties(args: argparse.Namespace) -> tuple[StateCountyPair, ...]:
    if args.dev_8_counties:
        print(f"[EB-IQ] dev 模式：{len(DEV_STATE_COUNTIES_8)} 县", flush=True)
        return DEV_STATE_COUNTIES_8
    if args.state_counties:
        pairs = parse_state_county_list(args.state_counties)
        print(f"[EB-IQ] 自定义清单：{len(pairs)} 个 state-county", flush=True)
        return tuple(pairs)
    if args.counties:
        counties = [x.strip().replace("_", " ") for x in args.counties.split(",") if x.strip()]
        state = str(args.state).strip() or "California"
        pairs = [StateCountyPair(state, c) for c in counties]
        print(f"[EB-IQ] 单州 {state}：{len(pairs)} 县", flush=True)
        return tuple(pairs)
    if args.mdp_ge5_xlsx:
        pairs = load_mdp_ge5_county_list(args.mdp_ge5_xlsx)
        pairs, missing = filter_pairs_with_grid_npz(pairs, args.grid_npz_dir)
        if missing:
            print(
                f"[EB-IQ] 警告: {len(missing)} 县缺少 npz，将训练已有数据的 {len(pairs)} 县",
                flush=True,
            )
        print(f"[EB-IQ] 从 Excel 加载 {len(pairs)} 个 state-county（已有 npz）", flush=True)
        return tuple(pairs)
    pairs = load_default_training_counties(grid_dir=args.grid_npz_dir)
    print(
        f"[EB-IQ] 默认 MDP≥5 清单：{len(pairs)} 个 state-county（已有 npz）",
        flush=True,
    )
    return tuple(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IRL_IQ-EB：联合 IQ-Learn + state/county embedding（默认 MDP≥5 全量县）"
    )
    parser.add_argument("--mdp-mode", type=str, default="legacy", choices=("legacy", "repeat"))
    parser.add_argument(
        "--dev-8-counties",
        action="store_true",
        help="8 县 CA 调试模式（覆盖默认 MDP≥5 全量清单）",
    )
    parser.add_argument(
        "--state-counties",
        type=str,
        default="",
        help='逗号分隔 "California/Los Angeles,California/Sacramento"',
    )
    parser.add_argument(
        "--mdp-ge5-xlsx",
        type=str,
        default="",
        help=f"指定 MDP≥5 Excel（默认 {DEFAULT_MDP_GE5_XLSX.name}）",
    )
    parser.add_argument("--state", type=str, default="California", help="与 --counties 联用（单州多县）")
    parser.add_argument(
        "--counties",
        type=str,
        default="",
        help="兼容旧 API：逗号分隔县名（须配合 --state）",
    )
    parser.add_argument("--grid-npz-dir", type=str, default=str(DEFAULT_GRID_NPZ_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_IQ_OUTPUT_DIR))
    parser.add_argument("--train-steps", type=int, default=50_000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument(
        "--eval-max-counties",
        type=int,
        default=DEFAULT_EVAL_MAX_COUNTIES,
        help="训练中周期性评估（含 policy rollout）最多县数；0=全量",
    )
    parser.add_argument(
        "--eval-final-max-counties",
        type=int,
        default=DEFAULT_EVAL_FINAL_MAX_COUNTIES,
        help="训练结束最终评估最多县数；0=全量 rollout 所有县",
    )
    parser.add_argument(
        "--rollout-max-sessions",
        type=int,
        default=DEFAULT_ROLLOUT_MAX_SESSIONS,
        help="iq-loss-mode=online 时 rollout 池活跃 session 上限",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=CNN_DROPOUT)
    parser.add_argument("--embed-dim", type=int, default=COUNTY_EMBED_DIM)
    parser.add_argument("--meta-dim", type=int, default=COUNTY_META_DIM)
    parser.add_argument("--residual-alpha", type=float, default=RESIDUAL_ALPHA)
    parser.add_argument(
        "--iq-loss-mode",
        type=str,
        default=DEFAULT_IQ_LOSS_MODE,
        choices=("online", "offline"),
        help="offline=仅专家轨迹（默认）；online=专家+策略 rollout 混合",
    )
    parser.add_argument("--no-chi", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu", "mps"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_main(
        MainConfig(
            state_counties=_resolve_state_counties(args),
            mdp_ge5_xlsx=str(args.mdp_ge5_xlsx),
            mdp_mode=str(args.mdp_mode),
            grid_npz_dir=args.grid_npz_dir,
            output_dir=args.output_dir,
            train_steps=int(args.train_steps),
            eval_every=int(args.eval_every),
            eval_max_counties=int(args.eval_max_counties),
            eval_final_max_counties=int(args.eval_final_max_counties),
            rollout_max_sessions=int(args.rollout_max_sessions),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            dropout=float(args.dropout),
            embed_dim=int(args.embed_dim),
            meta_dim=int(args.meta_dim),
            residual_alpha=float(args.residual_alpha),
            iq_loss_mode=str(args.iq_loss_mode),
            use_chi=not bool(args.no_chi),
            device=str(args.device),
            seed=int(args.seed),
        )
    )


if __name__ == "__main__":
    main()
