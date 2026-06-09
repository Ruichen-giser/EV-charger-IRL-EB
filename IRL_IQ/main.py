"""
IRL_IQ-EB 入口：8 县联合 IQ-Learn + SimpleGridCNN + county embedding。

共享 Q 网络，SimpleGridCNN 1×1 spatial head（无 GAP）。
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

from paths import DEFAULT_GRID_NPZ_DIR, DEFAULT_IQ_OUTPUT_DIR, ensure_grid_npz_dir  # noqa: E402
from mdp_config import apply_mdp_mode, current_mdp_mode  # noqa: E402
from cnn_config import (  # noqa: E402
    CNN_DROPOUT,
    COUNTY_EMBED_DIM,
    DEFAULT_OBS_CHANNEL_NAMES,
    TRAINING_COUNTIES,
)
from data_prep import prepare_counties_npz  # noqa: E402
from iq_learn.train_joint import JointTrainConfig, run_joint_training  # noqa: E402


@dataclass
class MainConfig:
    counties: tuple[str, ...] = TRAINING_COUNTIES
    mdp_mode: str = "legacy"
    grid_npz_dir: str = field(default_factory=lambda: str(DEFAULT_GRID_NPZ_DIR))
    output_dir: str = field(default_factory=lambda: str(DEFAULT_IQ_OUTPUT_DIR))
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
    county_embed_dim: int = COUNTY_EMBED_DIM
    obs_channel_names: tuple[str, ...] = DEFAULT_OBS_CHANNEL_NAMES
    expert_batch_fraction: float = 0.5
    policy_buffer_capacity: int = 20_000
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

    npz_paths = prepare_counties_npz(grid_dir, cfg.counties, log_prefix="EB-IQ")
    out = Path(cfg.output_dir)

    summary = run_joint_training(
        JointTrainConfig(
            grid_npz_paths=[str(p) for p in npz_paths],
            output_dir=str(out),
            counties=tuple(cfg.counties),
            gamma=float(cfg.gamma),
            alpha=float(cfg.alpha),
            alpha_reg=float(cfg.alpha_reg),
            use_chi=bool(cfg.use_chi),
            iq_loss_mode=str(cfg.iq_loss_mode),
            lr=float(cfg.lr),
            batch_size=int(cfg.batch_size),
            train_steps=int(cfg.train_steps),
            eval_every=int(cfg.eval_every),
            target_update_interval=int(cfg.target_update_interval),
            seed=int(cfg.seed),
            device=str(cfg.device),
            dropout=float(cfg.dropout),
            county_embed_dim=int(cfg.county_embed_dim),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="IRL_IQ-EB：8 县联合 IQ-Learn + county embedding")
    parser.add_argument("--mdp-mode", type=str, default="legacy", choices=("legacy", "repeat"))
    parser.add_argument(
        "--counties",
        type=str,
        default=",".join(TRAINING_COUNTIES),
        help="逗号分隔县名，如 Los_Angeles,Sacramento",
    )
    parser.add_argument("--grid-npz-dir", type=str, default=str(DEFAULT_GRID_NPZ_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_IQ_OUTPUT_DIR))
    parser.add_argument("--train-steps", type=int, default=50_000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=CNN_DROPOUT)
    parser.add_argument("--county-embed-dim", type=int, default=COUNTY_EMBED_DIM)
    parser.add_argument("--iq-loss-mode", type=str, default="online", choices=("online", "offline"))
    parser.add_argument("--no-chi", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu", "mps"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    counties = tuple(x.strip().replace(" ", "_") for x in args.counties.split(",") if x.strip())
    run_main(
        MainConfig(
            counties=counties,
            mdp_mode=str(args.mdp_mode),
            grid_npz_dir=args.grid_npz_dir,
            output_dir=args.output_dir,
            train_steps=int(args.train_steps),
            eval_every=int(args.eval_every),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            dropout=float(args.dropout),
            county_embed_dim=int(args.county_embed_dim),
            iq_loss_mode=str(args.iq_loss_mode),
            use_chi=not bool(args.no_chi),
            device=str(args.device),
            seed=int(args.seed),
        )
    )


if __name__ == "__main__":
    main()
