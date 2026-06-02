"""
IRL_IQ 入口：IQ-Learn + SimpleGridCNN。

本目录为独立包。默认 --mdp-mode legacy。
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
    default_output_dir,
    ensure_grid_npz_dir,
)
from mdp_config import apply_mdp_mode, current_mdp_mode  # noqa: E402

from cnn_config import CNN_DROPOUT, DEFAULT_COUNTY, DEFAULT_OBS_CHANNEL_NAMES  # noqa: E402
from data_prep import prepare_county_npz  # noqa: E402
from iq_learn.train_single import SingleCountyTrainConfig, run_single_county  # noqa: E402


@dataclass
class MainConfig:
    county: str = DEFAULT_COUNTY
    mdp_mode: str = "legacy"
    grid_npz_dir: str = field(default_factory=lambda: str(DEFAULT_GRID_NPZ_DIR))
    output_dir: str = ""
    gamma: float = 0.99
    alpha: float = 0.01
    alpha_reg: float = 0.5
    use_chi: bool = True
    lr: float = 5e-5
    batch_size: int = 64
    train_steps: int = 1_000_000
    eval_every: int = 200
    target_update_interval: int = 15
    seed: int = 0
    device: str = "auto"
    dropout: float = CNN_DROPOUT
    obs_channel_names: tuple[str, ...] = DEFAULT_OBS_CHANNEL_NAMES
    expert_batch_fraction: float = 0.5
    policy_buffer_capacity: int = 10_000
    policy_warmup_transitions: int = 100
    verbose: int = 1


def _print_mdp_mode() -> None:
    if current_mdp_mode() == "legacy":
        print(
            "[IQ] mdp-mode=legacy：每格仅建一次；专家 first_visit 去重；"
            "mask = valid_mask & ~stations",
            flush=True,
        )
    else:
        print(
            "[IQ] mdp-mode=repeat：完整 OpenDate 专家序列；允许同格重复；"
            "mask = valid_mask",
            flush=True,
        )


def run_main(cfg: MainConfig) -> dict:
    apply_mdp_mode(cfg.mdp_mode)
    grid_dir = Path(cfg.grid_npz_dir)
    ensure_grid_npz_dir(grid_dir)
    _print_mdp_mode()

    npz = prepare_county_npz(grid_dir, cfg.county, log_prefix="IQ")
    out = Path(cfg.output_dir) if cfg.output_dir else default_output_dir(cfg.county, DEFAULT_IQ_OUTPUT_DIR)

    summary = run_single_county(
        SingleCountyTrainConfig(
            county=cfg.county,
            grid_npz_paths=[str(npz)],
            output_dir=str(out),
            gamma=float(cfg.gamma),
            alpha=float(cfg.alpha),
            alpha_reg=float(cfg.alpha_reg),
            use_chi=bool(cfg.use_chi),
            lr=float(cfg.lr),
            batch_size=int(cfg.batch_size),
            train_steps=int(cfg.train_steps),
            eval_every=int(cfg.eval_every),
            target_update_interval=int(cfg.target_update_interval),
            seed=int(cfg.seed),
            device=str(cfg.device),
            dropout=float(cfg.dropout),
            obs_channel_names=tuple(cfg.obs_channel_names),
            expert_batch_fraction=float(cfg.expert_batch_fraction),
            policy_buffer_capacity=int(cfg.policy_buffer_capacity),
            policy_warmup_transitions=int(cfg.policy_warmup_transitions),
            verbose=int(cfg.verbose),
        )
    )
    summary["mdp_mode"] = current_mdp_mode()
    print(
        f"[IQ] {cfg.county} 完成 "
        f"expert_rollin: match={summary['mean_expert_match_rate']:.3f} "
        f"top10={summary['top10_accuracy']:.3f} "
        f"dist_km={summary['mean_distance_km']:.2f} | "
        f"policy_rollout: prec={summary['final_eval_policy_rollout'].get('policy_rollout_site_precision', 0):.3f} "
        f"recall={summary['final_eval_policy_rollout'].get('policy_rollout_site_recall', 0):.3f} "
        f"jaccard={summary['final_eval_policy_rollout'].get('policy_rollout_jaccard_similarity', 0):.3f} "
        f"dist_km={summary['final_eval_policy_rollout'].get('policy_rollout_mean_distance_km', 0):.2f} "
        f"ckpt={summary.get('policy_checkpoint', '')} → {out}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="IRL_IQ：IQ-Learn，默认 Los_Angeles + legacy MDP")
    parser.add_argument(
        "--mdp-mode",
        type=str,
        default="legacy",
        choices=("legacy", "repeat"),
        help="legacy=旧版(默认); repeat=完整专家序列可重复格",
    )
    parser.add_argument("--county", type=str, default=DEFAULT_COUNTY)
    parser.add_argument("--grid-npz-dir", type=str, default=str(DEFAULT_GRID_NPZ_DIR))
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--train-steps", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=CNN_DROPOUT)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu", "mps"))
    parser.add_argument("--seed", type=int, default=0, help="随机种子（影响训练与 checkpoint 文件名）")
    args = parser.parse_args()

    run_main(
        MainConfig(
            mdp_mode=str(args.mdp_mode),
            county=args.county.replace(" ", "_"),
            grid_npz_dir=args.grid_npz_dir,
            output_dir=args.output_dir,
            train_steps=int(args.train_steps),
            eval_every=int(args.eval_every),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            dropout=float(args.dropout),
            device=str(args.device),
            seed=int(args.seed),
        )
    )


if __name__ == "__main__":
    main()
