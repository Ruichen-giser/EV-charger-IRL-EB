"""
IRL_BC 入口：SimpleGridCNN 行为克隆（BC）。

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
    DEFAULT_BC_OUTPUT_DIR,
    DEFAULT_GRID_NPZ_DIR,
    default_output_dir,
    ensure_grid_npz_dir,
)
from mdp_config import apply_mdp_mode, current_mdp_mode  # noqa: E402

import torch  # noqa: E402

from bc_learn.train_bc import BCTrainConfig, run_bc  # noqa: E402
from cnn_config import CNN_DROPOUT, DEFAULT_OBS_CHANNEL_NAMES, DEFAULT_STATE_COUNTY  # noqa: E402
from data_prep import prepare_state_county_npz  # noqa: E402
from state_county import StateCountyPair  # noqa: E402


@dataclass
class MainConfig:
    """BC 训练超参数。"""

    state_county: StateCountyPair = field(default_factory=lambda: DEFAULT_STATE_COUNTY)
    mdp_mode: str = "legacy"
    grid_npz_dir: str = field(default_factory=lambda: str(DEFAULT_GRID_NPZ_DIR))
    output_dir: str = ""
    lr: float = 1e-4
    batch_size: int = 64
    train_steps: int = 1_000_000
    eval_every: int = 200
    seed: int = 0
    device: str = "auto"
    dropout: float = CNN_DROPOUT
    obs_channel_names: tuple[str, ...] = DEFAULT_OBS_CHANNEL_NAMES
    verbose: int = 1


def _print_mdp_mode() -> None:
    mode = current_mdp_mode()
    if mode == "legacy":
        print(
            "[BC] mdp-mode=legacy：每格仅建一次；专家 first_visit 去重；"
            "mask = valid_mask & ~stations",
            flush=True,
        )
    else:
        print(
            "[BC] mdp-mode=repeat：完整 OpenDate 专家序列；允许同格重复；"
            "mask = valid_mask",
            flush=True,
        )


def run_main(cfg: MainConfig) -> dict:
    apply_mdp_mode(cfg.mdp_mode)
    grid_dir = Path(cfg.grid_npz_dir)
    ensure_grid_npz_dir(grid_dir)
    _print_mdp_mode()

    if torch.cuda.is_available():
        print(f"[BC] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    pair = cfg.state_county
    npz = prepare_state_county_npz(grid_dir, pair, log_prefix="BC")
    out = (
        Path(cfg.output_dir)
        if cfg.output_dir
        else default_output_dir(pair, DEFAULT_BC_OUTPUT_DIR)
    )

    summary = run_bc(
        BCTrainConfig(
            grid_npz_path=str(npz),
            output_dir=str(out),
            obs_channel_names=tuple(cfg.obs_channel_names),
            lr=float(cfg.lr),
            batch_size=int(cfg.batch_size),
            train_steps=int(cfg.train_steps),
            eval_every=int(cfg.eval_every),
            seed=int(cfg.seed),
            device=str(cfg.device),
            dropout=float(cfg.dropout),
            verbose=int(cfg.verbose),
        )
    )
    summary["mdp_mode"] = current_mdp_mode()
    print(
        f"[BC] {pair.key} 完成 "
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
    parser = argparse.ArgumentParser(description="IRL_BC：行为克隆，默认 Los_Angeles + legacy MDP")
    parser.add_argument(
        "--mdp-mode",
        type=str,
        default="legacy",
        choices=("legacy", "repeat"),
        help="legacy=旧版(默认): 掩膜排除已建站+专家去重; repeat=完整专家序列可重复格",
    )
    parser.add_argument("--state", type=str, default=DEFAULT_STATE_COUNTY.state_name)
    parser.add_argument("--county", type=str, default=DEFAULT_STATE_COUNTY.county_name.replace(" ", "_"))
    parser.add_argument(
        "--state-county",
        type=str,
        default="",
        help='覆盖 --state/--county，如 "California/Los Angeles"',
    )
    parser.add_argument("--grid-npz-dir", type=str, default=str(DEFAULT_GRID_NPZ_DIR))
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--train-steps", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=CNN_DROPOUT)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu", "mps"))
    parser.add_argument("--seed", type=int, default=0, help="随机种子（影响训练与 checkpoint 文件名）")
    args = parser.parse_args()

    if args.state_county:
        pair = StateCountyPair.parse(args.state_county)
    else:
        pair = StateCountyPair(args.state, args.county.replace("_", " "))

    run_main(
        MainConfig(
            mdp_mode=str(args.mdp_mode),
            state_county=pair,
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
