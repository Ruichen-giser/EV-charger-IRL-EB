"""Smoke test: EB 联合 IQ 输出与 EV-charger-IRL 对齐（评估指标 / 可视化 / Q 网络存储）。"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from cnn_config import DEFAULT_OBS_CHANNEL_NAMES  # noqa: E402
from iq_learn.discrete_soft_q import DiscreteSoftQAgent  # noqa: E402
from iq_learn.evaluate import evaluate_joint_all  # noqa: E402
from iq_learn.expert_data import CountyLayout  # noqa: E402
from iq_learn.grid_align import JointGridCanvas  # noqa: E402
from iq_learn.metrics import (  # noqa: E402
    EXPERT_ROLLIN_CORE_KEYS,
    POLICY_ROLLOUT_CORE_KEYS,
    POLICY_ROLLOUT_LCSS_KEYS,
    aggregate_joint_eval_metrics,
    build_joint_final_eval,
    eval_metrics_for_log,
    merge_eval_dicts,
    empty_eval_dict,
)
from iq_learn.plotting import plot_iq_metrics  # noqa: E402
from obs_channels import ObsChannelConfig  # noqa: E402
from paths import policy_checkpoint_filename  # noqa: E402

# EV-charger-IRL train_single.py 中 metrics_log 每步写入的键（跨县版为 aggregate 后的同名键）
IRL_METRICS_LOG_TRAIN_KEYS = (
    "step",
    "Q_mean",
    "Q_std",
    "Q_max",
    "policy_entropy",
    "loss",
    *(f"expert_rollin_{k}" for k in EXPERT_ROLLIN_CORE_KEYS),
    *(f"policy_rollout_{k}" for k in POLICY_ROLLOUT_CORE_KEYS),
    *(f"policy_rollout_{k}" for k in POLICY_ROLLOUT_LCSS_KEYS),
)

# plot_iq_metrics 读取的评估曲线键
PLOT_EVAL_KEYS = (
    *(f"expert_rollin_{k}" for k in EXPERT_ROLLIN_CORE_KEYS),
    *(f"policy_rollout_{k}" for k in POLICY_ROLLOUT_CORE_KEYS),
    *(f"policy_rollout_{k}" for k in POLICY_ROLLOUT_LCSS_KEYS),
)


def _write_toy_npz(path: Path, *, county_name: str, h: int, w: int, expert_actions: list[int]) -> None:
    np.savez(
        path,
        grid_features=np.random.rand(h, w, 5).astype(np.float32),
        valid_mask=np.ones((h, w), dtype=bool),
        expert_actions=np.asarray(expert_actions, dtype=np.int64),
        grid_cell_km=np.float32(2.0),
        county_name=np.asarray([county_name]),
    )


def _expected_metric_keys() -> set[str]:
    return set(PLOT_EVAL_KEYS)


def _assert_numeric_metrics(row: dict, *, label: str) -> None:
    expected = _expected_metric_keys()
    missing = expected - set(row)
    if missing:
        raise AssertionError(f"{label}: 缺少指标键 {sorted(missing)}")
    for key in expected:
        val = row[key]
        if not isinstance(val, (int, float)):
            raise AssertionError(f"{label}: {key}={val!r} 不是数值")
    print(f"[smoke] {label}: OK ({len(expected)} 个数值指标)")


def _assert_metrics_log_row(row: dict) -> None:
    missing = [k for k in IRL_METRICS_LOG_TRAIN_KEYS if k not in row]
    if missing:
        raise AssertionError(f"metrics_log 行缺少 EV-charger-IRL 对齐键: {missing}")
    _assert_numeric_metrics(row, label="metrics_log row (IRL-aligned)")


def main() -> None:
    per_mock = [
        {
            "county_name": "A",
            **merge_eval_dicts(
                empty_eval_dict(eval_mode="expert_rollin"),
                empty_eval_dict(eval_mode="policy_rollout"),
            ),
        }
    ]
    agg_mock = aggregate_joint_eval_metrics(per_mock)
    _assert_numeric_metrics(agg_mock, label="aggregate_joint_eval_metrics(mock)")

    single = eval_metrics_for_log(per_mock[0])
    if set(single) != set(agg_mock):
        raise AssertionError("aggregate_joint_eval_metrics 与 eval_metrics_for_log 键不一致")
    print("[smoke] 评估指标键集与 EV-charger-IRL eval_metrics_for_log 一致")

    channel_cfg = ObsChannelConfig(names=tuple(DEFAULT_OBS_CHANNEL_NAMES))
    with tempfile.TemporaryDirectory(prefix="eb_iq_smoke_") as tmp:
        tmp_path = Path(tmp)
        specs = [
            ("Smoke_A", 4, 5, [0, 1, 6, 11]),
            ("Smoke_B", 3, 6, [0, 7, 13]),
        ]
        counties: list[CountyLayout] = []
        for cid, (name, h, w, actions) in enumerate(specs):
            npz = tmp_path / f"{name}_grid_cropped.npz"
            _write_toy_npz(npz, county_name=name, h=h, w=w, expert_actions=actions)
            counties.append(
                CountyLayout(
                    county_id=cid,
                    county_name=name,
                    grid_npz=str(npz),
                    H=h,
                    W=w,
                    cell_km=2.0,
                    n_obs_channels=channel_cfg.n_channels,
                )
            )

        canvas = JointGridCanvas(max_h=max(c.H for c in counties), max_w=max(c.W for c in counties))
        agent = DiscreteSoftQAgent(
            obs_dim=int(canvas.max_h * canvas.max_w * channel_cfg.n_channels),
            n_actions=int(canvas.n_actions),
            device="cpu",
            grid_h=int(canvas.max_h),
            grid_w=int(canvas.max_w),
            in_channels=int(channel_cfg.n_channels),
            n_counties=len(counties),
            county_embed_dim=8,
            county_names=[c.county_name for c in counties],
        )

        per_county, _ = evaluate_joint_all(agent, counties, canvas, channel_cfg)
        for p in per_county:
            _assert_numeric_metrics(
                {k: p[k] for k in _expected_metric_keys()},
                label=f"per_county {p['county_name']}",
            )

        row = {
            "step": 1,
            "Q_mean": 0.1,
            "Q_std": 0.2,
            "Q_max": 0.3,
            "policy_entropy": 1.5,
            "loss": 0.4,
            **aggregate_joint_eval_metrics(per_county),
        }
        _assert_metrics_log_row(row)

        final_ev = build_joint_final_eval(per_county)
        for prefix in ("expert_rollin_", "policy_rollout_"):
            block = {k: final_ev[k] for k in final_ev if str(k).startswith(prefix)}
            if not block:
                raise AssertionError(f"final_eval 缺少 {prefix}* 块")
        print("[smoke] final_eval / final_eval_expert_rollin / final_eval_policy_rollout 结构 OK")

        plot_path = tmp_path / "joint_iq_training_metrics.png"
        plot_iq_metrics([row, {**row, "step": 2}], plot_path, title="smoke")
        if not plot_path.is_file() or plot_path.stat().st_size <= 0:
            raise AssertionError("plot_iq_metrics 未生成 PNG")
        print(f"[smoke] 评估指标可视化 OK → {plot_path.name}")

        ckpt_name = policy_checkpoint_filename("IQ", seed=0)
        ckpt_path = tmp_path / ckpt_name
        agent.save(str(ckpt_path))
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        required = {
            "network",
            "q_net",
            "target_net",
            "alpha",
            "gamma",
            "grid_h",
            "grid_w",
            "in_channels",
            "n_actions",
            "n_counties",
            "county_embed_dim",
            "county_names",
        }
        missing_ckpt = required - set(blob)
        if missing_ckpt:
            raise AssertionError(f"checkpoint 缺少字段: {sorted(missing_ckpt)}")
        agent2 = DiscreteSoftQAgent(
            obs_dim=int(canvas.max_h * canvas.max_w * channel_cfg.n_channels),
            n_actions=int(canvas.n_actions),
            device="cpu",
            grid_h=int(canvas.max_h),
            grid_w=int(canvas.max_w),
            in_channels=int(channel_cfg.n_channels),
            n_counties=len(counties),
            county_embed_dim=8,
            county_names=[c.county_name for c in counties],
        )
        agent2.load(str(ckpt_path))
        print(f"[smoke] Q 网络存储 OK → {ckpt_name} (network={blob['network']})")

    print("[smoke] 全部通过")


if __name__ == "__main__":
    main()
