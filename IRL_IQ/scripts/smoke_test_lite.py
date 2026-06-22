"""无 torch 轻量 smoke test：评估指标汇总 / 输出结构 / 可视化键 / checkpoint 字段。

仅需 Python 3.10+ 标准库；若已安装 matplotlib 会额外测试绘图。
"""
from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

_PKG_ROOT = Path(__file__).resolve().parents[1]

# --- 与 iq_learn/metrics.py 保持一致的键定义（纯 Python，不 import metrics）---
EXPERT_ROLLIN_CORE_KEYS = (
    "expert_greedy_match_rate",
    "top10_accuracy",
    "mean_reciprocal_rank",
    "mean_distance_km",
)
POLICY_ROLLOUT_CORE_KEYS = (
    "site_precision",
    "site_recall",
    "site_f1",
    "jaccard_similarity",
    "grid_hausdorff_km",
    "mean_distance_km",
)
POLICY_ROLLOUT_LCSS_KEYS = ("lcss_eps0_km", "lcss_eps2_km", "lcss_eps2_829_km")

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

PLOT_EVAL_KEYS = (
    *(f"expert_rollin_{k}" for k in EXPERT_ROLLIN_CORE_KEYS),
    *(f"policy_rollout_{k}" for k in POLICY_ROLLOUT_CORE_KEYS),
    *(f"policy_rollout_{k}" for k in POLICY_ROLLOUT_LCSS_KEYS),
)

CKPT_REQUIRED_KEYS = {
    "network",
    "q_net",
    "target_net",
    "alpha",
    "gamma",
    "iq_loss_mode",
    "use_chi",
    "alpha_reg",
    "grid_h",
    "grid_w",
    "in_channels",
    "n_actions",
    "n_counties",
    "embed_dim",
    "meta_dim",
    "n_states",
    "n_residual",
    "residual_alpha",
    "embed_mode",
    "film_hidden",
    "county_names",
    "location_labels",
    "lr_schedule",
}


def _prefix_eval_dict(ev: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in ev.items():
        if k == "eval_mode":
            out[f"{prefix}eval_mode"] = v
        else:
            out[f"{prefix}{k}"] = v
    return out


def _merge_eval_dicts(expert: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    return {
        **_prefix_eval_dict(expert, "expert_rollin_"),
        **_prefix_eval_dict(policy, "policy_rollout_"),
    }


def _empty_eval_dict(*, eval_mode: str) -> dict[str, Any]:
    if eval_mode == "policy_rollout":
        base = {
            "site_precision": 0.0,
            "site_recall": 0.0,
            "site_f1": 0.0,
            "jaccard_similarity": 0.0,
            "grid_hausdorff_km": 0.0,
            "mean_distance_km": 0.0,
            "lcss_eps0_km": 0.0,
            "lcss_eps2_km": 0.0,
            "lcss_eps2_829_km": 0.0,
        }
    else:
        base = {
            "expert_greedy_match_rate": 0.0,
            "top10_accuracy": 0.0,
            "mean_reciprocal_rank": 0.0,
            "mean_distance_km": 0.0,
        }
    return {"eval_mode": eval_mode, **base}


def _eval_metrics_for_log(ev: dict[str, Any]) -> dict[str, float]:
    row: dict[str, float] = {}
    for k in EXPERT_ROLLIN_CORE_KEYS:
        row[f"expert_rollin_{k}"] = float(ev.get(f"expert_rollin_{k}", 0.0))
    for k in POLICY_ROLLOUT_CORE_KEYS:
        row[f"policy_rollout_{k}"] = float(ev.get(f"policy_rollout_{k}", 0.0))
    for k in POLICY_ROLLOUT_LCSS_KEYS:
        row[f"policy_rollout_{k}"] = float(ev.get(f"policy_rollout_{k}", 0.0))
    return row


def _aggregate_joint_eval_metrics(per_county: list[dict[str, Any]]) -> dict[str, float]:
    row: dict[str, float] = {}
    for prefix, shorts in (
        ("expert_rollin_", EXPERT_ROLLIN_CORE_KEYS),
        ("policy_rollout_", POLICY_ROLLOUT_CORE_KEYS + POLICY_ROLLOUT_LCSS_KEYS),
    ):
        for short in shorts:
            full = f"{prefix}{short}"
            vals = [float(p.get(full, 0.0)) for p in per_county]
            row[full] = float(mean(vals)) if vals else 0.0
    return row


def _build_joint_final_eval(per_county: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **_aggregate_joint_eval_metrics(per_county),
        "expert_rollin_eval_mode": "expert_rollin",
        "policy_rollout_eval_mode": "policy_rollout",
    }


def _assert_numeric_metrics(row: dict[str, Any], *, label: str) -> None:
    expected = set(PLOT_EVAL_KEYS)
    missing = expected - set(row)
    if missing:
        raise AssertionError(f"{label}: 缺少指标键 {sorted(missing)}")
    for key in expected:
        if not isinstance(row[key], (int, float)):
            raise AssertionError(f"{label}: {key}={row[key]!r} 不是数值")
    print(f"[lite] {label}: OK ({len(expected)} 个数值指标)")


def _assert_metrics_log_row(row: dict[str, Any]) -> None:
    missing = [k for k in IRL_METRICS_LOG_TRAIN_KEYS if k not in row]
    if missing:
        raise AssertionError(f"metrics_log 行缺少键: {missing}")
    _assert_numeric_metrics(row, label="metrics_log row")


def _test_old_bug_fixed() -> None:
    """复现 Windows 上报错：eval_mode 字符串不能被 float()。"""
    per = [
        {
            "county_name": "A",
            **_merge_eval_dicts(
                _empty_eval_dict(eval_mode="expert_rollin"),
                _empty_eval_dict(eval_mode="policy_rollout"),
            ),
        }
    ]
    # 旧逻辑：遍历所有 expert_rollin_* 键并 float() → 在 eval_mode 处崩溃
    for prefix in ("expert_rollin_", "policy_rollout_"):
        keys = {k[len(prefix) :] for k in per[0] if k.startswith(prefix)}
        for short in keys:
            full = f"{prefix}{short}"
            for p in per:
                val = p.get(full, 0.0)
                if short == "eval_mode":
                    if isinstance(val, str):
                        continue  # 旧代码这里会 float('expert_rollin') 崩溃
                    raise AssertionError("eval_mode 应为字符串")
                float(val)
    # 新逻辑：只汇总数值键
    agg = _aggregate_joint_eval_metrics(per)
    _assert_numeric_metrics(agg, label="aggregate（修复后）")
    print("[lite] eval_mode 字符串不再触发 float() 错误")


def _test_train_joint_source() -> None:
    src = (_PKG_ROOT / "iq_learn" / "train_joint.py").read_text(encoding="utf-8")
    required_snippets = (
        "aggregate_joint_eval_metrics",
        "build_joint_final_eval",
        "format_eval_log",
        "final_eval",
        "final_eval_expert_rollin",
        "final_eval_policy_rollout",
        "plot_iq_metrics",
        "policy_checkpoint_path",
    )
    for s in required_snippets:
        if s not in src:
            raise AssertionError(f"train_joint.py 缺少: {s}")
    if "for short in keys:" in src and "float(p.get(full" in src:
        raise AssertionError("train_joint.py 仍含旧的动态 float() 汇总逻辑")
    print("[lite] train_joint.py 结构与 EV-charger-IRL 输出对齐")


def _test_checkpoint_save_keys() -> None:
    src = (_PKG_ROOT / "iq_learn" / "discrete_soft_q.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    save_keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "save":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                save_keys.add(child.value)
    missing = CKPT_REQUIRED_KEYS - save_keys
    if missing:
        raise AssertionError(f"DiscreteSoftQAgent.save() 缺少字段: {sorted(missing)}")
    if "embed_dim" not in save_keys or "meta_dim" not in save_keys:
        raise AssertionError("checkpoint 应包含 embed_dim / meta_dim（CountyLocationEmbed）")
    print("[lite] checkpoint 字段定义 OK（未实际写 .pt 文件）")


def _test_plotting_optional() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("[lite] 跳过绘图测试（未安装 matplotlib）")
        return

    if str(_PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(_PKG_ROOT))

    from iq_learn.plotting import plot_iq_metrics  # noqa: WPS433

    per = [
        {
            "county_name": "A",
            **_merge_eval_dicts(
                _empty_eval_dict(eval_mode="expert_rollin"),
                _empty_eval_dict(eval_mode="policy_rollout"),
            ),
        },
        {
            "county_name": "B",
            **_merge_eval_dicts(
                {**_empty_eval_dict(eval_mode="expert_rollin"), "expert_greedy_match_rate": 0.5},
                {**_empty_eval_dict(eval_mode="policy_rollout"), "site_f1": 0.3},
            ),
        },
    ]
    row = {
        "step": 200,
        "Q_mean": 0.1,
        "Q_std": 0.2,
        "Q_max": 0.3,
        "policy_entropy": 1.5,
        "loss": 0.4,
        **_aggregate_joint_eval_metrics(per),
    }
    with tempfile.TemporaryDirectory(prefix="eb_iq_lite_") as tmp:
        plot_path = Path(tmp) / "joint_iq_training_metrics.png"
        plot_iq_metrics([row, {**row, "step": 400}], plot_path, title="lite-smoke")
        if not plot_path.is_file() or plot_path.stat().st_size <= 0:
            raise AssertionError("plot_iq_metrics 未生成 PNG")
    print("[lite] 评估指标可视化 OK")


def main() -> None:
    per = [
        {
            "county_name": "A",
            **_merge_eval_dicts(
                _empty_eval_dict(eval_mode="expert_rollin"),
                _empty_eval_dict(eval_mode="policy_rollout"),
            ),
        },
        {
            "county_name": "B",
            **_merge_eval_dicts(
                {**_empty_eval_dict(eval_mode="expert_rollin"), "expert_greedy_match_rate": 0.8},
                {**_empty_eval_dict(eval_mode="policy_rollout"), "site_recall": 0.6},
            ),
        },
    ]
    single = _eval_metrics_for_log(per[0])
    agg = _aggregate_joint_eval_metrics(per)
    if set(single) != set(agg.keys()) & set(single):
        raise AssertionError("单县 eval_metrics_for_log 键与跨县 aggregate 数值键不一致")

    row = {
        "step": 1,
        "Q_mean": 0.1,
        "Q_std": 0.2,
        "Q_max": 0.3,
        "policy_entropy": 1.5,
        "loss": 0.4,
        **agg,
    }
    _assert_metrics_log_row(row)

    final_ev = _build_joint_final_eval(per)
    summary = {
        "final_eval": final_ev,
        "mean_expert_match_rate": final_ev["expert_rollin_expert_greedy_match_rate"],
        "final_eval_expert_rollin": {k: v for k, v in final_ev.items() if k.startswith("expert_rollin_")},
        "final_eval_policy_rollout": {k: v for k, v in final_ev.items() if k.startswith("policy_rollout_")},
    }
    for block in ("final_eval_expert_rollin", "final_eval_policy_rollout"):
        if not summary[block]:
            raise AssertionError(f"summary 缺少 {block}")
    print("[lite] summary / final_eval 结构 OK")

    _test_old_bug_fixed()
    _test_train_joint_source()
    _test_checkpoint_save_keys()
    _test_plotting_optional()
    print("[lite] 全部通过（无需 torch / GPU）")


if __name__ == "__main__":
    main()
