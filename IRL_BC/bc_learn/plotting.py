"""BC training curves (smoothed line + light oscillation band)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from bc_learn.metrics import EXPERT_ROLLIN_CORE_KEYS, POLICY_ROLLOUT_CORE_KEYS, POLICY_ROLLOUT_LCSS_KEYS
from metrics_viz import plot_metric_with_band, style_metric_axes, training_steps_million

_EVAL_RATE_KEYS = frozenset(
    {
        f"expert_rollin_{k}"
        for k in EXPERT_ROLLIN_CORE_KEYS
        if k != "mean_distance_km"
    }
    | {
        f"policy_rollout_{k}"
        for k in POLICY_ROLLOUT_CORE_KEYS
        if k not in ("grid_hausdorff_km", "mean_distance_km")
    }
)

_EXPERT_PLOT_SPECS: list[tuple[str, str, str]] = [
    ("expert_rollin_expert_greedy_match_rate", "Match (expert roll-in)", "#4C78A8"),
    ("expert_rollin_top10_accuracy", "Top-10 (expert roll-in)", "#F58518"),
    ("expert_rollin_mean_reciprocal_rank", "MRR (expert roll-in)", "#B279A2"),
    ("expert_rollin_mean_distance_km", "Dist (expert roll-in, km)", "#9D755D"),
]

_POLICY_PLOT_SPECS: list[tuple[str, str, str]] = [
    ("policy_rollout_site_precision", "Site precision (policy rollout)", "#4C78A8"),
    ("policy_rollout_site_recall", "Site recall (policy rollout)", "#F58518"),
    ("policy_rollout_site_f1", "Site F1 (policy rollout)", "#B279A2"),
    ("policy_rollout_jaccard_similarity", "Jaccard (policy rollout)", "#72B7B2"),
    ("policy_rollout_grid_hausdorff_km", "Hausdorff (policy rollout, km)", "#E45756"),
    ("policy_rollout_mean_distance_km", "Chamfer (policy rollout, km)", "#9D755D"),
]

_POLICY_RATE_KEYS = frozenset(
    {
        *(f"policy_rollout_{k}" for k in POLICY_ROLLOUT_CORE_KEYS if k not in ("grid_hausdorff_km", "mean_distance_km")),
        *(f"policy_rollout_{k}" for k in POLICY_ROLLOUT_LCSS_KEYS),
    }
)


def _configure_plot_font() -> bool:
    """Register a CJK-capable font when available (Windows / macOS)."""
    from matplotlib import font_manager

    candidates: list[Path] = []
    if sys.platform == "win32":
        win_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        candidates.extend(
            [
                win_fonts / "msyh.ttc",
                win_fonts / "msyhbd.ttc",
                win_fonts / "simhei.ttf",
                win_fonts / "simsun.ttc",
            ]
        )
    elif sys.platform == "darwin":
        candidates.extend(
            [
                Path("/System/Library/Fonts/PingFang.ttc"),
                Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            ]
        )

    for path in candidates:
        if not path.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            continue

    plt.rcParams["axes.unicode_minus"] = False
    return False


def _plot_metric_panel(
    axes,
    metrics_log: list[dict[str, Any]],
    steps_m,
    specs: list[tuple[str, str, str]],
    *,
    rate_keys: frozenset[str] = frozenset(),
) -> None:
    for ax, (key, label, color) in zip(axes.flat, specs):
        vals = [float(r.get(key, 0)) for r in metrics_log]
        plot_metric_with_band(ax, steps_m, vals, color=color, label=label)
        if key in rate_keys:
            ax.set_ylim(0.0, 1.0)


def plot_bc_metrics(metrics_log: list[dict[str, Any]], out_path: str | Path, *, title: str) -> None:
    if not metrics_log:
        return
    _configure_plot_font()

    steps = [int(r["step"]) for r in metrics_log]
    steps_m = training_steps_million(steps)
    fig, axes = plt.subplots(3, 6, figsize=(18, 10))

    plot_metric_with_band(
        axes[0, 0],
        steps_m,
        [float(r.get("loss", 0)) for r in metrics_log],
        color="#54A24B",
        label="BC loss (NLL)",
    )
    plot_metric_with_band(
        axes[0, 1],
        steps_m,
        [float(r.get("policy_entropy", 0)) for r in metrics_log],
        color="#72B7B2",
        label="Policy entropy (train batch)",
    )
    for j in range(2, 6):
        axes[0, j].axis("off")

    _plot_metric_panel(axes[1:2], metrics_log, steps_m, _EXPERT_PLOT_SPECS, rate_keys=_EVAL_RATE_KEYS)
    _plot_metric_panel(axes[2:3], metrics_log, steps_m, _POLICY_PLOT_SPECS, rate_keys=_POLICY_RATE_KEYS)

    style_metric_axes(fig)
    axes[1, 0].set_ylabel("Expert roll-in", fontsize=10, labelpad=8)
    axes[2, 0].set_ylabel("Policy rollout", fontsize=10, labelpad=8)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
