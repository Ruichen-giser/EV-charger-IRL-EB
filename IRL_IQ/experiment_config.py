"""联合 IQ 实验预设：baseline（无县残差）与 S1（显式县残差）。"""
from __future__ import annotations

from paths import DEFAULT_IQ_OUTPUT_BASELINE, DEFAULT_IQ_OUTPUT_S1

EXPERIMENT_BASELINE = "baseline"
EXPERIMENT_S1 = "s1"
DEFAULT_EXPERIMENT = EXPERIMENT_BASELINE

RESIDUAL_ALPHA_BASELINE = 0.0
RESIDUAL_ALPHA_S1 = 1.0

EXPERIMENT_CHOICES = (EXPERIMENT_BASELINE, EXPERIMENT_S1)

EXPERIMENT_DESCRIPTIONS: dict[str, str] = {
    EXPERIMENT_BASELINE: (
        "Baseline：Bottleneck FiLM + state/meta（residual_alpha=0，无县残差）"
    ),
    EXPERIMENT_S1: (
        "S1：concat 或 FiLM + 显式县残差 e_fused = e_base + residual_alpha * residual_embed"
    ),
}


def residual_alpha_for_experiment(experiment: str) -> float:
    exp = str(experiment).strip().lower()
    if exp == EXPERIMENT_S1:
        return float(RESIDUAL_ALPHA_S1)
    if exp == EXPERIMENT_BASELINE:
        return float(RESIDUAL_ALPHA_BASELINE)
    raise ValueError(f"未知 experiment: {experiment!r}，可选: {EXPERIMENT_CHOICES}")


def output_dir_for_experiment(experiment: str) -> str:
    exp = str(experiment).strip().lower()
    if exp == EXPERIMENT_S1:
        return str(DEFAULT_IQ_OUTPUT_S1)
    if exp == EXPERIMENT_BASELINE:
        return str(DEFAULT_IQ_OUTPUT_BASELINE)
    raise ValueError(f"未知 experiment: {experiment!r}，可选: {EXPERIMENT_CHOICES}")


def resolve_experiment_settings(
    *,
    experiment: str,
    residual_alpha: float | None = None,
    output_dir: str | None = None,
) -> tuple[str, float, str]:
    """返回 (experiment, residual_alpha, output_dir)。"""
    exp = str(experiment).strip().lower()
    alpha = float(residual_alpha_for_experiment(exp) if residual_alpha is None else residual_alpha)
    out = str(output_dir or output_dir_for_experiment(exp))
    return exp, alpha, out
