"""Smoothed metric curves with a light band for oscillation (RL-style plots)."""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter

MILLION = 1_000_000.0
TRAINING_STEP_XLABEL = "Training step (million)"


def training_steps_million(steps: np.ndarray | list) -> np.ndarray:
    """将原始训练步数转为 million 单位，避免横轴出现 1e6 科学计数。"""
    return np.asarray(steps, dtype=np.float64) / MILLION


def set_training_step_xaxis(ax: Axes) -> None:
    """横轴标注 million（如 0.2, 0.4, …, 1.0），不用 matplotlib 默认的 1e6 offset。"""
    ax.set_xlabel(TRAINING_STEP_XLABEL)
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt)


def default_smooth_window(n: int) -> int:
    if n <= 4:
        return 1
    w = max(3, min(31, n // 6))
    return w if w % 2 == 1 else w + 1


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    w = max(1, int(window))
    if w <= 1 or n == 0:
        return y.copy()
    if w % 2 == 0:
        w += 1
    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(ypad, kernel, mode="valid")


def rolling_envelope(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    w = max(1, int(window))
    if w <= 1 or n == 0:
        return y.copy(), y.copy()
    if w % 2 == 0:
        w += 1
    half = w // 2
    lo = np.empty(n, dtype=np.float64)
    hi = np.empty(n, dtype=np.float64)
    for i in range(n):
        sl = y[max(0, i - half) : min(n, i + half + 1)]
        lo[i], hi[i] = float(sl.min()), float(sl.max())
    return lo, hi


def plot_metric_with_band(
    ax: Axes,
    steps: np.ndarray | list,
    values: np.ndarray | list,
    *,
    color: str,
    label: str,
    smooth_window: int | None = None,
    band_alpha: float = 0.25,
    line_width: float = 2.2,
    show_band: bool = True,
) -> None:
    """Light fill = local min–max envelope; solid line = rolling mean.

    steps 应为 training_steps_million() 转换后的横轴（单位 million）。
    """
    x = np.asarray(steps, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if y.size == 0:
        return

    w = smooth_window if smooth_window is not None else default_smooth_window(int(y.size))
    y_smooth = rolling_mean(y, w)

    if show_band and y.size > 1 and w > 1:
        y_lo, y_hi = rolling_envelope(y, w)
        ax.fill_between(x, y_lo, y_hi, color=color, alpha=band_alpha, linewidth=0, zorder=1)

    ax.plot(
        x,
        y_smooth,
        color=color,
        linewidth=line_width,
        solid_capstyle="round",
        zorder=3,
    )
    ax.set_ylabel(label)


def style_metric_axes(fig) -> None:
    for ax in fig.axes:
        if ax.axison:
            set_training_step_xaxis(ax)
        ax.grid(True, linestyle="-", alpha=0.28, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
