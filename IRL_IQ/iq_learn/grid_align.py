"""多县联合训练：统一画布上的观测 padding 与动作索引对齐。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointGridCanvas:
    """8 县共享的最大网格尺寸（左上角对齐 padding）。"""

    max_h: int
    max_w: int

    @property
    def n_actions(self) -> int:
        return int(self.max_h) * int(self.max_w)


def build_canvas(heights: list[int], widths: list[int]) -> JointGridCanvas:
    if not heights or not widths:
        raise ValueError("build_canvas 需要至少一个县的 H/W")
    return JointGridCanvas(max_h=max(heights), max_w=max(widths))


def pad_obs_hwc(obs_hwc: np.ndarray, canvas: JointGridCanvas, fill: float = 0.0) -> np.ndarray:
    """(h, w, c) → (max_h, max_w, c)。"""
    h, w, c = obs_hwc.shape
    out = np.full((canvas.max_h, canvas.max_w, int(c)), fill, dtype=np.float32)
    out[: int(h), : int(w), :] = obs_hwc.astype(np.float32, copy=False)
    return out


def pad_mask_flat(
    mask_local: np.ndarray,
    local_h: int,
    local_w: int,
    canvas: JointGridCanvas,
) -> np.ndarray:
    """本地 (h*w,) bool mask → 画布 (max_h*max_w,) bool mask。"""
    m2d = np.asarray(mask_local, dtype=bool).reshape(int(local_h), int(local_w))
    padded = np.zeros((canvas.max_h, canvas.max_w), dtype=bool)
    padded[: int(local_h), : int(local_w)] = m2d
    return padded.reshape(-1)


def local_action_to_canvas(local_action: int, local_w: int, canvas_w: int) -> int:
    """本地 flat action → 画布 flat action（gy*canvas_w+gx 对齐）。"""
    gx = int(local_action) % int(local_w)
    gy = int(local_action) // int(local_w)
    return int(gy * int(canvas_w) + gx)


def canvas_action_to_local(
    canvas_action: int,
    local_h: int,
    local_w: int,
    canvas_w: int,
) -> int | None:
    """画布 action → 本地 action；落在 padding 区则返回 None。"""
    gx = int(canvas_action) % int(canvas_w)
    gy = int(canvas_action) // int(canvas_w)
    if gx >= int(local_w) or gy >= int(local_h):
        return None
    return int(gy * int(local_w) + gx)
