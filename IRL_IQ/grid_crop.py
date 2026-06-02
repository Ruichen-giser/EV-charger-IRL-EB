"""裁剪县网格：保留与专家建站格连通的 valid 网络，再取最小外接矩形。"""
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


def _action_to_xy(action: int, w: int) -> tuple[int, int]:
    a = int(action)
    return a // int(w), a % int(w)


def station_network_mask(valid_mask: np.ndarray, expert_actions: np.ndarray, w: int) -> np.ndarray:
    """从专家建站格出发，在 valid 格上做 4 邻 BFS，得到「有充电站的网络」。"""
    h, width = valid_mask.shape
    network = np.zeros((h, width), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for a in np.asarray(expert_actions, dtype=np.int64).reshape(-1):
        gy, gx = _action_to_xy(int(a), w)
        if 0 <= gy < h and 0 <= gx < width and valid_mask[gy, gx]:
            if not network[gy, gx]:
                network[gy, gx] = True
                q.append((gy, gx))

    while q:
        y, x = q.popleft()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = y + dy, x + dx
            if ny < 0 or nx < 0 or ny >= h or nx >= width:
                continue
            if not valid_mask[ny, nx] or network[ny, nx]:
                continue
            network[ny, nx] = True
            q.append((ny, nx))
    return network


def crop_npz_to_station_network(
    src_npz: str | Path,
    dst_npz: str | Path,
    *,
    margin: int = 0,
) -> dict[str, Any]:
    """
    裁剪 npz：
      1. BFS 得到建站连通网络
      2. 取网络最小外接矩形（可选 margin）
      3. 矩形外网格剔除；矩形内非网络格 valid=False
    """
    src, dst = Path(src_npz), Path(dst_npz)
    blob = np.load(src, allow_pickle=False)
    grid_features = blob["grid_features"].astype(np.float32)
    valid_mask = blob["valid_mask"].astype(bool)
    expert_actions = blob["expert_actions"].astype(np.int64).reshape(-1)
    h, w = grid_features.shape[:2]

    network = station_network_mask(valid_mask, expert_actions, w)
    if not network.any():
        raise ValueError(f"专家建站格在 valid 网络中为空：{src}")

    ys, xs = np.where(network)
    y0 = max(0, int(ys.min()) - int(margin))
    y1 = min(h, int(ys.max()) + 1 + int(margin))
    x0 = max(0, int(xs.min()) - int(margin))
    x1 = min(w, int(xs.max()) + 1 + int(margin))

    crop_feat = grid_features[y0:y1, x0:x1].copy()
    crop_valid = network[y0:y1, x0:x1].copy()
    new_h, new_w = crop_valid.shape

    new_actions: list[int] = []
    for a in expert_actions:
        gy, gx = _action_to_xy(int(a), w)
        ny, nx = gy - y0, gx - x0
        if not (0 <= ny < new_h and 0 <= nx < new_w and crop_valid[ny, nx]):
            continue
        new_actions.append(int(ny * new_w + nx))
    if not new_actions:
        raise ValueError(f"裁剪后无有效 expert_actions：{src}")

    gx0 = int(blob["gx0"]) + x0 if "gx0" in blob else x0
    gy0 = int(blob["gy0"]) + y0 if "gy0" in blob else y0

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        grid_features=crop_feat,
        valid_mask=crop_valid,
        expert_actions=np.asarray(new_actions, dtype=np.int64),
        grid_cell_km=blob["grid_cell_km"],
        gx0=np.int32(gx0),
        gy0=np.int32(gy0),
        county_name=blob["county_name"] if "county_name" in blob else np.array(["Siskiyou"]),
        state_name=blob["state_name"] if "state_name" in blob else np.array([""]),
        crop_meta=np.array(
            [
                {
                    "src": str(src),
                    "orig_h": h,
                    "orig_w": w,
                    "crop_y0": y0,
                    "crop_x0": x0,
                    "new_h": new_h,
                    "new_w": new_w,
                    "n_network_cells": int(crop_valid.sum()),
                    "n_expert_steps": len(new_actions),
                }
            ],
            dtype=object,
        ),
    )
    meta = {
        "path": str(dst),
        "orig_h": h,
        "orig_w": w,
        "new_h": new_h,
        "new_w": new_w,
        "crop_box": (y0, y1, x0, x1),
        "n_network_cells": int(crop_valid.sum()),
        "n_expert_steps": len(new_actions),
    }
    return meta
