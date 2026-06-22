"""pipeline 步骤 3：稀疏 grids → 稠密 (H,W,5) npz + expert_actions，供 IRL_BC / IRL_IQ。"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from county_list import county_grid_npz_filename
from county_prepare import CountyTrajectoryPack
from paths import DEFAULT_GRID_NPZ_DIR, DEFAULT_PREPARED_DATA_PKL
from schema import GRID_FEATURE_PCT_COLS

# npz 中 5 个通道顺序（与 ChargingDeploymentEnv 一致）
FEATURE_CHANNELS = GRID_FEATURE_PCT_COLS


def grids_to_dense_tensor(
    grids: pd.DataFrame,
    *,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """将 events/grids 表中的 grid_x, grid_y 铺到 (H,W)，返回特征张量与有效格掩膜。"""
    g = grids.sort_values(["grid_y", "grid_x"]).reset_index(drop=True)
    gx = g["grid_x"].astype(int).to_numpy()
    gy = g["grid_y"].astype(int).to_numpy()
    gx0, gy0 = int(gx.min()), int(gy.min())
    H = int(gy.max() - gy0 + 1)
    W = int(gx.max() - gx0 + 1)

    grid_features = np.full((H, W, len(FEATURE_CHANNELS)), fill_value, dtype=np.float32)
    valid_mask = np.zeros((H, W), dtype=bool)

    for row in g.itertuples(index=False):
        ix = int(row.grid_x) - gx0
        iy = int(row.grid_y) - gy0
        valid_mask[iy, ix] = True
        for c, col in enumerate(FEATURE_CHANNELS):
            val = getattr(row, col, np.nan)
            grid_features[iy, ix, c] = 0.0 if pd.isna(val) else float(val)

    meta = {
        "grid_size_h": H,
        "grid_size_w": W,
        "gx0": gx0,
        "gy0": gy0,
        "n_valid_cells": int(valid_mask.sum()),
    }
    return grid_features, valid_mask, meta


def events_to_expert_action_sequence(
    events: pd.DataFrame,
    *,
    gx0: int,
    gy0: int,
    W: int,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """按 OpenDate 将建站事件转为 DQN 扁平动作下标：action = grid_y * W + grid_x。"""
    ev = events.sort_values("OpenDate")
    actions: list[int] = []
    for row in ev.itertuples(index=False):
        ix = int(row.grid_x) - int(gx0)
        iy = int(row.grid_y) - int(gy0)
        if ix < 0 or iy < 0 or iy >= valid_mask.shape[0] or ix >= valid_mask.shape[1]:
            continue
        if not valid_mask[iy, ix]:
            continue
        actions.append(int(iy * W + ix))
    return np.asarray(actions, dtype=np.int64)


def export_county_grid_npz(
    pack: CountyTrajectoryPack,
    out_path: str | Path,
    *,
    grid_cell_km: float = 2.0,
) -> dict[str, Any]:
    """写出单县 .npz：grid_features, valid_mask, expert_actions 等。"""
    grid_features, valid_mask, meta = grids_to_dense_tensor(pack.grids)
    H, W = meta["grid_size_h"], meta["grid_size_w"]
    expert_actions = events_to_expert_action_sequence(
        pack.events,
        gx0=int(meta["gx0"]),
        gy0=int(meta["gy0"]),
        W=int(W),
        valid_mask=valid_mask,
    )
    origin = pack.grid_origin or pack.grids.attrs.get("grid_origin") or {}
    cell_km = float(origin.get("grid_cell_km", grid_cell_km))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        grid_features=grid_features,
        valid_mask=valid_mask,
        expert_actions=expert_actions,
        grid_cell_km=np.float32(cell_km),
        gx0=np.int32(meta["gx0"]),
        gy0=np.int32(meta["gy0"]),
        county_name=np.array([pack.county_name]),
        state_name=np.array([pack.state_name]),
    )
    return {**meta, "path": str(out), "n_expert_steps": int(len(expert_actions)), "grid_cell_km": cell_km}


def pack_from_prepared_row(row: dict) -> CountyTrajectoryPack:
    """从 prepared_irl_dataset.pkl 中单县 dict 恢复 CountyTrajectoryPack。"""
    grids = row["grids"].copy()
    origin = row.get("grid_origin")
    if origin:
        grids.attrs["grid_origin"] = origin
    return CountyTrajectoryPack(
        county_name=str(row["county_name"]),
        state_name=str(row.get("state_name", "")),
        events=row["events"].copy(),
        grids=grids,
        grid_prune_meta=row.get("grid_prune_meta"),
        grid_origin=origin,
    )


def export_grids_from_prepared_pkl(
    prepared_pkl: str | Path = DEFAULT_PREPARED_DATA_PKL,
    out_dir: str | Path = DEFAULT_GRID_NPZ_DIR,
) -> None:
    """已有 pickle 时仅重导出 npz，无需重新跑 GIS。"""
    pkl = Path(prepared_pkl)
    with pkl.open("rb") as f:
        payload = pickle.load(f)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_km = float(payload.get("config", {}).get("grid_cell_km", 2))
    for row in payload.get("packs", []):
        pack = pack_from_prepared_row(row)
        npz_name = county_grid_npz_filename(pack.state_name, pack.county_name)
        meta = export_county_grid_npz(pack, out_dir / npz_name, grid_cell_km=cell_km)
        print(
            f"[IRL_data] exported {meta['path']} "
            f"({pack.state_name} / {pack.county_name}) H={meta['grid_size_h']} W={meta['grid_size_w']}"
        )


if __name__ == "__main__":
    export_grids_from_prepared_pkl()
