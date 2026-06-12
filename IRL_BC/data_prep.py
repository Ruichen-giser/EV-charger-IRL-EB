"""训练前数据准备：按 state-county 裁剪到建站连通区域。"""
from __future__ import annotations

from pathlib import Path

from grid_crop import crop_npz_to_station_network
from state_county import StateCountyPair, resolve_state_county_npz_paths


def state_county_npz_paths(grid_dir: Path, pair: StateCountyPair) -> tuple[Path, Path]:
    """原始特征 npz 与裁剪后 npz 路径（{State}__{County}_*.npz）。"""
    return resolve_state_county_npz_paths(grid_dir, pair)


def prepare_state_county_npz(
    grid_dir: Path,
    pair: StateCountyPair,
    *,
    log_prefix: str = "BC",
) -> Path:
    src, dst = state_county_npz_paths(grid_dir, pair)
    if not src.is_file():
        raise FileNotFoundError(
            f"缺少 {pair.key} 的网格数据: {src}\n"
            "请将 <州>__<县>_grid_features.npz 放入 --grid-npz-dir 指定目录。"
        )
    if not dst.is_file() or src.stat().st_mtime > dst.stat().st_mtime:
        meta = crop_npz_to_station_network(src, dst)
        print(
            f"[{log_prefix}] 裁剪 {pair.key}: {meta['orig_h']}×{meta['orig_w']} → "
            f"{meta['new_h']}×{meta['new_w']}，专家步 {meta['n_expert_steps']}",
            flush=True,
        )
    return dst


def prepare_state_counties_npz(
    grid_dir: Path,
    pairs: list[StateCountyPair] | tuple[StateCountyPair, ...],
    *,
    log_prefix: str = "BC",
) -> list[Path]:
    return [prepare_state_county_npz(grid_dir, p, log_prefix=log_prefix) for p in pairs]


# --- 兼容旧 API（仅县名，默认 California）---
def county_npz_paths(grid_dir: Path, county: str, *, state_name: str = "California") -> tuple[Path, Path]:
    return state_county_npz_paths(grid_dir, StateCountyPair(state_name, county.replace("_", " ")))


def prepare_county_npz(
    grid_dir: Path,
    county: str,
    *,
    state_name: str = "California",
    log_prefix: str = "BC",
) -> Path:
    return prepare_state_county_npz(
        grid_dir,
        StateCountyPair(state_name, county.replace("_", " ")),
        log_prefix=log_prefix,
    )
