"""训练前数据准备：按 state-county 裁剪 + 多县路径。"""
from __future__ import annotations

from pathlib import Path

from grid_crop import crop_npz_to_station_network
from state_county import (
    StateCountyPair,
    cropped_npz_filename,
    resolve_state_county_npz_paths,
)


def state_county_npz_paths(grid_dir: Path, pair: StateCountyPair) -> tuple[Path, Path]:
    src, _ = resolve_state_county_npz_paths(grid_dir, pair)
    dst = grid_dir / cropped_npz_filename(pair.state_name, pair.county_name)
    return src, dst


def prepare_state_county_npz(
    grid_dir: Path,
    pair: StateCountyPair,
    *,
    log_prefix: str = "EB",
) -> Path:
    src, dst = state_county_npz_paths(grid_dir, pair)
    if not src.is_file():
        raise FileNotFoundError(
            f"缺少 {pair.key} 网格: {src}\n请先运行 IRL_data/main.py 生成 grid_tensors。"
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
    log_prefix: str = "EB",
) -> list[Path]:
    return [prepare_state_county_npz(grid_dir, p, log_prefix=log_prefix) for p in pairs]


def county_npz_paths(grid_dir: Path, county: str, *, state_name: str = "California") -> tuple[Path, Path]:
    return state_county_npz_paths(grid_dir, StateCountyPair(state_name, county.replace("_", " ")))


def prepare_county_npz(
    grid_dir: Path,
    county: str,
    *,
    state_name: str = "California",
    log_prefix: str = "EB",
) -> Path:
    return prepare_state_county_npz(
        grid_dir,
        StateCountyPair(state_name, county.replace("_", " ")),
        log_prefix=log_prefix,
    )


def prepare_counties_npz(
    grid_dir: Path,
    counties: tuple[str, ...] | list[str],
    *,
    state_name: str = "California",
    log_prefix: str = "EB",
) -> list[Path]:
    pairs = [StateCountyPair(state_name, c.replace("_", " ")) for c in counties]
    return prepare_state_counties_npz(grid_dir, pairs, log_prefix=log_prefix)
