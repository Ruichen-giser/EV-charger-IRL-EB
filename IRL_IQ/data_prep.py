"""训练前数据准备：各县裁剪 + 多县路径。"""
from __future__ import annotations

from pathlib import Path

from grid_crop import crop_npz_to_station_network


def county_npz_paths(grid_dir: Path, county: str) -> tuple[Path, Path]:
    safe = county.replace(" ", "_")
    return (
        grid_dir / f"{safe}_grid_features.npz",
        grid_dir / f"{safe}_grid_cropped.npz",
    )


def prepare_county_npz(grid_dir: Path, county: str, *, log_prefix: str = "EB") -> Path:
    src, dst = county_npz_paths(grid_dir, county)
    if not src.is_file():
        raise FileNotFoundError(
            f"缺少 {county} 网格: {src}\n请先运行 IRL_data/main.py 生成 grid_tensors。"
        )
    if not dst.is_file() or src.stat().st_mtime > dst.stat().st_mtime:
        meta = crop_npz_to_station_network(src, dst)
        print(
            f"[{log_prefix}] 裁剪 {county}: {meta['orig_h']}×{meta['orig_w']} → "
            f"{meta['new_h']}×{meta['new_w']}，专家步 {meta['n_expert_steps']}",
            flush=True,
        )
    return dst


def prepare_counties_npz(
    grid_dir: Path,
    counties: tuple[str, ...] | list[str],
    *,
    log_prefix: str = "EB",
) -> list[Path]:
    return [prepare_county_npz(grid_dir, c, log_prefix=log_prefix) for c in counties]
