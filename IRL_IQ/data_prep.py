"""训练前数据准备：裁剪到建站连通区域。"""
from __future__ import annotations

from pathlib import Path

from grid_crop import crop_npz_to_station_network


def county_npz_paths(grid_dir: Path, county: str) -> tuple[Path, Path]:
    """原始特征 npz 与裁剪后 npz 路径。"""
    safe = county.replace(" ", "_")
    return (
        grid_dir / f"{safe}_grid_features.npz",
        grid_dir / f"{safe}_grid_cropped.npz",
    )


def prepare_county_npz(grid_dir: Path, county: str, *, log_prefix: str = "IQ") -> Path:
    """
    若裁剪文件不存在或源文件更新，则从 _grid_features 生成 _grid_cropped。
    返回用于训练的 npz 路径。

    注意：本函数不绘制网格特征分布图（见 viz_config.ENABLE_GRID_FEATURE_VIZ）。
    """
    src, dst = county_npz_paths(grid_dir, county)
    if not src.is_file():
        raise FileNotFoundError(
            f"缺少 {county} 的网格数据: {src}\n"
            "请将 <县名>_grid_features.npz 放入 --grid-npz-dir 指定目录。"
        )
    if not dst.is_file() or src.stat().st_mtime > dst.stat().st_mtime:
        meta = crop_npz_to_station_network(src, dst)
        print(
            f"[{log_prefix}] 裁剪 {county}: {meta['orig_h']}×{meta['orig_w']} → "
            f"{meta['new_h']}×{meta['new_w']}，专家步 {meta['n_expert_steps']}",
            flush=True,
        )
    return dst
