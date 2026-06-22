"""
IRL_data 入口（三步 pipeline）：

1. 导入 GIS/EVCS，构建 2 km 网格特征（county_prepare）
2. 按 Excel（MDP≥5）或 county_names 逐县切分轨迹与 grids
3. 存储 prepared_irl_dataset.pkl 与 grid_tensors/*.npz，供 IRL_BC / IRL_IQ 读取
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from county_list import county_grid_npz_filename, load_mdp_ge5_county_list  # noqa: E402
from county_prepare import (  # noqa: E402
    CountyTrajectoryPack,
    prepare_county_trajectory_packs,
    prepare_state_county_trajectory_packs,
)
from grid_tensor import export_county_grid_npz  # noqa: E402
from paths import (  # noqa: E402
    DATA_ROOT,
    DEFAULT_CACHE_DIR,
    DEFAULT_GRID_NPZ_DIR,
    DEFAULT_MDP_GE5_COUNTY_LIST_XLSX,
    DEFAULT_PREPARED_DATA_PKL,
    validate_project_layout,
)
from schema import SCHEMA_VERSION  # noqa: E402


@dataclass
class DataConfig:
    """修改下列路径与县名后，直接运行本文件。"""

    pkl_path: str = field(
        default_factory=lambda: str(DATA_ROOT / "US-EV-Station-2014-2025" / "Daily" / "EVCS_sequence.pkl")
    )
    usa_map_path: str = field(default_factory=lambda: str(DATA_ROOT / "US-map" / "usa_map.geojson"))
    # True：从 data/us_mdp_ge5_state_county_list.xlsx 读取 MDP≥5 的 state-county 列表
    use_mdp_ge5_county_list: bool = True
    mdp_ge5_county_list_xlsx: str = field(
        default_factory=lambda: str(DEFAULT_MDP_GE5_COUNTY_LIST_XLSX)
    )

    # use_mdp_ge5_county_list=False 时沿用单州 + 逗号分隔县名
    state_name: str = "California"
    county_names: str = (
        "Siskiyou,Modoc,Shasta,Lassen,Los Angeles,Sacramento,San Diego,Kern"
    )

    worldpop_tif_path: str = field(
        default_factory=lambda: str(
            DATA_ROOT / "US-WorldPOP-2014-2020" / "raw" / "usa_ppp_2020_1km_Aggregated_UNadj.tif"
        )
    )
    gdp_tif_path: str = field(default_factory=lambda: str(DATA_ROOT / "US-GDP" / "rast_gdpTot_1990_2024_30arcsec.tif"))
    gdp_band: int = 6
    poi_geoparquet_path: str = field(
        default_factory=lambda: str(DATA_ROOT / "US-poi-2014-2024" / "data_2024_geoparquet.parquet")
    )
    highway_geojson_path: str = field(
        default_factory=lambda: str(DATA_ROOT / "US-highway" / "NTAD_National_Highway_System.geojson")
    )

    output_pkl: str = field(default_factory=lambda: str(DEFAULT_PREPARED_DATA_PKL))
    cache_dir: str | None = field(default_factory=lambda: str(DEFAULT_CACHE_DIR))
    enable_gis_cache: bool = True
    county_data_workers: int = 8

    grid_cell_km: int = 2
    grid_subsample_mod: int = 1
    # 剔除人口/GDP/POI/加油站全为 0 且非建站格；剔除后重新县内归一化
    prune_zero_activity_grids: bool = True

    export_grid_npz: bool = True
    grid_npz_dir: str = field(default_factory=lambda: str(DEFAULT_GRID_NPZ_DIR))


CONFIG = DataConfig()


def _pack_to_dict(pack: CountyTrajectoryPack) -> dict[str, Any]:
    grids = pack.grids.copy()
    grid_origin = grids.attrs.get("grid_origin", pack.grid_origin)
    return {
        "county_name": pack.county_name,
        "state_name": pack.state_name,
        "events": pack.events.copy(),
        "grids": grids,
        "grid_prune_meta": pack.grid_prune_meta,
        "grid_origin": grid_origin,
    }


def save_prepared_dataset(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "schema_version": payload["schema_version"],
        "output_pkl": str(path),
        "counties": [
            {
                "state_name": p.get("state_name"),
                "county_name": p["county_name"],
                "n_events": int(len(p["events"])),
                "n_grids": int(len(p["grids"])),
                "grid_cell_km": (p.get("grid_origin") or {}).get("grid_cell_km"),
            }
            for p in payload["packs"]
        ],
        "config": payload["config"],
    }
    with path.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def main(config: DataConfig | None = None) -> None:
    cfg = config if config is not None else CONFIG
    validate_project_layout()
    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else None

    if cfg.use_mdp_ge5_county_list:
        state_county_pairs = load_mdp_ge5_county_list(Path(cfg.mdp_ge5_county_list_xlsx))
        print(
            f"[IRL_data] MDP≥5 county list: {len(state_county_pairs)} state-county pairs "
            f"from {cfg.mdp_ge5_county_list_xlsx}",
            flush=True,
        )
        n_listed = len(state_county_pairs)
        packs = prepare_state_county_trajectory_packs(
            Path(cfg.pkl_path),
            Path(cfg.usa_map_path),
            state_county_pairs,
            Path(cfg.worldpop_tif_path),
            Path(cfg.gdp_tif_path),
            int(cfg.gdp_band),
            Path(cfg.poi_geoparquet_path),
            Path(cfg.highway_geojson_path),
            grid_subsample_mod=int(cfg.grid_subsample_mod),
            grid_cell_km=int(cfg.grid_cell_km),
            prune_zero_activity_grids=bool(cfg.prune_zero_activity_grids),
            cache_dir=cache_dir,
            use_cache=bool(cfg.enable_gis_cache),
            n_jobs=int(cfg.county_data_workers),
        )
        if len(packs) < n_listed:
            print(
                f"[IRL_data] 警告: Excel 列出 {n_listed} 县，实际产出 {len(packs)} 县 "
                f"（{n_listed - len(packs)} 县因无 EVCS 事件被跳过）",
                flush=True,
            )
    else:
        county_names = [x.strip() for x in str(cfg.county_names).split(",") if x.strip()]
        packs = prepare_county_trajectory_packs(
            Path(cfg.pkl_path),
            Path(cfg.usa_map_path),
            cfg.state_name,
            county_names,
            Path(cfg.worldpop_tif_path),
            Path(cfg.gdp_tif_path),
            int(cfg.gdp_band),
            Path(cfg.poi_geoparquet_path),
            Path(cfg.highway_geojson_path),
            grid_subsample_mod=int(cfg.grid_subsample_mod),
            grid_cell_km=int(cfg.grid_cell_km),
            prune_zero_activity_grids=bool(cfg.prune_zero_activity_grids),
            cache_dir=cache_dir,
            use_cache=bool(cfg.enable_gis_cache),
            n_jobs=int(cfg.county_data_workers),
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(cfg),
        "packs": [_pack_to_dict(p) for p in packs],
    }
    save_prepared_dataset(Path(cfg.output_pkl), payload)

    if cfg.export_grid_npz:
        npz_dir = Path(cfg.grid_npz_dir)
        npz_dir.mkdir(parents=True, exist_ok=True)
        for pack in packs:
            npz_name = county_grid_npz_filename(pack.state_name, pack.county_name)
            meta = export_county_grid_npz(
                pack, npz_dir / npz_name, grid_cell_km=float(cfg.grid_cell_km)
            )
            print(
                f"[IRL_data] {pack.state_name} / {pack.county_name}: npz H={meta['grid_size_h']} "
                f"W={meta['grid_size_w']} expert_steps={meta['n_expert_steps']}",
                flush=True,
            )

    print(f"[IRL_data] pickle → {cfg.output_pkl}")


if __name__ == "__main__":
    main()
