"""
步骤 2 — 按县编排 pipeline（步骤 1 见 geo / gis_align / grid_ops；步骤 3 见 main + grid_tensor）。

单县流程：
  load_evcs_events → filter_events_within_county
  → build_county_grid_1km（GIS 对齐 1 km）
  → 聚合 2 km、归一化、补缺格（grid_ops）
  → 输出该县 events + grids
"""
import hashlib
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None  # type: ignore
    delayed = None  # type: ignore

from geo import filter_events_within_county, load_county_gdf
from gis_align import attach_fuel_station_counts, build_county_grid_1km, raster_file_info
from grid_ops import (
    GridSystem,
    add_missing_grid_cells,
    normalize_county_feature_percentiles,
    prune_grids_zero_non_highway_keep_stations,
    subsample_grids_mod,
)
from schema import FUEL_STATION_LABEL, SCHEMA_VERSION


def load_evcs_events(pkl_path: Path, *, state: str = "CA") -> pd.DataFrame:
    """从 EVCS_sequence.pkl 读取建站序列：去重、排序，默认仅加州。"""
    df = pickle.load(Path(pkl_path).open("rb"))
    df = df.copy()
    df["OpenDate"] = pd.to_datetime(df["OpenDate"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude", "OpenDate"])
    if state:
        df = df[df["State"] == state]
    df = df.drop_duplicates(subset=["station_ID"])
    df["operator"] = df["EV Network"].fillna("Unknown")
    return df.sort_values("OpenDate").reset_index(drop=True)


@dataclass
class CountyTrajectoryPack:
    """一县一条建站轨迹 + 该县动作空间网格表。"""

    county_name: str
    state_name: str
    events: pd.DataFrame
    grids: pd.DataFrame
    grid_prune_meta: dict | None = None
    grid_origin: dict | None = None


def _file_signature(path: Path) -> dict[str, Any]:
    p = Path(path)
    try:
        st = p.stat()
        return {"path": str(p.resolve()), "mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)}
    except OSError:
        return {"path": str(p), "missing": True}


def cache_key_for_county_dataset(
    *,
    pkl_path: Path,
    usa_map_path: Path,
    state_name: str,
    county_name: str,
    worldpop_tif_path: Path,
    gdp_tif_path: Path,
    gdp_band_1based: int,
    poi_geoparquet_path: Path,
    highway_geojson_path: Path,
    grid_subsample_mod: int,
    prune_zero_activity_grids: bool,
    grid_cell_km: int,
) -> str:
    payload = {
        "schema": f"{SCHEMA_VERSION}_county_cache",
        "state_name": state_name,
        "county_name": county_name,
        "gdp_band_1based": int(gdp_band_1based),
        "grid_subsample_mod": int(grid_subsample_mod),
        "prune_zero_activity_grids": bool(prune_zero_activity_grids),
        "grid_cell_km": int(grid_cell_km),
        "fuel_station_label": FUEL_STATION_LABEL,
        "files": {
            "pkl": _file_signature(pkl_path),
            "usa_map": _file_signature(usa_map_path),
            "worldpop": _file_signature(worldpop_tif_path),
            "gdp": _file_signature(gdp_tif_path),
            "poi": _file_signature(poi_geoparquet_path),
            "highway": _file_signature(highway_geojson_path),
        },
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"county_dataset_v6_{key}.pkl"


def load_county_dataset_cache(cache_dir: Path, key: str) -> tuple[pd.DataFrame, pd.DataFrame, dict | None] | None:
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    with path.open("rb") as f:
        payload = pickle.load(f)
    grids = payload["grids"]
    if payload.get("grid_origin"):
        grids.attrs["grid_origin"] = payload["grid_origin"]
    return payload["events"], grids, payload.get("prune_meta")


def save_county_dataset_cache(
    cache_dir: Path,
    key: str,
    events: pd.DataFrame,
    grids: pd.DataFrame,
    prune_meta: dict | None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "events": events,
        "grids": grids,
        "prune_meta": prune_meta,
        "grid_origin": grids.attrs.get("grid_origin"),
    }
    tmp = _cache_path(cache_dir, key).with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(_cache_path(cache_dir, key))


def prepare_county_dataset(
    pkl_path: Path,
    usa_map_path: Path,
    state_name: str,
    county_name: str,
    worldpop_tif_path: Path,
    gdp_tif_path: Path,
    gdp_band_1based: int,
    poi_geoparquet_path: Path,
    highway_geojson_path: Path,
    *,
    grid_subsample_mod: int = 1,
    prune_zero_activity_grids: bool = False,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    grid_cell_km: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    grid_cell_km = max(1, int(grid_cell_km))

    # --- 磁盘缓存（输入文件指纹 + 县名）---
    cache_key = None
    if cache_dir is not None and use_cache:
        cache_key = cache_key_for_county_dataset(
            pkl_path=pkl_path,
            usa_map_path=usa_map_path,
            state_name=state_name,
            county_name=county_name,
            worldpop_tif_path=worldpop_tif_path,
            gdp_tif_path=gdp_tif_path,
            gdp_band_1based=gdp_band_1based,
            poi_geoparquet_path=poi_geoparquet_path,
            highway_geojson_path=highway_geojson_path,
            grid_subsample_mod=grid_subsample_mod,
            prune_zero_activity_grids=prune_zero_activity_grids,
            grid_cell_km=grid_cell_km,
        )
        cached = load_county_dataset_cache(Path(cache_dir), cache_key)
        if cached is not None:
            return cached

    # --- 1. 事件：加州 EVCS → 县界内 ---
    county_gdf = load_county_gdf(usa_map_path, state_name, county_name)
    bounds = county_gdf.total_bounds

    events = load_evcs_events(pkl_path, state="CA")
    events = filter_events_within_county(events, usa_map_path, state_name, county_name)
    if events.empty:
        raise ValueError(f"No EVCS events in county {county_name} after boundary filter.")

    # --- 2. 1 km 网格：人口/GDP 栅格 + POI/公路/加油站矢量 ---
    grids_1km, grid_1km, poi_1km = build_county_grid_1km(
        usa_map_path,
        state_name,
        county_name,
        worldpop_tif_path,
        gdp_tif_path,
        gdp_band_1based,
        poi_geoparquet_path,
        highway_geojson_path,
    )
    grids_1km = attach_fuel_station_counts(grids_1km, poi_geoparquet_path, grid_1km, tuple(bounds))
    grids_1km = normalize_county_feature_percentiles(grids_1km)

    grid_1km_events = GridSystem(
        lon_min=grid_1km.lon_min,
        lat_min=grid_1km.lat_min,
        km_per_deg_lon=grid_1km.km_per_deg_lon,
        km_per_deg_lat=grid_1km.km_per_deg_lat,
        base_cell_km=1,
        target_cell_km=1,
    )
    events_1km = grid_1km_events.assign_events(events)
    grids_1km = add_missing_grid_cells(
        grids_1km,
        events_1km,
        grid_1km_events,
        usa_map_path=usa_map_path,
        state_name=state_name,
        county_name=county_name,
        worldpop_tif_path=worldpop_tif_path,
        gdp_tif_path=gdp_tif_path,
        gdp_band_1based=gdp_band_1based,
        poi_geoparquet_path=poi_geoparquet_path,
        highway_geojson_path=highway_geojson_path,
        poi_by_cell=poi_1km,
    )
    grids_1km = normalize_county_feature_percentiles(grids_1km)

    # --- 3. 聚合到 target_cell_km（默认 2 km），事件同步赋格 ---
    grid_target = GridSystem(
        lon_min=grid_1km.lon_min,
        lat_min=grid_1km.lat_min,
        km_per_deg_lon=grid_1km.km_per_deg_lon,
        km_per_deg_lat=grid_1km.km_per_deg_lat,
        base_cell_km=1,
        target_cell_km=grid_cell_km,
    )
    grids = grid_target.aggregate_base_to_target(grids_1km)
    events = grid_target.assign_events(events)

    poi_target = {
        str(gid): float(v)
        for gid, v in zip(grids["grid_id"].astype(str), grids.get("poi_entropy_raw", pd.Series(0.0)))
    }
    grids = add_missing_grid_cells(
        grids,
        events,
        grid_target,
        usa_map_path=usa_map_path,
        state_name=state_name,
        county_name=county_name,
        worldpop_tif_path=worldpop_tif_path,
        gdp_tif_path=gdp_tif_path,
        gdp_band_1based=gdp_band_1based,
        poi_geoparquet_path=poi_geoparquet_path,
        highway_geojson_path=highway_geojson_path,
        poi_by_cell=poi_target,
    )
    grids = normalize_county_feature_percentiles(grids)

    meta: dict[str, Any] = {
        "v6_schema": SCHEMA_VERSION,
        "implementation": "IRL_data",
        "raster_worldpop": raster_file_info(worldpop_tif_path),
        "raster_gdp": raster_file_info(gdp_tif_path),
        "vector_poi": str(poi_geoparquet_path),
        "vector_highway": str(highway_geojson_path),
        "fuel_station_label": FUEL_STATION_LABEL,
        "n_events": int(len(events)),
        "n_grids_1km": int(len(grids_1km)),
        "n_grids_target": int(len(grids)),
    }

    if prune_zero_activity_grids:
        before = grids.copy()
        grids, pst = prune_grids_zero_non_highway_keep_stations(grids, events)
        grids = normalize_county_feature_percentiles(grids)
        meta["prune_stats"] = pst
        meta["grids_before_prune"] = before

    if int(grid_subsample_mod) > 1:
        grids = subsample_grids_mod(grids, events, int(grid_subsample_mod))
        grids = add_missing_grid_cells(
            grids,
            events,
            grid_target,
            usa_map_path=usa_map_path,
            state_name=state_name,
            county_name=county_name,
            worldpop_tif_path=worldpop_tif_path,
            gdp_tif_path=gdp_tif_path,
            gdp_band_1based=gdp_band_1based,
            poi_geoparquet_path=poi_geoparquet_path,
            highway_geojson_path=highway_geojson_path,
            poi_by_cell=poi_target,
        )
        grids = normalize_county_feature_percentiles(grids)

    origin = grid_target.to_origin_dict()
    grids.attrs["grid_origin"] = origin
    meta["grid_origin"] = origin
    meta["grid_cell_km"] = int(grid_cell_km)

    if cache_dir is not None and use_cache and cache_key is not None:
        save_county_dataset_cache(Path(cache_dir), cache_key, events, grids, meta)

    return events, grids, meta


def prepare_county_trajectory_packs(
    pkl_path: Path,
    usa_map_path: Path,
    state_name: str,
    county_names: list[str],
    worldpop_tif_path: Path,
    gdp_tif_path: Path,
    gdp_band_1based: int,
    poi_geoparquet_path: Path,
    highway_geojson_path: Path,
    *,
    grid_subsample_mod: int = 1,
    prune_zero_activity_grids: bool = False,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    n_jobs: int = 1,
    grid_cell_km: int = 2,
) -> list[CountyTrajectoryPack]:
    cleaned = [c.strip() for c in county_names if c.strip()]

    def one(cname: str) -> CountyTrajectoryPack:
        ev, gr, m = prepare_county_dataset(
            pkl_path,
            usa_map_path,
            state_name,
            cname,
            worldpop_tif_path,
            gdp_tif_path,
            gdp_band_1based,
            poi_geoparquet_path,
            highway_geojson_path,
            grid_subsample_mod=grid_subsample_mod,
            prune_zero_activity_grids=prune_zero_activity_grids,
            cache_dir=cache_dir,
            use_cache=use_cache,
            grid_cell_km=grid_cell_km,
        )
        return CountyTrajectoryPack(
            county_name=cname,
            state_name=state_name,
            events=ev,
            grids=gr,
            grid_prune_meta=m,
            grid_origin=gr.attrs.get("grid_origin"),
        )

    workers = max(1, int(n_jobs))
    if workers <= 1 or len(cleaned) <= 1:
        packs = [one(c) for c in cleaned]
    elif Parallel is not None and delayed is not None:
        packs = Parallel(n_jobs=min(workers, len(cleaned)), prefer="threads")(delayed(one)(c) for c in cleaned)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(cleaned))) as ex:
            packs = list(ex.map(one, cleaned))
    if not packs:
        raise ValueError("county_names is empty.")
    return packs
