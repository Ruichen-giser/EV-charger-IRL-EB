"""
步骤 1c — 网格运算：格网坐标、1→2 km 聚合、县内归一化、补缺/裁剪。

补缺格在格心重采样 GIS 特征，运行时从 gis_align 导入采样函数（避免与 gis_align 循环依赖）。
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio

from geo import county_union_geometry
from schema import GRID_AGG_MEAN_COLS


@dataclass(frozen=True)
class GridSystem:
    """以县 bbox 左下角为原点的公里网格；target_cell_km 为研究格边长（默认 2 km）。"""

    lon_min: float
    lat_min: float
    km_per_deg_lon: float
    km_per_deg_lat: float
    base_cell_km: int = 1
    target_cell_km: int = 2

    def to_origin_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["axis_unit"] = "km"
        d["grid_cell_km"] = int(self.target_cell_km)
        d["base_grid_cell_km"] = int(self.base_cell_km)
        return d

    def cell_center_lon(self, gx: int, gy: int, *, cell_km: int | None = None) -> float:
        ck = max(1, int(cell_km if cell_km is not None else self.target_cell_km))
        return float(self.lon_min + (int(gx) * ck + ck * 0.5) / self.km_per_deg_lon)

    def cell_center_lat(self, gx: int, gy: int, *, cell_km: int | None = None) -> float:
        ck = max(1, int(cell_km if cell_km is not None else self.target_cell_km))
        return float(self.lat_min + (int(gy) * ck + ck * 0.5) / self.km_per_deg_lat)

    def cell_centers_lon(self, gx: np.ndarray, gy: np.ndarray, *, cell_km: int | None = None) -> np.ndarray:
        return np.array([self.cell_center_lon(int(x), int(y), cell_km=cell_km) for x, y in zip(gx, gy)])

    def cell_centers_lat(self, gx: np.ndarray, gy: np.ndarray, *, cell_km: int | None = None) -> np.ndarray:
        return np.array([self.cell_center_lat(int(x), int(y), cell_km=cell_km) for x, y in zip(gx, gy)])

    def cell_index_1km(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_km = (np.asarray(lon, dtype=float) - self.lon_min) * self.km_per_deg_lon
        y_km = (np.asarray(lat, dtype=float) - self.lat_min) * self.km_per_deg_lat
        return np.floor(x_km).astype(int), np.floor(y_km).astype(int)

    def cell_index_target(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_km, y_km = self.km_from_lonlat(lon, lat)
        cell = max(1, int(self.target_cell_km))
        return np.floor(x_km / cell).astype(int), np.floor(y_km / cell).astype(int)

    def km_from_lonlat(self, lon: np.ndarray | float, lat: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        x_km = (np.asarray(lon, dtype=float) - self.lon_min) * self.km_per_deg_lon
        y_km = (np.asarray(lat, dtype=float) - self.lat_min) * self.km_per_deg_lat
        return x_km, y_km

    @staticmethod
    def grid_id(gx: np.ndarray | int, gy: np.ndarray | int) -> np.ndarray:
        gx_a = np.asarray(gx).astype(int)
        gy_a = np.asarray(gy).astype(int)
        return np.char.add(np.char.add(gx_a.astype(str), "_"), gy_a.astype(str))

    def assign_events(self, events: pd.DataFrame) -> pd.DataFrame:
        out = events.copy()
        gx1, gy1 = self.cell_index_1km(out["Longitude"].to_numpy(), out["Latitude"].to_numpy())
        gxt, gyt = self.cell_index_target(out["Longitude"].to_numpy(), out["Latitude"].to_numpy())
        out["grid_x_1km"] = gx1
        out["grid_y_1km"] = gy1
        out["grid_x"] = gxt
        out["grid_y"] = gyt
        out["grid_id"] = self.grid_id(gxt, gyt)
        return out

    def aggregate_base_to_target(self, grids_base: pd.DataFrame) -> pd.DataFrame:
        cell = max(1, int(self.target_cell_km))
        if cell == 1:
            out = grids_base.copy().reset_index(drop=True)
            out["grid_id"] = self.grid_id(out["grid_x"], out["grid_y"])
            return out

        g = grids_base.copy()
        if "grid_x_1km" not in g.columns:
            g["grid_x_1km"] = g["grid_x"].astype(int)
            g["grid_y_1km"] = g["grid_y"].astype(int)
        g["grid_x"] = np.floor(g["grid_x_1km"].astype(int) / cell).astype(int)
        g["grid_y"] = np.floor(g["grid_y_1km"].astype(int) / cell).astype(int)

        mean_cols = [c for c in GRID_AGG_MEAN_COLS if c in g.columns]
        grouped = g.groupby(["grid_x", "grid_y"], as_index=False)
        out = grouped[mean_cols].mean() if mean_cols else grouped.size().rename(columns={"size": "n_covered_1km_cells"})
        out["n_covered_1km_cells"] = grouped.size()["size"].to_numpy(dtype=int)
        out["grid_id"] = self.grid_id(out["grid_x"], out["grid_y"])
        lead = ["grid_id", "grid_x", "grid_y", "n_covered_1km_cells"]
        rest = [c for c in out.columns if c not in lead]
        return out[lead + rest].reset_index(drop=True)


def normalize_county_feature_percentiles(grids: pd.DataFrame) -> pd.DataFrame:
    """县内 min-max 到 [0,1]；公路通道越大表示越靠近公路。"""
    g = grids.copy()

    def _minmax(col_raw: str, col_pct: str, default: float = 0.5) -> None:
        valid = g[col_raw].notna()
        if valid.sum() == 0:
            g[col_pct] = default
            return
        vmin = g.loc[valid, col_raw].min()
        vmax = g.loc[valid, col_raw].max()
        if vmax > vmin:
            g.loc[valid, col_pct] = (g.loc[valid, col_raw] - vmin) / (vmax - vmin)
        else:
            g.loc[valid, col_pct] = default
        g.loc[~valid, col_pct] = 0.0

    _minmax("population_raw", "population_pct")
    _minmax("gdp_raw", "gdp_pct")
    if "poi_entropy_raw" not in g.columns:
        g["poi_entropy_raw"] = 0.0
    _minmax("poi_entropy_raw", "poi_pct", default=0.0)

    valid = g["highway_dist_m"].notna()
    if valid.sum() > 0:
        dmin = g.loc[valid, "highway_dist_m"].min()
        dmax = g.loc[valid, "highway_dist_m"].max()
        if dmax > dmin:
            rem = (g.loc[valid, "highway_dist_m"] - dmin) / (dmax - dmin)
            g.loc[valid, "highway_dist_pct"] = 1.0 - rem
        else:
            g.loc[valid, "highway_dist_pct"] = 0.5
        g.loc[~valid, "highway_dist_pct"] = 0.0
    else:
        g["highway_dist_pct"] = 0.5

    if "fuel_station_count_raw" in g.columns:
        valid = g["fuel_station_count_raw"].notna()
        if valid.sum() > 0:
            fmin = g.loc[valid, "fuel_station_count_raw"].min()
            fmax = g.loc[valid, "fuel_station_count_raw"].max()
            if fmax > fmin:
                g.loc[valid, "fuel_station_count_pct"] = (
                    g.loc[valid, "fuel_station_count_raw"] - fmin
                ) / (fmax - fmin)
            else:
                g.loc[valid, "fuel_station_count_pct"] = 0.0
            g.loc[~valid, "fuel_station_count_pct"] = 0.0
        else:
            g["fuel_station_count_pct"] = 0.0

    return g


def add_missing_grid_cells(
    grids: pd.DataFrame,
    events: pd.DataFrame,
    grid: GridSystem,
    *,
    usa_map_path: Path,
    state_name: str,
    county_name: str,
    worldpop_tif_path: Path,
    gdp_tif_path: Path,
    gdp_band_1based: int,
    poi_geoparquet_path: Path,
    highway_geojson_path: Path,
    poi_by_cell: dict[str, float],
) -> pd.DataFrame:
    from gis_align import nearest_line_distance_m, read_highways_near_geometry, sample_raster_at_lonlat

    missing = sorted(set(events["grid_id"].astype(str)) - set(grids["grid_id"].astype(str)))
    if not missing:
        return grids

    county_geom = county_union_geometry(usa_map_path, state_name, county_name)
    highways = read_highways_near_geometry(highway_geojson_path, county_geom)
    cell_km = max(1, int(grid.target_cell_km))
    new_rows: list[dict] = []

    with rasterio.open(worldpop_tif_path) as src_wp:
        for gid in missing:
            gx, gy = (int(x) for x in str(gid).split("_"))
            lon = grid.cell_center_lon(gx, gy, cell_km=cell_km)
            lat = grid.cell_center_lat(gx, gy, cell_km=cell_km)
            pop = float(next(src_wp.sample([(lon, lat)]))[0])
            if not np.isfinite(pop) or pop < 0:
                pop = np.nan
            gdp = float(
                sample_raster_at_lonlat(
                    gdp_tif_path, np.array([lon]), np.array([lat]), band_index_1based=gdp_band_1based
                )[0]
            )
            hdist = float(nearest_line_distance_m(np.array([lon]), np.array([lat]), highways)[0])
            new_rows.append(
                {
                    "grid_x": gx,
                    "grid_y": gy,
                    "grid_id": gid,
                    "population_raw": pop,
                    "gdp_raw": gdp,
                    "poi_entropy_raw": float(poi_by_cell.get(gid, 0.0)),
                    "highway_dist_m": hdist if np.isfinite(hdist) else np.nan,
                    "fuel_station_count_raw": 0.0,
                }
            )
    return pd.concat([grids, pd.DataFrame(new_rows)], ignore_index=True)


def prune_grids_zero_non_highway_keep_stations(
    grids: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    station_ids = set(events["grid_id"].astype(str))
    cols = [c for c in ("population_raw", "gdp_raw", "poi_entropy_raw", "fuel_station_count_raw") if c in grids.columns]
    if not cols:
        cols = [c for c in ("population_pct", "gdp_pct", "poi_pct", "fuel_station_count_pct") if c in grids.columns]
    vals = grids[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    zero = (vals <= 0.0).all(axis=1)
    on_station = grids["grid_id"].astype(str).isin(station_ids)
    drop = zero & ~on_station
    stats = {
        "rule": "drop if all non-highway features are zero; keep station cells",
        "n_cells_total": int(len(grids)),
        "n_dropped_cells": int(drop.sum()),
        "n_kept_cells": int((~drop).sum()),
    }
    return grids.loc[~drop].copy().reset_index(drop=True), stats


def subsample_grids_mod(grids: pd.DataFrame, events: pd.DataFrame, mod: int) -> pd.DataFrame:
    if mod <= 1:
        return grids
    need = set(events["grid_id"].astype(str))
    gx = grids["grid_x"].astype(int)
    gy = grids["grid_y"].astype(int)
    keep = (((gx % mod) == 0) & ((gy % mod) == 0)) | grids["grid_id"].astype(str).isin(need)
    return grids.loc[keep].copy().reset_index(drop=True)
