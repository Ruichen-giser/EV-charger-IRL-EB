"""
步骤 1b — GIS 对齐：栅格(TIF)与矢量(POI/公路)采样到 1 km 格网。

对外主入口：build_county_grid_1km → 县界内 1 km 格 + 五类原始特征列。
"""
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point

from geo import (
    CRS_ALBERS_CONUS,
    CRS_WGS84,
    ensure_wgs84,
    km_per_degree_at_latitude,
    load_county_gdf,
    representative_latitude_rad,
)
from grid_ops import GridSystem
from schema import FUEL_STATION_LABEL


# ---------------------------------------------------------------------------
# 栅格（WorldPOP / GDP）
# ---------------------------------------------------------------------------


def raster_file_info(tif_path: Path) -> dict:
    with rasterio.open(tif_path) as src:
        return {
            "path": str(Path(tif_path).resolve()),
            "crs": str(src.crs),
            "bounds": list(src.bounds),
            "size": [int(src.width), int(src.height)],
            "bands": int(src.count),
        }


def sample_raster_at_lonlat(
    tif_path: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    band_index_1based: int = 1,
) -> np.ndarray:
    coords = list(zip(lons.astype(float), lats.astype(float)))
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        raw = np.array([v[0] for v in src.sample(coords, indexes=[int(band_index_1based)])], dtype=float)
    out = raw.astype(float)
    out[~np.isfinite(out)] = np.nan
    if nodata is not None and np.isfinite(nodata):
        out[np.isclose(out, float(nodata))] = np.nan
    if (out < 0).any():
        out[out < 0] = np.nan
    return out


# ---------------------------------------------------------------------------
# 矢量（POI 熵 / 公路距离 / 加油站）
# ---------------------------------------------------------------------------


def read_highways_near_geometry(
    highway_geojson: Path,
    county_geom_wgs84,
    *,
    pad_deg: float = 0.05,
) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = county_geom_wgs84.bounds
    bbox = (minx - pad_deg, miny - pad_deg, maxx + pad_deg, maxy + pad_deg)
    hw = gpd.read_file(highway_geojson, bbox=bbox)
    if hw.empty:
        bbox = (minx - 0.5, miny - 0.5, maxx + 0.5, maxy + 0.5)
        hw = gpd.read_file(highway_geojson, bbox=bbox)
    hw = ensure_wgs84(hw)
    return hw[hw.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()


def nearest_line_distance_m(
    lons: np.ndarray,
    lats: np.ndarray,
    lines_wgs84: gpd.GeoDataFrame,
) -> np.ndarray:
    n = len(lons)
    if lines_wgs84.empty:
        return np.full(n, np.nan, dtype=float)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons, lats), crs=CRS_WGS84)
    pts_m = pts.to_crs(CRS_ALBERS_CONUS).reset_index(drop=True)
    pts_m["_row"] = np.arange(n, dtype=int)
    lines_m = ensure_wgs84(lines_wgs84).to_crs(CRS_ALBERS_CONUS)
    joined = gpd.sjoin_nearest(pts_m, lines_m[["geometry"]], how="left", distance_col="dist_m")
    joined = joined.sort_values(["_row", "dist_m"]).drop_duplicates("_row", keep="first")
    joined = joined.sort_values("_row")
    d = joined["dist_m"].astype(float).to_numpy()
    if len(d) != n:
        raise RuntimeError(f"Highway distance join size mismatch: expected {n}, got {len(d)}")
    return d


def poi_category_entropy_by_cell(
    poi_geoparquet: Path,
    grid: GridSystem,
    bbox_wgs84: tuple[float, float, float, float],
    *,
    category_col: str = "category_label",
) -> dict[str, float]:
    minx, miny, maxx, maxy = bbox_wgs84
    filters = [
        ("latitude", ">=", miny),
        ("latitude", "<=", maxy),
        ("longitude", ">=", minx),
        ("longitude", "<=", maxx),
    ]
    try:
        pois = pd.read_parquet(
            poi_geoparquet,
            columns=["latitude", "longitude", category_col],
            filters=filters,
        )
    except (ValueError, TypeError):
        pois = pd.read_parquet(poi_geoparquet, columns=["latitude", "longitude", category_col])
        pois = pois[
            (pois["latitude"] >= miny)
            & (pois["latitude"] <= maxy)
            & (pois["longitude"] >= minx)
            & (pois["longitude"] <= maxx)
        ]
    if pois.empty:
        return {}
    gx, gy = grid.cell_index_1km(
        pois["longitude"].to_numpy(dtype=float),
        pois["latitude"].to_numpy(dtype=float),
    )
    pois = pois.assign(_gx=gx, _gy=gy)
    out: dict[str, float] = {}
    for (ix, iy), sub in pois.groupby(["_gx", "_gy"]):
        cats = sub[category_col].fillna("unknown").astype(str)
        counts = cats.value_counts().to_numpy(dtype=float)
        p = counts / counts.sum()
        k = int((p > 0).sum())
        if k <= 1:
            h = 0.0
        else:
            h = float(-(p * np.log(p + 1e-12)).sum() / np.log(k))
        out[f"{int(ix)}_{int(iy)}"] = h
    return out


def _fuel_station_mask(labels: Any) -> bool:
    if labels is None:
        return False
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    if isinstance(labels, (list, tuple, set)):
        return any(str(x).strip() == FUEL_STATION_LABEL for x in labels)
    return FUEL_STATION_LABEL in str(labels)


def fuel_station_counts_by_cell(
    poi_geoparquet: Path,
    grid: GridSystem,
    bbox_wgs84: tuple[float, float, float, float],
) -> dict[str, int]:
    minx, miny, maxx, maxy = bbox_wgs84
    filters = [
        ("latitude", ">=", miny),
        ("latitude", "<=", maxy),
        ("longitude", ">=", minx),
        ("longitude", "<=", maxx),
    ]
    cols = ["latitude", "longitude", "fsq_category_labels"]
    try:
        pois = pd.read_parquet(poi_geoparquet, columns=cols, filters=filters)
    except (ValueError, TypeError):
        pois = pd.read_parquet(poi_geoparquet, columns=cols)
        pois = pois[
            (pois["latitude"] >= miny)
            & (pois["latitude"] <= maxy)
            & (pois["longitude"] >= minx)
            & (pois["longitude"] <= maxx)
        ]
    if pois.empty:
        return {}
    mask = pois["fsq_category_labels"].map(_fuel_station_mask)
    pois = pois.loc[mask]
    if pois.empty:
        return {}
    gx, gy = grid.cell_index_1km(
        pois["longitude"].to_numpy(dtype=float),
        pois["latitude"].to_numpy(dtype=float),
    )
    pois = pois.assign(_gx=gx, _gy=gy)
    vc = pois.groupby(["_gx", "_gy"]).size()
    return {f"{int(ix)}_{int(iy)}": int(n) for (ix, iy), n in vc.items()}


def attach_highway_and_poi_to_grids(
    grids: pd.DataFrame,
    grid: GridSystem,
    poi_geoparquet_path: Path,
    highway_geojson_path: Path,
    county_geom_wgs84,
    poi_bbox_wgs84: tuple[float, float, float, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    lons = grid.cell_centers_lon(grids["grid_x"].to_numpy(int), grids["grid_y"].to_numpy(int))
    lats = grid.cell_centers_lat(grids["grid_x"].to_numpy(int), grids["grid_y"].to_numpy(int))
    poi_map = poi_category_entropy_by_cell(poi_geoparquet_path, grid, poi_bbox_wgs84)
    highways = read_highways_near_geometry(highway_geojson_path, county_geom_wgs84)
    out = grids.copy()
    out["poi_entropy_raw"] = out["grid_id"].astype(str).map(poi_map).fillna(0.0)
    out["highway_dist_m"] = nearest_line_distance_m(lons, lats, highways)
    return out, poi_map


def attach_fuel_station_counts(
    grids: pd.DataFrame,
    poi_geoparquet_path: Path,
    grid: GridSystem,
    county_bounds: tuple[float, float, float, float],
    *,
    pad_deg: float = 0.02,
) -> pd.DataFrame:
    minx, miny, maxx, maxy = county_bounds
    bbox = (minx - pad_deg, miny - pad_deg, maxx + pad_deg, maxy + pad_deg)
    counts = fuel_station_counts_by_cell(poi_geoparquet_path, grid, bbox)
    out = grids.copy()
    out["fuel_station_count_raw"] = out["grid_id"].astype(str).map(counts).fillna(0).astype(float)
    return out


# ---------------------------------------------------------------------------
# 1 km 网格构建（栅格 + 矢量合一）
# ---------------------------------------------------------------------------


def _iter_cells_inside_polygon(
    geom,
    lon_min: float,
    lat_min: float,
    km_lon: float,
    km_lat: float,
    nx: int,
    ny: int,
) -> list[tuple[int, int, float, float]]:
    cells: list[tuple[int, int, float, float]] = []
    for gx in range(nx):
        for gy in range(ny):
            lon = lon_min + (gx + 0.5) / km_lon
            lat = lat_min + (gy + 0.5) / km_lat
            if geom.covers(Point(lon, lat)):
                cells.append((gx, gy, lon, lat))
    return cells


def build_county_grid_1km(
    usa_map_path: Path,
    state_name: str,
    county_name: str,
    worldpop_tif_path: Path,
    gdp_tif_path: Path,
    gdp_band_1based: int,
    poi_geoparquet_path: Path,
    highway_geojson_path: Path,
    *,
    poi_bbox_pad_deg: float = 0.02,
) -> tuple[pd.DataFrame, GridSystem, dict[str, float]]:
    county = load_county_gdf(usa_map_path, state_name, county_name)
    geom = county.geometry.unary_union
    minx, miny, maxx, maxy = geom.bounds
    lat_rad = representative_latitude_rad(county)
    km_lon, km_lat = km_per_degree_at_latitude(lat_rad)

    nx = int(np.ceil((maxx - minx) * km_lon))
    ny = int(np.ceil((maxy - miny) * km_lat))
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Invalid county grid extent for {county_name}")

    grid = GridSystem(
        lon_min=float(minx),
        lat_min=float(miny),
        km_per_deg_lon=float(km_lon),
        km_per_deg_lat=float(km_lat),
        base_cell_km=1,
        target_cell_km=1,
    )

    cell_list = _iter_cells_inside_polygon(geom, minx, miny, km_lon, km_lat, nx, ny)
    if not cell_list:
        raise ValueError(f"No 1 km cells inside county polygon: {county_name}")

    rows = []
    lons_list: list[float] = []
    lats_list: list[float] = []
    for gx, gy, lon, lat in cell_list:
        rows.append({"grid_x": gx, "grid_y": gy, "grid_id": f"{gx}_{gy}"})
        lons_list.append(lon)
        lats_list.append(lat)

    grids = pd.DataFrame(rows)
    lons = np.asarray(lons_list, dtype=float)
    lats = np.asarray(lats_list, dtype=float)

    with rasterio.open(worldpop_tif_path) as src_pop:
        pop = np.array([v[0] for v in src_pop.sample(list(zip(lons, lats)))], dtype=float)
    pop[~np.isfinite(pop)] = np.nan
    pop[pop < 0] = np.nan
    grids["population_raw"] = pop

    grids["gdp_raw"] = sample_raster_at_lonlat(
        gdp_tif_path, lons, lats, band_index_1based=int(gdp_band_1based)
    )

    poi_bbox = (
        minx - poi_bbox_pad_deg,
        miny - poi_bbox_pad_deg,
        maxx + poi_bbox_pad_deg,
        maxy + poi_bbox_pad_deg,
    )
    grids, poi_by_gid = attach_highway_and_poi_to_grids(
        grids,
        grid,
        poi_geoparquet_path,
        highway_geojson_path,
        geom,
        poi_bbox,
    )
    return grids, grid, poi_by_gid
