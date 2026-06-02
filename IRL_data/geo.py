"""
步骤 1a — 空间基底：县界 GeoJSON、坐标系、建站事件落在县内。
"""
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

CRS_WGS84 = "EPSG:4326"
CRS_ALBERS_CONUS = "EPSG:5070"


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gdf
    if gdf.crs is None:
        return gdf.set_crs(CRS_WGS84, allow_override=True)
    try:
        if gdf.crs.to_epsg() != 4326:
            return gdf.to_crs(CRS_WGS84)
    except Exception:
        return gdf.to_crs(CRS_WGS84)
    return gdf


@lru_cache(maxsize=4)
def _load_usa_map(usa_map_path: str) -> gpd.GeoDataFrame:
    return ensure_wgs84(gpd.read_file(usa_map_path))


def load_county_gdf(usa_map_path: Path, state_name: str, county_name: str) -> gpd.GeoDataFrame:
    usa = _load_usa_map(str(Path(usa_map_path).resolve()))
    county = usa[
        (usa["NAME_1"].astype(str).str.lower() == state_name.strip().lower())
        & (usa["NAME_2"].astype(str).str.lower() == county_name.strip().lower())
    ].copy()
    county = ensure_wgs84(county)
    if county.empty:
        raise ValueError(f"County not found in map: {county_name}, {state_name}")
    return county


def county_union_geometry(usa_map_path: Path, state_name: str, county_name: str):
    return load_county_gdf(usa_map_path, state_name, county_name).geometry.unary_union


def representative_latitude_rad(county_gdf: gpd.GeoDataFrame) -> float:
    gc = ensure_wgs84(county_gdf).to_crs(CRS_ALBERS_CONUS)
    cent = gc.geometry.centroid
    cent_wgs = gpd.GeoDataFrame(geometry=cent, crs=CRS_ALBERS_CONUS).to_crs(CRS_WGS84)
    return float(np.deg2rad(cent_wgs.geometry.iloc[0].y))


def km_per_degree_at_latitude(lat_rad: float) -> tuple[float, float]:
    km_lat = 111.32
    km_lon = 111.32 * float(np.cos(lat_rad))
    return km_lon, km_lat


def filter_events_within_county(
    events: pd.DataFrame,
    usa_map_path: Path,
    state_name: str,
    county_name: str,
) -> pd.DataFrame:
    county = load_county_gdf(usa_map_path, state_name, county_name)
    pts = gpd.GeoDataFrame(
        events.copy(),
        geometry=gpd.points_from_xy(events["Longitude"], events["Latitude"]),
        crs=CRS_WGS84,
    )
    joined = gpd.sjoin(pts, county[["geometry"]], how="inner", predicate="within")
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore")).reset_index(drop=True)
