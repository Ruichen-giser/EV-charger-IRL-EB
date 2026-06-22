"""v6 数据产物版本号与网格字段名常量。"""
from __future__ import annotations

# prepared_irl_dataset.pkl 的 schema 标识
SCHEMA_VERSION = "v6_prepared_irl_dataset_v1"

# Foursquare 加油站 POI 类别标签（用于 gis_align）
FUEL_STATION_LABEL = "Travel and Transportation > Fuel Station"

# 县内 min-max 归一化后的 5 通道顺序（与 npz grid_features[:,:,k] 一致）
# k=0 人口, 1 GDP, 2 POI 熵, 3 公路接近度(越大越近), 4 加油站密度(县内 min-max)
GRID_FEATURE_PCT_COLS = (
    "population_pct",
    "gdp_pct",
    "poi_pct",
    "highway_dist_pct",
    "fuel_station_count_pct",
)

# 1 km → target km 聚合时仅对栅格特征取均值；POI/公路/加油站在 target 格网上重算
GRID_AGG_MEAN_COLS = (
    "population_raw",
    "gdp_raw",
)
