"""v6 数据产物版本号与网格字段名常量。"""
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

# 1 km → 2 km 聚合时对下列列取均值
GRID_AGG_MEAN_COLS = (
    "population_raw",
    "gdp_raw",
    "poi_entropy_raw",
    "highway_dist_m",
    "fuel_station_count_raw",
    *GRID_FEATURE_PCT_COLS,
)
