# Data layout

Place raw GIS and EV charging station datasets under this directory. The default paths in `IRL_data/main.py` expect:

```text
data/
├── US-EV-Station-2014-2025/Daily/EVCS_sequence.pkl
├── US-map/usa_map.geojson
├── US-WorldPOP-2014-2020/clipped/usa_ppp_2015_1km_Aggregated_UNadj_clipped_usa.tif
├── US-GDP/rast_gdpTot_1990_2024_30arcsec.tif
├── US-poi-2014-2024/data_2015_geoparquet.parquet
└── US-highway/NTAD_National_Highway_System.geojson
├── joint_usa_map.geojson              # 县社会经济属性（IQ county_meta 三维）
└── us_mdp_ge5_state_county_list.xlsx  # MDP≥5 训练县清单
```

`joint_usa_map.geojson` 每个县 feature 的 `properties` 提供：

- `education_bachelors_pct_2019_23` — 受教育程度（本科及以上占比 %）
- `population_census_2020` — 人口数量
- `median_hh_income_2023` — 家庭收入中位数

`us_mdp_ge5_state_county_list.xlsx` 已纳入版本库（小文件）。`joint_usa_map.geojson` 体积较大（~95MB），需自行放置到本目录。

After running `IRL_data/main.py`, prepared artifacts are written to `outputs/prepared_data/`.
