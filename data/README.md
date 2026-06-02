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
```

After running `IRL_data/main.py`, prepared artifacts are written to `outputs/prepared_data/`.
