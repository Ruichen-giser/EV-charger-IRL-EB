"""县元特征（3 维社会经济）与美国 51 州（50 州 + DC）固定 state_id 词表。"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paths import DEFAULT_JOINT_USA_MAP_GEOJSON

# 50 州 + DC，state_id ∈ [0, 50]
US_STATE_NAMES: tuple[str, ...] = (
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "District of Columbia",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
)

N_US_STATES = len(US_STATE_NAMES)  # 51

STATE_NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(US_STATE_NAMES)}

STATE_ALIASES: dict[str, str] = {
    "DC": "District of Columbia",
    "District Of Columbia": "District of Columbia",
}

# joint_usa_map.geojson 属性字段
GEOJSON_EDUCATION_FIELD = "education_bachelors_pct_2019_23"
GEOJSON_POPULATION_FIELD = "population_census_2020"
GEOJSON_INCOME_FIELD = "median_hh_income_2023"

# county_meta 三维（社会经济，部署前可观测）
COUNTY_META_DIM = 3
COUNTY_META_FEATURE_NAMES: tuple[str, ...] = (
    "education_bachelors_pct_norm",  # 受教育程度（本科及以上占比 / 100）
    "log_population_z",              # 人口数量 log1p + z-score
    "log_median_hh_income_z",        # 家庭收入（中位数）log1p + z-score
)


def state_name_to_id(state_name: str) -> int:
    key = normalize_state_name(state_name)
    if key not in STATE_NAME_TO_ID:
        raise KeyError(f"未知州名 {state_name!r}，须为 US 51 州/DC 之一")
    return int(STATE_NAME_TO_ID[key])


def normalize_state_name(state_name: str) -> str:
    key = str(state_name).strip()
    return STATE_ALIASES.get(key, key)


def normalize_county_name(county_name: str) -> str:
    return str(county_name).strip().replace("_", " ")


def county_name_variants(county_name: str) -> tuple[str, ...]:
    """匹配 geojson NAME_2 与 Excel 县名（St./Saint 等变体）。"""
    base = normalize_county_name(county_name)
    variants = {base}
    if base.startswith("St. "):
        variants.add("Saint " + base[4:])
    if base.startswith("Saint "):
        variants.add("St. " + base[6:])
    return tuple(variants)


def _as_finite_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(x):
        return float(default)
    return float(x)


@dataclass(frozen=True)
class CountySocioEconomicIndex:
    """(state, county) → 标准化后的 3 维 county_meta。"""

    geojson_path: Path
    lookup: dict[tuple[str, str], np.ndarray]
    raw_stats: dict[str, dict[str, float]]

    @classmethod
    def from_geojson(cls, geojson_path: str | Path) -> CountySocioEconomicIndex:
        path = Path(geojson_path)
        if not path.is_file():
            raise FileNotFoundError(f"County socioeconomic geojson not found: {path}")

        with path.open(encoding="utf-8") as f:
            payload = json.load(f)

        features = payload.get("features", [])
        if not features:
            raise ValueError(f"GeoJSON 无 features: {path}")

        raw_rows: list[tuple[tuple[str, str], np.ndarray]] = []
        for feat in features:
            props = feat.get("properties") or {}
            state = normalize_state_name(props.get("NAME_1", ""))
            county = normalize_county_name(props.get("NAME_2", ""))
            if not state or not county:
                continue
            raw = np.asarray(
                [
                    _as_finite_float(props.get(GEOJSON_EDUCATION_FIELD)),
                    _as_finite_float(props.get(GEOJSON_POPULATION_FIELD)),
                    _as_finite_float(props.get(GEOJSON_INCOME_FIELD)),
                ],
                dtype=np.float64,
            )
            key = (state, county)
            raw_rows.append((key, raw))

        if not raw_rows:
            raise ValueError(f"GeoJSON 未解析到有效县记录: {path}")

        mat = np.stack([row[1] for row in raw_rows], axis=0)
        edu = mat[:, 0]
        pop_log = np.log1p(np.maximum(mat[:, 1], 0.0))
        inc_log = np.log1p(np.maximum(mat[:, 2], 0.0))

        edu_fill = float(np.median(edu[edu > 0])) if np.any(edu > 0) else 0.0
        pop_fill = float(np.median(pop_log[pop_log > 0])) if np.any(pop_log > 0) else 0.0
        inc_fill = float(np.median(inc_log[inc_log > 0])) if np.any(inc_log > 0) else 0.0

        edu = np.where(edu > 0, edu, edu_fill)
        pop_log = np.where(pop_log > 0, pop_log, pop_fill)
        inc_log = np.where(inc_log > 0, inc_log, inc_fill)

        pop_mean, pop_std = float(pop_log.mean()), float(pop_log.std())
        inc_mean, inc_std = float(inc_log.mean()), float(inc_log.std())
        pop_std = pop_std if pop_std > 1e-8 else 1.0
        inc_std = inc_std if inc_std > 1e-8 else 1.0

        lookup: dict[tuple[str, str], np.ndarray] = {}
        for (state, county), raw in raw_rows:
            edu_v = float(raw[0]) if raw[0] > 0 else edu_fill
            pop_v = float(np.log1p(max(raw[1], 0.0)))
            inc_v = float(np.log1p(max(raw[2], 0.0)))
            if pop_v <= 0:
                pop_v = pop_fill
            if inc_v <= 0:
                inc_v = inc_fill
            meta = np.asarray(
                [
                    edu_v / 100.0,
                    (pop_v - pop_mean) / pop_std,
                    (inc_v - inc_mean) / inc_std,
                ],
                dtype=np.float32,
            )
            lookup[(state, county)] = meta

        stats = {
            "education_bachelors_pct_median": edu_fill,
            "log_population_mean": pop_mean,
            "log_population_std": pop_std,
            "log_median_hh_income_mean": inc_mean,
            "log_median_hh_income_std": inc_std,
            "n_counties": float(len(lookup)),
        }
        return cls(geojson_path=path, lookup=lookup, raw_stats=stats)

    def get(self, state_name: str, county_name: str) -> np.ndarray:
        state = normalize_state_name(state_name)
        for county in county_name_variants(county_name):
            key = (state, county)
            if key in self.lookup:
                return self.lookup[key].copy()
        raise KeyError(
            f"geojson 中未找到县 socioeconomic 元数据: {state_name}/{county_name} "
            f"(path={self.geojson_path})"
        )


_INDEX: CountySocioEconomicIndex | None = None
_INDEX_PATH: Path | None = None


def get_county_socioeconomic_index(
    geojson_path: str | Path | None = None,
) -> CountySocioEconomicIndex:
    global _INDEX, _INDEX_PATH
    path = Path(geojson_path) if geojson_path is not None else DEFAULT_JOINT_USA_MAP_GEOJSON
    if _INDEX is None or _INDEX_PATH != path:
        _INDEX = CountySocioEconomicIndex.from_geojson(path)
        _INDEX_PATH = path
    return _INDEX


def reset_county_socioeconomic_index() -> None:
    """测试用：清空索引缓存。"""
    global _INDEX, _INDEX_PATH
    _INDEX = None
    _INDEX_PATH = None


def compute_county_meta(
    state_name: str,
    county_name: str,
    *,
    geojson_path: str | Path | None = None,
) -> np.ndarray:
    """返回 (3,) float32：受教育程度、人口、家庭收入（标准化）。"""
    meta = get_county_socioeconomic_index(geojson_path).get(state_name, county_name)
    if meta.shape[0] != COUNTY_META_DIM:
        raise ValueError(f"county_meta dim mismatch: {meta.shape}")
    return meta


def validate_socioeconomic_coverage(
    pairs: list,
    *,
    geojson_path: str | Path | None = None,
) -> None:
    """训练前检查：所有 state-county 均能在 geojson 中查到 socioeconomic meta。"""
    idx = get_county_socioeconomic_index(geojson_path)
    missing: list[str] = []
    for pair in pairs:
        state = getattr(pair, "state_name", None) or pair[0]
        county = getattr(pair, "county_name", None) or pair[1]
        try:
            idx.get(str(state), str(county))
        except KeyError:
            missing.append(f"{state}/{county}")
    if missing:
        preview = ", ".join(missing[:12])
        more = f" ... (+{len(missing) - 12})" if len(missing) > 12 else ""
        raise ValueError(
            f"{len(missing)} 个县在 {idx.geojson_path} 中缺少 socioeconomic 字段: "
            f"{preview}{more}"
        )


def compute_county_meta_from_npz(
    grid_npz: str | Path,
    *,
    geojson_path: str | Path | None = None,
) -> np.ndarray:
    """从 npz 读取 state/county 名称，再查 geojson socioeconomic 元数据。"""
    path = Path(grid_npz)
    blob = np.load(path, allow_pickle=False)
    if "state_name" in blob and "county_name" in blob:
        state = str(blob["state_name"][0])
        county = str(blob["county_name"][0])
    else:
        raise ValueError(f"npz 缺少 state_name/county_name，无法查 socioeconomic meta: {path}")
    return compute_county_meta(state, county, geojson_path=geojson_path)
