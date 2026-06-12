"""县元特征（5 维）与美国 51 州（50 州 + DC）固定 state_id 词表。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

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

# county_meta 五维（部署前可观测，无专家标签泄漏）
COUNTY_META_DIM = 5
COUNTY_META_FEATURE_NAMES: tuple[str, ...] = (
    "log_grid_cells",       # log1p(H*W)，县网格规模
    "log_aspect_ratio",     # log(H/W)，形状
    "valid_cell_ratio",     # valid_mask 占比
    "log_expert_steps",     # log1p(专家建站步数)
    "mean_population_pct",  # 有效格上 population_pct 均值
)


def state_name_to_id(state_name: str) -> int:
    key = str(state_name).strip()
    if key not in STATE_NAME_TO_ID:
        raise KeyError(f"未知州名 {state_name!r}，须为 US 51 州/DC 之一")
    return int(STATE_NAME_TO_ID[key])


def compute_county_meta_from_npz(grid_npz: str | Path) -> np.ndarray:
    """
    从裁剪后/原始 npz 计算 5 维 county_meta（float32）。
    仅用网格尺寸、valid_mask、expert_actions 长度与 grid_features 人口通道。
    """
    path = Path(grid_npz)
    blob = np.load(path, allow_pickle=False)
    h, w = int(blob["grid_features"].shape[0]), int(blob["grid_features"].shape[1])
    valid = blob["valid_mask"].astype(bool)
    n_cells = max(h * w, 1)
    valid_ratio = float(valid.sum()) / float(n_cells)

    expert = blob["expert_actions"] if "expert_actions" in blob else np.array([], dtype=np.int64)
    n_steps = int(np.asarray(expert).reshape(-1).size)

    pop = blob["grid_features"][:, :, 0].astype(np.float32)
    if valid.any():
        mean_pop = float(pop[valid].mean())
    else:
        mean_pop = float(pop.mean()) if pop.size else 0.0

    aspect = float(h) / max(float(w), 1.0)
    meta = np.asarray(
        [
            np.log1p(float(h * w)),
            np.log(max(aspect, 1e-6)),
            valid_ratio,
            np.log1p(float(n_steps)),
            mean_pop,
        ],
        dtype=np.float32,
    )
    if meta.shape[0] != COUNTY_META_DIM:
        raise ValueError(f"county_meta dim mismatch: {meta.shape}")
    return meta
