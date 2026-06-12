"""从 MDP≥5 县清单 Excel 读取 state-county 对，并生成唯一 npz 文件名。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _safe_token(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(name).strip())


def county_grid_npz_stem(state_name: str, county_name: str) -> str:
    """多州同名县不冲突：California__Los_Angeles。"""
    return f"{_safe_token(state_name)}__{_safe_token(county_name)}"


def county_grid_npz_filename(state_name: str, county_name: str) -> str:
    return f"{county_grid_npz_stem(state_name, county_name)}_grid_features.npz"


def load_mdp_ge5_county_list(
    excel_path: str | Path,
    *,
    sheet_name: str | int = 0,
) -> list[tuple[str, str]]:
    """
    读取 us_mdp_ge5_state_county_list.xlsx，返回 [(state, county), ...]。
    列名须含 state、county（与 explore_data/export_mdp_ge5_counties.py 一致）。
    """
    path = Path(excel_path)
    if not path.is_file():
        raise FileNotFoundError(f"County list Excel not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name)
    required = {"state", "county"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel {path} missing columns: {sorted(missing)}")

    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in df.itertuples(index=False):
        state = str(getattr(row, "state", "")).strip()
        county = str(getattr(row, "county", "")).strip()
        if not state or not county:
            continue
        key = (state, county)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    if not out:
        raise ValueError(f"No valid state-county rows in {path}")
    return out
