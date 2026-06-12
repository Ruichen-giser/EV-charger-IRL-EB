"""State–county 标识、npz 路径解析与训练词表（与 IRL_data/county_list.py 命名一致）。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StateCountyPair:
    state_name: str
    county_name: str

    def __post_init__(self) -> None:
        if not str(self.state_name).strip() or not str(self.county_name).strip():
            raise ValueError(f"invalid StateCountyPair: {self!r}")

    @property
    def key(self) -> str:
        return f"{self.state_name}/{self.county_name}"

    @classmethod
    def parse(cls, text: str) -> StateCountyPair:
        raw = str(text).strip()
        if not raw:
            raise ValueError("empty state-county string")
        for sep in ("/", "|", ":"):
            if sep in raw:
                state, county = raw.split(sep, 1)
                return cls(state.strip(), county.strip())
        raise ValueError(f"state-county 须含分隔符 / | : ，当前: {text!r}")


def _safe_token(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(name).strip())


def county_grid_npz_stem(state_name: str, county_name: str) -> str:
    return f"{_safe_token(state_name)}__{_safe_token(county_name)}"


def county_grid_npz_filename(state_name: str, county_name: str) -> str:
    return f"{county_grid_npz_stem(state_name, county_name)}_grid_features.npz"


def cropped_npz_filename(state_name: str, county_name: str) -> str:
    return f"{county_grid_npz_stem(state_name, county_name)}_grid_cropped.npz"


def parse_npz_stem(stem: str) -> StateCountyPair | None:
    """从 California__Los_Angeles 或旧版 Los_Angeles 解析。"""
    text = str(stem).strip()
    if "__" in text:
        state_tok, county_tok = text.split("__", 1)
        return StateCountyPair(state_tok.replace("_", " "), county_tok.replace("_", " "))
    return None


def state_county_npz_paths(grid_dir: Path, pair: StateCountyPair) -> tuple[Path, Path]:
    stem = county_grid_npz_stem(pair.state_name, pair.county_name)
    return (
        grid_dir / f"{stem}_grid_features.npz",
        grid_dir / f"{stem}_grid_cropped.npz",
    )


def resolve_state_county_npz_paths(grid_dir: Path, pair: StateCountyPair) -> tuple[Path, Path]:
    src = grid_dir / county_grid_npz_filename(pair.state_name, pair.county_name)
    dst = grid_dir / cropped_npz_filename(pair.state_name, pair.county_name)
    if src.is_file():
        return src, dst

    legacy_safe = pair.county_name.replace(" ", "_")
    legacy_src = grid_dir / f"{legacy_safe}_grid_features.npz"
    if legacy_src.is_file():
        return legacy_src, grid_dir / f"{legacy_safe}_grid_cropped.npz"

    raise FileNotFoundError(
        f"缺少 {pair.key} 网格: {src}\n"
        f"（亦尝试旧命名 {legacy_src}）\n"
        "请先运行 IRL_data/main.py 生成 grid_tensors。"
    )


def parse_state_county_list(text: str) -> list[StateCountyPair]:
    out: list[StateCountyPair] = []
    seen: set[tuple[str, str]] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        pair = StateCountyPair.parse(part)
        key = (pair.state_name, pair.county_name)
        if key in seen:
            continue
        seen.add(key)
        out.append(pair)
    if not out:
        raise ValueError("state-county 列表为空")
    return out


def load_mdp_ge5_county_list(excel_path: str | Path, *, sheet_name: str | int = 0) -> list[StateCountyPair]:
    path = Path(excel_path)
    if not path.is_file():
        raise FileNotFoundError(f"County list Excel not found: {path}")
    df = pd.read_excel(path, sheet_name=sheet_name)
    required = {"state", "county"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel {path} missing columns: {sorted(missing)}")
    out: list[StateCountyPair] = []
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
        out.append(StateCountyPair(state, county))
    if not out:
        raise ValueError(f"No valid state-county rows in {path}")
    return out


def discover_grid_npz_pairs(grid_dir: Path) -> list[StateCountyPair]:
    pairs: list[StateCountyPair] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(grid_dir.glob("*_grid_features.npz")):
        stem = path.name[: -len("_grid_features.npz")]
        pair = parse_npz_stem(stem)
        if pair is None:
            blob = np.load(path, allow_pickle=False)
            if "state_name" in blob and "county_name" in blob:
                pair = StateCountyPair(str(blob["state_name"][0]), str(blob["county_name"][0]))
            else:
                pair = StateCountyPair("", str(blob["county_name"][0]) if "county_name" in blob else stem)
        key = (pair.state_name, pair.county_name)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(pair)
    return pairs


def build_location_vocab(
    pairs: list[StateCountyPair],
) -> tuple[dict[tuple[str, str], int], dict[str, int], list[str], list[str]]:
    """返回 (pair→location_id, state→state_id, state_names, location_labels)。"""
    state_names = sorted({p.state_name for p in pairs if p.state_name})
    state_to_id = {s: i for i, s in enumerate(state_names)}
    location_labels = [p.key for p in pairs]
    pair_to_location_id = {(p.state_name, p.county_name): i for i, p in enumerate(pairs)}
    return pair_to_location_id, state_to_id, state_names, location_labels


def default_output_stem(pair: StateCountyPair) -> str:
    return county_grid_npz_stem(pair.state_name, pair.county_name).lower()
