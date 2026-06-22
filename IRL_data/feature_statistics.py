"""
IRL_data 特征分布统计与可视化。

前提：已运行 main.py 生成 prepared_irl_dataset.pkl（覆盖的县即统计范围内的「全美」网格样本）。

输出：
  1. 全部 2 km 网格的五维特征直方图（柱状图）
  2. 历史充电站所在网格的五维特征直方图（柱状图）
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paths import DEFAULT_GRID_NPZ_DIR, DEFAULT_PREPARED_DATA_PKL, OUTPUTS_DIR, validate_project_layout
from schema import GRID_FEATURE_PCT_COLS

# 特征列 → 图例中文名（值为县内 min-max 归一化后的分位，约在 [0,1]）
FEATURE_LABELS: dict[str, str] = {
    "population_pct": "人口",
    "gdp_pct": "GDP",
    "poi_pct": "POI 熵",
    "highway_dist_pct": "公路接近度",
    "fuel_station_count_pct": "加油站密度",
}

DEFAULT_STATS_DIR = OUTPUTS_DIR / "statistics"

# macOS / 常见系统中文字体候选（按优先级）
_CJK_FONT_CANDIDATES = (
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Micro Hei",
)

# macOS 系统字体路径（按优先级；用 fname 加载，避免字体名与系统登记不一致）
_MACOS_FONT_PATHS = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
)

_CHINESE_FONT_PROP = None


def configure_matplotlib_chinese():
    """
    从系统字体文件加载 CJK 字体，并写入 rcParams。
    返回 FontProperties，绘图时需传给 title/xlabel/suptitle。
    """
    global _CHINESE_FONT_PROP
    if _CHINESE_FONT_PROP is not None:
        return _CHINESE_FONT_PROP

    import matplotlib
    from matplotlib import font_manager

    prop = None
    for path in _MACOS_FONT_PATHS:
        p = Path(path)
        if not p.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(p))
            prop = font_manager.FontProperties(fname=str(p))
            break
        except Exception:
            continue

    if prop is None:
        available = {f.name for f in font_manager.fontManager.ttflist}
        chosen = next((name for name in _CJK_FONT_CANDIDATES if name in available), None)
        if chosen:
            prop = font_manager.FontProperties(family=chosen)
        else:
            prop = font_manager.FontProperties(family="DejaVu Sans")

    fam = prop.get_name()
    matplotlib.rcParams["font.family"] = fam
    matplotlib.rcParams["font.sans-serif"] = [fam]
    matplotlib.rcParams["axes.unicode_minus"] = False
    _CHINESE_FONT_PROP = prop
    return prop


@dataclass
class StatisticsConfig:
    prepared_pkl: str = field(default_factory=lambda: str(DEFAULT_PREPARED_DATA_PKL))
    grid_npz_dir: str | None = field(default_factory=lambda: str(DEFAULT_GRID_NPZ_DIR))
    output_dir: str = field(default_factory=lambda: str(DEFAULT_STATS_DIR))
    n_bins: int = 5
    value_min: float = 0.0
    value_max: float = 1.0
    dpi: int = 120
    use_npz_if_no_pkl: bool = True


def load_prepared_packs(prepared_pkl: str | Path) -> list[dict[str, Any]]:
    path = Path(prepared_pkl)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 {path}，请先运行: python IRL_data/main.py\n"
            "（全美范围取决于 main 中 county_names 覆盖的县数）"
        )
    with path.open("rb") as f:
        payload = pickle.load(f)
    packs = payload.get("packs", [])
    if not packs:
        raise ValueError(f"{path} 中 packs 为空")
    return packs


def _feature_columns(grids: pd.DataFrame) -> list[str]:
    return [c for c in GRID_FEATURE_PCT_COLS if c in grids.columns]


def all_grids_feature_table(packs: list[dict[str, Any]]) -> pd.DataFrame:
    """各县 2 km 动作空间内全部网格的特征（长表，含 county_name）。"""
    parts: list[pd.DataFrame] = []
    for row in packs:
        grids = row["grids"]
        cols = _feature_columns(grids)
        if not cols:
            continue
        sub = grids[cols].copy()
        sub["county_name"] = str(row.get("county_name", ""))
        sub["grid_id"] = grids["grid_id"].astype(str)
        parts.append(sub)
    if not parts:
        raise ValueError("packs 中无可用特征列")
    return pd.concat(parts, ignore_index=True)


def station_grids_feature_table(packs: list[dict[str, Any]]) -> pd.DataFrame:
    """
    历史上出现过充电站的网格：按 (county_name, grid_id) 去重后取 grids 表中的特征。
    同一格多次建站只计一格。
    """
    parts: list[pd.DataFrame] = []
    for row in packs:
        grids = row["grids"]
        events = row["events"]
        cols = _feature_columns(grids)
        if not cols or events.empty:
            continue
        station_ids = events["grid_id"].astype(str).unique()
        g = grids[grids["grid_id"].astype(str).isin(station_ids)].copy()
        if g.empty:
            continue
        g = g.drop_duplicates(subset=["grid_id"], keep="first")
        sub = g[cols].copy()
        sub["county_name"] = str(row.get("county_name", ""))
        sub["grid_id"] = g["grid_id"].astype(str)
        parts.append(sub)
    if not parts:
        raise ValueError("未匹配到任何充电站所在网格，请检查 events 与 grids 的 grid_id")
    return pd.concat(parts, ignore_index=True)


def load_features_from_npz_dir(npz_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    从 grid_tensors/*.npz 读取特征（仅当有 expert_actions 时可区分建站格）。
    返回 (all_cells_df, station_cells_df|None)。
    """
    npz_dir = Path(npz_dir)
    paths = sorted(npz_dir.glob("*_grid_features.npz"))
    if not paths:
        raise FileNotFoundError(f"未找到 npz: {npz_dir}")

    all_parts: list[pd.DataFrame] = []
    station_parts: list[pd.DataFrame] = []

    for p in paths:
        blob = np.load(p, allow_pickle=False)
        feats = blob["grid_features"].astype(np.float32)
        valid = blob["valid_mask"].astype(bool)
        county = str(blob["county_name"][0]) if "county_name" in blob else p.stem

        H, W, _ = feats.shape
        iy, ix = np.where(valid)
        row = {"county_name": county}
        for ci, col in enumerate(GRID_FEATURE_PCT_COLS):
            row[col] = feats[iy, ix, ci]
        all_parts.append(pd.DataFrame(row))

        if "expert_actions" in blob:
            actions = blob["expert_actions"].astype(np.int64)
            gy0 = int(blob["gy0"]) if "gy0" in blob else 0
            gx0 = int(blob["gx0"]) if "gx0" in blob else 0
            seen: set[tuple[int, int]] = set()
            rows: list[dict[str, float]] = []
            for a in actions:
                ix = int(a) % W
                iy = int(a) // W
                gy, gx = iy + gy0, ix + gx0
                key = (gy, gx)
                if key in seen:
                    continue
                seen.add(key)
                # 用局部索引取特征
                ly, lx = iy, ix
                if ly < 0 or lx < 0 or ly >= H or lx >= W:
                    continue
                row_dict = {
                    col: float(feats[ly, lx, ci]) for ci, col in enumerate(GRID_FEATURE_PCT_COLS)
                }
                row_dict["county_name"] = county
                rows.append(row_dict)
            if rows:
                station_parts.append(pd.DataFrame(rows))

    all_df = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    station_df = pd.concat(station_parts, ignore_index=True) if station_parts else None
    return all_df, station_df


def summarize_features(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_cells": int(len(df)), "features": {}}
    for col in feature_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        out["features"][col] = {
            "label": FEATURE_LABELS.get(col, col),
            "count": int(len(s)),
            "mean": float(s.mean()) if len(s) else None,
            "std": float(s.std()) if len(s) else None,
            "min": float(s.min()) if len(s) else None,
            "max": float(s.max()) if len(s) else None,
        }
    return out


def plot_feature_distributions(
    df: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    feature_cols: list[str] | None = None,
    n_bins: int = 5,
    value_min: float = 0.0,
    value_max: float = 1.0,
    dpi: int = 120,
) -> None:
    """五维特征分布柱状图（直方图）。"""
    import matplotlib.pyplot as plt

    fp = configure_matplotlib_chinese()

    cols = feature_cols or [c for c in GRID_FEATURE_PCT_COLS if c in df.columns]
    if not cols:
        raise ValueError("DataFrame 中无特征列")

    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.8), squeeze=False)
    edges = np.linspace(value_min, value_max, int(n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = (edges[1] - edges[0]) * 0.9

    for ax, col in zip(axes[0], cols):
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        counts, _ = np.histogram(vals, bins=edges)
        ax.bar(centers, counts, width=width, color="#4C78A8", edgecolor="white", linewidth=0.4)
        ax.set_title(FEATURE_LABELS.get(col, col), fontsize=11, fontproperties=fp)
        ax.set_xlabel("县内归一化分位", fontproperties=fp)
        ax.set_ylabel("网格数", fontproperties=fp)
        ax.set_xlim(value_min, value_max)
        ax.ticklabel_format(style="plain", axis="y")

    counties = df["county_name"].nunique() if "county_name" in df.columns else 0
    subtitle = f"n={len(df):,} 格"
    if counties:
        subtitle += f"，{counties} 个县"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=12, y=1.02, fontproperties=fp)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_statistics(cfg: StatisticsConfig | None = None) -> dict[str, Any]:
    c = cfg or StatisticsConfig()
    font_used = configure_matplotlib_chinese().get_name()
    validate_project_layout()
    out_dir = Path(c.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = Path(c.prepared_pkl)
    if pkl_path.exists():
        packs = load_prepared_packs(pkl_path)
        df_all = all_grids_feature_table(packs)
        df_station = station_grids_feature_table(packs)
        data_source = str(pkl_path)
        county_list = sorted({str(p.get("county_name", "")) for p in packs})
    elif c.use_npz_if_no_pkl and c.grid_npz_dir and Path(c.grid_npz_dir).exists():
        df_all, df_station = load_features_from_npz_dir(c.grid_npz_dir)
        if df_station is None:
            raise ValueError("npz 模式下无法区分建站格，请先生成 prepared_irl_dataset.pkl")
        data_source = str(c.grid_npz_dir)
        county_list = sorted(df_all["county_name"].unique().tolist()) if "county_name" in df_all.columns else []
    else:
        raise FileNotFoundError(f"无可用数据: {pkl_path} 或 {c.grid_npz_dir}")

    feature_cols = [col for col in GRID_FEATURE_PCT_COLS if col in df_all.columns]

    plot_feature_distributions(
        df_all,
        title="全美 2 km 网格特征分布（已处理县汇总）",
        out_path=out_dir / "all_grids_feature_distribution.png",
        feature_cols=feature_cols,
        n_bins=int(c.n_bins),
        value_min=float(c.value_min),
        value_max=float(c.value_max),
        dpi=int(c.dpi),
    )
    plot_feature_distributions(
        df_station,
        title="历史充电站所在网格特征分布",
        out_path=out_dir / "station_grids_feature_distribution.png",
        feature_cols=feature_cols,
        n_bins=int(c.n_bins),
        value_min=float(c.value_min),
        value_max=float(c.value_max),
        dpi=int(c.dpi),
    )

    summary = {
        "data_source": data_source,
        "matplotlib_font": str(font_used),
        "n_counties": len(county_list),
        "counties": county_list,
        "note": "特征为各县内 min-max 归一化分位；跨县汇总 histogram 表示「县内相对水平」的分布。",
        "all_grids": summarize_features(df_all, feature_cols),
        "station_grids": summarize_features(df_station, feature_cols),
        "figures": {
            "all_grids": str(out_dir / "all_grids_feature_distribution.png"),
            "station_grids": str(out_dir / "station_grids_feature_distribution.png"),
        },
    }
    with (out_dir / "feature_distribution_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[IRL_data/feature_statistics] 字体: {font_used}")
    print(f"[IRL_data/feature_statistics] 全部网格 n={summary['all_grids']['n_cells']:,}")
    print(f"[IRL_data/feature_statistics] 建站网格 n={summary['station_grids']['n_cells']:,}")
    print(f"[IRL_data/feature_statistics] 图 → {out_dir}")
    return summary


def main() -> None:
    run_statistics()


if __name__ == "__main__":
    main()
