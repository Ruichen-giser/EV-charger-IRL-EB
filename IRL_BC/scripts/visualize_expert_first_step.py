#!/usr/bin/env python3
"""
Los Angeles（或其它县）2 km 网格：人口 / 路网 + 建站格 + 县界与图层说明。

图层（Population 图）：
  - 高速公路线状叠加（细线）
  - Triple-zero / CNN padding 阴影 + 斜线填充
  - 已部署充电站：格心黑色小圆点
  - 浅灰网格细线、细线县界

示例（独立脚本，不参与 main.py 训练流程）：
  python IRL_BC/scripts/visualize_expert_first_step.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap

_PKG = Path(__file__).resolve().parents[1]
_REPO = _PKG.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from data_prep import county_npz_paths, prepare_county_npz  # noqa: E402

DEFAULT_GRID_DIR = _REPO / "outputs" / "prepared_data" / "grid_tensors"
DEFAULT_OUT_DIR = _REPO / "outputs" / "figures"
DEFAULT_USA_MAP = _REPO / "data" / "US-map" / "usa_map.geojson"
DEFAULT_HIGHWAY = _REPO / "data" / "US-highway" / "NTAD_National_Highway_System.geojson"

TRIPLE_ZERO_RGBA = (0.55, 0.55, 0.55, 0.35)
CNN_PAD_RGBA = (0.72, 0.86, 0.98, 0.38)
TRIPLE_ZERO_HATCH = "///"
CNN_PAD_HATCH = "\\\\\\"
HIGHWAY_COLOR = "#F9A825"
HIGHWAY_LINEWIDTH = 0.45
STATION_DOT_SIZE = 2.8

# 图面字号（精简文案 + 放大）
FS_PANEL_TITLE = 14
FS_AXIS_LABEL = 13
FS_TICK = 12
FS_SUPTITLE = 15
FS_LEGEND = 11
FS_CBAR = 12


@dataclass
class DisplayLayers:
    triple_zero: np.ndarray
    cnn_padding: np.ndarray
    network: np.ndarray
    in_county: np.ndarray


def _action_to_xy(action: int, grid_w: int) -> tuple[int, int]:
    a = int(action)
    return a % int(grid_w), a // int(grid_w)


def _masked_layer(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float64, copy=True)
    out[~valid_mask.astype(bool)] = np.nan
    return out


def _crop_window_offsets(blob: np.lib.npyio.NpzFile) -> tuple[int, int]:
    if "crop_meta" in blob:
        cm = blob["crop_meta"].item()
        if isinstance(cm, dict):
            return int(cm.get("crop_x0", 0)), int(cm.get("crop_y0", 0))
    return 0, 0


def _grid_origin_from_county(
    usa_map_path: Path,
    state_name: str,
    county_name: str,
    *,
    grid_cell_km: float,
) -> dict[str, float]:
    import geopandas as gpd

    usa = gpd.read_file(usa_map_path)
    if usa.crs is None:
        usa = usa.set_crs(4326)
    else:
        usa = usa.to_crs(4326)
    county = usa[
        (usa["NAME_1"].astype(str).str.lower() == state_name.strip().lower())
        & (usa["NAME_2"].astype(str).str.lower() == county_name.strip().lower())
    ]
    if county.empty:
        raise ValueError(f"usa_map 中未找到 {county_name}, {state_name}")
    minx, miny, _, _ = county.total_bounds
    union = county.geometry.union_all() if hasattr(county.geometry, "union_all") else county.geometry.unary_union
    lat0 = math.radians(float(union.centroid.y))
    return {
        "lon_min": float(minx),
        "lat_min": float(miny),
        "km_per_deg_lon": float(111.32 * math.cos(lat0)),
        "km_per_deg_lat": 111.32,
        "grid_cell_km": float(grid_cell_km),
    }


def _load_county_union(usa_map_path: Path, state_name: str, county_name: str):
    import geopandas as gpd

    usa = gpd.read_file(usa_map_path)
    if usa.crs is None:
        usa = usa.set_crs(4326)
    else:
        usa = usa.to_crs(4326)
    county = usa[
        (usa["NAME_1"].astype(str).str.lower() == state_name.strip().lower())
        & (usa["NAME_2"].astype(str).str.lower() == county_name.strip().lower())
    ]
    if county.empty:
        raise ValueError(f"usa_map 中未找到 {county_name}, {state_name}")
    return county.geometry.union_all() if hasattr(county.geometry, "union_all") else county.geometry.unary_union


def _cell_center_lonlat(gx: int, gy: int, origin: dict[str, float]) -> tuple[float, float]:
    cell = float(origin["grid_cell_km"])
    lon = float(origin["lon_min"]) + (int(gx) * cell + cell * 0.5) / float(origin["km_per_deg_lon"])
    lat = float(origin["lat_min"]) + (int(gy) * cell + cell * 0.5) / float(origin["km_per_deg_lat"])
    return lon, lat


def _lonlat_to_grid_xy(
    lon: np.ndarray,
    lat: np.ndarray,
    origin: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    cell = float(origin["grid_cell_km"])
    x_km = (np.asarray(lon, dtype=float) - float(origin["lon_min"])) * float(origin["km_per_deg_lon"])
    y_km = (np.asarray(lat, dtype=float) - float(origin["lat_min"])) * float(origin["km_per_deg_lat"])
    return x_km / cell, y_km / cell


def _rasterize_in_county_mask(
    h: int,
    w: int,
    gx0: int,
    gy0: int,
    origin: dict[str, float],
    county_geom,
) -> np.ndarray:
    from shapely.geometry import Point
    from shapely.prepared import prep

    prepared = prep(county_geom)
    mask = np.zeros((h, w), dtype=bool)
    for iy in range(h):
        for ix in range(w):
            lon, lat = _cell_center_lonlat(gx0 + ix, gy0 + iy, origin)
            if prepared.contains(Point(lon, lat)):
                mask[iy, ix] = True
    return mask


def _binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    m = mask.astype(bool)
    if radius <= 0:
        return m.copy()
    try:
        from scipy.ndimage import binary_dilation

        return binary_dilation(m, iterations=int(radius))
    except ImportError:
        out = m.copy()
        for _ in range(int(radius)):
            nxt = out.copy()
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                shifted = np.zeros_like(out)
                if dy == -1:
                    shifted[:-1, :] = out[1:, :]
                elif dy == 1:
                    shifted[1:, :] = out[:-1, :]
                else:
                    shifted[:, :] = out[:, :]
                if dx == -1:
                    s2 = np.zeros_like(out)
                    s2[:, :-1] = shifted[:, 1:]
                    shifted = s2
                elif dx == 1:
                    s2 = np.zeros_like(out)
                    s2[:, 1:] = shifted[:, :-1]
                    shifted = s2
                nxt |= shifted
            out = nxt
        return out


def _build_display_layers(
    *,
    full_npz_path: Path,
    crop_h: int,
    crop_w: int,
    crop_x0: int,
    crop_y0: int,
    network_mask: np.ndarray,
    origin: dict[str, float],
    county_geom,
    cnn_pad_radius: int = 1,
) -> DisplayLayers:
    """Triple-zero：县界内、建网格时因 POP/GDP/POI 全零被剔除的格（相对 full npz kept）。"""
    full = np.load(full_npz_path, allow_pickle=False)
    fh, fw = int(full["grid_features"].shape[0]), int(full["grid_features"].shape[1])
    fgx0 = int(full["gx0"]) if "gx0" in full else 0
    fgy0 = int(full["gy0"]) if "gy0" in full else 0
    kept_full = full["valid_mask"].astype(bool)

    in_county_full = _rasterize_in_county_mask(fh, fw, fgx0, fgy0, origin, county_geom)
    triple_full = in_county_full & ~kept_full

    y1, y2 = crop_y0, crop_y0 + crop_h
    x1, x2 = crop_x0, crop_x0 + crop_w
    triple = triple_full[y1:y2, x1:x2]
    in_county = in_county_full[y1:y2, x1:x2]

    dilated = _binary_dilate(network_mask.astype(bool), radius=cnn_pad_radius)
    cnn_pad = dilated & ~network_mask.astype(bool)

    return DisplayLayers(
        triple_zero=triple,
        cnn_padding=cnn_pad,
        network=network_mask.astype(bool),
        in_county=in_county,
    )


def _county_boundary_lines_grid(
    usa_map_path: Path,
    state_name: str,
    county_name: str,
    origin: dict[str, float],
    *,
    crop_x0: int = 0,
    crop_y0: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    import geopandas as gpd

    usa = gpd.read_file(usa_map_path)
    if usa.crs is None:
        usa = usa.set_crs(4326)
    else:
        usa = usa.to_crs(4326)
    county = usa[
        (usa["NAME_1"].astype(str).str.lower() == state_name.strip().lower())
        & (usa["NAME_2"].astype(str).str.lower() == county_name.strip().lower())
    ]
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for geom in county.geometry:
        if geom is None or geom.is_empty:
            continue
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            coords = np.asarray(poly.exterior.coords)
            x, y = _lonlat_to_grid_xy(coords[:, 0], coords[:, 1], origin)
            lines.append((x - float(crop_x0), y - float(crop_y0)))
    return lines


def _highway_lines_crop(
    highway_path: Path,
    county_geom,
    origin: dict[str, float],
    *,
    crop_x0: int,
    crop_y0: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    import geopandas as gpd

    minx, miny, maxx, maxy = county_geom.bounds
    pad = 0.05
    bbox = (minx - pad, miny - pad, maxx + pad, maxy + pad)
    hw = gpd.read_file(highway_path, bbox=bbox)
    if hw.crs is None:
        hw = hw.set_crs(4326)
    else:
        hw = hw.to_crs(4326)
    hw = hw[hw.geometry.geom_type.isin(["LineString", "MultiLineString"])]

    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for geom in hw.geometry:
        if geom is None or geom.is_empty:
            continue
        segs = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        for ls in segs:
            c = np.asarray(ls.coords, dtype=float)
            if c.shape[0] < 2:
                continue
            x, y = _lonlat_to_grid_xy(c[:, 0], c[:, 1], origin)
            lines.append((x - float(crop_x0), y - float(crop_y0)))
    return lines


def _overlay_rgba(ax: plt.Axes, mask: np.ndarray, rgba: tuple[float, float, float, float], extent: tuple[float, float, float, float]) -> None:
    if not mask.any():
        return
    layer = np.zeros((*mask.shape, 4), dtype=float)
    layer[mask] = rgba
    ax.imshow(layer, origin="lower", extent=extent, interpolation="nearest", zorder=2)


def _draw_hatched_mask(
    ax: plt.Axes,
    mask: np.ndarray,
    *,
    face_rgba: tuple[float, float, float, float],
    hatch: str,
    edgecolor: str,
    zorder: int,
) -> None:
    """在阴影格上叠加斜线，便于与底图区分。"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return
    patches = [
        mpatches.Rectangle((float(x) - 0.5, float(y) - 0.5), 1.0, 1.0)
        for x, y in zip(xs, ys)
    ]
    col = PatchCollection(
        patches,
        facecolor=face_rgba[:3],
        alpha=face_rgba[3],
        edgecolor=edgecolor,
        hatch=hatch,
        linewidth=0.25,
        zorder=zorder,
    )
    ax.add_collection(col)


def _deployed_station_xy(expert_actions: np.ndarray, grid_w: int) -> list[tuple[int, int]]:
    """专家轨迹中每格首次建站位置（已部署充电站）。"""
    seen: set[tuple[int, int]] = set()
    cells: list[tuple[int, int]] = []
    for a in np.asarray(expert_actions, dtype=np.int64).reshape(-1):
        gx, gy = _action_to_xy(int(a), grid_w)
        key = (gx, gy)
        if key in seen:
            continue
        seen.add(key)
        cells.append(key)
    return cells


def _draw_station_dots(ax: plt.Axes, stations: list[tuple[int, int]]) -> None:
    if not stations:
        return
    xs = [gx for gx, _ in stations]
    ys = [gy for _, gy in stations]
    ax.plot(
        xs,
        ys,
        linestyle="none",
        marker="o",
        markersize=STATION_DOT_SIZE,
        markerfacecolor="black",
        markeredgecolor="black",
        markeredgewidth=0.3,
        zorder=9,
    )


def _draw_grid_mesh(ax: plt.Axes, h: int, w: int, *, color: str = "#D0D0D0", linewidth: float = 0.35) -> None:
    xs = np.arange(-0.5, w + 0.5, 1.0)
    ys = np.arange(-0.5, h + 0.5, 1.0)
    segs: list[np.ndarray] = []
    for x in xs:
        segs.append(np.column_stack([np.full_like(ys, x), ys]))
    for y in ys:
        segs.append(np.column_stack([xs, np.full_like(xs, y)]))
    lc = LineCollection(segs, colors=color, linewidths=linewidth, zorder=3)
    ax.add_collection(lc)


def _draw_boundary(
    ax: plt.Axes,
    lines: list[tuple[np.ndarray, np.ndarray]],
    *,
    color: str = "#333333",
    linewidth: float = 1.0,
) -> None:
    for x, y in lines:
        ax.plot(x, y, color=color, linewidth=linewidth, solid_capstyle="round", zorder=7)


def _draw_highways(ax: plt.Axes, lines: list[tuple[np.ndarray, np.ndarray]]) -> None:
    for x, y in lines:
        ax.plot(
            x,
            y,
            color=HIGHWAY_COLOR,
            linewidth=HIGHWAY_LINEWIDTH,
            alpha=0.88,
            solid_capstyle="round",
            zorder=5,
        )


def _render_panel(
    ax: plt.Axes,
    data: np.ndarray,
    cmap: LinearSegmentedColormap,
    title: str,
    *,
    extent: tuple[float, float, float, float],
    layers: DisplayLayers,
    boundary: list[tuple[np.ndarray, np.ndarray]],
    station_xy: list[tuple[int, int]],
    highway_lines: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> plt.cm.ScalarMappable:
    h, w = data.shape
    _overlay_rgba(ax, layers.triple_zero, TRIPLE_ZERO_RGBA, extent)
    _overlay_rgba(ax, layers.cnn_padding, CNN_PAD_RGBA, extent)

    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
        interpolation="nearest",
        zorder=4,
    )
    _draw_hatched_mask(
        ax,
        layers.triple_zero,
        face_rgba=TRIPLE_ZERO_RGBA,
        hatch=TRIPLE_ZERO_HATCH,
        edgecolor="#666666",
        zorder=5,
    )
    _draw_hatched_mask(
        ax,
        layers.cnn_padding,
        face_rgba=CNN_PAD_RGBA,
        hatch=CNN_PAD_HATCH,
        edgecolor="#5C7A99",
        zorder=5,
    )
    _draw_grid_mesh(ax, h, w)
    if highway_lines:
        _draw_highways(ax, highway_lines)
    _draw_boundary(ax, boundary)
    _draw_station_dots(ax, station_xy)

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_title(title, fontsize=FS_PANEL_TITLE, pad=8)
    ax.set_xlabel("x", fontsize=FS_AXIS_LABEL)
    ax.set_ylabel("y", fontsize=FS_AXIS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    return im


def plot_expert_steps_map(
    npz_path: Path,
    *,
    out_path: Path,
    usa_map_path: Path,
    highway_path: Path,
    full_npz_path: Path,
    dpi: int = 170,
    grid_origin: dict[str, float] | None = None,
    cnn_pad_radius: int = 1,
) -> dict[str, Any]:
    blob = np.load(npz_path, allow_pickle=True)
    grid = blob["grid_features"].astype(np.float32)
    valid = blob["valid_mask"].astype(bool)
    expert_raw = blob["expert_actions"].astype(np.int64).reshape(-1)
    cell_km = float(blob["grid_cell_km"])
    county = str(blob["county_name"][0]) if "county_name" in blob else npz_path.stem
    state = str(blob["state_name"][0]) if "state_name" in blob else "California"

    crop_x0, crop_y0 = _crop_window_offsets(blob)
    h, w = int(grid.shape[0]), int(grid.shape[1])
    pop = _masked_layer(grid[:, :, 0], valid)
    highway = _masked_layer(grid[:, :, 3], valid)

    station_xy = _deployed_station_xy(expert_raw, w)
    if len(station_xy) == 0:
        raise ValueError(f"无专家建站格: {npz_path}")

    origin = grid_origin or _grid_origin_from_county(
        usa_map_path, state, county, grid_cell_km=cell_km
    )
    county_geom = _load_county_union(usa_map_path, state, county)
    layers = _build_display_layers(
        full_npz_path=full_npz_path,
        crop_h=h,
        crop_w=w,
        crop_x0=crop_x0,
        crop_y0=crop_y0,
        network_mask=valid,
        origin=origin,
        county_geom=county_geom,
        cnn_pad_radius=cnn_pad_radius,
    )
    boundary = _county_boundary_lines_grid(
        usa_map_path, state, county, origin, crop_x0=crop_x0, crop_y0=crop_y0
    )
    hw_lines = _highway_lines_crop(
        highway_path, county_geom, origin, crop_x0=crop_x0, crop_y0=crop_y0
    )

    cmap_pop = LinearSegmentedColormap.from_list(
        "pop", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    )
    cmap_hw = LinearSegmentedColormap.from_list(
        "hw", ["#fff5eb", "#fdd0a2", "#fd8d3c", "#e6550d", "#7f2709"]
    )

    extent = (-0.5, w - 0.5, -0.5, h - 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    im0 = _render_panel(
        axes[0],
        pop,
        cmap_pop,
        "Population",
        extent=extent,
        layers=layers,
        boundary=boundary,
        station_xy=station_xy,
        highway_lines=hw_lines,
    )
    im1 = _render_panel(
        axes[1],
        highway,
        cmap_hw,
        "Highway dist.",
        extent=extent,
        layers=layers,
        boundary=boundary,
        station_xy=station_xy,
        highway_lines=None,
    )
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.ax.tick_params(labelsize=FS_TICK)
    cbar0.set_label("pct", fontsize=FS_CBAR)
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=FS_TICK)
    cbar1.set_label("pct", fontsize=FS_CBAR)

    county_label = county.replace("_", " ")
    fig.suptitle(
        f"{county_label} · 2 km · {h}×{w} · {len(station_xy)} stations",
        fontsize=FS_SUPTITLE,
        y=1.02,
    )

    leg_handles = [
        mpatches.Patch(
            facecolor=TRIPLE_ZERO_RGBA[:3],
            alpha=TRIPLE_ZERO_RGBA[3],
            hatch=TRIPLE_ZERO_HATCH,
            edgecolor="#666666",
            label="Triple-zero",
        ),
        mpatches.Patch(
            facecolor=CNN_PAD_RGBA[:3],
            alpha=CNN_PAD_RGBA[3],
            hatch=CNN_PAD_HATCH,
            edgecolor="#5C7A99",
            label="CNN pad",
        ),
        plt.Line2D([0], [0], color=HIGHWAY_COLOR, linewidth=HIGHWAY_LINEWIDTH, label="Highway"),
        plt.Line2D([0], [0], color="#333333", linewidth=1.0, label="Boundary"),
        plt.Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor="black",
            markeredgecolor="black",
            label="Station",
        ),
    ]
    fig.legend(handles=leg_handles, loc="lower center", ncol=5, fontsize=FS_LEGEND, frameon=True)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "county": county,
        "state": state,
        "npz": str(npz_path),
        "full_npz": str(full_npz_path),
        "H": h,
        "W": w,
        "n_triple_zero_cells": int(layers.triple_zero.sum()),
        "n_cnn_pad_cells": int(layers.cnn_padding.sum()),
        "n_station_cells": len(station_xy),
        "output": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="县网格特征图 + 建站格 + 图层")
    parser.add_argument("--county", type=str, default="Los_Angeles")
    parser.add_argument("--grid-npz-dir", type=str, default=str(DEFAULT_GRID_DIR))
    parser.add_argument("--usa-map", type=str, default=str(DEFAULT_USA_MAP))
    parser.add_argument("--highway-geojson", type=str, default=str(DEFAULT_HIGHWAY))
    parser.add_argument("--use-full", action="store_true", help="显示全县未裁剪 npz")
    parser.add_argument("--grid-origin-json", type=str, default="")
    parser.add_argument("--cnn-pad-radius", type=int, default=1, help="CNN 3×3 等效 padding 圈宽度（格）")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--dpi", type=int, default=175)
    args = parser.parse_args()

    grid_dir = Path(args.grid_npz_dir)
    county = args.county.replace(" ", "_")
    full_npz, _ = county_npz_paths(grid_dir, county)

    if args.use_full:
        npz_path = full_npz
    else:
        npz_path = prepare_county_npz(grid_dir, county, log_prefix="viz")

    origin = None
    if args.grid_origin_json:
        origin = json.loads(Path(args.grid_origin_json).read_text(encoding="utf-8"))

    suffix = "grid_layers_stations"
    out = Path(args.output) if args.output else DEFAULT_OUT_DIR / f"{county}_{suffix}.png"

    meta = plot_expert_steps_map(
        Path(npz_path),
        out_path=out,
        usa_map_path=Path(args.usa_map),
        highway_path=Path(args.highway_geojson),
        full_npz_path=full_npz,
        dpi=int(args.dpi),
        grid_origin=origin,
        cnn_pad_radius=int(args.cnn_pad_radius),
    )
    print("Saved:", meta["output"])
    print(f"  triple-zero cells in view: {meta['n_triple_zero_cells']}")
    print(f"  CNN pad ring cells: {meta['n_cnn_pad_cells']}")
    print(f"  deployed station cells: {meta['n_station_cells']}")


if __name__ == "__main__":
    main()
