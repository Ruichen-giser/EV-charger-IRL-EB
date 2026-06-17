"""State embedding Ward 聚类 + 美国州级填色地图（县 polygon 按州同色）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage

_PKG_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _PKG_ROOT.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

_DEFAULT_EMB = (
    _PROJECT_ROOT
    / "outputs"
    / "iq_output"
    / "joint_mdp_ge5_s1_residual"
    / "embedding_export"
    / "state_embeddings.csv"
)
_DEFAULT_MAP = _PROJECT_ROOT / "data" / "joint_usa_map.geojson"
_DEFAULT_OUT = _DEFAULT_EMB.parent / "state_cluster_analysis"

REGION_BOUNDS = {
    "conus": {"lon_min": -125.0, "lon_max": -66.5, "lat_min": 23.5, "lat_max": 50.5},
    "AK": {"lon_min": -170.0, "lon_max": -129.0, "lat_min": 50.5, "lat_max": 71.5},
    "HI": {"lon_min": -161.0, "lon_max": -154.0, "lat_min": 18.5, "lat_max": 22.8},
}

# Tableau-style distinct colors for 6 clusters
CLUSTER_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
]

GEO_STATE_ALIASES = {
    "District of Columbia": "District of Columbia",
}


def _configure_plot_font() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans", "Arial"]


def load_state_embeddings(csv_path: Path) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    embed_cols = [c for c in df.columns if c.startswith("e")]
    if not embed_cols:
        raise ValueError(f"未找到 embedding 列: {csv_path}")
    names = df["state_name"].astype(str).tolist()
    X = df[embed_cols].to_numpy(dtype=np.float64)
    return names, X, df


def ward_cluster_labels(X: np.ndarray, n_clusters: int) -> np.ndarray:
    z = linkage(X, method="ward")
    labels = fcluster(z, t=int(n_clusters), criterion="maxclust")
    return labels.astype(int), z


def iter_rings(geometry: dict):
    gtype = geometry["type"]
    if gtype == "Polygon":
        for ring in geometry["coordinates"]:
            yield ring
    elif gtype == "MultiPolygon":
        for poly in geometry["coordinates"]:
            for ring in poly:
                yield ring


def geometry_patches(geometry: dict) -> list[MplPolygon]:
    patches: list[MplPolygon] = []
    for ring in iter_rings(geometry):
        if len(ring) < 3:
            continue
        xy = [(float(x), float(y)) for x, y in ring]
        patches.append(MplPolygon(xy, closed=True))
    return patches


def load_county_features(geojson_path: Path) -> list[dict]:
    with geojson_path.open(encoding="utf-8") as f:
        blob = json.load(f)
    rows = []
    for feat in blob["features"]:
        props = feat["properties"]
        state = str(props.get("NAME_1", "")).strip()
        county = str(props.get("NAME_2", "")).strip()
        rows.append(
            {
                "state": state,
                "county": county,
                "geometry": feat["geometry"],
            }
        )
    return rows


def _set_extent(ax, bounds: dict) -> None:
    ax.set_xlim(bounds["lon_min"], bounds["lon_max"])
    ax.set_ylim(bounds["lat_min"], bounds["lat_max"])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _draw_region(
    ax,
    counties: list[dict],
    state_to_cluster: dict[str, int],
    *,
    edgecolor: str = "white",
    linewidth: float = 0.15,
) -> None:
    patches: list[MplPolygon] = []
    colors: list[str] = []
    for row in counties:
        state = row["state"]
        cluster = state_to_cluster.get(state)
        if cluster is None:
            face = "#DDDDDD"
        else:
            face = CLUSTER_COLORS[(cluster - 1) % len(CLUSTER_COLORS)]
        for patch in geometry_patches(row["geometry"]):
            patches.append(patch)
            colors.append(face)
    if patches:
        coll = PatchCollection(
            patches,
            facecolor=colors,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=0.95,
        )
        ax.add_collection(coll)


def plot_cluster_map(
    counties: list[dict],
    state_to_cluster: dict[str, int],
    *,
    n_clusters: int,
    out_path: Path,
) -> None:
    conus = [c for c in counties if c["state"] not in {"Alaska", "Hawaii"}]
    ak = [c for c in counties if c["state"] == "Alaska"]
    hi = [c for c in counties if c["state"] == "Hawaii"]

    fig = plt.figure(figsize=(12, 7.5), facecolor="white")
    ax = fig.add_axes([0.04, 0.18, 0.92, 0.78])
    _draw_region(ax, conus, state_to_cluster)
    _set_extent(ax, REGION_BOUNDS["conus"])
    ax.set_title(
        f"State Embedding Ward Clusters (k={n_clusters})",
        fontsize=16,
        pad=10,
    )

    ax_ak = fig.add_axes([0.08, 0.04, 0.22, 0.16])
    _draw_region(ax_ak, ak, state_to_cluster, linewidth=0.1)
    _set_extent(ax_ak, REGION_BOUNDS["AK"])
    ax_ak.set_title("AK", fontsize=10)

    ax_hi = fig.add_axes([0.34, 0.04, 0.12, 0.16])
    _draw_region(ax_hi, hi, state_to_cluster, linewidth=0.1)
    _set_extent(ax_hi, REGION_BOUNDS["HI"])
    ax_hi.set_title("HI", fontsize=10)

    handles = []
    for i in range(1, n_clusters + 1):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=CLUSTER_COLORS[(i - 1) % len(CLUSTER_COLORS)],
                markersize=12,
                label=f"Cluster {i}",
            )
        )
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.08),
        ncol=min(n_clusters, 6),
        frameon=False,
        fontsize=11,
    )

    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    normed = X / np.clip(norms, 1e-12, None)
    return normed @ normed.T


def order_indices_by_ward_clusters(
    X: np.ndarray,
    labels: np.ndarray,
) -> list[int]:
    """按 cluster 分组；组内再用 Ward 叶序排列，使同类 state 在热力图上相邻。"""
    ordered: list[int] = []
    for cl in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == cl)
        if idx.size <= 1:
            ordered.extend(idx.tolist())
            continue
        sub_z = linkage(X[idx], method="ward")
        sub_order = leaves_list(sub_z)
        ordered.extend(idx[sub_order].tolist())
    return ordered


def plot_cosine_heatmap_ordered(
    state_names: list[str],
    X: np.ndarray,
    labels: np.ndarray,
    cos: np.ndarray,
    *,
    n_clusters: int,
    out_path: Path,
) -> np.ndarray:
    order = order_indices_by_ward_clusters(X, labels)
    ordered_names = [state_names[i] for i in order]
    ordered_labels = labels[order]
    cos_ord = cos[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(cos_ord, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="equal", interpolation="nearest")
    n = len(ordered_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ordered_names, rotation=90, ha="center", fontsize=8)
    ax.set_yticklabels(ordered_names, fontsize=8)
    ax.tick_params(axis="both", length=0, pad=2)

    for i in range(1, n):
        if ordered_labels[i] != ordered_labels[i - 1]:
            b = i - 0.5
            ax.axhline(b, color="black", linewidth=1.2, alpha=0.85)
            ax.axvline(b, color="black", linewidth=1.2, alpha=0.85)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Cosine similarity", fontsize=12)
    ax.set_title(
        f"State Embedding Cosine Similarity (ordered by Ward k={n_clusters})",
        fontsize=15,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return cos_ord


def plot_dendrogram(state_names: list[str], z: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    dendrogram(
        z,
        labels=state_names,
        leaf_rotation=90,
        leaf_font_size=9,
        color_threshold=0.0,
        above_threshold_color="#4E79A7",
        ax=ax,
    )
    ax.set_title("State Embedding Ward Hierarchical Clustering", fontsize=15, pad=12)
    ax.set_ylabel("Ward linkage distance", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="State embedding Ward 聚类与地图可视化")
    parser.add_argument("--embeddings", type=str, default=str(_DEFAULT_EMB))
    parser.add_argument("--geojson", type=str, default=str(_DEFAULT_MAP))
    parser.add_argument("--output-dir", type=str, default=str(_DEFAULT_OUT))
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=6)
    args = parser.parse_args()

    emb_path = Path(args.embeddings)
    geo_path = Path(args.geojson)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not emb_path.is_file():
        raise FileNotFoundError(f"state embeddings 不存在: {emb_path}")
    if not geo_path.is_file():
        raise FileNotFoundError(f"geojson 不存在: {geo_path}")

    _configure_plot_font()
    state_names, X, df = load_state_embeddings(emb_path)
    counties = load_county_features(geo_path)

    # Ward linkage once; cut at k=4..6
    z = linkage(X, method="ward")
    cos = cosine_similarity_matrix(X)
    np.save(out_dir / "state_embed_cosine.npy", cos)
    dendro_path = out_dir / "state_embed_ward_dendrogram.png"
    plot_dendrogram(state_names, z, dendro_path)

    summary_rows = []
    geo_states = {c["state"] for c in counties}
    emb_states = set(state_names)

    cluster_tables: dict[int, pd.DataFrame] = {}
    for k in range(int(args.k_min), int(args.k_max) + 1):
        labels = fcluster(z, t=k, criterion="maxclust").astype(int)
        assign_df = df.copy()
        assign_df["cluster"] = labels
        cluster_tables[k] = assign_df

        state_to_cluster = {
            str(row["state_name"]): int(row["cluster"]) for _, row in assign_df.iterrows()
        }
        map_path = out_dir / f"state_embed_ward_map_k{k}.png"
        plot_cluster_map(counties, state_to_cluster, n_clusters=k, out_path=map_path)

        heatmap_path = out_dir / f"state_embed_cosine_heatmap_k{k}.png"
        cos_ord = plot_cosine_heatmap_ordered(
            state_names, X, labels, cos, n_clusters=k, out_path=heatmap_path
        )
        np.save(out_dir / f"state_embed_cosine_ordered_k{k}.npy", cos_ord)
        order = order_indices_by_ward_clusters(X, labels)
        with (out_dir / f"state_embed_cosine_order_k{k}.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "k": k,
                    "ordered_state_names": [state_names[i] for i in order],
                    "ordered_clusters": labels[order].tolist(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        for cl in sorted(assign_df["cluster"].unique()):
            members = assign_df.loc[assign_df["cluster"] == cl, "state_name"].tolist()
            summary_rows.append({"k": k, "cluster": int(cl), "n_states": len(members), "states": "; ".join(members)})
            print(f"[k={k}] cluster {cl} ({len(members)}): {', '.join(members)}")

        assign_df.to_csv(out_dir / f"state_embed_ward_k{k}.csv", index=False)
        print(f"  map → {map_path}")
        print(f"  heatmap → {heatmap_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "state_embed_ward_summary.csv", index=False)

    with (out_dir / "state_embed_cosine.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "state_names": state_names,
                "cosine_similarity": cos.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    payload = {
        "embeddings_csv": str(emb_path.resolve()),
        "geojson": str(geo_path.resolve()),
        "method": "ward",
        "k_range": [int(args.k_min), int(args.k_max)],
        "n_states": len(state_names),
        "embed_dim": int(X.shape[1]),
        "states_missing_in_geojson": sorted(emb_states - geo_states),
        "states_missing_in_embeddings": sorted(geo_states - emb_states - {"District of Columbia"}),
        "clusters": {
            str(k): [
                {
                    "cluster": int(cl),
                    "states": assign_df.loc[assign_df["cluster"] == cl, "state_name"].tolist(),
                }
                for cl in sorted(assign_df["cluster"].unique())
            ]
            for k, assign_df in cluster_tables.items()
        },
    }
    with (out_dir / "state_embed_ward_clusters.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[state_embed_ward] states={len(state_names)} embed_dim={X.shape[1]}")
    print(f"  dendrogram → {dendro_path}")
    print(f"  output dir → {out_dir}")


if __name__ == "__main__":
    main()
