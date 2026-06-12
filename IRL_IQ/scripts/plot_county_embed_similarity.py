"""从训练 checkpoint 提取 county embedding，绘制余弦相似度热力图与层次聚类树状图。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

_DEFAULT_CKPT = _PKG_ROOT.parent / "outputs" / "iq_output" / "joint_8counties" / "IRL_IQ_20260610_seed0.pt"
_DEFAULT_OUT = _PKG_ROOT.parent / "outputs" / "iq_output" / "joint_8counties" / "embedding_analysis"


def _configure_plot_font() -> None:
    from matplotlib import font_manager

    candidates: list[Path] = []
    if sys.platform == "win32":
        win_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        candidates.extend(
            [
                win_fonts / "msyh.ttc",
                win_fonts / "msyhbd.ttc",
                win_fonts / "simhei.ttf",
            ]
        )
    elif sys.platform == "darwin":
        candidates.extend(
            [
                Path("/System/Library/Fonts/PingFang.ttc"),
                Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            ]
        )

    for path in candidates:
        if not path.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def load_county_embeddings(ckpt_path: Path) -> tuple[list[str], np.ndarray]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    names = list(blob["county_names"])
    q_sd = blob["q_net"]
    key = "net.county_embed.weight"
    if key not in q_sd:
        raise KeyError(f"checkpoint 缺少 {key!r}，可用键: {sorted(q_sd)[:8]} ...")
    weights = q_sd[key].numpy().astype(np.float64)
    return names, weights


def cosine_similarity_matrix(weights: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    normed = weights / np.clip(norms, 1e-12, None)
    return normed @ normed.T


def plot_heatmap(names: list[str], cos: np.ndarray, out_path: Path) -> None:
    n = len(names)
    # 仅保留下三角（不含对角线），避免 1.0 拉高色标、重复对称信息
    tril_idx = np.tril_indices(n, k=-1)
    tril_vals = cos[tril_idx]
    vlim = float(max(np.abs(tril_vals).max(), 0.05))

    lower_tri = np.tril(np.ones_like(cos, dtype=bool), k=-1)
    masked = np.ma.masked_where(~lower_tri, cos)

    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(
        masked,
        vmin=-vlim,
        vmax=vlim,
        cmap="RdBu_r",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_facecolor("white")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(names, fontsize=14)
    ax.tick_params(axis="both", length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, j in zip(*tril_idx):
        val = cos[i, j]
        color = "white" if abs(val) > 0.55 * vlim else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=13, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_visible(False)
    ax.set_title("County Embedding Cosine Similarity (lower triangle)", fontsize=16, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dendrogram(names: list[str], cos: np.ndarray, out_path: Path) -> None:
    dist = 1.0 - cos
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    fig, ax = plt.subplots(figsize=(9, 9))
    dendrogram(
        Z,
        labels=names,
        leaf_rotation=45,
        leaf_font_size=14,
        color_threshold=0.0,
        above_threshold_color="#4C78A8",
        ax=ax,
    )
    ax.set_ylabel("Cosine distance (1 − cos)", fontsize=14)
    ax.set_title("County Embedding Hierarchical Clustering (average linkage)", fontsize=16, pad=12)
    ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 county embedding 余弦相似度热力图与 dendrogram")
    parser.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CKPT))
    parser.add_argument("--output-dir", type=str, default=str(_DEFAULT_OUT))
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    _configure_plot_font()
    names, weights = load_county_embeddings(ckpt_path)
    cos = cosine_similarity_matrix(weights)

    heatmap_path = out_dir / "county_embed_cosine_heatmap.png"
    dendro_path = out_dir / "county_embed_dendrogram.png"
    plot_heatmap(names, cos, heatmap_path)
    plot_dendrogram(names, cos, dendro_path)

    np.save(out_dir / "county_embed_cosine.npy", cos)
    with (out_dir / "county_embed_cosine.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_path.resolve()),
                "county_names": names,
                "cosine_similarity": cos.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[plot_county_embed] counties={len(names)} embed_dim={weights.shape[1]}")
    print(f"  heatmap  → {heatmap_path}")
    print(f"  dendrogram → {dendro_path}")


if __name__ == "__main__":
    main()
