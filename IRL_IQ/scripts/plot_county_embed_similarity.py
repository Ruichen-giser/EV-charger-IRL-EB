"""从训练 checkpoint 提取 CountyLocationEmbed，绘制余弦相似度热力图与层次聚类树状图。"""
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

from county_meta import compute_county_meta, state_name_to_id  # noqa: E402
from models.county_location_embed import CountyLocationEmbed  # noqa: E402

_DEFAULT_CKPT = _PKG_ROOT.parent / "outputs" / "iq_output" / "joint_mdp_ge5_baseline" / "iq_learn_shared.pt"
_DEFAULT_OUT = _PKG_ROOT.parent / "outputs" / "iq_output" / "joint_mdp_ge5_baseline" / "embedding_analysis"


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


def _parse_location_label(label: str) -> tuple[str, str]:
    for sep in ("/", "|", ":"):
        if sep in label:
            state, county = label.split(sep, 1)
            return state.strip(), county.strip()
    raise ValueError(f"无法解析 location label: {label!r}")


def load_county_embeddings(ckpt_path: Path) -> tuple[list[str], np.ndarray]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    labels = list(blob.get("location_labels") or blob.get("county_names") or [])
    if not labels:
        raise ValueError(f"checkpoint 缺少 location_labels / county_names: {ckpt_path}")

    q_sd = blob["q_net"]
    prefix = "net.location_embed."
    loc_sd = {
        k[len(prefix) :]: v
        for k, v in q_sd.items()
        if k.startswith(prefix)
    }
    if not loc_sd:
        raise KeyError(
            "checkpoint 缺少 CountyLocationEmbed 权重；"
            f"可用键示例: {sorted(q_sd)[:12]}"
        )

    embed = CountyLocationEmbed(
        n_states=int(blob.get("n_states", 51)),
        embed_dim=int(blob.get("embed_dim", 16)),
        meta_dim=int(blob.get("meta_dim", 3)),
        meta_hidden=32,
        n_residual=int(blob.get("n_residual", 1149)),
        residual_alpha=float(blob.get("residual_alpha", 0.0)),
    )
    embed.load_state_dict(loc_sd, strict=True)
    embed.eval()

    names: list[str] = []
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for i, label in enumerate(labels):
            if "/" in label or "|" in label or ":" in label:
                state, county = _parse_location_label(label)
                display = label
            else:
                state, county = "California", str(label)
                display = str(label)
            meta = torch.as_tensor(
                compute_county_meta(state, county),
                dtype=torch.float32,
            ).unsqueeze(0)
            state_t = torch.tensor([state_name_to_id(state)], dtype=torch.long)
            county_t = torch.tensor([i], dtype=torch.long)
            vec = embed(state_t, meta, county_t).squeeze(0).numpy()
            names.append(display)
            vectors.append(vec.astype(np.float64))

    return names, np.stack(vectors, axis=0)


def cosine_similarity_matrix(weights: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    normed = weights / np.clip(norms, 1e-12, None)
    return normed @ normed.T


def plot_heatmap(names: list[str], cos: np.ndarray, out_path: Path) -> None:
    n = len(names)
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
    ax.set_title("County Location Embedding Cosine Similarity (lower triangle)", fontsize=16, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dendrogram(names: list[str], cos: np.ndarray, out_path: Path) -> None:
    dist = 1.0 - cos
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")

    fig, ax = plt.subplots(figsize=(9, 9))
    dendrogram(
        z,
        labels=names,
        leaf_rotation=45,
        leaf_font_size=14,
        color_threshold=0.0,
        above_threshold_color="#4C78A8",
        ax=ax,
    )
    ax.set_ylabel("Cosine distance (1 − cos)", fontsize=14)
    ax.set_title("County Location Embedding Hierarchical Clustering (average linkage)", fontsize=16, pad=12)
    ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 CountyLocationEmbed 余弦相似度热力图与 dendrogram")
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
