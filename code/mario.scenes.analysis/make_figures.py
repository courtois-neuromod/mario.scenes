"""Generate two static figures for the mario.scenes.analysis story.

Figure 1 — UMAP of all 313 scenes, colored by which session(s) of sub-01 they
were practiced in (ses-001 only / ses-002 only / both / neither).

Figure 2 — Three panels showing predicted Δy across the full scene-space under
three hypotheses about how hippocampal ripples affect post-sleep performance:
  H1: ripples improve performance uniformly,
  H2: ripples improve performance within a fixed kernel radius around practiced sources,
  H3: ripples *widen* the kernel radius.

Run:
    python code/mario.scenes.analysis/make_figures.py [--out outputs/]
"""

from __future__ import annotations

import argparse
import logging
import os
import os.path as op
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = op.dirname(op.abspath(__file__))
sys.path.insert(0, HERE)
import data_io  # noqa: E402
import kernel as krn  # noqa: E402

log = logging.getLogger("make_figures")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def categorize_scenes(scene_ids: list[str], pre: set[str], post: set[str]) -> np.ndarray:
    """Return a length-N array with category labels: pre, post, both, neither."""
    out = np.empty(len(scene_ids), dtype=object)
    for i, sid in enumerate(scene_ids):
        in_pre, in_post = sid in pre, sid in post
        if in_pre and in_post:
            out[i] = "both"
        elif in_pre:
            out[i] = "pre"
        elif in_post:
            out[i] = "post"
        else:
            out[i] = "neither"
    return out


def kernel_field(source_idx: np.ndarray, distance_matrix: np.ndarray, lam: float) -> np.ndarray:
    """max_{s'∈A} exp(−d(s,s') / λ) for each scene s.

    We use **max over sources** (not sum) for the illustrative figure: a
    scene's "kernel reach" should be governed by its nearest practiced
    exemplar, so every source dot saturates at 1.0 from its own
    contribution (d=0), regardless of how many other sources cluster
    nearby. The actual modeling formula in the master equation is a
    sum-of-kernels — switch back here if you want fidelity to that.
    """
    if len(source_idx) == 0:
        return np.zeros(distance_matrix.shape[0])
    d_to_sources = distance_matrix[:, source_idx]
    return np.exp(-d_to_sources / lam).max(axis=1)


# ---------------------------------------------------------------------------
# Figure 1 — UMAP with practiced-set color coding
# ---------------------------------------------------------------------------

def figure_practiced_scenes(cfg: dict, umap_df: pd.DataFrame, perf: pd.DataFrame,
                            out_path: str) -> None:
    sub = cfg["data"]["performance"]["subject"]
    pre_session = cfg["data"]["performance"]["pre_session"]
    post_session = cfg["data"]["performance"]["post_session"]
    palette = cfg["figure"]["palette"]

    scene_ids = list(umap_df.index)
    pre = set(data_io.practiced_scenes(perf, sub, pre_session, scene_ids))
    post = set(data_io.practiced_scenes(perf, sub, post_session, scene_ids))
    cats = categorize_scenes(scene_ids, pre, post)

    log.info("Figure 1: %d in %s only, %d in %s only, %d in both, %d in neither",
             (cats == "pre").sum(), pre_session,
             (cats == "post").sum(), post_session,
             (cats == "both").sum(), (cats == "neither").sum())

    fig, ax = plt.subplots(figsize=cfg["figure"]["figsize_umap"], dpi=cfg["figure"]["dpi"])

    # Plot in z-order: neither (back) → pre/post → both (front, most informative)
    for cat, color, label, size, alpha in [
        ("neither", palette["neither"], "not practiced", 75, 0.55),
        ("pre",     palette["pre_only"], f"practiced in {pre_session} only", 200, 0.95),
        ("post",    palette["post_only"], f"practiced in {post_session} only", 200, 0.95),
        ("both",    palette["both"], "practiced in both", 260, 1.0),
    ]:
        mask = cats == cat
        if not mask.any():
            continue
        ax.scatter(
            umap_df.loc[mask, "DR_1"], umap_df.loc[mask, "DR_2"],
            c=color, s=size, alpha=alpha, edgecolors="black",
            linewidths=0.6, label=f"{label}  (n={mask.sum()})",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Mario scene-space (UMAP) — practiced scenes for {sub}",
                 fontsize=16)
    ax.legend(loc="best", frameon=True, fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="datalim")
    # Explicit margins so the single panel fills the 16:9 slide.
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.07)
    fig.savefig(out_path)
    plt.close(fig)
    log.info("→ %s", out_path)


# ---------------------------------------------------------------------------
# Figure 2 — Three-panel ripple effect models
# ---------------------------------------------------------------------------

def figure_ripple_models(cfg: dict, umap_df: pd.DataFrame, perf: pd.DataFrame,
                         d_jaccard: np.ndarray, d_umap: np.ndarray,
                         out_path: str) -> None:
    """Color = ripple sensitivity ∂Δy/∂R (computed as Δy(R=1) − Δy(R=0)).

    Reading the panels:
    - H1: every dot the same color — performance changes uniformly with R.
    - H2: high color *only* near the sources — only scenes inside the (fixed)
      kernel radius are R-sensitive.
    - H3: a *ring* of high color at intermediate distance from sources —
      these scenes come INTO the kernel as it widens with R; scenes very
      close are already maxed out, scenes very far stay out of reach.
    """
    sub = cfg["data"]["performance"]["subject"]
    pre_session = cfg["data"]["performance"]["pre_session"]
    palette = cfg["figure"]["palette"]

    scene_ids = list(umap_df.index)
    sources = data_io.practiced_scenes(perf, sub, pre_session, scene_ids)
    source_idx = np.array([scene_ids.index(s) for s in sources], dtype=int)
    log.info("Figure 2: %d source scenes (sub-01 %s)", len(sources), pre_session)

    D = d_umap if cfg["kernel"]["distance"] == "umap" else d_jaccard
    n = len(scene_ids)

    # Sensitivity ∂Δy/∂R, computed analytically from each model.
    # H1: Δy = α + η R + ε  →  ∂Δy/∂R = η   (constant across scenes)
    h1cfg = cfg["simulation"]["hypotheses"]["h1"]
    sens_h1 = np.full(n, float(h1cfg["eta"]))

    # H2: Δy = (γ + δ R) · Σ exp(-d/λ)  →  ∂Δy/∂R = δ · Σ exp(-d/λ)
    h2cfg = cfg["simulation"]["hypotheses"]["h2"]
    lam_h2 = float(cfg["kernel"]["lambda_h2"])
    field_h2 = kernel_field(source_idx, D, lam_h2)
    sens_h2 = h2cfg["delta"] * field_h2

    # H3: Δy = γ · Σ exp(-d/(λ₀ + λ₁ R)).  Linearise endpoints:
    #     ∂Δy/∂R ≈ γ · (Σ exp(-d/λ_high) − Σ exp(-d/λ_low))   over R ∈ [0, 1]
    h3cfg = cfg["simulation"]["hypotheses"]["h3"]
    lam_low = float(cfg["kernel"]["lambda_h3_low"])
    lam_high = float(cfg["kernel"]["lambda_h3_high"])
    field_low = kernel_field(source_idx, D, lam_low)
    field_high = kernel_field(source_idx, D, lam_high)
    sens_h3 = h3cfg["gamma"] * (field_high - field_low)

    # Shared colorbar across all three panels — the three sensitivities are
    # in the same units (∂Δy/∂R), so a common scale makes magnitudes
    # comparable: H1's uniform η, H2's source-saturated peaks, and H3's
    # weaker-but-wider donut all read on the same axis.
    panels = [
        ("H1: ripples lift everything",          sens_h1, "uniform η across all scenes"),
        ("H2: ripples within the kernel radius", sens_h2, f"δ · max exp(−d/λ),  λ = {lam_h2:.2f}"),
        ("H3: ripples widen the kernel radius",  sens_h3, f"γ · (max_{{λ={lam_high:.1f}}} − max_{{λ={lam_low:.1f}}})"),
    ]
    vmin = 0.0
    vmax = float(max(sens_h1.max(), sens_h2.max(), sens_h3.max()))

    fig, axes = plt.subplots(1, 3, figsize=cfg["figure"]["figsize_panels"],
                             dpi=cfg["figure"]["dpi"])
    cmap = plt.get_cmap(palette["delta_y_cmap"])
    xlim = (umap_df["DR_1"].min() - 0.5, umap_df["DR_1"].max() + 0.5)
    ylim = (umap_df["DR_2"].min() - 0.5, umap_df["DR_2"].max() + 0.5)

    sc = None
    for ax, (title, sens, sub_caption) in zip(axes, panels):
        sc = ax.scatter(
            umap_df["DR_1"], umap_df["DR_2"],
            c=sens, s=85, cmap=cmap, vmin=vmin, vmax=vmax,
            alpha=0.95, edgecolors="black", linewidths=0.3,
        )
        ax.scatter(
            umap_df.iloc[source_idx]["DR_1"], umap_df.iloc[source_idx]["DR_2"],
            facecolors="none", edgecolors=palette["source"], s=240,
            linewidths=1.8, label=f"sources ({len(sources)})",
        )
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel(f"UMAP 1   ({sub_caption})")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("UMAP 2")

    fig.suptitle(
        f"How much does each scene's Δy depend on the ripple density R_i?\n"
        f"(sources: {sub}'s {pre_session} practiced scenes, n={len(sources)})",
        fontsize=15, y=0.97,
    )
    # Margins for the 3 panels; explicit right=0.91 leaves room for the
    # single shared colorbar placed via add_axes.
    fig.subplots_adjust(left=0.04, right=0.91, top=0.86, bottom=0.10,
                        wspace=0.18)
    # One shared colorbar at a fixed location on the right, beyond the panels.
    cax = fig.add_axes([0.925, 0.12, 0.012, 0.70])  # [left, bottom, width, height]
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"$\partial\,\Delta y\,/\,\partial R_i$  (R-sensitivity)",
                 fontsize=12)
    fig.savefig(out_path)
    plt.close(fig)
    log.info("→ %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=op.join(HERE, "outputs"),
                   help="output directory (default: code/mario.scenes.analysis/outputs/)")
    args = p.parse_args()

    cfg = data_io.load_config()
    os.makedirs(args.out, exist_ok=True)

    # Load cached artifacts (require compute_umap.py + aggregate_performance.py first)
    for path_fn, name in [
        (data_io.umap_path, "umap_2d.csv"),
        (data_io.perf_path, "per_subject_perf.csv"),
    ]:
        if not op.exists(path_fn()):
            log.error("Missing %s — run compute_umap.py and aggregate_performance.py first.", name)
            return 1

    umap_df = data_io.load_umap()
    perf = data_io.load_performance()
    d_jaccard, _ = data_io.load_distance_matrix()
    d_umap = data_io.load_umap_distance()

    # Align order: use UMAP index as canonical
    d_jaccard = d_jaccard      # same scene_id ordering, by construction
    d_umap = d_umap

    figure_practiced_scenes(
        cfg, umap_df, perf,
        out_path=op.join(args.out, "fig1_practiced_scenes.png"),
    )
    figure_ripple_models(
        cfg, umap_df, perf, d_jaccard, d_umap,
        out_path=op.join(args.out, "fig2_ripple_models.png"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
