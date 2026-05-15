# mario.scenes.analysis

UMAP of scene annotations + two static figures illustrating *scene-space generalization* and *hippocampal ripples*.

## Quick start

```bash
# from repo root, with env/ active
pip install -r code/mario.scenes.analysis/requirements.txt

# 1. Compute UMAP + distance matrices (cached in data/)
python code/mario.scenes.analysis/compute_umap.py

# 2. Aggregate per-scene performance from BIDS events.tsv
python code/mario.scenes.analysis/aggregate_performance.py

# 3. Render the two figures
python code/mario.scenes.analysis/make_figures.py
```

All knobs live in `config.yaml`.

## Outputs

### Cached data (`data/`, gitignored)

| File | What |
|---|---|
| `annotations_27d.csv` | 313 × 28 binary feature matrix indexed by `scene_id` |
| `distance_jaccard.npy` + `scene_index.csv` | Pairwise Jaccard distance (modeling) |
| `umap_2d.csv` | UMAP 2D coordinates (visualization) |
| `distance_umap.npy` | Euclidean distance in UMAP space |
| `per_subject_perf.csv` | Long-form per-(subject, session, scene) mastery |

### Figures (`outputs/`, gitignored)

| File | What |
|---|---|
| `fig1_practiced_scenes.png` | UMAP scatter, colored by which session(s) of `subject` had the scene as practiced |
| `fig2_ripple_models.png` | 3-panel scatter showing predicted Δy under H1 / H2 / H3 |

## Figure 1 — Practiced scenes in scene-space

UMAP of all 313 scenes; each point colored by whether `subject` (default `sub-01`)
practiced it in `pre_session` (default `ses-001`), `post_session` (default
`ses-002`), both, or neither. Configure via `config.yaml > data.performance`.

## Figure 2 — Three models of how ripples reshape generalization

Three panels, each showing predicted Δy across the full scene-space under one
hypothesis about how hippocampal ripples (`R_i`) modulate post-sleep performance:

- **H1 — global gain.** `Δy = α + η R_i + ε`. Ripples lift performance uniformly across the whole space, regardless of distance from practiced scenes.
- **H2 — structured gain.** `Δy = (γ + δ R_i) · Σ_{s'∈A} exp(−d(s,s′)/λ) + ε`. Ripples scale a generalization kernel summed over the practiced source set `A` (sub-01's `pre_session` scenes); only scenes within the kernel radius are affected.
- **H3 — kernel widening.** `Δy = γ · Σ_{s'∈A} exp(−d(s,s′)/(λ₀ + λ₁ R_i)) + ε`. Ripples *widen* the kernel radius — the same sources reach further in scene-space.

H2 and H3 share a color scale so that the "widening" reads as "more dots brightened beyond the H2 peaks." Cyan rings mark the source scenes (set `A`) in all three panels.

The kernel uses **UMAP-space Euclidean distance** by default (`config.kernel.distance: umap`) so radii read as circles in the visualization. Switch to `jaccard` for a faithful-to-modeling but visually less intuitive rendering.

## Configuration

Everything tunable lives in `config.yaml`:
- `data.performance.{subject, pre_session, post_session}` — which subject/sessions to highlight in figure 1 and which session defines the source set for figure 2.
- `umap.{n_neighbors, min_dist, metric, random_state}` — UMAP hyperparameters.
- `kernel.{distance, lambda_h2, lambda_h3_low, lambda_h3_high}` — distance choice and kernel widths for H2 and H3 (H3 ranges from λ_low to λ_high as `R_i` goes 0→1).
- `simulation.{seed, ripple_demo, hypotheses}` — RNG seed, demo R_i, and per-hypothesis parameters.
- `figure.{dpi, figsize_*, palette}` — output resolution, figure sizes, colors.

## Reuses from the parent repo

- `code/utils.py:ensure_scenes_data()` — Zenodo download
- `code/utils.py:load_scenes_info()` — canonical scene loader

## Conceptual model (recap)

Master equation:

```
Δy_{i,s} = (γ + δ R_i) · Σ_{s' ∈ A_i} ℓ_{i,s'} · exp(−d(s,s') / λ) + ε_{i,s}
```

`A_i` is computed from real `events.tsv` (scenes visited in `pre_session` for
`subject`); ℓ_{i,s′} defaults to completion rate (`mastery` column in
`per_subject_perf.csv`).
