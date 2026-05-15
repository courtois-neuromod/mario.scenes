"""Compute the 27-feature annotation matrix, Jaccard distance, UMAP embedding,
and UMAP-space distance. All artifacts cached under ``data/``.

Run:
    python code/mario.scenes.analysis/compute_umap.py [--force]
"""

from __future__ import annotations

import argparse
import logging
import os.path as op
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import umap.umap_ as umap

HERE = op.dirname(op.abspath(__file__))
sys.path.insert(0, HERE)
import data_io  # noqa: E402

log = logging.getLogger("compute_umap")


def jaccard_distance_matrix(annotations: pd.DataFrame) -> np.ndarray:
    """Pairwise Jaccard distance with explicit handling of all-zero rows.

    scipy returns NaN when comparing two all-zero binary vectors. We override:
    - all-zero vs all-zero → 0.0
    - all-zero vs non-empty → 1.0 (maximally far)
    """
    X = annotations.values.astype(bool)
    d = squareform(pdist(X, metric="jaccard"))
    # Fix all-zero rows
    is_empty = ~X.any(axis=1)
    if is_empty.any():
        d[is_empty, :] = 1.0
        d[:, is_empty] = 1.0
        # all-zero vs all-zero → 0
        ee = np.outer(is_empty, is_empty)
        d[ee] = 0.0
    np.fill_diagonal(d, 0.0)
    return d


def compute_umap_2d(
    annotations: pd.DataFrame,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> pd.DataFrame:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    coords = reducer.fit_transform(annotations.values.astype(float))
    return pd.DataFrame(coords, columns=["DR_1", "DR_2"], index=annotations.index)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="recompute even if cache exists")
    args = p.parse_args()

    cfg = data_io.load_config()
    data_io.ensure_data_dir()

    # 1) annotations
    if args.force or not op.exists(data_io.annotations_path()):
        log.info("Loading annotations from scenes_mastersheet.csv ...")
        annotations = data_io.load_annotation_data(worlds=cfg["data"]["worlds"])
        data_io.save_annotations(annotations)
    else:
        annotations = data_io.load_annotations()
    log.info("annotations: %d scenes × %d features", *annotations.shape)

    # 2) Jaccard distance (modeling)
    npy, _ = data_io.jaccard_path()
    if args.force or not op.exists(npy):
        log.info("Computing Jaccard distance matrix ...")
        d = jaccard_distance_matrix(annotations)
        data_io.save_distance_matrix(d, list(annotations.index))
    else:
        d, _ = data_io.load_distance_matrix()
    log.info("jaccard distance: %s, mean=%.3f, max=%.3f", d.shape, d.mean(), d.max())

    # 3) UMAP 2D (visualization)
    if args.force or not op.exists(data_io.umap_path()):
        log.info("Computing UMAP 2D embedding ...")
        u = cfg["umap"]
        umap_df = compute_umap_2d(
            annotations,
            n_neighbors=u["n_neighbors"],
            min_dist=u["min_dist"],
            metric=u["metric"],
            random_state=u["random_state"],
        )
        data_io.save_umap(umap_df)
    else:
        umap_df = data_io.load_umap()
    log.info("umap: %s, DR_1=[%.2f, %.2f] DR_2=[%.2f, %.2f]",
             umap_df.shape,
             umap_df["DR_1"].min(), umap_df["DR_1"].max(),
             umap_df["DR_2"].min(), umap_df["DR_2"].max())

    # 4) UMAP-space distance (alt kernel input)
    if args.force or not op.exists(data_io.umap_distance_path()):
        log.info("Computing UMAP-space Euclidean distance ...")
        coords = umap_df.loc[annotations.index].values
        d_umap = squareform(pdist(coords, metric="euclidean"))
        data_io.save_umap_distance(d_umap)
    else:
        d_umap = data_io.load_umap_distance()
    log.info("umap distance: %s, mean=%.3f, max=%.3f", d_umap.shape, d_umap.mean(), d_umap.max())

    log.info("Done. Cached artifacts under %s", data_io.DATA_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
