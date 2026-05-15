"""I/O helpers for the mario.scenes.analysis pipeline.

Reads/writes under ``code/mario.scenes.analysis/data/`` (cached artifacts)
and the parent repo (``sourcedata/`` and per-subject ``func/`` event TSVs).
"""

from __future__ import annotations

import os
import os.path as op
import sys

import numpy as np
import pandas as pd
import yaml

HERE = op.dirname(op.abspath(__file__))
REPO_CODE = op.dirname(HERE)
REPO_ROOT = op.dirname(REPO_CODE)
DATA_DIR = op.join(HERE, "data")
CONFIG_PATH = op.join(HERE, "config.yaml")

sys.path.insert(0, REPO_CODE)
from utils import ensure_scenes_data, load_scenes_info  # noqa: E402

FEATURE_COLS = [
    "Enemy", "2-Horde", "3-Horde", "4-Horde", "Roof", "Gap",
    "Multiple gaps", "Variable gaps", "Gap enemy", "Pillar gap", "Valley",
    "Pipe valley", "Empty valley", "Enemy valley", "Roof valley", "2-Path",
    "3-Path", "Risk/Reward", "Stair up", "Stair down", "Empty stair valley",
    "Enemy stair valley", "Gap stair valley", "Reward", "Moving platform",
    "Flagpole", "Beginning", "Bonus zone",
]


def load_config(path: str | None = None) -> dict:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_annotation_data(worlds: str | list = "all") -> pd.DataFrame:
    """Return a (n_scenes × 28) binary feature matrix indexed by ``scene_id``."""
    ensure_scenes_data()
    df = load_scenes_info(format="df").dropna(subset=["World", "Level", "Scene"])
    df["scene_id"] = df.apply(
        lambda r: f"w{int(r['World'])}l{int(r['Level'])}s{int(r['Scene'])}",
        axis=1,
    )
    if worlds != "all":
        df = df[df["World"].astype(int).isin(list(worlds))]
    annotations = df[FEATURE_COLS].astype(int).copy()
    annotations.index = df["scene_id"].values
    annotations.index.name = "scene_id"
    return annotations


# ---------------------------------------------------------------------------
# Cached-artifact paths
# ---------------------------------------------------------------------------

def annotations_path() -> str: return op.join(DATA_DIR, "annotations_27d.csv")
def umap_path() -> str: return op.join(DATA_DIR, "umap_2d.csv")
def jaccard_path() -> tuple[str, str]:
    return op.join(DATA_DIR, "distance_jaccard.npy"), op.join(DATA_DIR, "scene_index.csv")
def umap_distance_path() -> str: return op.join(DATA_DIR, "distance_umap.npy")
def perf_path() -> str: return op.join(DATA_DIR, "per_subject_perf.csv")


def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def save_distance_matrix(matrix: np.ndarray, scene_ids: list[str]) -> None:
    ensure_data_dir()
    npy, idx = jaccard_path()
    np.save(npy, matrix)
    pd.Series(scene_ids, name="scene_id").to_csv(idx, index=False)


def load_distance_matrix() -> tuple[np.ndarray, list[str]]:
    npy, idx = jaccard_path()
    return np.load(npy), pd.read_csv(idx)["scene_id"].tolist()


def save_umap_distance(matrix: np.ndarray) -> None:
    ensure_data_dir()
    np.save(umap_distance_path(), matrix)


def load_umap_distance() -> np.ndarray:
    return np.load(umap_distance_path())


def save_umap(coords: pd.DataFrame) -> None:
    ensure_data_dir()
    coords.to_csv(umap_path())


def load_umap() -> pd.DataFrame:
    return pd.read_csv(umap_path(), index_col=0)


def save_annotations(annotations: pd.DataFrame) -> None:
    ensure_data_dir()
    annotations.to_csv(annotations_path())


def load_annotations() -> pd.DataFrame:
    return pd.read_csv(annotations_path(), index_col=0)


def load_performance() -> pd.DataFrame:
    return pd.read_csv(perf_path())


def practiced_scenes(perf: pd.DataFrame, subject: str, session: str,
                     valid_scenes: list[str]) -> list[str]:
    """Return the scene_ids visited by ``subject`` in ``session`` (n_visits >= 1)."""
    sl = perf[(perf["subject"] == subject) & (perf["session"] == session)]
    sl = sl[sl["n_visits"].fillna(0) > 0]
    return [s for s in sl["scene_id"].tolist() if s in valid_scenes]
