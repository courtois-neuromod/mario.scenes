"""Walk the BIDS event TSVs and aggregate per-(subject, session, scene_id)
performance into a long-form CSV used as the source for A_i and ℓ.

Run:
    python code/mario.scenes.analysis/aggregate_performance.py
"""

from __future__ import annotations

import glob
import logging
import os.path as op
import sys

import pandas as pd

HERE = op.dirname(op.abspath(__file__))
sys.path.insert(0, HERE)
import data_io  # noqa: E402

log = logging.getLogger("aggregate_performance")

REPO_ROOT = data_io.REPO_ROOT


def discover_events_files() -> list[str]:
    pattern = op.join(REPO_ROOT, "sub-*", "ses-*", "func", "*_desc-scenes_events.tsv")
    return sorted(glob.glob(pattern))


def parse_subject_session(path: str) -> tuple[str, str]:
    parts = op.normpath(path).split(op.sep)
    sub = next(p for p in parts if p.startswith("sub-"))
    ses = next(p for p in parts if p.startswith("ses-"))
    return sub, ses


def aggregate_one(path: str) -> pd.DataFrame:
    sub, ses = parse_subject_session(path)
    df = pd.read_csv(path, sep="\t")
    df = df[df["trial_type"] == "scene"].copy()
    if df.empty:
        return pd.DataFrame()

    # Treat NaN gameplay stats as 0 for aggregation
    stat_cols = ["ScoreGained", "CoinsGained", "Lives_lost", "Hits_taken",
                 "Enemies_killed", "Powerups_collected", "Bricks_smashed", "X_Traveled"]
    for c in stat_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["completed"] = (df["outcome"] == "completed").astype(int)
    df["died"] = (df["outcome"] == "death").astype(int)

    grouped = df.groupby("scene_id", as_index=False).agg(
        n_visits=("scene_id", "size"),
        n_completed=("completed", "sum"),
        n_deaths=("died", "sum"),
        ScoreGained=("ScoreGained", "mean"),
        Enemies_killed=("Enemies_killed", "mean"),
        X_Traveled=("X_Traveled", "mean"),
    )
    grouped["completion_rate"] = grouped["n_completed"] / grouped["n_visits"]
    grouped["mastery"] = grouped["completion_rate"]    # default; override here for composite
    grouped.insert(0, "session", ses)
    grouped.insert(0, "subject", sub)
    return grouped


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    files = discover_events_files()
    log.info("Found %d events TSVs", len(files))
    if not files:
        log.error("No events files found under %s", REPO_ROOT)
        return 1

    frames = []
    for f in files:
        try:
            row = aggregate_one(f)
            if not row.empty:
                frames.append(row)
        except Exception as e:
            log.warning("skipping %s: %s", f, e)

    perf = pd.concat(frames, ignore_index=True)
    # Aggregate again across multiple runs in same session
    perf = perf.groupby(["subject", "session", "scene_id"], as_index=False).agg(
        n_visits=("n_visits", "sum"),
        n_completed=("n_completed", "sum"),
        n_deaths=("n_deaths", "sum"),
        ScoreGained=("ScoreGained", "mean"),
        Enemies_killed=("Enemies_killed", "mean"),
        X_Traveled=("X_Traveled", "mean"),
    )
    perf["completion_rate"] = perf["n_completed"] / perf["n_visits"].clip(lower=1)
    perf["mastery"] = perf["completion_rate"]

    data_io.ensure_data_dir()
    perf.to_csv(data_io.perf_path(), index=False)

    log.info("Aggregated rows: %d", len(perf))
    log.info("Subjects: %s", sorted(perf["subject"].unique()))
    log.info("Sessions for sub-01: %s",
             sorted(perf[perf["subject"] == "sub-01"]["session"].unique()))
    log.info("Cached at %s", data_io.perf_path())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
