#!/usr/bin/env python
"""Generate scene-level event annotations for the Mario dataset.

Reads ``_events.tsv`` files from the mario dataset and the clip
``_summary.json`` sidecars produced by ``generate_clips.py`` to build
``_desc-scenes_events.tsv`` files under ``sub-XX/ses-XX/func/``.

Scene events are derived directly from the clip metadata — no
secondary scene-boundary detection is performed.

Usage::

    python generate_annotations.py \\
        -d /path/to/mario \\
        -o /path/to/mario.scenes \\
        --subjects sub-01 --sessions ses-001
"""

import argparse
import json
import logging
import os
import os.path as op

import pandas as pd

FS = 60  # NES frame rate


# ---------------------------------------------------------------------------
# Build clip index
# ---------------------------------------------------------------------------

def build_clip_index(output_path):
    """Index all clip ``_summary.json`` files by their source .bk2 path.

    Parameters
    ----------
    output_path : str
        Root of the mario.scenes output directory.

    Returns
    -------
    dict
        ``{source_bk2_relative_path: [summary_dict, ...]}``
    """
    index = {}
    for root, _, files in os.walk(output_path):
        if "gamelogs" not in root:
            continue
        for fname in files:
            if not fname.endswith("_summary.json"):
                continue
            fpath = op.join(root, fname)
            try:
                with open(fpath) as f:
                    summary = json.load(f)
            except Exception as e:
                logging.warning(f"Could not read {fpath}: {e}")
                continue

            source = summary.get("SourceBk2")
            if not source:
                continue

            # Build the stim_file path (relative to output root)
            bk2_path = fpath.replace("_summary.json", ".bk2")
            summary["_stim_file"] = op.relpath(bk2_path, output_path)

            index.setdefault(source, []).append(summary)

    # Sort each list by StartFrame so events are in temporal order
    for key in index:
        index[key].sort(key=lambda s: s.get("StartFrame", 0))

    return index


# ---------------------------------------------------------------------------
# Per-events-file processing
# ---------------------------------------------------------------------------

def process_events_file(events_path, data_path, output_path, clip_index):
    """Process one ``_events.tsv`` and write the scenes annotation.

    Parameters
    ----------
    events_path : str
        Full path to the source ``_events.tsv`` file.
    data_path : str
        Root of the mario dataset.
    output_path : str
        Root of the mario.scenes output directory.
    clip_index : dict
        Mapping ``{source_bk2: [summary, ...]}``, built by
        :func:`build_clip_index`.
    """
    filename = op.basename(events_path)
    parts = filename.split("_")
    sub = parts[0]    # sub-01
    ses = parts[1]    # ses-001

    out_fname = filename.replace("_events.", "_desc-scenes_events.")
    out_dir = op.join(output_path, sub, ses, "func")
    out_path = op.join(out_dir, out_fname)

    if op.exists(out_path):
        logging.info(f"Skipping (exists): {out_path}")
        return

    logging.info(f"Processing: {filename}")

    events_df = pd.read_table(events_path)
    game_events = events_df[
        events_df["trial_type"] == "gym-retro_game"
    ].copy()

    if game_events.empty:
        logging.warning(f"No gym-retro_game events in {filename}")
        return

    cols = ["trial_type", "onset", "level", "stim_file"]
    if "duration" in game_events.columns:
        cols.insert(2, "duration")
    game_events = game_events[cols].reset_index(drop=True)

    # Pre-populate string columns to prevent float conversion on concat
    for col in ("clip_code", "scene_id", "outcome", "phase"):
        game_events[col] = pd.array([pd.NA] * len(game_events), dtype="string")

    game_events["rep_index"] = range(len(game_events))

    # Build scene event rows from clip summaries
    all_scene_rows = []

    for rep_idx, row in game_events.iterrows():
        bk2_file = row.get("stim_file")
        if not isinstance(bk2_file, str) or bk2_file == "Missing file":
            logging.debug(f"Skipping missing file at index {rep_idx}")
            continue

        rep_onset = float(row["onset"])
        summaries = clip_index.get(bk2_file, [])

        if not summaries:
            logging.debug(f"No clips found for {bk2_file}")
            continue

        for summary in summaries:
            start_frame = summary.get("StartFrame", 0)
            end_frame = summary.get("EndFrame", 0)
            n_frames = end_frame - start_frame
            onset = rep_onset + start_frame / FS
            duration = n_frames / FS

            scene_row = {
                "trial_type": "scene",
                "scene_id": summary.get("SceneID"),
                "onset": round(onset, 3),
                "duration": round(duration, 3),
                "frame_start": int(start_frame),
                "frame_stop": int(end_frame),
                "outcome": summary.get("Outcome"),
                "phase": summary.get("Phase"),
                "clip_code": summary.get("ClipCode"),
                "stim_file": summary.get("_stim_file"),
                "rep_index": rep_idx,
                "level": summary.get("Level"),
            }

            # Gameplay stats (present when --skip_variables was not used)
            for stat in ("ScoreGained", "CoinsGained", "Lives_lost",
                         "Hits_taken", "Enemies_killed", "Powerups_collected",
                         "Bricks_smashed", "X_Traveled"):
                if stat in summary:
                    scene_row[stat] = summary[stat]

            all_scene_rows.append(scene_row)

    # Combine base game events and scene events
    dfs = [game_events]
    if all_scene_rows:
        scene_df = pd.DataFrame(all_scene_rows)
        for col in ("clip_code", "scene_id", "outcome", "phase"):
            if col in scene_df.columns:
                scene_df[col] = scene_df[col].astype("string")
        dfs.append(scene_df)

    combined = (
        pd.concat(dfs, ignore_index=True)
        .sort_values(by="onset")
        .reset_index(drop=True)
    )

    combined["onset"] = combined["onset"].round(3)
    combined["duration"] = combined["duration"].round(3)

    for col in ("frame_start", "frame_stop", "rep_index"):
        if col in combined.columns:
            combined[col] = combined[col].astype("Int64")

    # Reorder columns
    priority = [
        "trial_type", "scene_id", "onset", "duration",
        "frame_start", "frame_stop", "phase", "rep_index",
        "stim_file", "clip_code", "outcome", "level",
        "ScoreGained", "CoinsGained", "Lives_lost", "Hits_taken",
        "Enemies_killed", "Powerups_collected", "Bricks_smashed", "X_Traveled",
    ]
    priority = [c for c in priority if c in combined.columns]
    other = [c for c in combined.columns if c not in priority]
    combined = combined[priority + other]

    os.makedirs(out_dir, exist_ok=True)
    combined.to_csv(out_path, sep="\t", index=False)
    logging.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    """Walk the mario dataset and generate scene annotations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    data_path = op.abspath(args.datapath)
    output_path = op.abspath(args.output_path)

    logging.info(f"Mario dataset: {data_path}")
    logging.info(f"Output: {output_path}")

    subjects = args.subjects
    sessions = args.sessions
    if subjects:
        logging.info(f"Subjects: {', '.join(subjects)}")
    if sessions:
        logging.info(f"Sessions: {', '.join(sessions)}")

    logging.info("Building clip index from _summary.json files ...")
    clip_index = build_clip_index(output_path)
    logging.info(f"Indexed {sum(len(v) for v in clip_index.values())} clips "
                 f"from {len(clip_index)} source bk2 files.")

    processed = 0
    for root, _, files in sorted(os.walk(data_path)):
        if "sourcedata" in root:
            continue

        if subjects and not any(sub in root for sub in subjects):
            continue
        if sessions and not any(ses in root for ses in sessions):
            continue

        for fname in sorted(files):
            if (fname.endswith("_events.tsv")
                    and "annotated" not in fname
                    and "scenes" not in fname):
                process_events_file(
                    op.join(root, fname), data_path, output_path, clip_index,
                )
                processed += 1

    logging.info(f"Processed {processed} events files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate scene-level event annotations from clip metadata."
    )
    parser.add_argument(
        "-d", "--datapath", required=True,
        help="Root of the mario dataset.",
    )
    parser.add_argument(
        "-o", "--output_path", default=".",
        help="Output directory (mario.scenes root). Default: current dir.",
    )
    parser.add_argument(
        "--subjects", "-sub", nargs="+", default=None,
        help="Subjects to process (e.g. sub-01 sub-02).",
    )
    parser.add_argument(
        "--sessions", "-ses", nargs="+", default=None,
        help="Sessions to process (e.g. ses-001 ses-002).",
    )
    main(parser.parse_args())
