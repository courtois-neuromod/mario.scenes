#!/usr/bin/env python
"""Extract scene clip replays from Super Mario Bros replay files.

For each .bk2 replay in the mario dataset, detect scene boundaries and
record one trimmed ``.bk2`` per scene traversal into a flat ``gamelogs/``
directory.

This script produces *only* the trimmed clip ``.bk2`` files plus a
dataset-level ``clips_manifest.tsv`` recording the scene-detection
metadata (outcome, source replay, phase) that cannot be recovered from a
``.bk2`` alone.  The replay sidecars (``_recording.mp4``, ``.state``,
``_variables.json``, ``_summary.json``) are generated separately by
``generate_replays.py``, which can run on any directory of clip ``.bk2``
files — including clips produced by artificial agents.

Usage::

    python generate_clips.py -d /path/to/mario -o /path/to/mario.scenes \\
        --subjects sub-01 --sessions ses-001 -nj 1 -v
"""

import argparse
import logging
import os
import os.path as op
import shutil
import tempfile
import traceback
import sys

import pandas as pd
import stable_retro as retro
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from videogames_utils.metadata import collect_bk2_files
from videogames_utils.replay import replay_bk2, reformat_info

# Add code/ to path so utils is importable
sys.path.insert(0, op.dirname(op.dirname(op.abspath(__file__))))
from utils import load_scenes_info, detect_scene_traversals, load_source_phase

GAME_NAME = "SuperMarioBros-Nes"
MIN_CLIP_FRAMES = 6  # Clips shorter than this (~0.1 s) are skipped
MANIFEST_NAME = "clips_manifest.tsv"

# Columns of clips_manifest.tsv — consumed by generate_replays.py
MANIFEST_COLUMNS = [
    "clip_code", "clip_file", "sub", "ses", "run", "rep_index",
    "level", "scene_id", "start_frame", "end_frame", "duration",
    "outcome", "phase", "source_bk2",
]


# ---------------------------------------------------------------------------
# Clip code generation
# ---------------------------------------------------------------------------

def get_clip_code(ses, bids_run, rep_index, start_frame):
    """Build a 14-character clip code: ``SSSRRBBNNNNNNN``.

    Parameters
    ----------
    ses : str or int
        Session number (3 digits).
    bids_run : str or int
        BIDS run number within the session (2 digits).
    rep_index : int
        1-based position of the .bk2 within its BIDS run, as given by
        the ``rep_index`` column in ``desc-annotated_events.tsv``.
    start_frame : int
        Frame index at which the scene traversal begins (7 digits).
    """
    return f"{int(ses):03d}{int(bids_run):02d}{int(rep_index):02d}{start_frame:07d}"


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def _build_clip_paths(output_dir, sub, ses, level, scene_id, clip_code):
    """Return a dict of output paths for one clip's .bk2."""
    gamelogs = op.join(output_dir, f"sub-{sub}", f"ses-{ses}", "gamelogs")
    scene_num = int(scene_id.split("s")[-1])
    entities = (
        f"sub-{sub}_ses-{ses}_task-mario_level-{level}"
        f"_scene-{scene_num}_clip-{clip_code}"
    )
    return {
        "bk2": op.join(gamelogs, f"{entities}.bk2"),
        "gamelogs": gamelogs,
        "entities": entities,
    }


# ---------------------------------------------------------------------------
# BK2 creation via stable-retro recording
# ---------------------------------------------------------------------------

def _create_clip_bk2(clip_state, clip_keys, output_path, stimuli_path):
    """Record a clean .bk2 by replaying *clip_keys* from *clip_state*.

    Creates a fresh emulator with ``record=True``, feeds the actions
    through it, and moves the resulting .bk2 to *output_path*.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        retro.data.Integrations.add_custom_path(stimuli_path)
        env = retro.make(
            GAME_NAME,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
            render_mode=False,
            record=tmpdir,
        )
        env.initial_state = clip_state
        env.reset()

        for keys in clip_keys:
            env.step(keys)
        env.unwrapped.stop_record()
        env.close()

        # Move the auto-generated .bk2 to the desired path
        generated = [f for f in os.listdir(tmpdir) if f.endswith(".bk2")]
        if not generated:
            raise RuntimeError("stable-retro did not produce a .bk2 file")
        os.makedirs(op.dirname(output_path), exist_ok=True)
        shutil.move(op.join(tmpdir, generated[0]), output_path)


# ---------------------------------------------------------------------------
# Per-bk2 processing
# ---------------------------------------------------------------------------

def process_bk2_file(bk2_info, scenes_info, data_path, output_dir,
                     stimuli_path, verbose):
    """Process one source .bk2 replay: detect scenes and record clip .bk2 files.

    Returns
    -------
    tuple
        ``(error_logs, stats, manifest_rows)`` where *stats* counts clips
        processed / skipped / errors and *manifest_rows* is a list of
        dicts (one per recorded clip) for ``clips_manifest.tsv``.
    """
    retro.data.Integrations.add_custom_path(stimuli_path)
    error_logs = []
    manifest_rows = []
    stats = {
        "bk2_file": bk2_info["bk2_file"],
        "clips_processed": 0,
        "clips_skipped": 0,
        "clips_too_short": 0,
        "errors": 0,
    }

    try:
        bk2_file = bk2_info["bk2_file"]
        logging.info(f"Processing: {bk2_file}")

        bk2_path = op.join(data_path, bk2_file)
        sub = bk2_info["sub"]
        ses = bk2_info["ses"]
        bids_run = bk2_info["bids_run"]
        rep_index = bk2_info["rep_index"]

        # -----------------------------------------------------------
        # Replay the bk2 using the low-level iterator so we capture
        # emulator states and raw keys alongside game variables.
        # Frames and audio are not needed here — the replay sidecars
        # are produced later by generate_replays.py.
        # -----------------------------------------------------------
        all_keys = []
        all_info = []
        all_states = []
        actions = None

        # skip_first_step=True only for the first bk2 in its BIDS run.
        is_first_rep = (rep_index == 1)
        for _, keys, annotations, _, _, _, acts, state in replay_bk2(
            bk2_path,
            skip_first_step=is_first_rep,
            game=GAME_NAME,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
        ):
            all_keys.append(keys)
            all_info.append(annotations["info"])
            all_states.append(state)
            if actions is None:
                actions = acts

        if actions is None:
            logging.warning(f"Empty replay: {bk2_file}")
            return error_logs, stats, manifest_rows

        rep_vars = reformat_info(all_info, all_keys, bk2_path, actions)

        logging.debug(
            f"  Replayed {len(all_keys)} frames, "
            f"variables keys: {sorted(k for k in rep_vars.keys() if k is not None)[:10]}..."
        )

        # Load phase from the source bk2's _summary.json companion
        source_phase = load_source_phase(data_path, bk2_file)

        # -----------------------------------------------------------
        # Detect scene clips
        # -----------------------------------------------------------
        curr_level = op.basename(bk2_file).split("_")[-2].split("-")[1]
        scenes_in_level = [s for s in scenes_info if curr_level in s]
        logging.debug(f"  Level: {curr_level}, scenes: {scenes_in_level}")

        for scene_id in scenes_in_level:
            scene_bounds = scenes_info[scene_id]
            traversals = detect_scene_traversals(rep_vars, scene_bounds)
            if traversals:
                logging.debug(f"  {scene_id}: {len(traversals)} traversal(s)")

            for start_idx, end_idx, outcome in traversals:
                n_clip_frames = end_idx - start_idx

                if n_clip_frames < MIN_CLIP_FRAMES:
                    logging.debug(
                        f"  Skipping too-short traversal of {scene_id} "
                        f"({n_clip_frames} frames)"
                    )
                    stats["clips_too_short"] += 1
                    continue

                clip_code = get_clip_code(ses, bids_run, rep_index, start_idx)
                paths = _build_clip_paths(
                    output_dir, sub, ses, curr_level, scene_id, clip_code,
                )
                clip_relpath = op.relpath(paths["bk2"], output_dir)

                # Manifest row is recorded even when the .bk2 already
                # exists, so the manifest stays complete across runs.
                manifest_rows.append({
                    "clip_code": clip_code,
                    "clip_file": clip_relpath,
                    "sub": f"sub-{sub}",
                    "ses": f"ses-{ses}",
                    "run": bids_run,
                    "rep_index": rep_index,
                    "level": curr_level,
                    "scene_id": scene_id,
                    "start_frame": int(start_idx),
                    "end_frame": int(end_idx),
                    "duration": round(n_clip_frames / 60, 3),
                    "outcome": outcome,
                    "phase": source_phase,
                    "source_bk2": bk2_file,
                })

                if op.exists(paths["bk2"]):
                    logging.debug(f"Skipping existing clip {clip_code}")
                    stats["clips_skipped"] += 1
                    continue

                try:
                    os.makedirs(paths["gamelogs"], exist_ok=True)
                    _create_clip_bk2(
                        all_states[start_idx],
                        all_keys[start_idx:end_idx],
                        paths["bk2"],
                        stimuli_path,
                    )
                    stats["clips_processed"] += 1

                except Exception as e:
                    msg = f"Error clip {clip_code}: {e}\n{traceback.format_exc()}"
                    logging.error(msg)
                    error_logs.append(msg)
                    stats["errors"] += 1

    except Exception as e:
        msg = (
            f"Error bk2 {bk2_info.get('bk2_file', '?')}: {e}\n"
            f"{traceback.format_exc()}"
        )
        logging.error(msg)
        error_logs.append(msg)
        stats["errors"] += 1

    return error_logs, stats, manifest_rows


# ---------------------------------------------------------------------------
# BK2 metadata enrichment from annotated events
# ---------------------------------------------------------------------------

def _enrich_bk2_info(bk2_files, data_path):
    """Add BIDS run number and rep_index to each bk2_info dict.

    Reads ``desc-annotated_events.tsv`` files from the mario dataset and
    extracts the BIDS run (from the filename) and the 1-based ``rep_index``
    column (temporal position of the .bk2 within its run).

    Parameters
    ----------
    bk2_files : list[dict]
        Output of :func:`collect_bk2_files`.  Each dict is mutated in-place
        to add ``bids_run`` (str) and ``rep_index`` (int, 1-based) keys.
    data_path : str
        Root of the mario dataset.
    """
    # Build mapping: bk2 relative path -> (bids_run, rep_index)
    bk2_to_info = {}

    seen_events = set()
    for info in bk2_files:
        parts = info["bk2_file"].split("/")
        sub, ses = parts[0], parts[1]
        func_dir = op.join(data_path, sub, ses, "func")
        if not op.exists(func_dir):
            continue
        for fname in sorted(os.listdir(func_dir)):
            if not (fname.endswith("_events.tsv")
                    and "annotated" in fname
                    and "scenes" not in fname):
                continue
            key = (sub, ses, fname)
            if key in seen_events:
                continue
            seen_events.add(key)

            # Extract BIDS run from filename, e.g.
            # sub-01_ses-004_task-mario_run-01_desc-annotated_events.tsv
            bids_run = None
            for part in fname.replace(".tsv", "").split("_"):
                if part.startswith("run-"):
                    bids_run = part.split("-", 1)[1]
                    break
            if bids_run is None:
                continue

            try:
                events_df = pd.read_table(op.join(func_dir, fname))
                game_rows = events_df[
                    events_df["trial_type"] == "gym-retro_game"
                ]
                for _, row in game_rows.iterrows():
                    stim = row.get("stim_file")
                    rep_idx = row.get("rep_index")
                    if (isinstance(stim, str) and stim != "Missing file"
                            and pd.notna(rep_idx)):
                        bk2_to_info[stim] = (bids_run, int(rep_idx))
            except Exception:
                pass

    for info in bk2_files:
        bk2_rel = info["bk2_file"]
        if bk2_rel in bk2_to_info:
            info["bids_run"], info["rep_index"] = bk2_to_info[bk2_rel]
        else:
            logging.warning(
                f"No annotated events entry for {bk2_rel}, "
                f"falling back to run='00', rep_index=1"
            )
            info["bids_run"] = "00"
            info["rep_index"] = 1


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------

def _write_manifest(manifest_rows, output_dir):
    """Merge *manifest_rows* into ``clips_manifest.tsv`` and write it.

    An existing manifest is loaded first and updated by ``clip_code`` so
    incremental runs (per subject/session) accumulate rather than
    overwrite earlier results.
    """
    manifest_path = op.join(output_dir, MANIFEST_NAME)

    rows_by_code = {}
    if op.exists(manifest_path):
        try:
            existing = pd.read_table(manifest_path, dtype={"clip_code": str})
            for row in existing.to_dict("records"):
                rows_by_code[row["clip_code"]] = row
        except Exception as e:
            logging.warning(f"Could not read existing manifest: {e}")

    for row in manifest_rows:
        rows_by_code[row["clip_code"]] = row

    df = pd.DataFrame(list(rows_by_code.values()), columns=MANIFEST_COLUMNS)
    df = df.sort_values(["sub", "ses", "run", "start_frame"])
    df.to_csv(manifest_path, sep="\t", index=False)
    logging.info(f"Manifest: {manifest_path} ({len(df)} clips)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    """Run the clip extraction pipeline."""
    levels = {0: logging.WARNING, 1: logging.INFO}
    logging.basicConfig(
        level=levels.get(args.verbose, logging.DEBUG),
        format="%(levelname)s: %(message)s",
    )

    data_path = op.abspath(args.datapath)
    output_dir = op.abspath(args.output)
    stimuli_path = (
        op.abspath(args.stimuli)
        if args.stimuli
        else op.join(data_path, "stimuli")
    )

    retro.data.Integrations.add_custom_path(stimuli_path)
    logging.info(f"Data: {data_path}")
    logging.info(f"Output: {output_dir}")
    logging.info(f"Stimuli: {stimuli_path}")

    scenes_info = load_scenes_info(format="dict")
    bk2_files = collect_bk2_files(data_path, args.subjects, args.sessions)
    _enrich_bk2_info(bk2_files, data_path)

    logging.info(f"Processing {len(bk2_files)} bk2 files ({args.n_jobs} jobs)")

    with tqdm_joblib(tqdm(desc="Processing bk2 files", total=len(bk2_files))):
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_bk2_file)(
                info, scenes_info, data_path, output_dir, stimuli_path,
                args.verbose,
            )
            for info in bk2_files
        )

    # Aggregate stats and manifest rows
    total_processed = sum(r[1]["clips_processed"] for r in results)
    total_skipped = sum(r[1]["clips_skipped"] for r in results)
    total_too_short = sum(r[1]["clips_too_short"] for r in results)
    total_errors = sum(r[1]["errors"] for r in results)
    all_errors = [msg for r in results for msg in r[0]]
    all_manifest_rows = [row for r in results for row in r[2]]

    if all_manifest_rows:
        _write_manifest(all_manifest_rows, output_dir)

    logging.warning(
        f"Done: {total_processed} clips created, "
        f"{total_skipped} skipped, {total_too_short} too short, "
        f"{total_errors} errors."
    )
    if all_errors:
        log_path = op.join(output_dir, "clip_extraction_errors.log")
        with open(log_path, "w") as f:
            f.write("\n".join(all_errors))
        logging.warning(f"Error log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract scene clip .bk2 files from Mario replay files.",
    )
    parser.add_argument(
        "-d", "--datapath", required=True,
        help="Root of the Mario dataset (contains sub-XX/ and stimuli/).",
    )
    parser.add_argument(
        "-o", "--output", default=".",
        help="Output directory (mario.scenes root). Default: current dir.",
    )
    parser.add_argument(
        "-sp", "--stimuli", default=None,
        help="Path to stimuli/ folder (ROM, data.json, etc.). "
             "Defaults to <datapath>/stimuli.",
    )
    parser.add_argument(
        "--subjects", "-sub", nargs="+", default=None,
        help="Subjects to process (e.g. sub-01 sub-02).",
    )
    parser.add_argument(
        "--sessions", "-ses", nargs="+", default=None,
        help="Sessions to process (e.g. ses-001 ses-002).",
    )
    parser.add_argument(
        "-nj", "--n_jobs", default=-1, type=int,
        help="Parallel jobs (-1 = all cores). Default: -1.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v = INFO, -vv = DEBUG).",
    )
    main(parser.parse_args())
