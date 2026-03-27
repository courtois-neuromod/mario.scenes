"""Extract scene clips from Super Mario Bros replay files."""

import argparse
import os
import os.path as op
import gzip
import stable_retro as retro
import pandas as pd
import numpy as np
import skvideo.io
from PIL import Image
from joblib import Parallel, delayed
import json
import logging
import re
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import traceback
from mario_scenes.load_data import load_scenes_info
# Import general-purpose functions directly from videogames.utils
from videogames_utils.video import make_mp4, make_gif, make_webp
from videogames_utils.replay import get_variables_from_replay
from videogames_utils.metadata import collect_bk2_files, create_sidecar_dict


def prune_variables(variables, start_idx, end_idx):
    """Slice list/array variables to frame range, copy scalars."""
    pruned = {}
    for key, value in variables.items():
        pruned[key] = value[start_idx:end_idx] if isinstance(value, (list, np.ndarray)) else value
    return pruned


def merge_metadata(additional_fields, base_dict):
    """Add fields from additional_fields to base_dict if not already present."""
    for key, value in additional_fields.items():
        if key not in base_dict:
            base_dict[key] = value
    return base_dict


def get_rep_order(ses, run, bk2_idx):
    """Generate repetition order string from session, run, and bk2 index."""
    return f"{int(ses):03d}{int(run):02d}{int(bk2_idx):02d}"


def _compute_player_x_pos(pos_hi, pos_lo):
    """Compute full x position from high and low byte arrays."""
    return [hi * 256 + lo for hi, lo in zip(pos_hi, pos_lo)]


def _is_scene_entry(x_pos_curr, x_pos_prev, scene_start, scene_end, layout_curr, layout_target):
    """Check if player just entered the scene."""
    crossed_start = x_pos_curr >= scene_start and x_pos_prev < scene_start
    within_scene = x_pos_curr < scene_end
    correct_layout = layout_curr == layout_target
    return crossed_start and within_scene and correct_layout


def _is_scene_exit(x_pos_curr, x_pos_prev, scene_end):
    """Check if player crossed scene end boundary."""
    return x_pos_curr >= scene_end and x_pos_prev < scene_end


def _is_death(lives_curr, lives_prev):
    """Check if player died (life count decreased)."""
    return lives_curr - lives_prev < 0


def cut_scene_clips(repetition_variables, rep_order_string, scene_bounds):
    """
    Extract clip boundaries for a scene from replay variables.

    Detects when player enters and exits the scene based on x position
    and level layout. A clip ends when player exits the scene or dies.
    Multiple clips can be found if player re-enters the scene.

    Returns dict mapping clip codes to (start_frame, end_frame) tuples.
    """
    x_pos = _compute_player_x_pos(
        repetition_variables["player_x_posHi"],
        repetition_variables["player_x_posLo"]
    )

    clips_found = {}
    start_found = False
    start_idx = None

    for frame_idx in range(1, len(x_pos)):
        x_curr, x_prev = x_pos[frame_idx], x_pos[frame_idx - 1]
        layout_curr = repetition_variables["level_layout"][frame_idx]
        lives_curr = repetition_variables["lives"][frame_idx]
        lives_prev = repetition_variables["lives"][frame_idx - 1]

        if not start_found:
            if _is_scene_entry(x_curr, x_prev, scene_bounds["start"],
                             scene_bounds["end"], layout_curr,
                             scene_bounds["level_layout"]):
                start_idx = frame_idx
                start_found = True
        else:
            if _is_scene_exit(x_curr, x_prev, scene_bounds["end"]) or _is_death(lives_curr, lives_prev):
                clip_code = f"{rep_order_string}{str(start_idx).zfill(7)}"
                clips_found[clip_code] = (start_idx, frame_idx)
                start_found = False
            elif _is_scene_entry(x_curr, x_prev, scene_bounds["start"],
                               scene_bounds["end"], layout_curr,
                               scene_bounds["level_layout"]):
                start_idx = frame_idx

    return clips_found


def _build_clip_paths(output_folder, sub, ses, run, level, scene, clip_code):
    """Construct BIDS-compliant file paths for all clip outputs."""
    beh_folder = op.join(output_folder, f"sub-{sub}", f"ses-{ses}", "beh")
    entities = f"sub-{sub}_ses-{ses}_run-{run}_level-{level}_scene-{scene}_clip-{clip_code}"

    return {
        "gif": op.join(beh_folder, "videos", f"{entities}.gif"),
        "mp4": op.join(beh_folder, "videos", f"{entities}.mp4"),
        "webp": op.join(beh_folder, "videos", f"{entities}.webp"),
        "savestate": op.join(beh_folder, "savestates", f"{entities}.state"),
        "ramdump": op.join(beh_folder, "ramdumps", f"{entities}.npz"),
        "json": op.join(beh_folder, "infos", f"{entities}.json"),
        "variables": op.join(beh_folder, "variables", f"{entities}.json"),
        "entities": entities
    }


def _build_clip_metadata(sub, ses, run, level, scene, clip_code, start_idx,
                         end_idx, total_frames, bk2_file, game_name, level_full, scene_full):
    """Build base metadata dictionary for a clip."""
    return {
        "Subject": sub, "Session": ses, "Run": run, "Level": level,
        "Scene": scene, "ClipCode": clip_code, "StartFrame": start_idx,
        "EndFrame": end_idx, "TotalFrames": total_frames, "Bk2File": bk2_file,
        "GameName": game_name, "LevelFullName": level_full, "SceneFullName": scene_full
    }


def _load_replay_sidecar(replays_path, sub, ses, bk2_filename):
    """Load replay metadata JSON if available."""
    if replays_path is None:
        return {}

    replay_path = op.join(replays_path, f"sub-{sub}", f"ses-{ses}", "beh",
                          "infos", bk2_filename.replace(".bk2", ".json"))

    if not os.path.exists(replay_path):
        logging.warning(f"Replay file not found: {replay_path}")
        return {}

    with open(replay_path, "r") as f:
        return json.load(f)


def _check_clip_exists(paths, args):
    """Check if all requested output files already exist for a clip."""
    # Always check JSON as it's always created
    if not os.path.exists(paths["json"]):
        return False

    # Check video files
    if args.save_videos:
        video_path = paths.get(args.video_format)
        if not os.path.exists(video_path):
            return False

    # Check savestate file
    if args.save_states:
        if not os.path.exists(paths["savestate"]):
            return False

    # Check ramdump file
    if args.save_ramdumps:
        if not os.path.exists(paths["ramdump"]):
            return False

    # Check variables file
    if args.save_variables:
        if not os.path.exists(paths["variables"]):
            return False

    return True


def _save_clip_outputs(paths, frames, states, audio_track, audio_rate,
                      start_idx, end_idx, variables, args):
    """Save requested output files for a clip."""
    if args.save_videos:
        os.makedirs(os.path.dirname(paths["mp4"]), exist_ok=True)
        if args.video_format == "gif":
            make_gif(frames, paths["gif"])
        elif args.video_format == "mp4":
            make_mp4(frames, paths["mp4"], audio=audio_track, sample_rate=audio_rate)
        elif args.video_format == "webp":
            make_webp(frames, paths["webp"])

    if args.save_states:
        os.makedirs(os.path.dirname(paths["savestate"]), exist_ok=True)
        with gzip.open(paths["savestate"], "wb") as fh:
            fh.write(states[start_idx])

    if args.save_ramdumps:
        os.makedirs(os.path.dirname(paths["ramdump"]), exist_ok=True)
        np.savez_compressed(paths["ramdump"], states[start_idx:end_idx])

    if args.save_variables:
        os.makedirs(os.path.dirname(paths["variables"]), exist_ok=True)
        with open(paths["variables"], "w") as f:
            json.dump(variables, f)


def process_bk2_file(bk2_info, args, scenes_info_dict, DATA_PATH, OUTPUT_FOLDER, STIMULI_PATH):
    """
    Process a single bk2 replay file to extract and save scene clips.

    Runs the replay, detects scene boundaries, and saves requested outputs
    (videos, savestates, ramdumps, variables) along with BIDS metadata.

    Returns (error_logs, processing_stats) for aggregation by main process.
    """
    retro.data.Integrations.add_custom_path(STIMULI_PATH)

    error_logs = []
    processing_stats = {"bk2_file": bk2_info["bk2_file"], "clips_processed": 0,
                       "clips_skipped": 0, "errors": 0}

    try:
        bk2_file = bk2_info["bk2_file"]
        logging.info(f"Processing bk2 file: {bk2_file}")

        rep_order = get_rep_order(bk2_info["ses"], bk2_info["run"], bk2_info["bk2_idx"])

        rep_vars, _, frames, states, audio, audio_rate = get_variables_from_replay(
            op.join(DATA_PATH, bk2_file),
            skip_first_step=(bk2_info["bk2_idx"] == 0),
            game=args.game_name,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
        )

        curr_level = op.basename(bk2_file).split("_")[-2].split("-")[1]
        scenes_in_level = [s for s in scenes_info_dict.keys() if curr_level in s]

        for scene_name in scenes_in_level:
            scene_bounds = {
                'start': scenes_info_dict[scene_name]['start'],
                'end': scenes_info_dict[scene_name]['end'],
                'level_layout': scenes_info_dict[scene_name]['level_layout']
            }

            clips_found = cut_scene_clips(rep_vars, rep_order, scene_bounds)

            for clip_code, (start_idx, end_idx) in clips_found.items():
                assert len(clip_code) == 14, f"Invalid clip code: {clip_code}"

                scene_num = int(scene_name.split('s')[1])
                paths = _build_clip_paths(OUTPUT_FOLDER, bk2_info["sub"], bk2_info["ses"],
                                        bk2_info["run"], rep_vars['level'], scene_num, clip_code)

                # Check if clip already exists with all requested outputs
                if _check_clip_exists(paths, args):
                    logging.debug(f"Skipping existing clip {clip_code}")
                    processing_stats["clips_skipped"] += 1
                    continue

                metadata = _build_clip_metadata(
                    bk2_info["sub"], bk2_info["ses"], bk2_info["run"], rep_vars["level"],
                    scene_num, clip_code, start_idx, end_idx, len(rep_vars['score']),
                    paths["entities"], args.game_name, curr_level, scene_name
                )

                scene_vars = prune_variables(rep_vars, start_idx, end_idx)
                scene_vars["metadata"] = paths["entities"]
                scene_sidecar = create_sidecar_dict(scene_vars)
                enriched_metadata = merge_metadata(metadata, scene_sidecar)

                replay_sidecar = _load_replay_sidecar(
                    args.replays_path, bk2_info["sub"], bk2_info["ses"],
                    bk2_file.split("/")[-1]
                )
                enriched_metadata = merge_metadata(replay_sidecar, enriched_metadata)

                os.makedirs(os.path.dirname(paths["json"]), exist_ok=True)
                with open(paths["json"], "w") as f:
                    json.dump(enriched_metadata, f, indent=4)

                if not any([args.save_states, args.save_ramdumps, args.save_videos]):
                    processing_stats["clips_skipped"] += 1
                    continue

                try:
                    # Calculate audio slice based on frame indices
                    # Assuming 60 FPS for NES emulation
                    samples_per_frame = audio_rate // 60
                    audio_start = start_idx * samples_per_frame
                    audio_end = end_idx * samples_per_frame
                    audio_slice = audio[audio_start:audio_end] if audio is not None else None

                    _save_clip_outputs(paths, frames[start_idx:end_idx], states, audio_slice,
                                     audio_rate, start_idx, end_idx, scene_vars, args)
                    processing_stats["clips_processed"] += 1

                except Exception as e:
                    error_logs.append(f"Error processing clip {clip_code}: {str(e)}")
                    processing_stats["errors"] += 1

    except Exception as e:
        error_logs.append(f"Error processing bk2 file {bk2_info.get('bk2_file', 'Unknown')}: {str(e)}")
        processing_stats["errors"] += 1
        traceback.print_exc()

    return error_logs, processing_stats


def _setup_logging(verbose_level):
    """Configure logging based on verbosity level."""
    levels = {0: logging.WARNING, 1: logging.INFO}
    level = levels.get(verbose_level, logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _setup_paths(args):
    """Setup and validate data/output/stimuli paths."""
    data_path = op.abspath(args.datapath)
    output_folder = op.join(op.abspath(args.output), args.output_name)
    os.makedirs(output_folder, exist_ok=True)

    stimuli_path = (op.abspath(args.stimuli)
                   if args.stimuli
                   else op.abspath(op.join(data_path, "stimuli")))

    return data_path, output_folder, stimuli_path


def _aggregate_results(results, total_bk2_files):
    """Aggregate processing statistics and error logs from parallel results."""
    stats = {
        "total_bk2_files": total_bk2_files,
        "total_clips_processed": 0,
        "total_clips_skipped": 0,
        "total_errors": 0,
    }
    error_logs = []

    for logs, proc_stats in results:
        stats["total_clips_processed"] += proc_stats.get("clips_processed", 0)
        stats["total_clips_skipped"] += proc_stats.get("clips_skipped", 0)
        stats["total_errors"] += proc_stats.get("errors", 0)
        error_logs.extend(logs)

    return stats, error_logs


def _save_dataset_metadata(output_folder, output_name, stats, error_logs):
    """Save BIDS dataset description and processing log."""
    dataset_desc = {
        "Name": output_name,
        "BIDSVersion": "1.6.0",
        "GeneratedBy": [{
            "Name": "Courtois Neuromod",
            "Version": "1.0.0",
            "CodeURL": "https://github.com/courtois-neuromod/mario.scenes/src/mario_scenes/clips_extraction/clip_extraction.py",
        }],
        "SourceDatasets": [{"URL": "https://github.com/courtois-neuromod/mario/"}],
        "License": "CC0",
    }

    with open(op.join(output_folder, "dataset_description.json"), "w") as f:
        json.dump(dataset_desc, f, indent=4)

    log_file = op.join(output_folder, "processing_log.txt")
    with open(log_file, "w") as f:
        f.write("Processing Log\n=================\n")
        f.write(f"Total bk2 files: {stats['total_bk2_files']}\n")
        f.write(f"Total clips processed: {stats['total_clips_processed']}\n")
        f.write(f"Total clips skipped: {stats['total_clips_skipped']}\n")
        f.write(f"Total errors: {stats['total_errors']}\n")
        f.write("\nError Details:\n")
        for error in error_logs:
            f.write(error + "\n")

    logging.info(f"Processing complete. Log file saved to {log_file}.")


def main(args):
    """
    Main entry point for clip extraction pipeline.

    Processes Mario replay files to extract scene clips with videos,
    savestates, and metadata in BIDS-compliant format.
    """
    _setup_logging(args.verbose)

    args.game_name = "SuperMarioBros-Nes"
    args.output_name = "."

    data_path, output_folder, stimuli_path = _setup_paths(args)

    logging.debug(f"Adding stimuli path: {stimuli_path}")
    retro.data.Integrations.add_custom_path(stimuli_path)
    logging.debug(f"Available games: {retro.data.list_games(inttype=retro.data.Integrations.CUSTOM_ONLY)}")
    logging.info(f"Game: {args.game_name}, Output: {args.output_name}")
    logging.info(f"Data: {data_path}, Stimuli: {stimuli_path}, Output: {output_folder}")

    scenes_info = load_scenes_info(format="dict")
    bk2_files = collect_bk2_files(data_path, args.subjects, args.sessions)

    if args.replays_path is None:
        logging.warning(
            "\n⚠️  No replay metadata path provided (--replays_path).\n"
            "   Scene clips will contain basic scene-level metadata only.\n"
            "   For enriched metadata with replay-level statistics (score, enemies killed, etc.),\n"
            "   first run mario.replays to generate replay-level metadata:\n"
            "     cd ../mario.replays && invoke create-replays --save-variables\n"
            "   Then re-run create-clips with: --replays-path <path_to_replays_output>\n"
        )
    else:
        logging.info(f"Using replay metadata from: {args.replays_path}")

    logging.info(f"Processing {len(bk2_files)} bk2 files using {args.n_jobs} job(s)...")

    with tqdm_joblib(tqdm(desc="Processing bk2 files", total=len(bk2_files))):
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_bk2_file)(info, args, scenes_info, data_path, output_folder, stimuli_path)
            for info in bk2_files
        )

    stats, error_logs = _aggregate_results(results, len(bk2_files))
    _save_dataset_metadata(output_folder, args.output_name, stats, error_logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract clips from Mario dataset based on scene information."
    )
    parser.add_argument(
        "-d",
        "--datapath",
        default="sourcedata/mario",
        type=str,
        help="Data path to look for events.tsv and .bk2 files. Should be the root of the Mario dataset.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default='outputdata/',
        type=str,
        help="Path to the derivatives folder, where the outputs will be saved.",
    )
    parser.add_argument(
        "-sp",
        "--stimuli",
        default=None,
        type=str,
        help="Path to the stimuli folder containing the game ROMs. Defaults to <datapath>/stimuli if not specified.",
    )
    parser.add_argument(
        "-nj",
        "--n_jobs",
        default=-1,
        type=int,
        help="Number of CPU cores to use for parallel processing.",
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save the playback video file (.mp4).",
    )
    parser.add_argument(
        "--save_variables",
        action="store_true",
        help="Save the variables file (.npz) that contains game variables.",
    )
    parser.add_argument(
        "--save_states",
        action="store_true",
        help="Save full RAM state at each frame into a *_states.npy file.",
    )
    parser.add_argument(
        "--save_ramdumps",
        action="store_true",
        help="Save RAM dumps at each frame into a *_ramdumps.npy file.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (can be specified multiple times)",
    )
    parser.add_argument(
        "--subjects",
        "-sub",
        nargs="+",
        default=None,
        help="List of subjects to process (e.g., sub-01 sub-02). If not specified, all subjects are processed.",
    )
    parser.add_argument(
        "--sessions",
        "-ses",
        nargs="+",
        default=None,
        help="List of sessions to process (e.g., ses-001 ses-002). If not specified, all sessions are processed.",
    )
    parser.add_argument(
        "--video_format",
        "-vf",
        default="mp4",
        choices=["gif", "mp4", "webp"],
        help="Video format to save (default: mp4).",
    )
    parser.add_argument(
        "--replays_path",
        "-rp",
        default=None,
        type=str,
        help="Path to the replay files (bk2) to extract metadata from.",
    )

    args = parser.parse_args()

    main(args)
