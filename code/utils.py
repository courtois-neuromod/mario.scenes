"""Shared utilities for mario.scenes generation scripts.

Provides scene metadata loading (with auto-download from Zenodo), scene
boundary detection, clip-level stat computation, and source-phase lookup
used by both clip extraction and annotation generation.
"""

import json
import logging
import os
import os.path as op
import urllib.request

import pandas as pd

BASE_DIR = op.dirname(op.dirname(op.abspath(__file__)))
SOURCEDATA = op.join(BASE_DIR, "sourcedata")
SCENES_INFO_DIR = op.join(SOURCEDATA, "scenes_info")
SCENES_MASTERSHEET = op.join(SCENES_INFO_DIR, "scenes_mastersheet.csv")

ZENODO_RECORD = "15586709"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"


# ---------------------------------------------------------------------------
# Zenodo download helpers
# ---------------------------------------------------------------------------

def ensure_scenes_data():
    """Download scenes_mastersheet.csv from Zenodo if not already present.

    Uses the Zenodo API to resolve the latest version of the record, then
    downloads the CSV file to ``sourcedata/scenes_info/``.

    Returns the path to the local CSV file.
    """
    if op.exists(SCENES_MASTERSHEET):
        return SCENES_MASTERSHEET

    os.makedirs(SCENES_INFO_DIR, exist_ok=True)

    # Resolve the latest version via the Zenodo API
    logging.info(f"Fetching Zenodo record {ZENODO_RECORD} metadata ...")
    with urllib.request.urlopen(ZENODO_API_URL) as resp:
        record = json.loads(resp.read().decode())

    # Follow the "latest" link if available
    latest_url = record.get("links", {}).get("latest")
    if latest_url:
        with urllib.request.urlopen(latest_url) as resp:
            record = json.loads(resp.read().decode())

    # Find the CSV file among the record's files
    csv_url = None
    for f in record.get("files", []):
        if f["key"].endswith("scenes_mastersheet.csv"):
            csv_url = f["links"]["self"]
            break

    if csv_url is None:
        raise FileNotFoundError(
            "Could not find scenes_mastersheet.csv in Zenodo record "
            f"{ZENODO_RECORD}."
        )

    logging.info(f"Downloading {csv_url} ...")
    urllib.request.urlretrieve(csv_url, SCENES_MASTERSHEET)
    logging.info(f"Saved to {SCENES_MASTERSHEET}")
    return SCENES_MASTERSHEET


# ---------------------------------------------------------------------------
# Scene metadata loading
# ---------------------------------------------------------------------------

def load_scenes_info(format="df"):
    """Load scene definitions with start/end positions and layouts.

    Calls :func:`ensure_scenes_data` first so the CSV is available.

    Parameters
    ----------
    format : str
        ``"df"`` returns a :class:`~pandas.DataFrame`.
        ``"dict"`` returns ``{scene_id: {"start", "end", "level_layout"}}``.
    """
    ensure_scenes_data()
    scenes_df = pd.read_csv(SCENES_MASTERSHEET)

    if format == "df":
        return scenes_df

    if format == "dict":
        scenes_dict = {}
        for _, row in scenes_df.iterrows():
            try:
                scene_id = (
                    f"w{int(row['World'])}l{int(row['Level'])}"
                    f"s{int(row['Scene'])}"
                )
                scenes_dict[scene_id] = {
                    "start": int(row["Entry point"]),
                    "end": int(row["Exit point"]),
                    "level_layout": int(row["Layout"]),
                }
            except (ValueError, KeyError):
                continue
        return scenes_dict

    raise ValueError('format must be "df" or "dict"')


# ---------------------------------------------------------------------------
# Scene boundary detection
# ---------------------------------------------------------------------------

def _compute_player_x_pos(pos_hi, pos_lo):
    """Compute full x position from high and low byte arrays."""
    return [hi * 256 + lo for hi, lo in zip(pos_hi, pos_lo)]


def _is_scene_entry(x_curr, x_prev, scene_start, scene_end,
                    layout_curr, layout_target):
    """Return True if the player just crossed into the scene."""
    crossed_start = x_curr >= scene_start and x_prev < scene_start
    within_scene = x_curr < scene_end
    correct_layout = layout_curr == layout_target
    return crossed_start and within_scene and correct_layout


def _is_scene_exit(x_curr, x_prev, scene_end):
    """Return True if the player crossed the scene end boundary."""
    return x_curr >= scene_end and x_prev < scene_end


def _is_death(player_state_curr, player_state_prev, lives_curr, lives_prev):
    """Check if the player died, using player_state (like mario/code/ scripts).

    Detects two kinds of death:
    - **Enemy / timeout**: ``player_state`` transitions to 11 (dying animation).
    - **Fall**: lives decrease without the dying animation (no ``player_state == 11``).
    """
    if player_state_curr == 11 and player_state_prev != 11:
        return True
    if lives_curr < lives_prev:
        return True
    return False


def detect_scene_traversals(rep_vars, scene_bounds):
    """Detect all traversals of a single scene within a repetition.

    Parameters
    ----------
    rep_vars : dict
        Per-frame game variables for one repetition.  Must contain
        ``player_x_posHi``, ``player_x_posLo``, ``level_layout``,
        ``player_state``, and ``lives``.
    scene_bounds : dict
        ``{"start": int, "end": int, "level_layout": int}`` for the scene.

    Returns
    -------
    list[tuple[int, int, str]]
        ``[(start_frame, end_frame, outcome), ...]`` where *outcome* is
        ``"completed"`` (player exited the scene) or ``"death"``.
    """
    x_pos = _compute_player_x_pos(
        rep_vars["player_x_posHi"],
        rep_vars["player_x_posLo"],
    )

    traversals = []
    start_found = False
    start_idx = None

    for i in range(1, len(x_pos)):
        x_curr, x_prev = x_pos[i], x_pos[i - 1]
        layout_curr = rep_vars["level_layout"][i]
        ps_curr = rep_vars["player_state"][i]
        ps_prev = rep_vars["player_state"][i - 1]
        lives_curr = rep_vars["lives"][i]
        lives_prev = rep_vars["lives"][i - 1]

        if not start_found:
            if _is_scene_entry(x_curr, x_prev, scene_bounds["start"],
                               scene_bounds["end"], layout_curr,
                               scene_bounds["level_layout"]):
                start_idx = i
                start_found = True
        else:
            if _is_scene_exit(x_curr, x_prev, scene_bounds["end"]):
                traversals.append((start_idx, i, "completed"))
                start_found = False
            elif _is_death(ps_curr, ps_prev, lives_curr, lives_prev):
                traversals.append((start_idx, i, "death"))
                start_found = False
            elif _is_scene_entry(x_curr, x_prev, scene_bounds["start"],
                                 scene_bounds["end"], layout_curr,
                                 scene_bounds["level_layout"]):
                # Re-entry without exit (e.g. after respawn) — reset start
                start_idx = i

    return traversals


# ---------------------------------------------------------------------------
# Clip-level gameplay statistics
# ---------------------------------------------------------------------------

def _count_enemy_kills_for_slot(clip_vars, slot_idx):
    """Count enemy kills for one NES enemy slot (enemy_kill3X)."""
    enemy_key = f"enemy_kill3{slot_idx}"
    if enemy_key not in clip_vars:
        return 0
    vals = clip_vars[enemy_key]
    kill_count = 0
    for idx in range(len(vals) - 1):
        val = vals[idx]
        # Values 4, 34, 132 indicate a kill event; count the falling edge
        if val in [4, 34, 132] and vals[idx + 1] != val:
            if slot_idx != 5:
                kill_count += 1
    return kill_count


def compute_clip_stats(clip_vars):
    """Compute gameplay statistics from clip-level per-frame variables.

    Mirrors the logic in the mario dataset's ``generate_replays.py`` but
    applied to a single clip slice rather than a full repetition.

    Parameters
    ----------
    clip_vars : dict
        Per-frame game variable lists sliced to the clip range.

    Returns
    -------
    dict
        Keys: ``ScoreGained``, ``CoinsGained``, ``Lives_lost``,
        ``Hits_taken``, ``Enemies_killed``, ``Powerups_collected``,
        ``Bricks_smashed``, ``X_Traveled``.  Empty dict on failure.
    """
    score = clip_vars.get("score")
    if not isinstance(score, list) or not score:
        return {}

    # Simple difference stats
    score_gained = score[-1] - score[0]
    coins_gained = clip_vars["coins"][-1] - clip_vars["coins"][0]
    lives = clip_vars["lives"]
    lives_lost = lives[0] - lives[-1]

    # Hits taken: powerstate drops (big negative) + life losses
    hits = 0
    ps = clip_vars.get("powerstate", [])
    for i in range(1, len(ps)):
        if ps[i] - ps[i - 1] < -10000:
            hits += 1
    for i in range(1, len(lives)):
        if lives[i] < lives[i - 1]:
            hits += 1

    # Enemies killed across all 6 NES enemy slots
    enemies = sum(_count_enemy_kills_for_slot(clip_vars, i) for i in range(6))

    # Powerups collected: player_state exits [9, 12, 13]
    powerups = 0
    pstate = clip_vars.get("player_state", [])
    for i in range(len(pstate) - 1):
        if pstate[i] in [9, 12, 13] and pstate[i + 1] != pstate[i]:
            powerups += 1

    # Bricks smashed: score increment of 5 while airborne
    bricks = 0
    ja = clip_vars.get("jump_airborne", [])
    for i in range(len(score) - 1):
        if score[i + 1] - score[i] == 5 and i < len(ja) and ja[i] == 1:
            bricks += 1

    # X distance traveled (scroll register difference)
    x_traveled = (
        clip_vars["xscrollLo"][-1] + 256 * clip_vars["xscrollHi"][-1]
        - clip_vars["xscrollLo"][0] - 256 * clip_vars["xscrollHi"][0]
    )

    return {
        "ScoreGained": score_gained,
        "CoinsGained": coins_gained,
        "Lives_lost": lives_lost,
        "Hits_taken": hits,
        "Enemies_killed": enemies,
        "Powerups_collected": powerups,
        "Bricks_smashed": bricks,
        "X_Traveled": x_traveled,
    }


# ---------------------------------------------------------------------------
# Source phase lookup
# ---------------------------------------------------------------------------

def load_source_phase(data_path, bk2_file):
    """Read ``Phase`` from the ``_summary.json`` companion of a source .bk2.

    Parameters
    ----------
    data_path : str
        Root of the mario dataset.
    bk2_file : str
        Path to the .bk2 relative to *data_path*.

    Returns
    -------
    str or None
        ``"discovery"``, ``"practice"``, or ``None`` if the sidecar is absent.
    """
    summary_path = op.join(
        data_path,
        bk2_file.replace(".bk2", "_summary.json"),
    )
    if op.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f).get("Phase")
    return None
