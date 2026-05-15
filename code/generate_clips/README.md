# generate_clips

Extract scene-level clips from Super Mario Bros replay files (.bk2).

For each source `.bk2` in the mario dataset, this script detects scene
boundaries and records one trimmed clip `.bk2` per scene traversal. It
produces **only** the clip `.bk2` files plus a dataset-level
`clips_manifest.tsv`.

The replay sidecars (`_recording.mp4`, `.state`, `_variables.json`,
`_summary.json`) are generated in a separate step by
[`generate_replays`](../generate_replays/), which can run on any directory
of clip `.bk2` files — including clips produced by artificial agents.

## Prerequisites

- The [mario](https://github.com/courtois-neuromod/mario) dataset (provides `.bk2` replays and `stimuli/`)
- Python 3.10+

## Installation

```bash
pip install -r requirements.txt
```

The game ROM must be integrated into stable-retro. See the mario dataset's `stimuli/` directory.

## Usage

```bash
python generate_clips.py \
    -d /path/to/mario \
    -o /path/to/mario.scenes \
    --subjects sub-01 \
    --sessions ses-001 \
    -nj 4 -v
```

| Flag | Description |
|------|-------------|
| `-d, --datapath` | Root of the mario dataset (required) |
| `-o, --output` | Output directory (mario.scenes root, default: `.`) |
| `-sp, --stimuli` | Path to stimuli/ (default: `<datapath>/stimuli`) |
| `--subjects` | Filter by subject(s) |
| `--sessions` | Filter by session(s) |
| `-nj, --n_jobs` | Parallel workers (`-1` = all cores) |
| `-v` | Verbosity (`-v` INFO, `-vv` DEBUG) |

## Output

Clip `.bk2` files are written to a flat `gamelogs/` directory per
subject/session:

```
sub-01/ses-001/gamelogs/
    sub-01_ses-001_task-mario_level-w1l1_scene-0_clip-00100000000122.bk2
```

A `clips_manifest.tsv` is written at the output root. It records, for
every detected clip, the scene-detection metadata that cannot be
recovered from a `.bk2` alone:

| Column | Description |
|--------|-------------|
| `clip_code` | 14-char clip code (`SSSRRBBNNNNNNN`) |
| `clip_file` | Clip `.bk2` path, relative to the output root |
| `sub`, `ses`, `run`, `rep_index` | BIDS entities |
| `level`, `scene_id` | Scene identification |
| `start_frame`, `end_frame`, `duration` | Clip frame range |
| `outcome` | `completed` or `death` |
| `phase` | `discovery` / `practice` (from the source replay) |
| `source_bk2` | Path of the source replay, relative to the mario dataset |

Run [`generate_replays`](../generate_replays/) next to produce the per-clip
sidecars (video, savestate, variables, summary). It reads the
`clips_manifest.tsv` to fold the scene metadata into each `_summary.json`.
