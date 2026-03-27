# generate_clips

Extract scene-level clips from Super Mario Bros replay files (.bk2).

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
| `--skip_variables` | Skip generating per-clip `_variables.json` files |
| `-nj, --n_jobs` | Parallel workers (`-1` = all cores) |
| `-v` | Verbosity (`-v` INFO, `-vv` DEBUG) |

## Output

Files are written to a flat `gamelogs/` directory per subject/session:

```
sub-01/ses-001/gamelogs/
    sub-01_ses-001_task-mario_level-w1l1_scene-0_clip-00100000000122.bk2
    sub-01_ses-001_task-mario_level-w1l1_scene-0_clip-00100000000122_recording.mp4
    sub-01_ses-001_task-mario_level-w1l1_scene-0_clip-00100000000122.state
    sub-01_ses-001_task-mario_level-w1l1_scene-0_clip-00100000000122_summary.json
    sub-01_ses-001_task-mario_level-w1l1_scene-0_clip-00100000000122_variables.json
```

- `_variables.json` — Frame-by-frame game variables sliced to the clip range
- `.bk2` — Deterministic replay recorded via stable-retro
- `_recording.mp4` — Video playback of the clip
- `.state` — Gzipped emulator RAM at scene entry
- `_summary.json` — Metadata (scene ID, outcome, frame range, etc.)
