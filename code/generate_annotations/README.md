# generate_annotations

Generate scene-level event annotations for the Mario dataset.

## Prerequisites

- The [mario](https://github.com/courtois-neuromod/mario) dataset with replay variables
  (`gamelogs/*_variables.json` — run `mario/code/replays/generate_replays.py` first)
- Python 3.10+

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python generate_annotations.py \
    -d /path/to/mario \
    -o /path/to/mario.scenes \
    --subjects sub-01 \
    --sessions ses-001
```

| Flag | Description |
|------|-------------|
| `-d, --datapath` | Root of the mario dataset (required) |
| `-o, --output_path` | Output directory (mario.scenes root, default: `.`) |
| `--subjects` | Filter by subject(s) |
| `--sessions` | Filter by session(s) |

## Output

```
sub-01/ses-001/func/
    sub-01_ses-001_task-mario_run-001_desc-scenes_events.tsv
```

Each TSV contains the base `gym-retro_game` rows from the source events file,
interleaved with `scene` rows for each detected scene traversal.

| Column | Description |
|--------|-------------|
| `trial_type` | `gym-retro_game` or `scene` |
| `scene_id` | Scene identifier, e.g. `w1l1s3` (scene rows only) |
| `onset` | Event onset in seconds |
| `duration` | Event duration in seconds |
| `frame_start` | Start frame index |
| `frame_stop` | End frame index |
| `phase` | `discovery` or `practice` |
| `rep_index` | Repetition index (0-based) |
| `stim_file` | Relative path to the clip `.bk2` (scene rows only) |
| `clip_code` | 14-character clip identifier (scene rows only) |
| `outcome` | `completed` or `death` (scene rows only) |
