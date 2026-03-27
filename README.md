# mario.scenes

A BIDS derivatives dataset containing atomic scene-level clips and annotations extracted from Super Mario Bros gameplay replays, part of the [Courtois NeuroMod](https://www.cneuromod.ca/) project.

Derived from the [mario](https://github.com/courtois-neuromod/mario) dataset.

## Data structure

```
mario.scenes/
├── code/
│   ├── utils.py                              # Shared utilities
│   ├── generate_clips/                       # Scene clip extraction
│   ├── generate_annotations/                 # Scene event annotations
│   └── archives/                             # Pack/unpack gamelogs archives
├── sub-{01..06}/ses-{XXX}/
│   ├── gamelogs.tar                          # Archived scene clips (see below)
│   └── func/
│       └── *_desc-scenes_events.tsv          # Scene event annotations
├── dataset_description.json
└── README.md
```

### gamelogs.tar

Each session's clip files are bundled into a single uncompressed `.tar` archive
(`gamelogs.tar`) to keep the number of tracked files manageable.  Once
extracted, the archive produces a flat `gamelogs/` directory containing one set
of files per scene traversal:

```
sub-01_ses-001_task-mario_level-w1l1_scene-0_clip-00100000000122
```

- `.bk2` — Deterministic replay recorded via stable-retro, starting from the scene entry state
- `_recording.mp4` — Video playback of the clip
- `.state` — Gzipped emulator RAM at the frame of scene entry
- `_summary.json` — Metadata (scene ID, outcome, frame range, source .bk2, etc.)

**To access clip files**, first `datalad get` the archive for your sessions of
interest, then extract with `code/archives/decompress.py` (see
[Accessing the data](#accessing-the-data) below).

### func/

BIDS-compatible event files with scene traversal annotations interleaved with base `gym-retro_game` repetition events:

| Column | Description |
|--------|-------------|
| `trial_type` | `gym-retro_game` (base) or `scene` (traversal) |
| `scene_id` | e.g. `w1l1s3` — scene identifier |
| `onset` | Event onset in seconds |
| `duration` | Event duration in seconds |
| `outcome` | `completed` or `death` |
| `clip_code` | 14-character clip identifier |
| `stim_file` | Relative path to the clip `.bk2` |

## Accessing the data

### 1. Get archives

Use `datalad get` to download the sessions you need:

```bash
# One session
datalad get sub-01/ses-001/gamelogs.tar

# All sessions for a subject
datalad get sub-01/*/gamelogs.tar

# Everything
datalad get */*/gamelogs.tar
```

### 2. Extract clip files

```bash
pip install -r code/archives/requirements.txt

# Extract all downloaded archives
python code/archives/decompress.py -o .

# Or restrict to specific subjects/sessions
python code/archives/decompress.py -o . --subjects sub-01 --sessions ses-001
```

This unpacks each `gamelogs.tar` into the corresponding `gamelogs/` directory.
Sessions that are already extracted are skipped automatically.

Note: paths in `stim_file` columns of the events TSVs point to individual
files inside `gamelogs/` and will only resolve after extraction.

---

## Generating the data

### 1. Extract scene clips

```bash
cd code/generate_clips
pip install -r requirements.txt
python generate_clips.py -d /path/to/mario -o /path/to/mario.scenes -nj 4 -v
```

### 2. Generate scene annotations

Requires `_variables.json` files in the mario dataset (run `mario/code/replays/generate_replays.py` first).

```bash
cd code/generate_annotations
pip install -r requirements.txt
python generate_annotations.py -d /path/to/mario -o /path/to/mario.scenes
```

### 3. Archive clip files for distribution

Pack each session's `gamelogs/` into a `gamelogs.tar` before pushing to datalad:

```bash
pip install -r code/archives/requirements.txt
python code/archives/compress.py -o /path/to/mario.scenes --remove-source
datalad save -m "add gamelogs archives"
datalad push
```

See `code/archives/README.md` for more options.

## Scene annotation schema

Scenes are atomic gameplay segments defined by spatial boundaries within each level. Each of the 74 scenes across worlds 1 and 2 is annotated with 27 binary gameplay features:

**Enemies** (5): Enemy, 2-Horde, 3-Horde, 4-Horde, Gap enemy
**Terrain** (5): Gap, Multiple gaps, Variable gaps, Pillar gap, Valley
**Valleys** (5): Pipe valley, Empty valley, Enemy valley, Roof valley, Roof
**Paths** (2): 2-Path, 3-Path
**Stairs** (5): Stair up, Stair down, Empty stair valley, Enemy stair valley, Gap stair valley
**Other** (5): Risk/Reward, Reward, Moving platform, Flagpole, Beginning, Bonus zone

Scene definitions are hosted on [Zenodo](https://zenodo.org/records/15586709) and auto-downloaded by the scripts.

## References

- Courtois NeuroMod: [cneuromod.ca](https://www.cneuromod.ca/)
- Mario dataset: [github.com/courtois-neuromod/mario](https://github.com/courtois-neuromod/mario)
- Scene metadata: [Zenodo record 15586709](https://zenodo.org/records/15586709)

## License

CC0 — See LICENSE file for details.
