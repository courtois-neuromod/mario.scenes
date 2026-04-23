# archives

Pack and unpack per-session `gamelogs/` directories.

Each session's `gamelogs/` folder contains several thousand files (`.bk2`,
`.mp4`, `.state`, `.json`).  To keep the git-annex object count manageable,
these are bundled into a single uncompressed `.tar` archive per session
(`sub-XX/ses-YYY/gamelogs.tar`) before being tracked by datalad.

## Installation

```bash
pip install -r requirements.txt
```

No non-stdlib dependencies beyond `tqdm` for progress display.

## compress.py

Packs each `gamelogs/` directory into a `gamelogs.tar` archive at the session
level.  Run this **before** pushing data to datalad.

```bash
python compress.py -o /path/to/mario.scenes
```

| Flag | Description |
|------|-------------|
| `-o, --output` | Root of the mario.scenes dataset (default: `.`) |
| `--subjects` | Restrict to these subjects (e.g. `sub-01 sub-02`) |
| `--sessions` | Restrict to these sessions (e.g. `ses-001 ses-002`) |
| `--force` | Overwrite existing archives |
| `--remove-source` | Delete `gamelogs/` after archiving (irreversible — use with care) |
| `-v` | Verbose output (`-v` INFO, `-vv` DEBUG) |

The archive is written atomically: a `.tmp` file is created first and renamed
on success, so an interrupted run never leaves a corrupt archive.

## decompress.py

Extracts `gamelogs.tar` archives back into `gamelogs/` directories.  Run this
after `datalad get` to access individual clip files.

```bash
python decompress.py -o /path/to/mario.scenes
```

| Flag | Description |
|------|-------------|
| `-o, --output` | Root of the mario.scenes dataset (default: `.`) |
| `--subjects` | Restrict to these subjects (e.g. `sub-01 sub-02`) |
| `--sessions` | Restrict to these sessions (e.g. `ses-001 ses-002`) |
| `--force` | Re-extract even if `gamelogs/` already exists |
| `-v` | Verbose output (`-v` INFO, `-vv` DEBUG) |

Sessions whose `gamelogs/` directory already exists are skipped by default.

## Typical workflow

### Producer (data curator)

```bash
# 1. Generate clips as usual
python code/generate_clips/generate_clips.py -d /path/to/mario -o .

# 2. Archive the gamelogs/ directories
python code/archives/compress.py -o . --remove-source

# 3. Track archives with datalad and push
datalad save -m "add gamelogs archives"
datalad push
```

### Consumer (dataset user)

```bash
# 1. Get the archives for sessions of interest
datalad get sub-01/ses-001/gamelogs.tar

# 2. Extract
python code/archives/decompress.py -o . --subjects sub-01 --sessions ses-001

# 3. Access clip files normally
ls sub-01/ses-001/gamelogs/
```
