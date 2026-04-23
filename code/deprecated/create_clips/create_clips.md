# Mario Scenes Extraction

This repository contains a script to extract specific scenes from the Super Mario Bros game dataset. The script processes `.bk2` game recording files to generate video clips and savestates corresponding to predefined scenes.

This repo is designed to be installed as a submodule of https://github.com/courtois-neuromod/mario_annotations.git, which creates annotations for the cneuromod.mario dataset.

## Prerequisites
Install the [cneuromod.mario](https://github.com/courtois-neuromod/mario) dataset and the related [stimuli]https://github.com/courtois-neuromod/mario.stimuli.
Make sure the mario dataset is on the branch `events`.

## Usage

The `clip_extractor.py` script extracts clips and savestates from the Mario dataset based on scene information provided in the `scenes_mastersheet.csv` file.

### Command-Line Arguments

```bash
python clip_extractor.py [-h]
                         [-d DATAPATH]
                         [-o OUTPUT]
                         [-s SCENESFILE]
                         [-sp STIMULI_PATH]
                         [--filetypes {savestate,ramdump,gif,mp4,webp,json} ...]
                         [-n N_JOBS]
                         [-v]
                         [--subjects SUBJECTS [SUBJECTS ...]]
                         [--sessions SESSIONS [SESSIONS ...]]
```

- `-d`, `--datapath`: Data path to look for `events.tsv` and `.bk2` files. Should be the root of the Mario dataset. Default is the current directory (`.`).
- `-o`, `--output`: Path to the derivatives folder where the outputs will be saved. If not specified, defaults to `<datapath>/derivatives`.
- `-s`, `--scenesfile`: Path to the scenes file (`.csv`) containing information about the start and end positions to clip. If not specified, it defaults to `ressources/scenes_mastersheet.csv`.
- `-sp`, `--stimuli_path`: Path to the stimuli folder containing the game ROMs. Defaults to `<datapath>/stimuli` if not specified.
- `--filetypes`: Files to save. Options are `gif`, `mp4`, etc. Default is to save all possible files.
- `-n`, `--n_jobs`: Number of CPU cores to use for parallel processing. Default is `1`.
- `-v`, `--verbose`: Increase verbosity level. Can be specified multiple times (e.g., `-vv` for more verbosity).
- `--subjects`, `-sub`: List of subjects to process (e.g., `sub-01 sub-02`). If not specified, all subjects are processed.
- `--sessions`, `-ses`: List of sessions to process (e.g., `ses-001 ses-002`). If not specified, all sessions are processed.

### Examples

#### Process All Subjects and Sessions

```bash
python clip_extractor.py -d /path/to/mario/dataset -o /path/to/derivatives -n 4 -vv
```

#### Process Specific Subjects

```bash
python clip_extractor.py -d /path/to/mario/dataset -o /path/to/derivatives -n 4 -vv --subjects sub-01 sub-02
```

#### Process Specific Sessions

```bash
python clip_extractor.py -d /path/to/mario/dataset -o /path/to/derivatives -n 4 -vv --sessions ses-001 ses-002
```

#### Specify a Custom Scenes File

```bash
python clip_extractor.py -d /path/to/mario/dataset -s resources/scenes_mastersheet.csv -o /path/to/derivatives -n 4 -vv
```

#### Specify a Custom Stimuli Path

```bash
python clip_extractor.py -d /path/to/mario/dataset -sp /path/to/stimuli -o /path/to/derivatives -n 4 -vv
```

### Scenes Mastersheet

The `scenes_mastersheet.csv` file, located in the `resources/` folder, contains information about the scenes to extract, including the start and end positions in the game levels.

If you wish to use a custom scenes file, you can specify it using the `-s` or `--scenesfile` argument.

### Output Structure

The script generates a BIDS-compliant dataset under the derivatives folder specified by the `-o` or `--output` argument.
By default, the script will target the derivatives folder in the base (input) mario dataset.

The directory structure is as follows:

```
derivatives/
  mario_scenes/
    dataset_description.json
    processing_log.txt
    sub-<subject>/
      ses-<session>/
        beh/
          clips/
            sub-<subject>_ses-<session>_run-<run>_level-<level>_scene-<scene>_clip-<clipcode>_beh.<ext>
            sub-<subject>_ses-<session>_run-<run>_level-<level>_scene-<scene>_clip-<clipcode>_beh.json
          savestates/
            sub-<subject>_ses-<session>_run-<run>_level-<level>_scene-<scene>_clip-<clipcode>_beh.state
            sub-<subject>_ses-<session>_run-<run>_level-<level>_scene-<scene>_clip-<clipcode>_ramdump.npz

```

- **Clips**: Video files of the extracted scenes, saved in the format specified by `--clip_extension` (e.g., `.gif`, `.mp4`).
- **Savestates**: Game savestates corresponding to the start of each clip.
- **Metadata**: JSON sidecar files containing metadata for each clip.

### Logs and Metadata

- `dataset_description.json`: Contains metadata about the dataset, following the BIDS derivatives specification.
- `processing_log.txt`: Contains logs of the processing, including the number of files processed, skipped, and any errors encountered.


## Acknowledgements

- This script uses the [Gym Retro](https://github.com/openai/retro) library for replaying game recordings.
- The BIDS standard is used for organizing the output dataset.
- This project was developed as part of the [Courtois Neuromod project](https://www.cneuromod.ca/)