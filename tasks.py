"""
Mario Scenes - Invoke Tasks

This module defines reproducible workflow tasks for the mario.scenes pipeline,
following airoh pipeline conventions for task organization and documentation.

Available Tasks:
    Setup & Configuration:
        - setup-env: Create virtual environment and install dependencies
        - setup-env-on-hpc: HPC-specific environment setup for computing clusters
        - setup-mario-dataset: Download and configure Mario dataset via datalad
        - get-scenes-data: Download scene metadata and background images from Zenodo

    Analysis & Processing:
        - dimensionality-reduction: Apply PCA, UMAP, and t-SNE to scene annotations
        - cluster-scenes: Perform hierarchical clustering on scene features
        - create-clips: Extract video clips from replay files for each scene
        - make-scene-images: Generate background images for levels and scenes

    Workflows:
        - full-pipeline: Execute complete analysis workflow
"""

from invoke import task
import os
import os.path as op

# Import airoh utility tasks
#from airoh.utils import setup_env_python, ensure_dir_exist, clean_folder # not used in this script and creat a error on HPC
#from airoh.datalad import get_data # same as previous line

BASE_DIR = op.dirname(op.abspath(__file__))


# ===============================
# 📦 Setup & Configuration
# ===============================


@task
def setup_env(c):
    """🔧 Set up Python virtual environment and install dependencies.

    Creates a virtual environment in ./env/ and installs all required packages
    from requirements.txt, then installs the mario_scenes package in editable mode.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.

    Examples
    --------
    ```bash
    invoke setup-env
    ```

    Notes
    -----
    This task creates a new virtual environment from scratch. If you need to update
    dependencies in an existing environment, activate it manually and run pip install.
    """
    c.run(
        f"python -m venv {BASE_DIR}/env && "
        f"source {BASE_DIR}/env/bin/activate && "
        "pip install -e ."
    )

@task
def setup_env_on_hpc(c):
    """🖥️ Set up environment on HPC computing clusters.

    Configures the environment on HPC systems with specific Python module loading
    and stable-retro installation from source. This task is tailored for HPC
    environments with module systems (e.g., Compute Canada, SLURM clusters).

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.

    Examples
    --------
    ```bash
    invoke setup-env-on-hpc
    ```

    Notes
    -----
    This task assumes you're running on an HPC cluster with access to the
    module system and git repositories.
    """
    c.run("module load StdEnv/2023 gcc/12.3 python/3.10 opencv/4.10.0 qt/6.5.3 && "
        f"python -m venv {BASE_DIR}/env && "
        "cd env/lib/python3.10/site-packages && "
        "git clone git@github.com:farama-foundation/stable-retro && "
        "git clone git@github.com:courtois-neuromod/videogames_utils.git &&"
        "cd ../../../.. && "
        "source env/bin/activate && "
        "pip install -e env/lib/python3.10/site-packages/stable-retro/. &&"
        "pip install -e env/lib/python3.10/site-packages/videogames_utils/. &&"
        "pip install -r requirements_hpc.txt && "
        "pip install -e . &&"
    )


@task
def setup_mario_dataset(c):
    """📥 Download and configure the Mario dataset using datalad.

    Installs the Courtois NeuroMod Mario dataset including:
    - All .bk2 replay files
    - Event timing .tsv files
    - Game ROM stimuli

    The dataset is installed into sourcedata/mario/ following BIDS conventions.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.

    Examples
    --------
    ```bash
    invoke setup-mario-dataset
    ```

    Notes
    -----
    Requires datalad to be installed and SSH access to the Courtois NeuroMod
    repositories. The dataset is large (~several GB) and may take time to download.
    """
    command = (
        f"source {BASE_DIR}/env/bin/activate && "
        "mkdir -p sourcedata && "
        "cd sourcedata && "
        "datalad install git@github.com:courtois-neuromod/mario && "
        "cd mario && "
        "git checkout events && "
        "datalad get */*/*/*.bk2 && "
        "datalad get */*/*/*.tsv && "
        "rm -rf stimuli && "
        "datalad install git@github.com:courtois-neuromod/mario.stimuli && "
        "mv mario.stimuli stimuli && "
        "cd stimuli && "
        "git checkout scenes_states && "
        "datalad get ."
    )
    c.run(command)


@task
def get_scenes_data(c):
    """📊 Download scene metadata and background images from Zenodo.

    Downloads and extracts:
    - scenes_mastersheet.csv: Scene boundary definitions and layouts
    - scenes_mastersheet.json: Same data in JSON format
    - mario_scenes_manual_annotation.pdf: Annotation documentation
    - level_backgrounds/: Background images for each level
    - scene_backgrounds/: Background images for individual scenes

    All files are saved to sourcedata/scenes_info/ and sourcedata/*_backgrounds/.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.

    Examples
    --------
    ```bash
    invoke get-scenes-data
    ```

    Notes
    -----
    This must be run before other analysis tasks that depend on scene definitions.
    Downloads from Zenodo record 15586709.
    """
    c.run("mkdir -p sourcedata/scenes_info")
    c.run(
        'wget "https://zenodo.org/records/15586709/files/mario_scenes_manual_annotation.pdf?download=1" -O sourcedata/scenes_info/mario_scenes_manual_annotation.pdf'
    )
    c.run(
        'wget "https://zenodo.org/records/15586709/files/scenes_mastersheet.json?download=1" -O sourcedata/scenes_info/scenes_mastersheet.json'
    )
    c.run(
        'wget "https://zenodo.org/records/15586709/files/scenes_mastersheet.csv?download=1" -O sourcedata/scenes_info/scenes_mastersheet.csv'
    )
    c.run(
        'wget "https://zenodo.org/records/15586709/files/scene_backgrounds.tar.gz?download=1" -O sourcedata/scene_backgrounds.tar.gz'
    )
    c.run(
        'wget "https://zenodo.org/records/15586709/files/level_backgrounds.tar.gz?download=1" -O sourcedata/level_backgrounds.tar.gz'
    )
    c.run("tar -xvf sourcedata/scene_backgrounds.tar.gz -C sourcedata/")
    c.run("tar -xvf sourcedata/level_backgrounds.tar.gz -C sourcedata/")
    c.run("rm sourcedata/scene_backgrounds.tar.gz")
    c.run("rm sourcedata/level_backgrounds.tar.gz")


# ===============================
# 🔬 Analysis & Processing
# ===============================


@task
def dimensionality_reduction(c):
    """📉 Apply dimensionality reduction to scene annotation features.

    Reduces the 27-dimensional scene annotation feature space to 2D using three methods:
    - PCA (Principal Component Analysis)
    - UMAP (Uniform Manifold Approximation and Projection)
    - t-SNE (t-distributed Stochastic Neighbor Embedding)

    Results are saved as CSV files in outputs/dimensionality_reduction/.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.

    Examples
    --------
    ```bash
    invoke dimensionality-reduction
    ```

    Notes
    -----
    Requires scene annotations to be available (run get-scenes-data first).
    Output files: pca.csv, umap.csv, tsne.csv
    """
    c.run(
        f"python {BASE_DIR}/code/mario_scenes/scenes_analysis/dimensionality_reduction.py"
    )


@task
def cluster_scenes(c, n_clusters=None):
    """🗂️ Perform hierarchical clustering on scene features.

    Groups scenes into clusters based on their annotation features using
    hierarchical clustering with Ward linkage. Generates cluster assignments
    and summary statistics for multiple cluster counts (default: 5-30).

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.
    n_clusters : str, optional
        Space-separated list of cluster counts to generate (e.g., "5 10 15 20").
        If not provided, generates clusters for all values from 5 to 30.

    Examples
    --------
    ```bash
    invoke cluster-scenes
    invoke cluster-scenes --n-clusters "10 15 20"
    ```

    Notes
    -----
    Output saved to outputs/cluster_scenes/hierarchical_clusters.pkl
    """
    if n_clusters is None:
        cluster_arg = " ".join(str(i) for i in range(5, 31))
    else:
        cluster_arg = n_clusters
    c.run(
        f"python {BASE_DIR}/code/mario_scenes/scenes_analysis/cluster_scenes.py --n_clusters {cluster_arg}"
    )


@task
def create_clips(
    c,
    datapath=None,
    output=None,
    subjects=None,
    sessions=None,
    n_jobs=None,
    save_videos=None,
    save_variables=None,
    save_states=None,
    save_ramdumps=None,
    video_format=None,
    replays_path=None,
    stimuli=None,
    verbose=None,
):
    """🎬 Extract scene clips from Mario replay files.

    Processes .bk2 replay files to identify and extract individual scene clips,
    saving them as video files, savestates, ramdumps, or variables. Uses parallel
    processing for efficiency.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.
    datapath : str, optional
        Path to Mario dataset root directory. Defaults to mario_dataset from invoke.yaml.
    output : str, optional
        Path for output derivatives. Defaults to output_dir from invoke.yaml.
    subjects : str, optional
        Space-separated subject IDs to process (e.g., "sub-01 sub-02").
        If None, processes all subjects.
    sessions : str, optional
        Space-separated session IDs to process (e.g., "ses-001 ses-002").
        If None, processes all sessions.
    n_jobs : int, optional
        Number of parallel jobs. Defaults to n_jobs from invoke.yaml.
    save_videos : bool, optional
        Whether to save video files. Defaults to save_videos from invoke.yaml.
    save_variables : bool, optional
        Whether to save game variables as JSON. Defaults to save_variables from invoke.yaml.
    save_states : bool, optional
        Whether to save savestates (gzipped RAM at clip start). Defaults to save_states from invoke.yaml.
    save_ramdumps : bool, optional
        Whether to save full RAM dumps per frame. Defaults to save_ramdumps from invoke.yaml.
    video_format : str, optional
        Video format to save: "mp4", "gif", or "webp". Defaults to video_format from invoke.yaml.
    replays_path : str, optional
        Path to mario.replays output to enrich metadata with replay-level info.
        Defaults to replays_path from invoke.yaml (optional).
    stimuli : str, optional
        Path to stimuli folder containing game ROMs. Defaults to stimuli_path from invoke.yaml.
    verbose : int, optional
        Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG). Defaults to verbose from invoke.yaml.

    Examples
    --------
    ```bash
    invoke create-clips
    invoke create-clips --subjects "sub-01 sub-02" --video-format gif
    invoke create-clips --save-states --save-variables --replays-path outputdata/replays
    invoke create-clips --n-jobs 4
    ```

    Notes
    -----
    Output follows BIDS structure: sub-{sub}/ses-{ses}/beh/
    Generates JSON sidecars with metadata for each clip.

    For enriched metadata, first run mario.replays:
        cd ../mario.replays && invoke create-replays --save-variables
    Then pass the output path via --replays-path to include replay-level
    statistics (score gained, enemies killed, etc.) in clip metadata.
    """
    # Resolve paths from configuration or arguments
    if datapath is None:
        datapath = c.config.get("mario_dataset", "sourcedata/mario")

    if output is None:
        output = c.config.get("output_dir", "outputdata/clips")

    if stimuli is None:
        stimuli = c.config.get("stimuli_path", None)

    if replays_path is None:
        replays_path = c.config.get("replays_path", None)

    if n_jobs is None:
        n_jobs = c.config.get("n_jobs", -1)

    # Resolve boolean/string flags from config if not explicitly set via CLI
    if save_videos is None:
        save_videos = c.config.get("save_videos", True)

    if save_variables is None:
        save_variables = c.config.get("save_variables", False)

    if save_states is None:
        save_states = c.config.get("save_states", False)

    if save_ramdumps is None:
        save_ramdumps = c.config.get("save_ramdumps", False)

    if video_format is None:
        video_format = c.config.get("video_format", "mp4")

    if verbose is None:
        verbose = c.config.get("verbose", 0)

    cmd = f"python {BASE_DIR}/code/mario_scenes/create_clips/create_clips.py -d {datapath} -o {output} -nj {n_jobs}"
    if save_videos:
        cmd += " --save_videos"
    if save_variables:
        cmd += " --save_variables"
    if save_states:
        cmd += " --save_states"
    if save_ramdumps:
        cmd += " --save_ramdumps"
    cmd += f" --video_format {video_format}"
    if subjects:
        cmd += f" --subjects {subjects}"
    if sessions:
        cmd += f" --sessions {sessions}"
    if replays_path:
        cmd += f" --replays_path {replays_path}"
    if stimuli:
        cmd += f" --stimuli {stimuli}"
    if verbose > 0:
        cmd += " " + "-v" * verbose
    c.run(cmd)


@task
def make_scene_images(c, data_path=None, subjects="all", level=None):
    """🖼️ Generate background images for levels and scenes.

    Processes replay files to create canonical background images by averaging
    pixel columns across multiple playthroughs. Generates both full level
    backgrounds and individual scene backgrounds.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.
    data_path : str, optional
        Path to Mario dataset. If None, uses sourcedata/mario.
    subjects : str, optional
        Subjects to include in averaging. Default: "all"
    level : str, optional
        Specific level to process (e.g., "w1l1"). If None, processes all levels.

    Examples
    --------
    ```bash
    invoke make-scene-images
    invoke make-scene-images --level w1l1 --subjects sub-03
    ```

    Notes
    -----
    Outputs saved to sourcedata/level_backgrounds/ and sourcedata/scene_backgrounds/
    """
    cmd = f"python {BASE_DIR}/code/mario_scenes/make_images/make_scene_img.py"
    if data_path:
        cmd += f" -d {data_path}"
    if subjects != "all":
        cmd += f" -s {subjects}"
    if level:
        cmd += f" -l {level}"
    c.run(cmd)


# ===============================
# 🔄 Workflows
# ===============================


@task
def full_pipeline(c):
    """🚀 Execute the complete mario.scenes analysis pipeline.

    Runs all processing steps in sequence:
    1. Download scene data and metadata (get-scenes-data)
    2. Apply dimensionality reduction to features (dimensionality-reduction)
    3. Generate hierarchical clusters (cluster-scenes)
    4. Extract scene clips from replays (create-clips)

    Parameters
    ----------
    c : invoke.Context
        The Invoke context object.

    Examples
    --------
    ```bash
    invoke full-pipeline
    ```

    Notes
    -----
    Assumes the Mario dataset is already downloaded (setup-mario-dataset).
    This workflow may take several hours depending on dataset size and hardware.
    """
    c.run("invoke get-scenes-data dimensionality-reduction cluster-scenes create-clips")
