"""Load scene metadata, annotations, and background images for Mario levels."""

import pandas as pd
import os.path as op
import os
import json
from PIL import Image

BASE_DIR = op.dirname(op.dirname(op.dirname(op.dirname(op.abspath(__file__)))))
SOURCEDATA = op.join(BASE_DIR, 'sourcedata')
SCENES_MASTERSHEET = op.join(SOURCEDATA, 'scenes_info', 'scenes_mastersheet.csv')


def load_scenes_info(format='df'):
    """Load scene definitions with start/end positions and layouts.

    Args:
        format: 'df' for DataFrame or 'dict' for {scene_id: {start, end, level_layout}}
    """
    # Check if file exists
    assert op.exists(SCENES_MASTERSHEET), f"File not found: {SCENES_MASTERSHEET}, make sure you run 'invoke get-scenes-data' first."
    
    # Load the data
    scenes_df = pd.read_csv(SCENES_MASTERSHEET)
    if format == 'df':
        return scenes_df
    elif format == 'dict':
        scenes_dict = {}
        for idx, row in scenes_df.iterrows():
            try:
                scene_id = f'w{int(row["World"])}l{int(row["Level"])}s{int(row["Scene"])}'
                scenes_dict[scene_id] = {
                    'start': int(row['Entry point']),
                    'end': int(row['Exit point']),
                    'level_layout': int(row['Layout'])
                }
            except:
                continue
        return scenes_dict
    else:
        raise ValueError('format must be either "df" or "dict"')
    

def load_background_images(level='level'):
    """Load background images as PIL objects.

    Args:
        level: 'level' for full levels or 'scene' for individual scenes

    Returns dict mapping IDs (e.g., 'w1l1') to PIL Image objects.
    """
    # load images
    if level == 'level':
        folder = 'level_backgrounds'
    if level == 'scene':
        folder = 'scene_backgrounds'

    backgrounds = []
    backgrounds_names = []
    for img in sorted(os.listdir(os.path.join(SOURCEDATA, folder))):
        if img.endswith('.png'):
            backgrounds.append(Image.open(os.path.join(SOURCEDATA, folder, img)))
            backgrounds_names.append(img.split('.')[0])
    # create dict
    backgrounds_dict = {}
    for i, name in enumerate(backgrounds_names):
        backgrounds_dict[name] = backgrounds[i]
    return backgrounds_dict



def load_annotation_data():
    """Load 27 binary feature annotations (enemies, gaps, platforms, etc) for all scenes.

    Returns DataFrame indexed by scene_ID with shape (n_scenes, 27).
    """
    # Create the 'scene_ID' column
    df = load_scenes_info(format='df')
    df['scene_ID'] = df.apply(
        lambda row: f"w{int(row['World'])}l{int(row['Level'])}s{int(row['Scene'])}",
        axis=1
    )
    
    # List of feature columns to keep (features and identifying variables)
    feature_cols = [
        'Enemy', '2-Horde', '3-Horde', '4-Horde', 'Roof', 'Gap',
        'Multiple gaps', 'Variable gaps', 'Gap enemy', 'Pillar gap', 'Valley',
        'Pipe valley', 'Empty valley', 'Enemy valley', 'Roof valley', '2-Path',
        '3-Path', 'Risk/Reward', 'Stair up', 'Stair down', 'Empty stair valley',
        'Enemy stair valley', 'Gap stair valley', 'Reward', 'Moving platform',
        'Flagpole', 'Beginning', 'Bonus zone'
    ]
    
    annotations_df = df[feature_cols]
    annotations_df.index = df['scene_ID']

    return annotations_df

def load_reduced_data(method='umap'):
    """Load 2D dimensionality-reduced coordinates.

    Args:
        method: 'umap', 'pca', or 'tsne'

    Returns DataFrame with columns [DR_1, DR_2] indexed by scene_ID.
    """
    fname = op.join(BASE_DIR, 'outputs', 'dimensionality_reduction', f'{method}.csv')
    assert op.exists(fname), f"File not found: {fname}, make sure you run 'invoke dimensionality-reduction' first."
    return pd.read_csv(fname, index_col=0)



