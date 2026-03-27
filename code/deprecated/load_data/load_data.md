## Data loading functions

### Examples


#### Load a dataframe and curate it to retain only annotations (e.g. for scenes analysis)
```
from mario_scenes.load_data import load_scenes_info, curate_dataframe  
scenes_df = load_scenes_info(format='df')  
scenes_df = curate_dataframe(scenes_df)
```

#### Load a dict with entry and exit points, without annotations (e.g. for scene-clips extraction)
```
from mario_scenes.load_data import load_scenes_info
scenes_dict = load_scenes_info(format='dict')
```