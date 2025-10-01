# Core Selection

Core selection is the process of choosing regions of interest (ROIs), called “cores” , from a whole multiplex image.  
These cores are saved and later used for **core cutting**. 

Core Selection combines:
-  **Automated detection** via SAM2 
-  **Manual review and editing** via napari

This page documents how to use the `00_core_selection_demo.ipynb` notebook to detect and select cores interactively using the **SAM2 model** and **napari**.

It’s designed reproduce the ROI/core selection process before running core cutting and other downstream analysis.

Goal: 

1. Launching the notebook and loading configurations  
2. Running SAM2 to suggest candidate cores  
3. Refining ROIs in napari (rectangles or polygons)  
4. Saving the final selected cores for downstream processing  

## Prerequisites

Before you start, ensure :
-  SAM2 model is installed and configured  
-  Multiplex Pipeline is installed and accessible  
-  Analysis and workstation config files exist (`pipeline_settings.yaml`, `analysis_settings.yaml`)

## Input Files

| File | Description |
|------|--------------|
| `pipeline_settings.yaml` | Defines SAM2 model path and Python environment |
| `analysis_settings.yaml` | Defines directories, detection params, and image name |

**Example input snippet:**

      core_detection:
      im_level: 6
      min_area: 2000
      max_area: 10000
      min_iou: 0.8
      min_st: 0.9
      min_int: 15
      frame: 4

## Step by Step Guide
- > *For the full workflow, see `00_core_selection_demo.ipynb`.*

## 1. Load Dependencies

Enable autoreload to automatically refresh imports when changes are made, and import all required libraries.

These include:
- **Standard Python modules**: `os`, `pickle`, `subprocess`  
- **Visualization**: `napari`  
- **GUI utilities**: `magicgui`  
- **Multiplex Pipeline utilities**: core selection, image handling, ROI processing, and configuration management

```python
%load_ext autoreload
%autoreload 2

import os
import pickle as pkl
import subprocess

import napari
from magicgui import magicgui

from multiplex_pipeline.core_selection.viewer_utils import (
    display_saved_rois,
    redo_bbox_layer,
    redo_cores_layer,
    save_rois_from_viewer,
)
from multiplex_pipeline.im_utils import get_org_im_shape, prepare_rgb_image
from multiplex_pipeline.roi_utils import (
    get_refined_rectangles,
    get_visual_rectangles,
    prepare_poly_df_for_saving,
    xywh_to_corners,
)
from multiplex_pipeline.utils.utils import (
    change_to_wsl_path,
    load_analysis_settings,
    load_workstation_config,
)
```

---

### 2. Load Configuration Files

Load both pipeline and analysis configuration files:

```python
config = load_workstation_config("path/to/pipeline_settings.yaml")
settings = load_analysis_settings("path/to/analysis_settings.yaml")
```

These files define:
- SAM2 environment and model path  
- Image directories and detection image name  
- Core detection parameters (e.g. `min_area`, `min_iou`, etc.)  
- ROI output location


Prepare the Detection Image

Combine the directory and image name from your configuration, then prepare it for display:

```python
from multiplex_pipeline.im_utils import get_org_im_shape, prepare_rgb_image

image_path = os.path.join(settings["image_dir"], settings["detection_image"])
org_im_shape = get_org_im_shape(image_path)
im_rgb = prepare_rgb_image(image_path, req_level=im_level)
```
- The image is automatically downsampled for efficient rendering

### 4. Launch napari Viewer and Add Interactive Controls

Initialize a `napari` viewer to display the detection image, visualize any previously saved ROIs, and add interactive buttons for managing selections.
Your napari viewer displays:
- The detection image  
- Any previously saved ROIs
```python
import napari
from magicgui import magicgui
from multiplex_pipeline.core_selection.viewer_utils import (
    display_saved_rois,
    save_rois_from_viewer,
)
from multiplex_pipeline.roi_utils import xywh_to_corners

# Launch viewer and add base image
viewer = napari.Viewer()
viewer.add_image(im_rgb)

# Add a white frame outlining the full image
frame_rect = xywh_to_corners([0, 0, im_rgb.shape[1], im_rgb.shape[0]], frame=0)
viewer.add_shapes(
    frame_rect,
    edge_color="white",
    face_color="transparent",
    shape_type="rectangle",
    edge_width=2,
    name="frame"
)

# Display any previously saved ROIs
display_saved_rois(viewer, IM_LEVEL=im_level, save_path=save_path)
```
GUI buttons:

- **Display Saved Cores** reloads ROIs saved in prior sessions  
- **Save Cores**  writes current ROIs to disk

  
```python  
# Add interactive buttons
@magicgui(call_button="Display Saved Cores")
def display_button(viewer: napari.Viewer):
    display_saved_rois(viewer, IM_LEVEL=im_level, save_path=save_path)

@magicgui(call_button="Save Cores")
def save_button(viewer: napari.Viewer):
    save_rois_from_viewer(viewer, org_im_shape=org_im_shape, req_level=im_level, save_path=save_path)

viewer.window.add_dock_widget(display_button, area="left")
viewer.window.add_dock_widget(save_button, area="left")
```


### 5. Generate Core Suggestions 

Use SAM2 segmentation model to generate an initial set of ROI proposals based on image content
  - These suggested “cores” can be refined interactively in the following steps

```python
from multiplex_pipeline.utils.utils import change_to_wsl_path
import subprocess, pickle as pkl, os

im_wsl = change_to_wsl_path(image_path)
out_win = os.path.join(os.getcwd(), "masks.pkl")
out_wsl = change_to_wsl_path(out_win)

command = [
    "wsl",
    config["sam_env"],
    "-m", "multiplex_pipeline.core_selection.suggest_cores",
    im_wsl, str(im_level), config["model_path"], out_wsl
]

process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
```

*Processing may take 1–2 minutes. Output will appear in the cell once SAM2 completes*

Load the generated masks:
```python
masks = pkl.load(open(out_win, "rb"))
os.remove(out_win)
```
### 6. Refine and Visualize Core Suggestions

Filter candidate ROIs by geometric and intensity thresholds:

```python
from multiplex_pipeline.roi_utils import get_refined_rectangles

rect_list = get_refined_rectangles(
    masks,
    im=im_rgb[:, :, 0],
    frame=settings["core_detection"]["frame"],
    min_area=settings["core_detection"]["min_area"],
    max_area=settings["core_detection"]["max_area"],
    min_iou=settings["core_detection"]["min_iou"],
    min_stability=settings["core_detection"]["min_st"],
    min_int=settings["core_detection"]["min_int"]
)
```

Visualize refined ROIs:

```python
from multiplex_pipeline.roi_utils import prepare_poly_df_for_saving, get_visual_rectangles
from multiplex_pipeline.core_selection.viewer_utils import redo_cores_layer, redo_bbox_layer

df = prepare_poly_df_for_saving(rect_list, ["rectangle"] * len(rect_list), org_im_shape, im_level)
rect_list = get_visual_rectangles(df, im_level)

redo_cores_layer(viewer, rect_list, shape_type=df.poly_type.to_list())
redo_bbox_layer(viewer, rect_list, df["core_name"].tolist())
```

- You can now inspect, adjust, or delete ROIs interactively in napari.

### 7. Save Final ROIs

When satisfied with selections, click **Save Cores** in the napari sidebar.  
This exports ROI metadata and geometry for downstream processing.

**Expected outputs:**
| File | Description |
|------|--------------|
| `cores.csv` | ROI metadata (name, coordinates) |
| `core_polygons.pkl` | Serialized polygons |

## ROI Selection and Editing

| ROI Type | Description | Use Case |
|-----------|--------------|----------|
| Rectangle | Simple box ROI | Uniform, grid cores |
| Polygon | Arbitrary ROI | Non-rectangular cores or irregular regions |

Use the **Shapes Layer toolbar** in napari to add, delete, modify ROIs
 
- [Napari Shapes Tools Guide](https://napari.org/stable/howtos/layers/shapes.html)








