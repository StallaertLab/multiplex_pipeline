# Core Selection

Core selection is the process of choosing regions of interest (ROIs), called “cores”,from a whole multiplex image.  
These cores are saved and later used for **core cutting**. 

Core Selection combines:
-  **Automated detection** via SAM2 
-  **Manual review and editing** via napari

This page documents how to use the `00_core_selection_demo.ipynb` notebook to detect and select cores interactively using the **SAM2 model** and **napari**.

It’s designed reproduce the ROI/core selection process before running core cutting and other downstream analysis.

## Goal

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


