# Usage

This pipeline supports both command-line execution and interactive analysis via Jupyter notebooks.

---

## Command-Line Usage

To run the pipeline from the command line, use the provided scripts:

### Prepare Cores

```bash
python scripts/prepare_cores.py --config config/analysis_pipeline.yaml
```

This script wraps the `CoreController` class, which handles:

* Reading image and metadata files.
* Cutting out cores with appropriate margins and masking.
* Writing intermediate TIFFs.
* Assembling per-core Zarr datasets using the [SpatialData](https://spatialdata.scverse.org/) model.

For more details, see the `core_cutter.py` source and its [configuration](configuration/core-cutting.md).

---

## Jupyter Notebook Usage

For interactive inspection, prototyping, or educational purposes, the following notebooks illustrate how to use the components directly:

* **`core_selection_demo.ipynb`**: An interactive notebook that enables users to define cores as rectangles or polygons using the Napari viewer. It supports automatic core detection via [Segment Anything v2](https://github.com/facebookresearch/sam2), with the option to manually correct the detected shapes or draw new ones from scratch. This step is interactive and only available as a notebook. The result is a `core_info.csv` file containing core metadata for use in subsequent steps via either Jupyter or CLI.
* **`core_cutting_demo.ipynb`**: Demonstrates how to load a single image and metadata entry and apply the core cutting logic.
