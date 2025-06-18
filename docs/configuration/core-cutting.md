# Core Cutting Configuration

Core cutting supports two modes of operation:

* **Local mode**: Input files are available on the local filesystem.
* **Globus mode**: Input files are accessed remotely using Globus endpoints.

---

## Shared Configuration

These fields are common to both modes:

```yaml
temp_dir: "C:/path_to_temporary_directory"
output_dir: "C:/path_to_output_directory"
core_info: "C:/path_to/core_metadata.csv"
include_channels: ['002_DAPI']
exclude_channels: ['005_pRB']
use_channels: ['DAPI','CD3']
margin: 10
mask_value: 0
max_pyramid_levels: 3

transfer_cleanup_enabled: true
core_cleanup_enabled: true
```

| Key                        | Type        | Description                                                             |
| -------------------------- | ----------- | ----------------------------------------------------------------------- |
| `temp_dir`                 | `str`       | Temporary folder to store extracted TIFFs for each core                 |
| `output_dir`               | `str`       | Final destination for SpatialData (Zarr) outputs                        |
| `core_info`                | `str`       | Path to the CSV file containing core cutting metadata                   |
| `include_channels`         | `list[str]` | Optional list of full channel names to include (e.g., `001_DAPI`)       |
| `exclude_channels`         | `list[str]` | Optional list of full channel names to exclude                          |
| `use_channels`             | `list[str]` | Optional list to restrict final base marker names (e.g., `DAPI`, `CD3`) |
| `margin`                   | `int`       | Number of pixels to pad around each bounding box when cutting cores     |
| `mask_value`               | `int`       | Value used to fill background for polygonal core masks                  |
| `max_pyramid_levels`       | `int`       | Number of downsampled pyramid levels (0 means no pyramid)               |
| `transfer_cleanup_enabled` | `bool`      | Whether to delete temporary files downloaded via Globus after the run   |
| `core_cleanup_enabled`     | `bool`      | Whether to delete TIFFs from `temp_dir` after core assembly             |

* For detailed explanation of `include_channels`, `exclude_channels`, and `use_channels`, see the [channel selection logic](channel-selection.md).

---

## Sourcing Image Files

Differences in local vs. Globus mode come from how OME-TIFF files are sourced.

### 📰 Local Mode

In local mode, `image_dir` should point to a folder with accessible OME-TIFF files on disk.

```yaml
image_dir: "C:/path_to_image_directory"
```

* On Windows, use **forward slashes `/`** in paths: `C:/path/to/folder`. This avoids issues with escape characters.

---

### ☁️ Globus Mode

In Globus mode, you must also specify the path to a Globus configuration directory. The `image_dir` field should reflect the remote directory on the Globus endpoint:

```yaml
image_dir: "/my_globus/path_to_image_directory"
globus_config: "C:/path_to_globus_config_directory"
```

This allows the pipeline to discover and transfer files on demand via Globus.
