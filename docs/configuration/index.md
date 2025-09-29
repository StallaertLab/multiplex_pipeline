# 🔧 Configuration

You can explore parameters related to different parts of the pipeline:

- [Core cutting](core-cutting.md)
- Understand [channel selection logic](channel-selection.md) for core cutting


---

### Full configuration file example:

```yaml

# core cutting

image_dir: "C:/path_to_image_directory"
temp_dir: "C:/path_to_temporary_directory"
output_dir: "C:/path_to_output_directory"
core_info: "C:/path_to/core_metadata.csv"

include_channels: ['002_DAPI']
exclude_channels: ['005_pRB']
use_channels: ['DAPI','CD3']

margin: 10
mask_value: 0
max_pyramid_levels: 3

transfer_cleanup_enabled: True
core_cleanup_enabled: True
```
