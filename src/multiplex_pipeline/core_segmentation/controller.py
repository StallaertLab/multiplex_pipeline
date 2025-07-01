from typing import List
from pathlib import Path
import numpy as np
import dask.array as da
import spatialdata as sd
from spatialdata.models import Labels2DModel
from .segmenters import BaseSegmenter
from skimage.transform import resize
import warnings

class SegmentationController:
    def __init__(self, segmenter: BaseSegmenter, resolution_level: int = 0, pyramid_levels: int = 3, downscale: int = 2):
        self.segmenter = segmenter
        self.resolution_level = resolution_level
        self.pyramid_levels = pyramid_levels
        self.downscale = downscale

    def segment_spatial_data(self, sdata_path: Path, channels: List[str], mask_name: str = "mask") -> None:
        
        sdata = sd.read_zarr(sdata_path)

        # Validate resolution level
        for ch in channels:
            img = sdata.images[ch]
            if len(img.items()) <= self.resolution_level:
                raise ValueError(f"Channel '{ch}' does not have resolution level {self.resolution_level}.")

        # Load images from specified resolution level
        images = [np.array(sd.get_pyramid_levels(sdata[ch], n=self.resolution_level)).squeeze() for ch in channels]

        # Run segmenter (expects list of np.ndarrays)
        mask = self.segmenter.run(images)  # Output shape: (H, W)

        # Upscale to level 0 if needed
        if self.resolution_level > 0:
            scale_factor = self.downscale ** self.resolution_level
            new_shape = tuple(dim * scale_factor for dim in mask.shape)
            mask = resize(mask, new_shape, order=0, preserve_range=True, anti_aliasing=False)

            if mask.shape != new_shape:
                warnings.warn(f"Upscaled mask shape {mask.shape} does not match expected {new_shape} — check alignment.")

        # Convert to multiscale image with pyramid levels
        mask_model = Labels2DModel.parse(
            data=mask,
            dims=("y", "x"),
            scale_factors=[self.downscale] * (self.pyramid_levels - 1),
        )

        sdata[mask_name] = mask_model
        sdata.write_element(mask_name)
