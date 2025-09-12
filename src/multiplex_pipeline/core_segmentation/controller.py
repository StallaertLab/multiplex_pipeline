from typing import List, Union, Optional
from pathlib import Path
import numpy as np
import dask.array as da
import spatialdata as sd
from spatialdata.models import Labels2DModel
from .segmenters import BaseSegmenter
from .cleaners import DbscanEllipseCleaner
from skimage.transform import resize
import warnings
from loguru import logger

class SegmentationController:
    
    def __init__(
        self,
        segmenter: BaseSegmenter,
        resolution_level: int = 0,
        pyramid_levels: int = 3,
        downscale: int = 2,
        cleaner: Optional[object] = None
    ):
        self.segmenter = segmenter
        self.resolution_level = resolution_level
        self.pyramid_levels = pyramid_levels
        self.downscale = downscale
        self.cleaner = cleaner
        logger.info(
            f"Initialized SegmentationController with segmenter={segmenter}, "
            f"resolution_level={resolution_level}, pyramid_levels={pyramid_levels}, downscale={downscale}, "
            f"cleaner={cleaner}"
        )

    def segment_spatial_data(
        self,
        sdata_path: Path,
        channels: List[str],
        mask_name: Union[str, List[str]] = "mask",
    ) -> None:
        sdata = sd.read_zarr(sdata_path)
        
        sdata = sd.read_zarr(sdata_path)

        # Validate requested channels exist
        missing = [ch for ch in channels if ch not in sdata.images]
        if missing:
            logger.error(f"Requested channels not found in sdata: {missing}")
            raise ValueError(f"Requested channels not found in sdata: {missing}")
        logger.info("All requested channels are present.")

        # Validate resolution level
        for ch in channels:
            img = sdata.images[ch]
            if len(img.items()) <= self.resolution_level:
                logger.error(f"Channel '{ch}' does not have resolution level {self.resolution_level}.")
                raise ValueError(f"Channel '{ch}' does not have resolution level {self.resolution_level}.")
        logger.info(f"All channels have required resolution level: {self.resolution_level}")

        # Load images from specified resolution level
        images = [np.array(sd.get_pyramid_levels(sdata[ch], n=self.resolution_level)).squeeze() for ch in channels]

        # Run segmenter (expects list of np.ndarrays)
        logger.info(f"Running segmentation on {len(images)} images at resolution level {self.resolution_level}.")
        masks = self.segmenter.run(images) 

        # Ensure masks is a list for consistent handling
        if not isinstance(masks, (list, tuple)):
            masks = [masks]

        # cleaning if a cleaner is provided
        if self.cleaner:  
            logger.info("Cleaning masks using the provided cleaner.")  
            masks[0] = self.cleaner.run(masks[0])
            if len(masks) > 1:
                masks[1:] = [self.cleaner.clean_with_mask(mask) for mask in masks[1:]]

        # Handle mask names
        if isinstance(mask_name, list):
            if len(mask_name) != len(masks):
                logger.warning("Provided mask_name list does not match the number of masks; auto-generating mask names.")
                mask_names = [f"mask_{i:02d}" for i in range(len(masks))]
            else:
                mask_names = mask_name
        elif len(masks) == 1:
            mask_names = [mask_name if isinstance(mask_name, str) else "mask"]
        else:
            mask_names = [f"mask_{i:02d}" for i in range(len(masks))]
            logger.info(f"Multiple masks detected, auto-generated mask names: {mask_names}")

        # Check for existing mask names
        for name in mask_names:
            if name in sdata:
                logger.error(f"Mask name '{name}' already exists in sdata. Please provide unique mask names.")
                raise ValueError(
                    f"Mask name '{name}' already exists in sdata. Please provide unique mask names."
                )

        # Process and save each mask
        for mask, name in zip(masks, mask_names):
            # Upscale to level 0 if needed
            if self.resolution_level > 0:
                scale_factor = self.downscale ** self.resolution_level
                new_shape = tuple(dim * scale_factor for dim in mask.shape)
                mask = resize(mask, new_shape, order=0, preserve_range=True, anti_aliasing=False)
            # Convert to multiscale image with pyramid levels
            mask_model = Labels2DModel.parse(
                data=mask,
                dims=("y", "x"),
                scale_factors=[self.downscale] * (self.pyramid_levels - 1),
            )
            try:
                sdata[name] = mask_model
                sdata.write_element(name)
                logger.success(f"Segmentation complete. Mask '{name}' written to {sdata_path}")
            except Exception as e:
                logger.error(f"Failed to write mask '{name}': {e}")
                raise