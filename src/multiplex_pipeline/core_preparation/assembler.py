import os
import numpy as np
import tifffile
from loguru import logger

from spatialdata import SpatialData
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity

class CoreAssembler:
    def __init__(self, temp_dir, output_dir,
                 max_pyramid_levels=4,
                 downscale=2,
                 allowed_channels=None,
                 cleanup: bool = False):
        """
        temp_dir: base directory containing temp core folders
        output_dir: where final .zarr SpatialData will be written
        max_pyramid_levels: max number of downsampling levels (0 = no pyramid)
        downscale: downscaling factor per level (typically 2)
        allowed_channels: optional list of channel names to process (e.g., ['pRB', '009_DAPI'])
        cleanup: if True, delete per-channel TIFFs after assembling
        """
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.max_pyramid_levels = max_pyramid_levels
        self.downscale = downscale
        self.allowed_channels = allowed_channels
        self.cleanup = cleanup


    def assemble_core(self, core_id):
        """
        Build one SpatialData object from per-channel TIFFs.
        Each channel is a multiscale image entry in the SpatialData.images.
        """
        core_path = os.path.join(self.temp_dir, core_id)
        if not os.path.exists(core_path):
            raise FileNotFoundError(f"No temp folder found for core: {core_id}")

        # Collect and sort TIFFs
        channel_files = sorted([
            f for f in os.listdir(core_path)
            if f.lower().endswith(".tiff") or f.lower().endswith(".tif")
        ])
        if not channel_files:
            raise ValueError(f"No TIFFs found for core: {core_id}")

        images = {}
        used_channels = []

        for fname in channel_files:
            channel_name = os.path.splitext(fname)[0]

            # Skip if not in allowed list
            if self.allowed_channels and channel_name not in self.allowed_channels:
                continue

            full_path = os.path.join(core_path, fname)

            # Read base image
            base_img = tifffile.imread(full_path)

            # Parse into SpatialData model
            image_model = Image2DModel.parse(np.expand_dims(base_img, axis=0), dims=("c","y","x"), scale_factors=[2] * (self.max_pyramid_levels-1))

            images[channel_name] = image_model
            used_channels.append(channel_name)

        # log the info
        logger.info(f"Core '{core_id}' assembled with channels: {used_channels}")

        # Construct and write SpatialData
        sdata = SpatialData(images=images)

        output_path = os.path.join(self.output_dir, f"{core_id}.zarr")
        sdata.write(output_path, overwrite=True)

        if self.cleanup:
            self._cleanup_core_files(core_path, used_channels)

        return output_path

    def _cleanup_core_files(self, core_path, channels):
        for ch in channels:
            tiff_path = os.path.join(core_path, f"{ch}.tiff")
            try:
                os.remove(tiff_path)
                logger.debug(f"Deleted intermediate TIFF: {tiff_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {tiff_path}: {e}")
