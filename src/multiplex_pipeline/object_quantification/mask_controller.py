from loguru import logger
import numpy as np
import spatialdata as sd
from spatialdata.models import Labels2DModel

from multiplex_pipeline.object_quantification.mask_builders import BaseBuilder


class MaskBuildingController:
    def __init__(
        self,
        mask_builder: BaseBuilder,
        source,
        mask_name: str,
        keep: bool = False,
        overwrite: bool = True,
        pyramid_levels: int = 3,
        downscale: int = 2,
        resolution_level: int = 0,
    ) -> None:

        self.mask_builder = mask_builder
        self.source = source
        self.mask_name = mask_name
        self.pyramid_levels = pyramid_levels
        self.downscale = downscale
        self.keep = keep
        self.overwrite = overwrite
        self.resolution_level = resolution_level

    def validate_channels(self):
        for src in self.source:
            if src not in self.sdata:
                raise ValueError(f"Requested source mask '{src}' not found.")

    def prepare_to_overwrite(self):
        if self.mask_name in self.sdata:
            if not self.overwrite:
                logger.error(
                    f"Mask name '{self.mask_name}' already exists in sdata. Please provide unique mask names."
                )
                raise ValueError(
                    f"Mask name '{self.mask_name}' already exists in sdata. Please provide unique mask names."
                )
            else:
                logger.warning(
                    f"Mask name '{self.mask_name}' already exists and will be overwritten."
                )
                del self.sdata[self.mask_name]
                self.sdata.delete_element_from_disk(self.mask_name)
                logger.info(f"Existing mask '{self.mask_name}' deleted from sdata.")

    def run(self,sdata):

        self.sdata = sdata

        # validate requested channels
        self.validate_channels()

        # Handle overwiting
        self.prepare_to_overwrite()

        # Build the mask
        data_sources = [np.array(sd.get_pyramid_levels(self.sdata[ch], n=self.resolution_level)).squeeze() for ch in self.source]
        new_mask = self.mask_builder.run(*data_sources)

        logger.info(f"New mask '{self.mask_name}' has been created.")

        # put into the sdata object
        mask_model = Labels2DModel.parse(
                data=new_mask,
                dims=("y", "x"),
                scale_factors=[self.downscale] * (self.pyramid_levels - 1),
            )

        # Put the mask in sdata
        self.sdata[self.mask_name] = mask_model

        # Save to disk if requested
        if self.keep:
            self.sdata.write_element(self.mask_name)
            logger.info(f"Mask '{self.mask_name}' has been saved to disk.")

        return self.sdata
