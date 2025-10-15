from typing import Optional, Sequence

import numpy as np
import spatialdata as sd
from loguru import logger
from skimage.transform import resize
from spatialdata.models import Image2DModel, Labels2DModel

from multiplex_pipeline.processors.base import BaseOp


class ResourceBuildingController:
    def __init__(
        self,
        builder: BaseOp,
        input_names,
        output_names,
        resolution_level: int = 0,
        keep: bool = False,
        overwrite: bool = False,
        pyramid_levels: int = 1,
        downscale: int = 2,
        chunk_size: Optional[Sequence[int]] = None,
    ) -> None:

        self.builder = builder
        self.input_names = input_names
        self.output_names = output_names
        self.resolution_level = resolution_level

        self.pyramid_levels = pyramid_levels
        self.downscale = downscale
        self.chunk_size = list(chunk_size) if chunk_size else [1, 512, 512]

        self.keep = keep
        self.overwrite = overwrite

    def validate_elements_present(self):
        for src in self.input_names:
            if src not in self.sdata:
                raise ValueError(f"Requested source mask '{src}' not found.")

    def validate_resolution_present(self):
        for src in self.input_names:
            el = self.sdata[src]
            if len(el.items()) <= self.resolution_level:
                logger.error(
                    f"Channel '{src}' does not have resolution level {self.resolution_level}."
                )
                raise ValueError(
                    f"Channel '{src}' does not have resolution level {self.resolution_level}."
                )
        logger.info(
            f"All channels have required resolution level: {self.resolution_level}"
        )

    def validate_sdata_as_input(self):

        self.validate_elements_present()
        self.validate_resolution_present()

    def prepare_to_overwrite(self):

        for out_name in self.output_names:
            if out_name in self.sdata:
                if not self.overwrite:
                    message = f"Mask name '{out_name}' already exists in sdata. Please provide unique mask names."
                    logger.error(message)
                    raise ValueError(message)
                else:
                    logger.warning(
                        f"Mask name '{out_name}' already exists and will be overwritten."
                    )
                    del self.sdata[out_name]
                    self.sdata.delete_element_from_disk(out_name)
                    logger.info(
                        f"Existing mask '{out_name}' deleted from sdata."
                    )

    def bring_to_max_resolution(self, el):

        scale_factor = self.downscale**self.resolution_level
        new_shape = tuple(dim * scale_factor for dim in el.shape)
        el_res0 = resize(
            el,
            new_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )

        return el_res0

    def pack_into_model(self, el):
        if self.builder.OUTPUT_TYPE.value == "labels":
            el_model = Labels2DModel.parse(
                data=el.astype(int),
                dims=("y", "x"),
                scale_factors=[self.downscale] * (self.pyramid_levels - 1),
                chunks=self.chunk_size[1:],
            )
        elif self.builder.OUTPUT_TYPE.value == "image":

            el_model = Image2DModel.parse(
                data=el[None],
                dims=("c", "y", "x"),
                scale_factors=[self.downscale] * (self.pyramid_levels - 1),
                chunks=self.chunk_size,
            )

        return el_model

    def run(self, sdata):

        # get sdata
        self.sdata = sdata

        # validate builder settings
        in_list, out_list = self.builder.validate_io(
            inputs=self.input_names, outputs=self.output_names
        )
        self.input_names = in_list
        self.output_names = out_list

        # validate sdata as input
        self.validate_sdata_as_input()

        # Handle overwiting
        self.prepare_to_overwrite()

        # Build the mask
        data_sources = [
            np.array(
                sd.get_pyramid_levels(self.sdata[ch], n=self.resolution_level)
            ).squeeze()
            for ch in self.input_names
        ]
        new_elements = self.builder.run(*data_sources)

        if not isinstance(new_elements, Sequence):
            new_elements = [new_elements]

        logger.info(f"New element(s) '{self.output_names}' have been created.")

        # save output
        for el, el_name in zip(new_elements, self.output_names):

            # bring to max resolution level
            if self.resolution_level > 0:
                el = self.bring_to_max_resolution(el)

            # pack into the data model
            el_model = self.pack_into_model(el)

            # put the data model into the sdata
            self.sdata[el_name] = el_model

            # save to disk if requested
            if self.keep:
                self.sdata.write_element(el_name)
                logger.info(f"Mask '{el_name}' has been saved to disk.")

        return self.sdata
