from typing import Optional, Sequence, Union

import numpy as np
import spatialdata as sd
from loguru import logger
from skimage.transform import resize
from spatialdata.models import Image2DModel, Labels2DModel

from plex_pipe.processors.base import BaseOp


class ResourceBuildingController:
    """Controls the execution of a builder op on a SpatialData object.

    This class orchestrates the process of running a `BaseOp` subclass. It
    manages input validation, fetching data from the correct resolution,
    handling overwrites, executing the operation, and saving the results back
    into the SpatialData object.
    """

    def __init__(
        self,
        builder: BaseOp,
        input_names: Union[str, Sequence[str]],
        output_names: Union[str, Sequence[str]],
        resolution_level: int = 0,
        keep: bool = False,
        overwrite: bool = False,
        pyramid_levels: int = 1,
        downscale: int = 2,
        chunk_size: Optional[Sequence[int]] = None,
    ) -> None:
        """Initializes the ResourceBuildingController.

        Args:
            builder: The processing operation instance (e.g., a filter).
            input_names: Name or sequence of names of input elements.
            output_names: Name or sequence of names for output elements.
            resolution_level: The resolution level to fetch input data from.
            keep: Whether to save the generated element to disk.
            overwrite: Whether to overwrite existing elements with the same name.
            pyramid_levels: The number of pyramid levels for the output.
            downscale: The downscaling factor between pyramid levels.
            chunk_size: The chunk size for the output Dask array.
        """

        self.builder = builder
        self.input_names = input_names
        self.output_names = output_names
        self.resolution_level = resolution_level

        self.pyramid_levels = pyramid_levels
        self.downscale = downscale
        self.chunk_size = list(chunk_size) if chunk_size else [1, 512, 512]

        self.keep = keep
        self.overwrite = overwrite

    def validate_elements_present(self, sdata):
        """Checks if all specified input elements exist in the sdata object.

        Raises:
            ValueError: If an input element is not found.
        """
        for src in self.input_names:
            if src not in sdata:
                raise ValueError(f"Requested source mask '{src}' not found.")

    def validate_resolution_present(self, sdata):
        """Checks if inputs have the required resolution level.

        Raises:
            ValueError: If an input element does not have the specified
                resolution level.
        """
        for src in self.input_names:
            el = sdata[src]
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

    def validate_sdata_as_input(self, sdata):
        """Runs all input validation checks."""

        self.validate_elements_present(sdata)
        self.validate_resolution_present(sdata)

    def prepare_to_overwrite(self, sdata):
        """Handles existing output elements based on the `overwrite` flag.

        If an output name already exists and `overwrite` is False, it raises
        an error. If `overwrite` is True, it deletes the existing element.

        Raises:
            ValueError: If an output element exists and `overwrite` is False.
        """

        for out_name in self.output_names:
            if out_name in sdata:
                if not self.overwrite:
                    message = f"Mask name '{out_name}' already exists in sdata. Please provide unique mask names."
                    logger.error(message)
                    raise ValueError(message)
                else:
                    logger.warning(
                        f"Mask name '{out_name}' already exists and will be overwritten."
                    )
                    del sdata[out_name]
                    logger.info(f"Existing element '{out_name}' deleted from sdata.")
                    if out_name in [
                        x.split("/")[-1] for x in sdata.elements_paths_on_disk()
                    ]:
                        sdata.delete_element_from_disk(out_name)
                        logger.info(f"Existing element '{out_name}' deleted from disk.")

    def bring_to_max_resolution(self, el):
        """Upscales an element to the base resolution (level 0).

        Args:
            el: The numpy array to upscale.

        Returns:
            The upscaled numpy array.
        """

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
        """Packs a numpy array into the appropriate SpatialData model.

        Args:
            el: The numpy array to pack.

        Returns:
            An `Image2DModel` or `Labels2DModel` instance.
        """
        if self.builder.OUTPUT_TYPE.value == "labels":
            el_model = Labels2DModel.parse(
                data=el.astype(np.int32),
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
        """Executes the full pipeline for the processor. It starts with running validation of a compatibility of a processor with the sdata object to process.

        Args:
            sdata: The SpatialData object to process.

        Returns:
            The processed SpatialData object.
        """

        # validate builder settings
        in_list, out_list = self.builder.validate_io(
            inputs=self.input_names, outputs=self.output_names
        )
        self.input_names = in_list
        self.output_names = out_list

        # validate sdata as input
        self.validate_sdata_as_input(sdata)

        # Handle overwiting
        self.prepare_to_overwrite(sdata)

        # Build
        data_sources = [
            np.array(
                sd.get_pyramid_levels(sdata[ch], n=self.resolution_level)
            ).squeeze()
            for ch in self.input_names
        ]
        # data_sources = [
        #         sd.get_pyramid_levels(sdata[ch], n=self.resolution_level).squeeze()
        #     for ch in self.input_names
        # ]
        new_elements = self.builder.run(*data_sources)

        # forced cleanup
        del data_sources

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

            # forced cleanup
            del el

            # put the data model into the sdata
            sdata[el_name] = el_model

            # save to disk if requested
            if self.keep:
                sdata.write_element(el_name)
                logger.info(f"Mask '{el_name}' has been saved to disk.")

        return sdata
