import os

import numpy as np
import tifffile
from loguru import logger
from spatialdata import SpatialData
from spatialdata.models import Image2DModel


class CoreAssembler:
    """Assemble per-channel TIFFs into a ``SpatialData`` object."""

    def __init__(
        self,
        temp_dir: str,
        output_dir: str,
        max_pyramid_levels: int = 4,
        downscale: int = 2,
        chunk_size: tuple[int, int, int] = (
            1,
            256,
            256,
        ),  # SpatialData default chunk size
        allowed_channels: list[str] | None = None,
        cleanup: bool = False,
    ) -> None:
        """Initialize the assembler.

        Args:
            temp_dir (str): Directory containing temporary per-core folders.
            output_dir (str): Location where ``.zarr`` outputs are written.
            max_pyramid_levels (int, optional): Maximum number of
                multiscale levels. ``0`` disables pyramid creation.
            downscale (int, optional): Downsampling factor per level.
            allowed_channels (list[str] | None, optional): Restrict processing
                to these channels. If ``None`` all channels are used.
            cleanup (bool, optional): Remove intermediate TIFFs when ``True``.
        """
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.max_pyramid_levels = max_pyramid_levels
        self.chunk_size = chunk_size
        self.downscale = downscale
        self.allowed_channels = allowed_channels
        self.cleanup = cleanup

    def assemble_core(self, core_id: str) -> str:
        """Assemble a single core from its per-channel TIFF images.

        Args:
            core_id (str): Identifier of the core's temporary folder.

        Returns:
            str: Path to the written ``.zarr`` dataset.

        Raises:
            FileNotFoundError: If the temporary core folder is missing.
            ValueError: If no TIFF files are found in the folder.
        """
        core_path = os.path.join(self.temp_dir, core_id)
        if not os.path.exists(core_path):
            raise FileNotFoundError(f"No temp folder found for core: {core_id}")

        # Collect and sort TIFFs
        channel_files = sorted(
            [
                f
                for f in os.listdir(core_path)
                if f.lower().endswith(".tiff") or f.lower().endswith(".tif")
            ]
        )
        if not channel_files:
            raise ValueError(f"No TIFFs found for core: {core_id}")

        images = {}
        used_channels = []

        # initialize object
        sdata = SpatialData()   

        # save to drive
        output_path = os.path.join(self.output_dir, f"{core_id}.zarr")
        sdata.write(output_path, overwrite=True)     

        for fname in channel_files:
            channel_name = os.path.splitext(fname)[0]

            # Skip if not in allowed list
            if self.allowed_channels and channel_name not in self.allowed_channels:
                continue

            full_path = os.path.join(core_path, fname)

            # Read base image
            base_img = tifffile.imread(full_path)

            # Parse into SpatialData model
            image_model = Image2DModel.parse(
                np.expand_dims(base_img, axis=0),
                dims=("c", "y", "x"),
                scale_factors=[2] * (self.max_pyramid_levels - 1),
                chunks=self.chunk_size,
            )

            sdata[channel_name] = image_model
            sdata.write_element(channel_name, overwrite = True)
            used_channels.append(channel_name)

            # release memory
            del sdata[channel_name]

        # log the info
        logger.info(f"Core '{core_id}' assembled with channels: {used_channels}")

        if self.cleanup:
            self._cleanup_core_files(core_path, used_channels)

        return output_path

    def _cleanup_core_files(self, core_path: str, channels: list[str]) -> None:
        """Delete intermediate TIFF files for the given channels.

        Args:
            core_path (str): Temporary directory for the core.
            channels (list[str]): Channel names to remove.
        """
        for ch in channels:
            tiff_path = os.path.join(core_path, f"{ch}.tiff")
            try:
                os.remove(tiff_path)
                logger.debug(f"Deleted intermediate TIFF: {tiff_path}")
            except OSError as exc:
                logger.warning(f"Failed to delete {tiff_path}: {exc}")
