import os
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from plex_pipe.core_cutting.assembler import CoreAssembler
from plex_pipe.core_cutting.cutter import CoreCutter
from plex_pipe.core_cutting.file_io import (
    FileAvailabilityStrategy,
    read_ome_tiff,
    write_temp_tiff,
)


class CorePreparationController:
    """Coordinate cutting and assembly of cores from multiplex images."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        image_paths: dict[str, str],
        temp_dir: str,
        output_dir: str,
        file_strategy: FileAvailabilityStrategy,
        margin: int = 0,
        mask_value: int = 0,
        max_pyramid_levels: int = 3,
        chunk_size: tuple[int, int, int] = (1, 256, 256),
        downscale = 2,
        core_cleanup_enabled: bool = True,
    ) -> None:
        """Initialize the controller.

        Args:
            metadata_df (pandas.DataFrame): Table describing each core.
            image_paths (dict[str, str]): Mapping of channel names to image
                paths.
            temp_dir (str): Directory for temporary core files.
            output_dir (str): Destination for assembled ``.zarr`` outputs.
            file_strategy (FileAvailabilityStrategy): Strategy used to obtain
                image files.
            margin (int, optional): Pixels of padding around each core.
            mask_value (int, optional): Fill value for masked regions.
            max_pyramid_levels (int, optional): Number of pyramid levels.
            chunk_size (tuple[int, int, int], optional): Chunk size for the
                data storage.
            core_cleanup_enabled (bool, optional): Remove intermediate TIFFs
                after assembly.
        """

        self.metadata_df = metadata_df
        self.image_paths = image_paths
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.file_strategy = file_strategy
        self.margin = margin
        self.mask_value = mask_value

        os.makedirs(output_dir, exist_ok=True)

        self.cutter = CoreCutter(margin=margin, mask_value=mask_value)
        self.assembler = CoreAssembler(
            temp_dir=temp_dir,
            output_dir=output_dir,
            max_pyramid_levels=max_pyramid_levels,
            chunk_size=chunk_size,
            downscale = downscale,
            allowed_channels=list(self.image_paths.keys()),
            cleanup=core_cleanup_enabled,
        )

        self.completed_channels = set()
        self.ready_cores = {}  # core_id -> set of completed channels

    def cut_channel(self, channel, file_path):
        """Cut all cores from a single channel image.

        Args:
            channel (str): Name of the channel being processed.
            file_path (str | Path): Path to the OME-TIFF file.
        """

        full_img, store = read_ome_tiff(str(file_path))

        try:
            for _, row in self.metadata_df.iterrows():
                core_id = row["core_name"]
                core_img = self.cutter.extract_core(full_img, row)
                write_temp_tiff(core_img, core_id, channel, self.temp_dir)
                self.ready_cores.setdefault(core_id, set()).add(channel)
                logger.debug(
                    f"Cut and saved core {core_id}, channel {channel}."
                )
        finally:
            # Ensures file is closed even if something fails mid-cut
            if hasattr(store, "close"):
                store.close()
                logger.debug(f"Closed file handle for channel {channel}.")

    def try_assemble_ready_cores(self):
        """Assemble any cores whose channels are complete."""
        for core_id, channels_done in list(self.ready_cores.items()):
            if set(self.image_paths.keys()).issubset(channels_done):
                logger.info(f"Assembling full core {core_id}")
                self.assembler.assemble_core(core_id)
                del self.ready_cores[core_id]

    def run(self, poll_interval: float = 10.0) -> None:
        """Process all channels and assemble cores.

        This method blocks until all channels have been processed and the
        corresponding cores have been assembled.
        """

        logger.info("Starting controller run loop...")

        while True:
            all_ready = True

            for channel, path in self.image_paths.items():
                if channel in self.completed_channels:
                    continue

                if self.file_strategy.is_channel_ready(channel, path):
                    logger.info(f'Channel {channel} file available at {path}.')
                    self.cut_channel(channel, path)
                    self.file_strategy.cleanup(Path(path))
                    self.completed_channels.add(channel)
                else:
                    all_ready = False

            self.try_assemble_ready_cores()

            if all_ready and not self.ready_cores:
                logger.info("All channels processed and cores assembled.")
                break
            else:
                logger.info(f'All ready: {all_ready}, self.ready_cores')

            time.sleep(poll_interval)
