import os
import time
import pandas as pd
from pathlib import Path
from loguru import logger
from abc import ABC, abstractmethod

from multiplex_pipeline.core_preparation.cutter import CoreCutter
from multiplex_pipeline.core_preparation.assembler import CoreAssembler
from multiplex_pipeline.core_preparation.file_io import write_temp_tiff, read_ome_tiff, FileAvailabilityStrategy
from multiplex_pipeline.utils.globus_utils import GlobusConfig



class CorePreparationController:
    def __init__(self,
                 metadata_df: pd.DataFrame,
                 image_paths: dict,  # channel -> path
                 temp_dir: str,
                 output_dir: str,
                 file_strategy: FileAvailabilityStrategy,
                 margin: int = 0,
                 mask_value: int = 0,
                 max_pyramid_levels: int = 3,
                 core_cleanup_enabled: bool = True):

        self.metadata_df = metadata_df
        self.image_paths = image_paths
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.file_strategy = file_strategy
        self.margin = margin
        self.mask_value = mask_value

        os.makedirs(output_dir, exist_ok=True)

        self.cutter = CoreCutter(margin=margin, mask_value=mask_value)
        self.assembler = CoreAssembler(temp_dir=temp_dir,
                                       output_dir=output_dir,
                                       max_pyramid_levels=max_pyramid_levels,
                                       allowed_channels=list(self.image_paths.keys()),
                                       cleanup=core_cleanup_enabled)

        self.completed_channels = set()
        self.ready_cores = {}  # core_id -> set of completed channels

    def run(self):
        logger.info("Starting controller run loop...")

        while True:
            all_ready = True

            for channel, path in self.image_paths.items():
                if channel in self.completed_channels:
                    continue

                if self.file_strategy.fetch_or_wait(channel, path):
                    self.cut_channel(channel, path)
                    self.file_strategy.cleanup(Path(path))
                    self.completed_channels.add(channel)
                else:
                    all_ready = False

            self.try_assemble_ready_cores()

            if all_ready and not self.ready_cores:
                logger.info("All channels processed and cores assembled.")
                break

            time.sleep(10)

    def cut_channel(self, channel, file_path):
        full_img, store = read_ome_tiff(str(file_path))

        try:
            for _, row in self.metadata_df.iterrows():
                core_id = row["core_name"]
                core_img = self.cutter.extract_core(full_img, row)
                write_temp_tiff(core_img, core_id, channel, self.temp_dir)
                self.ready_cores.setdefault(core_id, set()).add(channel)
                logger.debug(f"Cut and saved core {core_id}, channel {channel}")
        finally:
            # Ensures file is closed even if something fails mid-cut
            if hasattr(store, "close"):
                store.close()
                logger.debug(f"Closed file handle for channel {channel}")


    def try_assemble_ready_cores(self):
        for core_id, channels_done in list(self.ready_cores.items()):
            if set(self.image_paths.keys()).issubset(channels_done):
                logger.info(f"Assembling full core {core_id}")
                self.assembler.assemble_core(core_id)
                del self.ready_cores[core_id]
