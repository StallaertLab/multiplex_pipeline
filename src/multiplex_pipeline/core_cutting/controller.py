import os
import pandas as pd
import dask.array as da
import re
from pathlib import Path
from datetime import datetime
from loguru import logger

from multiplex_pipeline.core_cutting.cutter import CoreCutter
from multiplex_pipeline.core_cutting.assembler import CoreAssembler
from multiplex_pipeline.core_cutting.io import write_temp_tiff, read_ome_tiff


class CoreController:
    def __init__(self,
                 metadata_df: pd.DataFrame,
                 image_dir: str,
                 temp_dir: str,
                 output_dir: str,
                 include_channels=None,
                 exclude_channels=None,
                 margin: int = 0,
                 mask_value: int = 0,
                 max_pyramid_levels: int = 3):

        self.metadata_df = metadata_df
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.margin = margin
        self.mask_value = mask_value

        include_channels = include_channels or []
        exclude_channels = exclude_channels or []

        # Set up logging in output directory
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logger.add(log_path, level="INFO", backtrace=True, diagnose=True)
        logger.info(f"Logging initialized: {log_path}")

        # Check for channel conflicts
        conflicts = set(include_channels).intersection(set(exclude_channels))
        if conflicts:
            raise ValueError(f"Channels specified in both include and exclude lists: {conflicts}")

        self.include_channels = include_channels
        self.exclude_channels = exclude_channels

        # Discover and filter image paths
        discovered = self.discover_images(image_dir)
        logger.info(f"Discovered channels before filtering: {list(discovered.keys())}")

        # Apply inclusion/exclusion and deduplication logic
        self.image_paths = self.filter_channels(discovered)

        self.cutter = CoreCutter(margin=margin, mask_value=mask_value)
        self.assembler = CoreAssembler(temp_dir=temp_dir,
                                       output_dir=output_dir,
                                       max_pyramid_levels=max_pyramid_levels,
                                       allowed_channels=list(self.image_paths.keys()))

        # logging
        logger.info("Final channels to process with source images:")
        for ch in sorted(self.image_paths, key=str.casefold):
            logger.info(f"  Channel: {ch} <- {self.image_paths[ch]}")

        used_files = set(self.image_paths.values())
        all_files = set(discovered.values())
        unused_files = all_files - used_files
        if unused_files:
            logger.info("OME-TIFF files not used in final channel selection:")
            for ch, path in sorted(discovered.items()):
                if path in unused_files:
                    logger.info(f"  Unused: Channel {ch} <- {path}")


    def discover_images(self, image_dir):
        """
        Discover OME-TIFFs from round 4 only (*.0.4), extract clean channel names.
        No hardcoded prefix assumptions (e.g., BLCA).
        """
        image_dir = Path(image_dir)
        image_dict = {}

        for file in image_dir.glob("*.ome.tif*"):
            fname = file.name

            # Match pattern: slideID_round.0.4_R000_MARKER-CHANNEL_...ome.tif
            match = re.match(r"[^_]+_(\d+)\.0\.4_R000_([^_]+)_(.*)\.ome\.tif", fname)
            if not match:
                continue

            round_num_str, dye_or_marker, _ = match.groups()
            round_num = int(round_num_str)

            if "DAPI" in dye_or_marker.upper():
                marker = 'DAPI'
            else:
                parts = fname.split('_')
                if len(parts) > 4:
                    h_parts = parts[4].split('-')
                    if len(h_parts) > 1:
                        marker = '-'.join(h_parts[:-1])
                    else:
                        marker = h_parts[0]
                else:
                    # Raise error explicitly
                    raise ValueError(
                        f"Cannot extract marker from filename '{fname}'. "
                        "Expected at least 5 underscore-separated parts."
                    )
                
            channel_name = f"{round_num:03d}_{marker}"
            image_dict[channel_name] = str(file)

        if not image_dict:
            logger.error(f"No valid .0.4 OME-TIFF files found in {image_dir}")
            raise ValueError(f"No valid .0.4 OME-TIFF files found in {image_dir}")

        return image_dict

    def filter_channels(self, image_dict):
        grouped = {}

        # Group channels by base name
        for ch in image_dict:
            if '_' not in ch:
                continue
            round_prefix, base = ch.split('_', 1)
            grouped.setdefault(base, []).append((int(round_prefix), ch))

        result = {}

        for base, items in grouped.items():
            items.sort()  # ascending by round

            # Handle inclusion/exclusion overrides
            included = [name for _, name in items if name in self.include_channels]
            excluded = [name for _, name in items if name in self.exclude_channels]

            if included:
                for name in included:
                    result[name] = image_dict[name]
                continue

            # Exclude forbidden first
            items = [(r, name) for r, name in items if name not in self.exclude_channels]
            if not items:
                continue

            # Choose preferred version
            if base.upper() == "DAPI":
                preferred = [name for _, name in items if name == "001_DAPI"]
                if preferred:
                    result["DAPI"] = image_dict[preferred[0]]
                else:
                    continue  # skip other DAPI variants
            else:
                round_num, name = items[-1]  # most senior
                result[base] = image_dict[name]

        return result

    def run(self):
        """
        Main execution logic: process all cores for each selected channel,
        save temporary TIFFs, and assemble final Zarrs.
        """
        logger.info(f"Starting run for {len(self.metadata_df)} cores and {len(self.image_paths)} channels.")
        
        for channel_name, file_path in self.image_paths.items():
            logger.info(f"Processing channel: {channel_name}")
            full_img = read_ome_tiff(file_path)  # Returns a Dask array

            for idx, row in self.metadata_df.iterrows():
                core_id = row['core_name']
                core_img = self.cutter.extract_core(full_img, row)
                write_temp_tiff(core_img, core_id, channel_name, self.temp_dir)
                logger.debug(f"Wrote TIFF for core {core_id}, channel {channel_name}")

        # Assemble one Zarr per core
        core_ids = self.metadata_df['core_name'].unique()
        for core_id in core_ids:
            logger.info(f"Assembling SpatialData for: {core_id}")
            self.assembler.assemble_core(core_id)

        logger.info("Processing complete.")
