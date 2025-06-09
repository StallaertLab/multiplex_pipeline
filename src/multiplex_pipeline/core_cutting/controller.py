import os
import pandas as pd
import dask.array as da
import re
from pathlib import Path

from multiplex_pipeline.core_cutting.cutter import CoreCutter
from multiplex_pipeline.core_cutting.assembler import CoreAssembler
from multiplex_pipeline.core_cutting.io import write_temp_tiff, read_ome_tiff


class CoreController:
    def __init__(self,
                 metadata_df: pd.DataFrame,
                 image_dir: str,
                 temp_dir: str,
                 output_dir: str,
                 allowed_channels=None,
                 margin: int = 0,
                 mask_value: int = 0,
                 max_pyramid_levels: int = 3):
        """
        metadata_df: pandas DataFrame with core metadata (must include 'core_id' and bbox/polygon info)
        image_dir: path to folder with raw OME-TIFFs
        temp_dir: path to write temporary per-core/channel TIFFs
        output_dir: path to write final Zarr per-core outputs
        allowed_channels: optional list of channel names to process (e.g., ['pRB', '009_DAPI'])
        margin: margin in pixels around bbox
        mask_value: value to assign outside polygon
        max_pyramid_levels: for multiscale generation
        """
        self.metadata_df = metadata_df
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.margin = margin
        self.mask_value = mask_value
        self.allowed_channels = allowed_channels

        self.cutter = CoreCutter(margin=margin, mask_value=mask_value)
        self.assembler = CoreAssembler(temp_dir=temp_dir,
                                       output_dir=output_dir,
                                       max_pyramid_levels=max_pyramid_levels,
                                       allowed_channels=self.allowed_channels)

        self.image_paths = self._discover_images(image_dir)

        if self.allowed_channels:
            missing = set(self.allowed_channels) - set(self.image_paths.keys())
            if missing:
                raise ValueError(f"The following requested channels are missing from available image files: {missing}")

            self.image_paths = {
                k: v for k, v in self.image_paths.items() if k in self.allowed_channels
            }


    def _discover_images(self, image_dir):
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
                channel_name = f"{round_num:03d}_DAPI"
            else:
                marker_match = re.search(r"_([a-zA-Z0-9\-]+)-AF\d+", fname)
                if marker_match:
                    marker = marker_match.group(1)
                else:
                    marker = dye_or_marker
                channel_name = marker

            image_dict[channel_name] = str(file)

        if not image_dict:
            raise ValueError(f"No valid .0.4 OME-TIFF files found in {image_dir}")

        return image_dict

    def run(self):
        """
        Main execution logic: process all cores for each selected channel,
        save temporary TIFFs, and assemble final Zarrs.
        """
        print(f"Found {len(self.image_paths)} channels to process.")
        for channel_name, file_path in self.image_paths.items():
            print(f"Processing channel: {channel_name}")
            full_img = read_ome_tiff(file_path)  # Returns a Dask array

            for idx, row in self.metadata_df.iterrows():
                core_id = row['core_name']
                core_img = self.cutter.extract_core(full_img, row)
                write_temp_tiff(core_img, core_id, channel_name, self.temp_dir)

        # Assemble one Zarr per core
        core_ids = self.metadata_df['core_name'].unique()
        for core_id in core_ids:
            print(f"Assembling SpatialData for: {core_id}")
            self.assembler.assemble_core(core_id)
