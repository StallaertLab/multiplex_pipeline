import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from multiplex_pipeline.core_cutting.channel_scanner import (
    build_transfer_map,
    discover_channels,
)
from multiplex_pipeline.core_cutting.controller import (
    CorePreparationController,
)
from multiplex_pipeline.core_cutting.file_io import (
    GlobusFileStrategy,
    LocalFileStrategy,
)
from multiplex_pipeline.utils.config_loaders import load_analysis_settings
from multiplex_pipeline.utils.globus_utils import (
    GlobusConfig,
    create_globus_tc
)


def configure_logging(settings):
    """
    Setup logging.
    """

    log_file = settings['log_dir'] / f"cores_cutting_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(log_file, level="DEBUG", enqueue=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare cores from OME-TIFFs using metadata and optional Globus transfers."
    )

    parser.add_argument("--exp_config", help="Path to experiment YAML config.", required=True)
    parser.add_argument("--remote_analysis", help="Use remote analysis directory as base.", default=False)
    parser.add_argument("--globus_config", help="Path to Globus config.")
    parser.add_argument("--from_collection", help="Key for source collection in Globus config.", default='r_collection_id')
    parser.add_argument("--to_collection", help="Key for destination collection in Globus config.", default='crcd_collection_id')

    return parser.parse_args()


def main():

    args = parse_args()

    # read config file
    settings = load_analysis_settings(args.exp_config)

    # setup logging
    configure_logging(settings)
    logger.info("Starting core cutting script.")

    # setup Globus if requested
    if args.globus_config:

        gc = GlobusConfig.from_config_files(args.globus_config, from_collection = args.from_collection, to_collection = args.to_collection)
        tc = create_globus_tc(gc.client_id, gc.transfer_tokens)

    else:
        
        gc = None

    # map channels to image paths
    channel_map = discover_channels(settings['image_dir'], 
                                include_channels=settings['include_channels'], 
                                exclude_channels=settings['exclude_channels'], 
                                use_markers=settings.get('use_markers'), 
                                ignore_markers=settings.get('ignore_markers'),
                                gc=gc)
    

    # get cores coordinates
    df_path = settings['core_info_file_path'].with_suffix('.pkl')
    df = pd.read_pickle(df_path)

    # build transfer map
    transfer_cache_dir = settings['temp_dir']
    transfer_map = build_transfer_map(channel_map, transfer_cache_dir)

    # define file access
    if gc:
        # initialize Globus transfer
        strategy = GlobusFileStrategy(tc=tc, transfer_map=transfer_map, gc=gc, cleanup_enabled=True)
        # build a dict for transfered images
        image_paths = {
            ch: str(Path(transfer_cache_dir) / Path(remote).name)
            for ch, (remote, _) in transfer_map.items()
        }
    else:
        strategy = LocalFileStrategy()
        # local files have not been moved
        image_paths = channel_map

    # setup cutting controller
    controller = CorePreparationController(
        metadata_df = df,
        image_paths = image_paths,
        temp_dir = settings['cores_dir_tif'],
        output_dir = settings['cores_dir_output'],
        file_strategy = strategy,
        margin = settings['core_cutting']['margin'],
        mask_value = settings['core_cutting']['mask_value'],
        max_pyramid_levels = settings['core_cutting']['max_pyramid_level'],
        chunk_size = settings['core_cutting']['chunk_size'],
    )

    # run core cutting
    controller.run()

if __name__ == "__main__":
    main()
