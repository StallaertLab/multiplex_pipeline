import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import spatialdata as sd
from loguru import logger

from multiplex_pipeline.utils.config_loaders import load_analysis_settings
from multiplex_pipeline.processors import build_processor
from multiplex_pipeline.processors.controller import ResourceBuildingController

def configure_logging(settings):
    """
    Setup logging.
    """

    log_file = settings.log_dir_path / f"cores_segmenation_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"

    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    logger.add(log_file, level="DEBUG", enqueue=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare cores from OME-TIFFs using metadata and optional Globus transfers."
    )

    parser.add_argument(
        "--exp_config", help="Path to experiment YAML config.", required=True
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Should the masks be overwritten.",
    )
    parser.add_argument(
        "--remote_analysis",
        action="store_true",
        help="Use remote analysis directory as base.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # read config file
    settings = load_analysis_settings(
        args.exp_config, remote_analysis=args.remote_analysis
    )

    # setup logging
    configure_logging(settings)
    logger.info("Starting object segmentation script.")

    # setup builders of additional data elements
    if getattr(settings,'additional_elements',None):
        
        builders_list = []

        for builder_settings in settings.additional_elements:
            
            params = dict(getattr(builder_settings,'parameters',None)) or {}

            builder = build_processor(builder_settings.category, builder_settings.type, **params) 
            
            builder_controller = ResourceBuildingController(builder=builder, 
                                                input_names=builder_settings.input, 
                                                output_names=builder_settings.output, 
                                                keep=builder_settings.keep, 
                                                overwrite=True,
                                                pyramid_levels=settings.sdata_storage.max_pyramid_level,
                                                downscale = settings.sdata_storage.downscale,
                                                chunk_size = settings.sdata_storage.chunk_size,
                                                )
            
            logger.info(f"Image transformer of type '{builder_settings.type}' for image '{builder_settings.input}' has been created.")

            builders_list.append(builder_controller)

    else:
        builders_list = []
        logger.info("No resource builders specified.")

    # define the cores for the analysis
    core_dir = settings.analysis_dir / 'cores'
    path_list = [core_dir / f for f in os.listdir(core_dir)]
    path_list.sort()

    # run processing
    for sd_path in path_list:
    
        logger.info(f"Processing {sd_path.name}")

        # get sdata
        sdata = sd.read_zarr(sd_path)

        # check that the pipeline can run on provide sdata
        settings.validate_pipeline(sdata)

        # run builders of additional elements
        for builder_controller in builders_list:
            sdata = builder_controller.run(sdata)

if __name__ == "__main__":
    main()
