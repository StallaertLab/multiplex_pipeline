import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import spatialdata as sd
from loguru import logger

from multiplex_pipeline.utils.config_loaders import load_analysis_settings
from multiplex_pipeline.object_quantification.controller import QuantificationController

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
    logger.info("Starting quantification script.")

    # setup quantification controllers
    quant_controller_list = [] 
    qc_prefix = settings.qc.prefix
    for quant in settings.quant:

        table_name = quant.name
        masks_keys = quant.masks
        connect_to_mask = quant.layer_connection

        logger.info(f"Setting up quantification controller for '{table_name}' table with masks {masks_keys} and connection to '{connect_to_mask}' mask")

        controller = QuantificationController(
            table_name=table_name,
            mask_keys=masks_keys,
            connect_to_mask=connect_to_mask,
            overwrite=True,
            quantify_qc = True,
            qc_prefix = qc_prefix,
        )

        quant_controller_list.append(controller) 


    # define the cores for the analysis
    core_dir = settings.analysis_dir / 'cores'
    path_list = [core_dir / f for f in os.listdir(core_dir)]
    path_list.sort()

    # run processing
    for sd_path in path_list:
    
        logger.info(f"Processing {sd_path.name}")

        # get sdata
        sdata = sd.read_zarr(sd_path)

        # run quantification
        for controller in quant_controller_list:
            controller.run(sdata)

if __name__ == "__main__":
    main()
