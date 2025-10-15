import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import spatialdata as sd
from loguru import logger

import multiplex_pipeline.object_segmentation.cleaners as cleaners
from multiplex_pipeline.object_segmentation.controller import (
    SegmentationController,
)
from multiplex_pipeline.object_segmentation.preprocessing_controller import (
    PreSegmentationProcessor,
)
from multiplex_pipeline.object_segmentation.segmenters import (
    InstansegSegmenter,
)
from multiplex_pipeline.utils.config_loaders import load_analysis_settings


def configure_logging(settings):
    """
    Setup logging.
    """

    log_file = (
        settings["log_dir"]
        / f"cores_cutting_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
    )

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
    logger.info("Starting core segmentation script.")

    # define pre-processor
    preseg_processor = PreSegmentationProcessor(
        normalize=settings["segmentation"]["preprocessing"]["normalize"],
        denoise=settings["segmentation"]["preprocessing"]["denoise"],
        mix=settings["segmentation"]["preprocessing"]["mix"],
    )

    # define segmenter
    if settings["segmentation"]["package"] == "instanseg":
        model = settings["segmentation"]["model"]
        segmenter = InstansegSegmenter(
            model_type=model, **settings["segmentation"]["kwargs"]
        )
    else:
        raise ValueError(
            f"Segmentation package {settings['segmentation']['package']} to be implemented."
        )

    # define cleaner
    cleaner_name = settings["segmentation"]["cleaner"]
    cleaner_class = getattr(cleaners, cleaner_name)
    cleaner = cleaner_class()

    # define controller
    segmentation_controller = SegmentationController(
        segmenter,
        cleaner=cleaner,
        resolution_level=settings["segmentation"]["resolution_level"],
        overwrite_mask=args.overwrite,
    )

    # channels used for segmentation
    channels = settings["segmentation"]["input"]
    input_channels = [f"{ch}_preseg" for ch in channels]

    # define the cores for the analysis
    core_dir = Path(settings["analysis_dir"]) / "cores"
    path_list = [core_dir / f for f in os.listdir(core_dir)]
    path_list.sort()

    for sd_path in path_list[:30]:

        # get sdata
        sdata = sd.read_zarr(sd_path)

        # run preprocessing
        sdata = preseg_processor.run(sdata, channels=channels)

        # run segmentation
        segmentation_controller.run(
            sdata,
            channels=input_channels,
            mask_name=settings["segmentation"]["output_name"],
        )


if __name__ == "__main__":
    main()
