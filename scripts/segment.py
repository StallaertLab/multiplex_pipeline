import argparse
from pathlib import Path

import yaml

from multiplex_pipeline.segmentation.controller import SegmentationController
from multiplex_pipeline.segmentation.segmenters import (
    CellposeSegmenter,
    DummySegmenter,
)

# Registry for available segmenters
SEGMENTER_REGISTRY = {
    "dummy": DummySegmenter,
    "cellpose": CellposeSegmenter,
    # "instanseg": InstanSegSegmenter,
}


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("segmentation", {})


def build_segmenter(method: str, kwargs: dict):
    if method not in SEGMENTER_REGISTRY:
        raise ValueError(
            f"Unknown segmentation method '{method}'. "
            f"Available: {list(SEGMENTER_REGISTRY.keys())}"
        )
    segmenter_class = SEGMENTER_REGISTRY[method]
    return segmenter_class(**kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Run segmentation on a SpatialData Zarr object."
    )
    parser.add_argument(
        "--sdata-path", required=True, help="Path to SpatialData .zarr folder"
    )
    parser.add_argument(
        "--config", required=True, help="Path to segmentation YAML config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    method = config.pop("method", None)
    channels = config.pop("channels", None)
    output_name = config.pop("output_name", "mask")

    if not method or not channels:
        raise ValueError(
            "Config must contain 'method' and 'channels' fields under 'segmentation'."
        )

    segmenter = build_segmenter(method, config)
    controller = SegmentationController(segmenter)

    print(
        f"Segmenting {args.sdata_path} using {method} on channels: {channels}"
    )
    controller.segment_spatial_data(
        Path(args.sdata_path), channels, mask_name=output_name
    )


if __name__ == "__main__":
    main()
