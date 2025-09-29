import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from multiplex_pipeline.core_preparation.channel_scanner import (
    build_transfer_map,
    discover_channels,
)
from multiplex_pipeline.core_preparation.controller import (
    CorePreparationController,
)
from multiplex_pipeline.core_preparation.file_io import (
    GlobusFileStrategy,
    LocalFileStrategy,
)
from multiplex_pipeline.utils.config_loader import load_config
from multiplex_pipeline.utils.globus_utils import (
    GlobusConfig,
    create_globus_tc,
)


def configure_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Add timestamp to the log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        output_dir, f"pipeline_{Path(output_dir).stem}_{timestamp}.log"
    )

    logger.add(log_path, level="INFO", backtrace=True, diagnose=True)
    logger.info(f"Logging initialized: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare cores from OME-TIFFs using metadata and optional Globus transfers."
    )

    parser.add_argument("--config", help="Path to optional YAML config")
    parser.add_argument("--core-info")
    parser.add_argument("--image-dir")
    parser.add_argument("--temp-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--include-channels", nargs="+")
    parser.add_argument("--exclude-channels", nargs="+")
    parser.add_argument("--margin", type=int)
    parser.add_argument("--mask-value", type=int)
    parser.add_argument("--max-pyramid-levels", type=int)
    parser.add_argument("--globus-config")
    parser.add_argument(
        "--transfer-cache-dir",
        help="Optional subfolder under temp-dir for downloaded files",
    )
    parser.add_argument(
        "--transfer-cleanup-enabled",
        default=True,
        help="Enable cleanup of transfer cache after processing",
    )
    parser.add_argument(
        "--core-cleanup-enabled",
        default=True,
        help="Enable cleanup of temporary core files after assembling SpatialData object",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, vars(args))

    required_keys = ["core_info", "image_dir", "temp_dir", "output_dir"]
    missing = [k for k in required_keys if k not in cfg or cfg[k] is None]
    if missing:
        raise ValueError(f"Missing required config values: {missing}")

    configure_logging(cfg["output_dir"])
    metadata_df = pd.read_csv(cfg["core_info"])

    gc = (
        GlobusConfig.from_config_files(Path(cfg["globus_config"]))
        if cfg.get("globus_config")
        else None
    )
    tc = create_globus_tc(gc.client_id, gc.transfer_tokens) if gc else None

    image_map = discover_channels(
        image_dir_or_path=cfg["image_dir"],
        include_channels=cfg.get("include_channels", []),
        exclude_channels=cfg.get("exclude_channels", []),
        use_channels=cfg.get("use_channels", []),
        gc=gc,
    )

    if gc and tc:
        transfer_cache_dir = cfg.get("transfer_cache_dir") or str(
            Path(cfg["temp_dir"]) / "input_cache"
        )
        transfer_map = build_transfer_map(image_map, transfer_cache_dir)
        strategy = GlobusFileStrategy(
            tc,
            transfer_map,
            gc,
            cleanup_enabled=cfg["transfer_cleanup_enabled"],
        )
        image_paths = {
            ch: str(Path(transfer_cache_dir) / Path(remote).name)
            for ch, (remote, _) in transfer_map.items()
        }
    else:
        strategy = LocalFileStrategy()
        image_paths = image_map

    controller = CorePreparationController(
        metadata_df=metadata_df,
        image_paths=image_paths,
        temp_dir=cfg["temp_dir"],
        output_dir=cfg["output_dir"],
        file_strategy=strategy,
        margin=cfg.get("margin", 0),
        mask_value=cfg.get("mask_value", 0),
        max_pyramid_levels=cfg.get("max_pyramid_levels", 3),
        core_cleanup_enabled=cfg.get("core_cleanup_enabled", True),
    )

    controller.run()


if __name__ == "__main__":
    main()
