import argparse
import json
import sys

from loguru import logger

from plex_pipe.core_preparation.channel_scanner import (
    discover_channels,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan a directory of OME-TIFFs and output a channel-to-path mapping."
    )
    parser.add_argument("image_dir", help="Directory path (local or Globus)")
    parser.add_argument("--include", nargs="+", help="Channel base names to include")
    parser.add_argument("--exclude", nargs="+", help="Channel base names to exclude")
    parser.add_argument("--output", help="Output JSON file to save result")
    parser.add_argument(
        "--config-dir",
        help="Path to Globus config directory (if using Globus)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time}</green> | <level>{message}</level>",
    )

    gc = None
    if args.config_dir:
        from plex_pipe.utils.globus_utils import GlobusConfig

        gc = GlobusConfig.from_config_files(args.config_dir)

    channel_map = discover_channels(
        image_dir_or_path=args.image_dir,
        include_channels=args.include,
        exclude_channels=args.exclude,
        gc=gc,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(channel_map, f, indent=2)
        logger.info(f"Saved channel map to: {args.output}")
    else:
        print(json.dumps(channel_map, indent=2))


if __name__ == "__main__":
    main()
