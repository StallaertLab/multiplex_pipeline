import os
import re
from typing import Union
from pathlib import Path, PurePosixPath
from loguru import logger

from multiplex_pipeline.utils.globus_utils import GlobusConfig,create_globus_tc
from multiplex_pipeline.core_preparation.file_io import list_local_files, list_globus_files

def scan_channels_from_list(files, include_channels=None, exclude_channels=None, use_channels=None) -> dict:
    include_channels = include_channels or []
    exclude_channels = exclude_channels or []
    use_channels = use_channels or []

    image_dict = {}

    for filepath in files:
        fname = os.path.basename(filepath)

        match = re.match(r"[^_]+_(\d+)\.0\.4_R000_([^_]+)_(.*)\.ome\.tif+", fname)
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
                marker = '-'.join(h_parts[:-1]) if len(h_parts) > 1 else h_parts[0]
            else:
                raise ValueError(
                    f"Cannot extract marker from filename '{fname}'. "
                    "Expected at least 5 underscore-separated parts."
                )

        channel_name = f"{round_num:03d}_{marker}"
        image_dict[channel_name] = filepath

    if not image_dict:
        raise ValueError("No valid .0.4 OME-TIFF files found.")

    logger.info(f"Discovered channels before filtering: {list(image_dict.keys())}")

    grouped = {}
    for ch in image_dict:
        if '_' not in ch:
            continue
        round_prefix, base = ch.split('_', 1)
        grouped.setdefault(base, []).append((int(round_prefix), ch))

    result = {}
    unused = set(image_dict.keys())

    for base, items in grouped.items():
        items.sort()

        included = [name for _, name in items if name in include_channels]
        if included:
            for name in included:
                result[name] = image_dict[name]
                unused.discard(name)
            continue

        items = [(r, name) for r, name in items if name not in exclude_channels]
        if not items:
            continue

        if base.upper() == "DAPI":
            preferred = [name for _, name in items if name == "001_DAPI"]
            if preferred:
                result["DAPI"] = image_dict[preferred[0]]
                unused.discard(preferred[0])
        else:
            _, name = items[-1]
            result[base] = image_dict[name]
            unused.discard(name)

    # Apply final-use channel filter (e.g. DAPI, CD44)
    if use_channels:
        result = {
            base: path
            for base, path in result.items()
            if base in use_channels
        }
        logger.info(f"Restricting to use_channels = {use_channels}")
        logger.info(f"Final filtered channels: {list(result.keys())}")

    logger.info("Final selected channels:")
    for ch in sorted(result, key=str.casefold):
        logger.info(f"  Channel: {ch} <- {result[ch]}")

    if unused:
        logger.info("OME-TIFF files not used in final channel selection:")
        for ch in sorted(unused):
            logger.info(f"  Unused: Channel {ch} <- {image_dict[ch]}")

    return result

def discover_channels(image_dir_or_path, include_channels=None, exclude_channels=None, gc=None, use_channels=None):
    """
    Unified scanner wrapper for local or Globus-based input.
    """
    if gc is not None:
        files = list_globus_files(gc, image_dir_or_path)
    else:
        files = list_local_files(image_dir_or_path)

    return scan_channels_from_list(files, include_channels, exclude_channels, use_channels)

def build_transfer_map(
    remote_paths: dict,
    full_local_path: Union[str, Path]
) -> dict:
    """
    Build a transfer map with fully qualified destination paths that are
    POSIX-formatted and compatible with Globus Transfer, including the /~/ prefix
    and drive letter representation for Windows paths.

    Parameters:
    - remote_paths: dict of {channel: remote_path}
    - full_local_path: absolute local path as str or Path (e.g., 'D:/multiplex_pipeline/tests/input_cache')

    Returns:
    - dict of {channel: (remote_path, Globus-friendly POSIX-style local path)}
    """
    base = Path(full_local_path).resolve()
    
    # Convert base to Globus-style POSIX path with /~/ prefix
    base_drive = base.drive.rstrip(':')  # Extract drive letter without colon
    base_suffix = base.relative_to(base.anchor).as_posix()  # Strip drive anchor and convert to POSIX
    base_globus = Path(f"/~/{base_drive}/{base_suffix}")

    return {
        ch: (
            str(Path(remote).as_posix()),
            str((base_globus / Path(remote).name).as_posix())
        )
        for ch, remote in remote_paths.items()
    }