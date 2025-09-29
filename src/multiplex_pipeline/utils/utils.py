"""General utilities for configuration loading and dynamic path handling."""

import os
import platform
from pathlib import Path

import yaml


def load_workstation_config(config_path=None):
    """
    Load workstation-specific configuration from a YAML file.

    Args:
        config_path (str or Path): Path to the YAML configuration file.
    Returns:
        dict: Dictionary containing the workstation configuration.
    """
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Handle 'workstations' separately
    if "workstations" in config:
        hostname = platform.node()
        workstation_config = config.get("workstations", {}).get(hostname, {})

        return workstation_config

    else:
        raise KeyError(
            "'workstations' key not found in the configuration file."
        )


def load_analysis_settings(settings_path):
    """
    Load analysis settings from a YAML file.

    Args:
        settings_path (str or Path): Path to the YAML settings file.
    Returns:
        dict: Dictionary containing the analysis settings.
    """
    with open(settings_path) as file:
        settings = yaml.safe_load(file)

    # Define defaults relative to analysis_dir
    analysis_dir = Path(settings["analysis_dir"])
    defaults = {
        "core_info_file_path": analysis_dir / "cores.csv",
        "cores_dir_tif": analysis_dir / "temp",
        "cores_dir_output": analysis_dir / "cores",
        "log_dir": analysis_dir / "logs",
        "temp_dir": analysis_dir / "temp",
    }

    # Fill in defaults + normalize to Path
    for key, default in defaults.items():
        settings[key] = Path(settings.get(key) or default)

    for path in [
        analysis_dir,
        settings["cores_dir_tif"],
        settings["cores_dir_output"],
        settings["log_dir"],
        settings["temp_dir"],
        settings["core_info_file_path"].parent,
    ]:
        os.makedirs(path, exist_ok=True)

    return settings


def change_to_wsl_path(path):
    """
    Converts a Windows path to a WSL path and checks if the drive is mounted.

    Args:
        path (str): The Windows path to convert.

    Returns:
        str: The WSL path if the drive is mounted, or an error message if not.
    """
    if ":" not in path:
        raise ValueError(
            "Invalid Windows path. Ensure the path includes a drive letter (e.g., C:\\)."
        )

    drive_letter = path[0].lower()
    if not ("a" <= drive_letter <= "z"):
        raise ValueError(
            f"Invalid drive letter: '{drive_letter}' in path '{path}'."
        )

    # Convert to WSL path
    wsl_path = f"/mnt/{drive_letter}" + path[2:].replace("\\", "/")

    return wsl_path
