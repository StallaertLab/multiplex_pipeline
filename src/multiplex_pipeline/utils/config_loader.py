import yaml
from pathlib import Path
from argparse import Namespace
from loguru import logger
from typing import Union


def load_config(config_path: Union[str, Path, None], cli_overrides: dict = None) -> dict:
    """
    Load pipeline configuration from YAML if provided, and apply CLI overrides.

    If no config_path is provided, returns a dict from CLI args only.

    Parameters:
        config_path: Optional path to YAML config file.
        cli_overrides: Dictionary of CLI args (e.g., vars(args))

    Returns:
        Final unified config dictionary
    """
    config = {}

    # Load from file if provided
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded configuration from: {config_path}")
    else:
        logger.info("No config file provided; using CLI arguments only.")

    # Apply CLI overrides
    cli_overrides = cli_overrides or {}
    for key, value in cli_overrides.items():
        if value is not None:
            logger.info(f"Overriding config: {key} = {value}")
            config[key] = value

    return config
