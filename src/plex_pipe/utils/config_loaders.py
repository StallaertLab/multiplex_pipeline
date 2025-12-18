"""General utilities for configuration loading and dynamic path handling."""

import copy
import os
import platform

import yaml

from plex_pipe.utils.config_schema import AnalysisConfig


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
        raise KeyError("'workstations' key not found in the configuration file.")


def load_analysis_settings(settings_path, remote_analysis=False):
    """
    Load analysis settings from a YAML file.

    Args:
        settings_path (str or Path): Path to the YAML settings file.
    Returns:
        dict: Dictionary containing the analysis settings.
    """
    with open(settings_path) as file:
        settings = yaml.safe_load(file)

    # expand placeholders
    settings = expand_pipeline(settings)

    # Validate and get the "blueprint" object
    config = AnalysisConfig.model_validate(
        settings, context={"remote_analysis": remote_analysis}
    )

    # create dirs necessary for the analysis
    for p in [
        config.analysis_dir,
        config.log_dir_path,
        config.core_info_file_path.parent,
        config.cores_dir_tif_path,
        config.cores_dir_output_path,
    ]:
        os.makedirs(p, exist_ok=True)

    return config


def contains_placeholder(obj, placeholder="${input}"):
    """Recursively check if the placeholder appears anywhere in the object."""
    if isinstance(obj, dict):
        return any(contains_placeholder(v, placeholder) for v in obj.values())
    if isinstance(obj, list):
        return any(contains_placeholder(v, placeholder) for v in obj)
    return isinstance(obj, str) and placeholder in obj


def replace_placeholders(obj, mapping):
    """Recursively substitute ${key} with mapping[key] in strings."""
    if isinstance(obj, dict):
        return {k: replace_placeholders(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_placeholders(v, mapping) for v in obj]
    if isinstance(obj, str):
        out = obj
        for k, v in mapping.items():
            out = out.replace("${" + k + "}", str(v))
        return out
    return obj


def expand_pipeline(cfg, section="additional_elements"):
    """
    Expand any step where:
      - input is a list
      - and the step contains a placeholder like ${input}
    """
    expanded = []
    for step in cfg.get(section, []):
        inputs = step.get("input")
        if isinstance(inputs, list) and contains_placeholder(step, "${input}"):
            for inp in inputs:
                inst = copy.deepcopy(step)
                inst["input"] = inp
                inst = replace_placeholders(inst, {"input": inp})
                expanded.append(inst)
        else:
            expanded.append(step)
    cfg[section] = expanded
    return cfg
