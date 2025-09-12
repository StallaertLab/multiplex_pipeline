"""General utilities for configuration loading and dynamic path handling."""

import os
import sys
import importlib
import platform
import yaml
from pathlib import Path


def load_config(config_path = None, verbose = True, namespace=None):
    """
    Load the configuration file from the specified path. 
    If no path is provided, the default configuration file is used.
    If verbose is set to True, the configuration values are printed.
    """
    if config_path is None:
        config_path = Path(os.getcwd()).resolve().parent / "config" / "defaults.yaml"
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Handle 'workstations' separately
    if 'workstations' in config:
        hostname = platform.node()
        workstation_config = config.get("workstations", {}).get(hostname, {})

        if verbose and workstation_config:
            print(f'Using workstation {hostname}\n')

    # Use the provided namespace or fallback to global namespace
    if namespace is None:
        namespace = globals()

    for key, value in config.items():
        namespace[key] = value
        if verbose and key != 'workstations':
            print(f'{key}: {value}')

    return config
    
def get_workstation_path(key="model_path", config_path=None):
    """
    Retrieves the specified path (e.g., model_path or python_env) for the current workstation.
    It can accept a config file from another location. 

    Args:
        key (str): The key to retrieve from the workstation configuration (e.g., "model_path", "python_env").

    Returns:
        str or None: The requested path for the current workstation, or None if not found.
    """
    # Locate the configuration file within the package
    if config_path is None:
        config_path = Path(os.getcwd()).resolve().parent / "config" / "defaults.yaml"
        #config_path = Path(__file__).resolve().parent.parent.parent / "config" / "defaults.yaml"
    
    hostname = platform.node()
    
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        # Fetch the configuration for the current workstation
        workstation_config = config.get("workstations", {}).get(hostname, {})
        
        # Retrieve the specified key
        return workstation_config.get(key)
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

    # Example usage:
    # model_path = get_workstation_path(key="model_path")
    # python_env = get_workstation_path(key="python_env")

def load_module_from_path(module_path, module_name, function_name):
    """
    Dynamically loads a module and a function from a specified package directory.
    
    Args:
        module_path (str): Path to the directory containing the module (e.g., 'sam2').
        module_name (str): The module name relative to the package (e.g., 'sam2.build_sam').
        function_name (str): The function name to import from the module (e.g., 'build_sam2').

    Returns:
        object: The requested function or object.
    """
    # Add the directory containing the package to sys.path
    base_dir = os.path.abspath(module_path)
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    try:
        # Dynamically import the module using the full module name
        module = importlib.import_module(module_name)

        # Retrieve the specified function
        if not hasattr(module, function_name):
            raise AttributeError(f"Module '{module_name}' does not have function '{function_name}'")
        return getattr(module, function_name)
    finally:
        # Clean up sys.path
        if base_dir in sys.path:
            sys.path.remove(base_dir)

def change_to_wsl_path(path):
    """
    Converts a Windows path to a WSL path and checks if the drive is mounted.

    Args:
        path (str): The Windows path to convert.

    Returns:
        str: The WSL path if the drive is mounted, or an error message if not.
    """
    if ":" not in path:
        raise ValueError("Invalid Windows path. Ensure the path includes a drive letter (e.g., C:\\).")
    
    drive_letter = path[0].lower()
    if not ('a' <= drive_letter <= 'z'):
        raise ValueError(f"Invalid drive letter: '{drive_letter}' in path '{path}'.")

    # Convert to WSL path
    wsl_path = f"/mnt/{drive_letter}" + path[2:].replace("\\", "/")

    return wsl_path
    
def get_package_path(package_name="multiplex_pipeline"):
    """
    Get the absolute path to an installed Python package.

    Args:
        package_name (str): The name of the package.

    Returns:
        str: The absolute path to the package.
    """
    spec = importlib.util.find_spec(package_name)
    if not spec or not spec.origin:
        raise ModuleNotFoundError(f"Package '{package_name}' not found.")
    
    # Resolve the package directory from the module's origin
    package_path = Path(spec.origin).parent
    return str(package_path)