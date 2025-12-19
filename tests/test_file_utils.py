import pytest
from unittest.mock import MagicMock
from globus_sdk import GlobusAPIError

import plex_pipe.utils.file_utils as file_utils

def test_change_to_wsl_path():
    """
    Verifies valid Windows paths convert to /mnt/x/ format.
    """
    # Standard C drive
    assert file_utils.change_to_wsl_path(r"C:\Users\Data") == "/mnt/c/Users/Data"

    # Relative path (no drive)
    with pytest.raises(ValueError, match="Invalid Windows path"):
        file_utils.change_to_wsl_path(r"\Users\Data")
    
    # Invalid drive letter (symbol)
    with pytest.raises(ValueError, match="Invalid drive letter"):
        file_utils.change_to_wsl_path(r"1:\Users")

def test_converter_multi_drive():
    """
    Strategy: 'multi_drive'
    Logic: C:\Data -> /C/Data
    """
    conv = file_utils.GlobusPathConverter(layout="multi_drive")
    
    # Check C drive
    assert conv.windows_to_globus(r"C:\Users\file.txt") == "/C/Users/file.txt"
    # Check D drive (should be dynamic)
    assert conv.windows_to_globus(r"D:\Data\raw.tif") == "/D/Data/raw.tif"
    
    # Error: Missing drive letter
    with pytest.raises(ValueError):
        conv.windows_to_globus("Relative\\Path")

def test_converter_single_drive():
    """
    Strategy: 'single_drive'
    Logic: C:\Data -> /Data (Drive letter is stripped, root is mapped to /)
    """
    conv = file_utils.GlobusPathConverter(layout="single_drive")
    
    assert conv.windows_to_globus(r"C:\Users\file.txt") == "/Users/file.txt"
    # Even if it's a network drive R:, it should map to root /
    assert conv.windows_to_globus(r"R:\Project\Data") == "/Project/Data"

def test_converter_subfolder_root_success():
    """
    Strategy: 'subfolder_root'
    Logic: Root is C:\Share. Path C:\Share\Project -> /Project.
    """
    shared_root = r"C:\Share"
    conv = file_utils.GlobusPathConverter(layout="subfolder_root", shared_root=shared_root)
    
    # Valid path inside the share
    win_path = r"C:\Share\Project\file.txt"
    assert conv.windows_to_globus(win_path) == "/Project/file.txt"

def test_converter_subfolder_root_security_check():
    """
    Critical Security Test: 'subfolder_root'
    Logic: Accessing C:\Windows when Root is C:\Share must FAIL.
    """
    shared_root = r"C:\Share"
    conv = file_utils.GlobusPathConverter(layout="subfolder_root", shared_root=shared_root)
    
    # Path is valid Windows path, but outside the allowed scope
    outside_path = r"C:\Windows\System32\secret.dll"
    
    with pytest.raises(ValueError, match="is outside the shared root"):
        conv.windows_to_globus(outside_path)