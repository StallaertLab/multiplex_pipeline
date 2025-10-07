import os
import re
from globus_sdk import GlobusAPIError
from pathlib import Path, PureWindowsPath, PurePosixPath

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

def globus_dir_exists(tc, endpoint_id, path):

    try:
        tc.operation_ls(endpoint_id, path=path)
        return True
    except GlobusAPIError as e:
        if e.code == "ClientError.NotFound":
            return False
        raise  # some other error, bubble it up

class GlobusPathConverter:
    """Converts local Windows file paths to Globus-compatible POSIX-style paths.

    This class converts Windows-style file paths (e.g., ``C:\\Users\\Kasia\\Documents\\file.txt``)
    into Globus-compatible POSIX paths (e.g., ``/C/Users/Kasia/Documents/file.txt``),
    according to how the Globus Connect Personal endpoint is configured.

    The converter supports three endpoint layout modes:

    - **multi_drive**:
        Each Windows drive (C, D, E, etc.) appears as a top-level folder under
        the Globus root (e.g., ``/C``, ``/D``). This layout is typical when
        the shared directory is set to the full system root (``C:\\``) and multiple
        drives are available in Globus Connect Personal.  
        Example:
            ``C:\\Users\\Kasia\\Documents\\file.txt`` → ``/C/Users/Kasia/Documents/file.txt``

    - **single_drive**:
        The endpoint root corresponds to a single drive (e.g., ``C:\\``), or to a
        network-mounted CIFS drive (e.g., ``R:\\``). In this case, the Globus root
        ``/`` maps directly to that drive, and no drive letter appears in the path.  
        Example:
            ``C:\\Users\\Kasia\\Documents\\file.txt`` → ``/Users/Kasia/Documents/file.txt``  
            ``R:\\Data\\Project1\\results.csv`` → ``/Data/Project1/results.csv``

    - **subfolder_root**:
        Only a specific subfolder of a drive (e.g., ``C:\\Users\\Kasia``) is shared.
        All accessible paths must lie within this folder, and conversion is relative
        to it. Attempting to convert a path outside the shared root will raise an error.  
        Example:
            ``C:\\Users\\Kasia\\Documents\\file.txt`` → ``/Documents/file.txt``

    Attributes:
        layout (str): The endpoint layout type. One of {"multi_drive", "single_drive", "subfolder_root"}.
        shared_root (str | None): The absolute Windows path of the shared root directory.
            Required only when ``layout == "subfolder_root"``.

    Example usage:
        >>> conv = GlobusPathConverter(layout="multi_drive")
        >>> conv.windows_to_globus(r"C:\\Users\\Kasia\\Documents\\file.txt")
        '/C/Users/Kasia/Documents/file.txt'

        >>> conv = GlobusPathConverter(layout="single_drive")
        >>> conv.windows_to_globus(r"C:\\Users\\Kasia\\Documents\\file.txt")
        '/Users/Kasia/Documents/file.txt'

        >>> conv = GlobusPathConverter(layout="single_drive")
        >>> conv.windows_to_globus(r"R:\\Project\\data.csv")
        '/Project/data.csv'

        >>> conv = GlobusPathConverter(layout="subfolder_root",
        ...                            shared_root=r"C:\\Users\\Kasia")
        >>> conv.windows_to_globus(r"C:\\Users\\Kasia\\Documents\\file.txt")
        '/Documents/file.txt'

    Raises:
        ValueError: If an invalid layout is provided, or if the path lies outside
            the shared root when using ``subfolder_root`` layout.
        RuntimeError: If ``shared_root`` is missing when required.
    """
    def __init__(self, layout: str, shared_root: str | None = None):
        """Initialize the GlobusPathConverter.

        Args:
            layout (str): The endpoint layout type. Must be one of
                {"multi_drive", "single_drive", "subfolder_root"}.
            shared_root (str | None): Absolute Windows path to the shared directory,
                required only for ``subfolder_root`` layout.

        Raises:
            ValueError: If an invalid layout is provided.
            RuntimeError: If ``shared_root`` is required but not provided.
        """
        self.layout = layout
        if layout not in {"multi_drive", "single_drive", "subfolder_root"}:
            raise ValueError("layout must be 'multi_drive', 'single_drive', or 'subfolder_root'")

        self.shared_root = os.path.normpath(shared_root) if shared_root else None

        if layout == "subfolder_root" and not shared_root:
            raise RuntimeError("shared_root is required when layout='subfolder_root'")

    def windows_to_globus(self, win_path: str) -> str:
        """Convert a Windows path to a Globus-compatible POSIX path.

        Args:
            win_path (str): Absolute Windows-style path (e.g., ``C:\\Data\\file.txt``).

        Returns:
            str: The equivalent Globus-compatible path (e.g., ``/C/Data/file.txt``).

        Raises:
            ValueError: If the path lies outside the shared root (in ``subfolder_root`` layout)
                or lacks a drive letter when ``multi_drive`` is used.
            RuntimeError: If ``shared_root`` is required but not initialized.
        """
        # Normalize Windows path
        p = PureWindowsPath(win_path)
        drive = (p.drive[:-1] if p.drive.endswith(':') else p.drive).upper()  # Extract drive letter
        parts_after_drive = p.parts[1:] if drive else p.parts

        if self.layout == "multi_drive":
            if not drive:
                raise ValueError("Path must include a drive (e.g., C:) for multi_drive layout.")
            posix = PurePosixPath("/", drive, *parts_after_drive)
            return str(posix)

        elif self.layout == "single_drive":
            # For single-drive or CIFS mounts (e.g., R:), root is mapped directly to /
            posix = PurePosixPath("/", *parts_after_drive)
            return str(posix)

        elif self.layout == "subfolder_root":
            if not self.shared_root:
                raise RuntimeError("shared_root not set for subfolder_root layout.")
            norm_win = os.path.normpath(win_path)
            if os.path.commonprefix([norm_win.lower(), self.shared_root.lower()]) != self.shared_root.lower():
                raise ValueError(f"Path is outside the shared root: {self.shared_root}")
            rel = os.path.relpath(norm_win, start=self.shared_root)
            rel_posix = PurePosixPath(*PureWindowsPath(rel).parts)
            return str(PurePosixPath("/", rel_posix))

        else:
            raise ValueError(f"Unsupported layout: {self.layout}")