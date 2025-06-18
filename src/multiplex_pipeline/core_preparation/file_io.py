import os
from io import BytesIO
from pathlib import Path, PurePosixPath
from abc import ABC, abstractmethod
from typing import Union
from globus_sdk import TransferData
import zarr
import dask.array as da
from tifffile import imread, imwrite
from loguru import logger

from multiplex_pipeline.utils.globus_utils import (
    GlobusConfig,
    create_globus_tc,
    get_with_globus_https,
)

class FileAvailabilityStrategy(ABC):
    @abstractmethod
    def fetch_or_wait(self, channel: str, path: str) -> bool:
        pass

    @abstractmethod
    def cleanup(self, path: Path):
        pass

class GlobusFileStrategy(FileAvailabilityStrategy):
    def __init__(self, tc, transfer_map: dict, gc: GlobusConfig, cleanup_enabled: bool = True):
        self.tc = tc
        self.gc = gc
        self.transfer_map = transfer_map
        self.source_endpoint = gc.r_collection_id
        self.destination_endpoint = gc.local_collection_id
        self.pending = []  # (task_id, local_path, channel)
        self.already_available = set()
        self.cleanup_enabled = cleanup_enabled
        self._submit_all_transfers()

    def _posix_to_windows_path(self, posix_path: str) -> Path:
        """
        Convert a Globus-style path like '/~/D/multiplex_pipeline/...' into a real Windows Path like 'D:/multiplex_pipeline/...'
        """
        try:
            parts = PurePosixPath(posix_path).parts
            if len(parts) >= 3 and parts[1] == "~":
                drive_letter = parts[2]
                relative_path = Path(*parts[3:])
                return Path(f"{drive_letter}:/") / relative_path
            return Path(posix_path)
        except Exception as e:
            logger.warning(f"Failed to convert {posix_path} to local path: {e}")
            return Path(posix_path)

    def _submit_all_transfers(self):
        for channel, (remote_path, local_path) in self.transfer_map.items():
            local_actual_path = self._posix_to_windows_path(local_path)

            if local_actual_path.exists():
                self.already_available.add(channel)
                logger.info(f"Skipping transfer for {channel}; file already exists: {local_actual_path}")
                continue

            task_id = self._submit_transfer(remote_path, local_path)
            self.pending.append((task_id, local_actual_path, channel))
            logger.info(f"Submitted transfer for {channel} to {local_path} (task_id={task_id})")

    def _submit_transfer(self, remote_path, local_path):
        transfer_data = TransferData(
            source_endpoint=self.source_endpoint,
            destination_endpoint=self.destination_endpoint,
            label="Core Image Transfer",
            sync_level="checksum",
            verify_checksum=True,
            notify_on_succeeded=False,  
            notify_on_failed=False,      
            notify_on_inactive=False    
        )
        transfer_data.add_item(remote_path, local_path)
        submission_result = self.tc.submit_transfer(transfer_data)
        return submission_result["task_id"]

    def fetch_or_wait(self, channel, path) -> bool:
        # Return immediately if we already verified it existed
        if channel in self.already_available:
            return True

        still_pending = []
        ready = False
        found_task = False

        for task_id, local_path, ch in self.pending:
            if ch != channel:
                still_pending.append((task_id, local_path, ch))
                continue

            found_task = True
            task = self.tc.get_task(task_id)

            if task["status"] == "SUCCEEDED":
                logger.info(f"Transfer for {channel} complete: {local_path}")
                ready = True
            elif task["status"] == "FAILED":
                logger.error(f"Transfer failed for {channel} (task {task_id}): {task.get('nice_status_details')}")
            else:
                still_pending.append((task_id, local_path, ch))

        self.pending = still_pending

        # If somehow no task was submitted AND it's not in already_available, check filesystem
        if not found_task:
            local_path = self._posix_to_windows_path(path)
            if Path(local_path).exists():
                self.already_available.add(channel)
                return True
            else:
                logger.warning(f"No transfer task for {channel}, and file not found: {local_path}")
                return False

        return ready

    def cleanup(self, path: Path, force: bool = False):
        if not self.cleanup_enabled and not force:
            logger.info(f"Skipping cleanup for {path}; cleanup is disabled.")
            return

        try:
            if path.exists():
                path.unlink()
                logger.info(f"Cleaned up file: {path}")
        except Exception as e:
            logger.warning(f"Cleanup failed for {path}: {e}")

class LocalFileStrategy(FileAvailabilityStrategy):
    def fetch_or_wait(self, channel, path) -> bool:
        return Path(path).exists()

    def cleanup(self, path: Path):
        pass

# Supporting file I/O functions
def write_temp_tiff(array, core_id: str, channel: str, temp_dir: str):
    """
    Save a NumPy array as TIFF in temp/core_id/channel.tiff
    """
    core_path = os.path.join(temp_dir, core_id)
    os.makedirs(core_path, exist_ok=True)
    fname = os.path.join(core_path, f"{channel}.tiff")
    imwrite(fname, array)


def read_ome_tiff(path: str,level_num: int = 0) -> da.Array:
    """
    Return a Dask array from an OME-TIFF file.
    Assumes channels are along axis 0.
    """
    store = imread(path,aszarr=True)
    group = zarr.open(store, mode='r')
    zattrs = group.attrs.asdict()
    path = zattrs['multiscales'][0]['datasets'][level_num]['path']
    
    return da.from_zarr(group[path]), store

def list_local_files(image_dir: Union[str, Path]) -> list[str]:
    """
    List local .ome.tif* files from a given directory path.
    Accepts either a string or a pathlib.Path object.
    """
    image_dir = Path(image_dir)  # Ensures uniform behavior
    return [str(p) for p in image_dir.glob("*.ome.tif*")]


def list_globus_files(gc: GlobusConfig, path: str) -> list[str]:
    tc = create_globus_tc(gc.client_id, gc.transfer_tokens)
    listing = tc.operation_ls(gc.r_collection_id, path=path)

    files = []
    for entry in listing:
        name = entry["name"]
        if name.endswith(".ome.tif") or name.endswith(".ome.tiff"):
            files.append(str(PurePosixPath(path) / name)) 

    return files