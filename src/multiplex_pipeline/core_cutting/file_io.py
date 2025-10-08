import os
import time
import random
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from typing import Any, Union

import dask.array as da
import zarr
from globus_sdk import TransferData, GlobusAPIError
from loguru import logger
from tifffile import imread, imwrite

from multiplex_pipeline.utils.globus_utils import (
    GlobusConfig,
    create_globus_tc,
)

RETRYABLE_STATUSES = {502, 503, 504}
MAX_TRIES = 6
BASE_DELAY = 2.0  # seconds
MAX_DELAY = 60.0  # seconds


class FileAvailabilityStrategy(ABC):
    """Strategy interface for ensuring image files are available."""

    @abstractmethod
    def fetch_or_wait(self, channel: str, path: str) -> bool:
        """Return ``True`` when the requested file is ready locally."""

    @abstractmethod
    def cleanup(self, path: Path) -> None:
        """Remove or close the given file path."""


class GlobusFileStrategy(FileAvailabilityStrategy):
    """Fetch files from a remote Globus endpoint."""

    def __init__(
        self,
        tc,
        transfer_map: dict[str, tuple[str, str]],
        gc: GlobusConfig,
        cleanup_enabled: bool = True,
    ) -> None:
        """Create the strategy and submit initial transfers.

        Args:
            tc: Authenticated ``TransferClient`` instance.
            transfer_map (dict[str, tuple[str, str]]): Mapping from channel to
                ``(remote_path, local_path)`` pairs.
            gc (GlobusConfig): Configuration containing endpoint identifiers.
            cleanup_enabled (bool, optional): Remove files after use.
        """

        self.tc = tc
        self.gc = gc
        self.transfer_map = transfer_map
        self.source_endpoint = gc.source_collection_id
        self.destination_endpoint = gc.destination_collection_id
        self.pending = []  # (task_id, local_path, channel)
        self.already_available = set()
        self.cleanup_enabled = cleanup_enabled
        self.failed = []
        self.submit_all_transfers()

    def submit_all_transfers(self) -> None:
        """Submit transfer tasks for all channels."""

        # Best-effort endpoint (re)activation (non-fatal if it fails)
        try:
            self.tc.endpoint_autoactivate(self.source_endpoint, if_expires_in=3600)
        except Exception as e:
            logger.warning(f"Autoactivate source failed (non-fatal): {e}")
        try:
            self.tc.endpoint_autoactivate(self.destination_endpoint, if_expires_in=3600)
        except Exception as e:
            logger.warning(f"Autoactivate dest failed (non-fatal): {e}")

        for channel, (remote_path, local_path) in self.transfer_map.items():

            try:
                task_id = self._submit_transfer(remote_path, local_path)
            except Exception as e:
                # Make sure one failing submit doesn't kill the whole run
                msg = f"Submit failed for {channel} ({remote_path} -> {local_path}): {e}"
                logger.error(msg)
                self.failed.append(channel)
                continue

            self.pending.append((task_id, local_path, channel))
            logger.info(
                f"Submitted transfer for {channel} to {local_path} (task_id={task_id})"
            )

    def _submit_transfer(self, remote_path: str, local_path: str) -> str:
        """Submit a single Globus transfer.

        Args:
            remote_path (str): Source path on the remote endpoint.
            local_path (str): Destination path on the local endpoint.

        Returns:
            str: ID of the submitted transfer task.
        """
        transfer_data = TransferData(
            source_endpoint=self.source_endpoint,
            destination_endpoint=self.destination_endpoint,
            label="Core Image Transfer",
            sync_level="checksum",
            verify_checksum=True,
            notify_on_succeeded=False,
            notify_on_failed=False,
            notify_on_inactive=False,
        )
        transfer_data.add_item(remote_path, local_path)

        delay = BASE_DELAY

        for attempt in range(1, MAX_TRIES + 1):
            try:
                # opportunistic auto-activation before each attempt
                try:
                    self.tc.endpoint_autoactivate(self.source_endpoint, if_expires_in=3600)
                    self.tc.endpoint_autoactivate(self.destination_endpoint, if_expires_in=3600)
                except Exception:
                    # non-fatal; the submit itself will surface real auth/activation problems
                    pass

                submission_result = self.tc.submit_transfer(transfer_data)
                return submission_result["task_id"]

            except GlobusAPIError as e:
                # Retry only on transient service/network side errors
                if e.http_status in RETRYABLE_STATUSES:
                    sleep_for = delay + random.uniform(0, 0.5 * delay)
                    logger.warning(
                        f"[submit retry {attempt}/{MAX_TRIES}] HTTP {e.http_status} "
                        f"for {remote_path} -> {local_path}; sleeping {sleep_for:.1f}s"
                    )
                    time.sleep(sleep_for)
                    delay = min(delay * 2, MAX_DELAY)
                    continue
                # Non-retryable Globus API error: re-raise to caller (caught in submit_all_transfers)
                raise

            except Exception as e:
                # Unknown / network hiccup → treat as transient
                sleep_for = delay + random.uniform(0, 0.5 * delay)
                logger.warning(
                    f"[submit retry {attempt}/{MAX_TRIES}] {type(e).__name__}: {e}; "
                    f"sleeping {sleep_for:.1f}s"
                )
                time.sleep(sleep_for)
                delay = min(delay * 2, MAX_DELAY)

        # Exhausted retries
        raise RuntimeError(f"Exhausted retries submitting {remote_path} -> {local_path}")


    def fetch_or_wait(self, channel: str, path: str) -> bool:
        """Check the status of a transfer and wait if necessary.

        Args:
            channel (str): Channel name being fetched.
            path (str): Local path expected for the file.

        Returns:
            bool: ``True`` when the file is ready.
        """

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
                msg = f"Transfer failed for {channel} (task {task_id})."
                logger.error(msg)
                self.failed.append(channel)
            else:
                still_pending.append((task_id, local_path, ch))

        self.pending = still_pending

        # If no task was submitted 
        if not found_task:
            logger.error(
                f"Transfer/check for {channel} failed. No task found."
            )
            ready = False

        return ready

    def cleanup(self, path: Path, force: bool = False) -> None:
        """Remove the specified file if cleanup is enabled."""

        if not self.cleanup_enabled and not force:
            logger.info(f"Skipping cleanup for {path}; cleanup is disabled.")
            return

        try:
            if path.exists():
                path.unlink()
                logger.info(f"Cleaned up file: {path}")
        except OSError as exc:
            logger.warning(f"Cleanup failed for {path}: {exc}")


class LocalFileStrategy(FileAvailabilityStrategy):
    """Strategy that relies on files already present locally."""

    def fetch_or_wait(self, channel: str, path: str) -> bool:
        """Return ``True`` if the file exists locally."""
        return Path(path).exists()

    def cleanup(self, path: Path) -> None:
        """Local files are left untouched."""


# Supporting file I/O functions
def write_temp_tiff(array, core_id: str, channel: str, temp_dir: str):
    """Save an array as ``temp/<core_id>/<channel>.tiff``.

    Args:
        array (numpy.ndarray): Image data to save.
        core_id (str): Core identifier.
        channel (str): Channel name.
        temp_dir (str): Base directory for temporary files.
    """
    core_path = os.path.join(temp_dir, core_id)
    os.makedirs(core_path, exist_ok=True)
    fname = os.path.join(core_path, f"{channel}.tiff")
    imwrite(fname, array)


def read_ome_tiff(path: str, level_num: int = 0) -> tuple[da.Array, Any]:
    """Load an OME-TIFF as a Dask array.

    Args:
        path (str): Path to the OME-TIFF file.
        level_num (int, optional): Multiscale level to read.

    Returns:
        tuple[dask.array.Array, Any]: The image array and the underlying store.
    """
    store = imread(path, aszarr=True)
    group = zarr.open(store, mode="r")
    zattrs = group.attrs.asdict()
    path = zattrs["multiscales"][0]["datasets"][level_num]["path"]

    return da.from_zarr(group[path]), store


def list_local_files(image_dir: Union[str, Path]) -> list[str]:
    """List ``*.ome.tif*`` files within a directory.

    Args:
        image_dir (str | Path): Directory to search.

    Returns:
        list[str]: Sorted list of matching file paths.
    """
    image_dir = Path(image_dir)  # Ensures uniform behavior
    return [str(p) for p in image_dir.glob("*.ome.tif*")]


def list_globus_files(gc: GlobusConfig, path: str) -> list[str]:
    """List ``*.ome.tif*`` files from a Globus endpoint.

    Args:
        gc (GlobusConfig): Globus configuration object.
        path (str): Remote directory to list.

    Returns:
        list[str]: Paths to files on the remote endpoint.
    """

    tc = create_globus_tc(gc.client_id, gc.transfer_tokens)
    listing = tc.operation_ls(gc.source_collection_id, path=str(path))

    files = []
    for entry in listing:
        name = entry["name"]
        if name.endswith((".ome.tif", ".ome.tiff")):
            files.append(str(PurePosixPath(path) / name))

    return files
