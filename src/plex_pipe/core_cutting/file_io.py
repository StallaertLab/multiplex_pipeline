import os
import random
import ssl
import time
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from typing import Any, Union

import dask.array as da
import zarr
from globus_sdk import (
    GlobusAPIError,
    GlobusConnectionError,
    GlobusTimeoutError,
    TransferData,
)
from loguru import logger
from requests.exceptions import (
    RequestException,  # if requests is available/used
)
from tifffile import imread, imwrite

from plex_pipe.utils.globus_utils import (
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
    def is_channel_ready(self, channel: str, path: str) -> bool:
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
        self.submit_all_transfers()

    def submit_all_transfers(self) -> None:
        """Submit transfer tasks for all channels."""

        for channel, (remote_path, local_path) in self.transfer_map.items():

            try:
                task_id = self._submit_transfer(remote_path, local_path)
            except (GlobusAPIError, RuntimeError) as e:
                # Fail immediately
                raise RuntimeError(
                    f"Transfer submission failed for {channel} "
                    f"({remote_path} -> {local_path}): {e}"
                ) from e

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
                self.tc.endpoint_autoactivate(self.source_endpoint, if_expires_in=3600)
                self.tc.endpoint_autoactivate(
                    self.destination_endpoint, if_expires_in=3600
                )

                submission_result = self.tc.submit_transfer(transfer_data)

                return submission_result["task_id"]

            except GlobusAPIError as e:
                # Retry only on transient service/network side errors
                if e.http_status in RETRYABLE_STATUSES:
                    delay = self._sleep_with_backoff(
                        attempt, delay, remote_path, local_path, e
                    )
                    continue
                # Non-retryable Globus API error: re-raise to caller (caught in submit_all_transfers)
                raise

            # ONLY transient, non-API exceptions are retried here
            except (
                GlobusConnectionError,
                GlobusTimeoutError,
                RequestException,
                TimeoutError,
                ConnectionError,
                ssl.SSLError,
                OSError,
            ) as e:
                # Unknown / network hiccup → treat as transient
                delay = self._sleep_with_backoff(
                    attempt, delay, remote_path, local_path, e
                )

        # Exhausted retries
        message = f"Exhausted retries submitting {remote_path} -> {local_path}"
        logger.error(message)
        raise RuntimeError(message)

    def _sleep_with_backoff(
        self,
        attempt: int,
        delay: float,
        remote_path: str,
        local_path: str,
        err: Exception,
    ) -> float:
        """Log, sleep, and compute the next backoff delay."""
        sleep_for = delay + random.uniform(0, 0.5 * delay)
        logger.warning(
            f"[submit retry {attempt}/{MAX_TRIES}] {type(err).__name__}: {err}; "
            f"{remote_path} -> {local_path}; sleeping {sleep_for:.1f}s"
        )
        time.sleep(sleep_for)
        return min(delay * 2, MAX_DELAY)

    def is_channel_ready(self, channel: str, path: str = None) -> bool:
        """Return True when the file for `channel` is ready; False if still pending.
        Raises on impossible states or hard failures."""

        if channel in self.already_available:
            return True

        # find the (single) pending task for this channel
        idx = next(
            (i for i, (_, _, ch) in enumerate(self.pending) if ch == channel),
            None,
        )
        if idx is None:
            # with fail-fast submit, this should never happen
            raise RuntimeError(f"No pending task for {channel}.")

        task_id, local_path, _ = self.pending[idx]
        task = self.tc.get_task(task_id)
        status = task["status"]

        if status == "SUCCEEDED":
            self.pending.pop(idx)
            self.already_available.add(channel)
            logger.info(f"Transfer for {channel} complete: {local_path}")
            return True

        if status == "FAILED":
            msg = f"Transfer failed for {channel} (task {task_id})."
            logger.error(msg)
            raise RuntimeError(msg)

        # e.g. ACTIVE, INACTIVE, QUEUED, etc.
        return False

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

    def is_channel_ready(self, channel: str, path: str) -> bool:
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
