import pytest
import os
from unittest.mock import MagicMock, patch, call, ANY
from globus_sdk import GlobusAPIError, GlobusConnectionError

# Import module under test
import multiplex_pipeline.core_cutting.file_io as file_io

# --- Fixtures ---

@pytest.fixture
def mock_globus_config():
    """Mocks the Globus configuration object."""
    gc = MagicMock()
    gc.source_collection_id = "source-uuid"
    gc.destination_collection_id = "dest-uuid"
    gc.client_id = "client-id"
    gc.transfer_tokens = {"transfer.api.globus.org": "token"}
    return gc

@pytest.fixture
def mock_tc():
    """Mocks the Globus TransferClient."""
    tc = MagicMock()
    # Default successful submission
    tc.submit_transfer.return_value = {"task_id": "task-123"}
    # Default autoactivate success
    tc.endpoint_autoactivate.return_value = {"code": "AutoActivated"}
    return tc

@pytest.fixture
def transfer_map():
    return {
        "DAPI": ("/remote/dapi.tif", "/local/dapi.tif"),
        "CD45": ("/remote/cd45.tif", "/local/cd45.tif")
    }

# --- Tests for GlobusFileStrategy: Submission & Retries ---

def test_globus_submit_success(mock_tc, mock_globus_config, transfer_map):
    """
    Verifies that initializing the strategy immediately submits transfers 
    for all channels in the map.
    """
    strategy = file_io.GlobusFileStrategy(mock_tc, transfer_map, mock_globus_config)
    
    # Check that transfer was submitted twice (once per channel)
    assert mock_tc.submit_transfer.call_count == 2
    
    # Verify internal state
    assert len(strategy.pending) == 2
    # pending item structure: (task_id, local_path, channel)
    assert strategy.pending[0] == ("task-123", "/local/dapi.tif", "DAPI")

@patch("multiplex_pipeline.core_cutting.file_io.time.sleep")
def test_globus_submit_retry_logic(mock_sleep, mock_tc, mock_globus_config):
    """
    Critical Test: Verifies that transient errors (e.g., 503 Service Unavailable)
    trigger a retry loop and eventually succeed.
    """
    # Setup transfer map with 1 item
    t_map = {"DAPI": ("/remote", "/local")}
    
    # Mock behavior: 
    # Call 1: 503 Error (Retryable)
    # Call 2: Connection Error (Retryable)
    # Call 3: Success
    
    # Construct a fake 503 error
    err_503 = GlobusAPIError(MagicMock())
    err_503.http_status = 503
    
    mock_tc.submit_transfer.side_effect = [
        err_503,
        # Use standard ConnectionError instead of the complex GlobusConnectionError
        # This triggers the exact same retry logic in your code
        ConnectionError("Simulated network failure"), 
        {"task_id": "task-retry-success"}
    ]
    
    # Initialize strategy
    strategy = file_io.GlobusFileStrategy(mock_tc, t_map, mock_globus_config)
    
    # Assertions
    # Should have called submit 3 times total
    assert mock_tc.submit_transfer.call_count == 3
    # Should have slept twice
    assert mock_sleep.call_count == 2
    # Verify it ultimately succeeded and stored the task
    assert strategy.pending[0][0] == "task-retry-success"

@patch("multiplex_pipeline.core_cutting.file_io.time.sleep")
def test_globus_submit_exhausted_retries(mock_sleep, mock_tc, mock_globus_config):
    """
    Verifies that after MAX_TRIES, the code raises a RuntimeError.
    """
    t_map = {"DAPI": ("/remote", "/local")}
    
    # Always raise 503
    err_503 = GlobusAPIError(MagicMock())
    err_503.http_status = 503
    mock_tc.submit_transfer.side_effect = err_503
    
    with pytest.raises(RuntimeError, match="Transfer submission failed"):
        file_io.GlobusFileStrategy(mock_tc, t_map, mock_globus_config)
        
    # Should have tried MAX_TRIES (6) times
    assert mock_tc.submit_transfer.call_count == 6

def test_globus_submit_fatal_error(mock_tc, mock_globus_config):
    """
    Verifies that non-retryable errors (e.g. 404 Not Found, 403 Forbidden)
    fail immediately without retrying.
    """
    t_map = {"DAPI": ("/remote", "/local")}
    
    # 404 Error
    err_404 = GlobusAPIError(MagicMock())
    err_404.http_status = 404
    mock_tc.submit_transfer.side_effect = err_404
    
    with pytest.raises(RuntimeError):
        file_io.GlobusFileStrategy(mock_tc, t_map, mock_globus_config)
        
    # Should only try once
    assert mock_tc.submit_transfer.call_count == 1

# --- Tests for GlobusFileStrategy: Status Checking ---

def test_is_channel_ready_succeeded(mock_tc, mock_globus_config, transfer_map):
    """
    Scenario: Globus task finishes successfully.
    Expected: Return True, remove from pending, add to available.
    """
    strategy = file_io.GlobusFileStrategy(mock_tc, transfer_map, mock_globus_config)
    
    # Mock get_task response
    mock_tc.get_task.return_value = {"status": "SUCCEEDED"}
    
    # Check DAPI
    is_ready = strategy.is_channel_ready("DAPI")
    
    assert is_ready is True
    assert "DAPI" in strategy.already_available
    # DAPI should be removed from pending list
    assert len([p for p in strategy.pending if p[2] == "DAPI"]) == 0

def test_is_channel_ready_failed(mock_tc, mock_globus_config, transfer_map):
    """
    Scenario: Globus task fails.
    Expected: Raise RuntimeError immediately to stop pipeline.
    """
    strategy = file_io.GlobusFileStrategy(mock_tc, transfer_map, mock_globus_config)
    
    mock_tc.get_task.return_value = {"status": "FAILED"}
    
    with pytest.raises(RuntimeError, match="Transfer failed for DAPI"):
        strategy.is_channel_ready("DAPI")

def test_is_channel_ready_active(mock_tc, mock_globus_config, transfer_map):
    """
    Scenario: Globus task is still running.
    Expected: Return False.
    """
    strategy = file_io.GlobusFileStrategy(mock_tc, transfer_map, mock_globus_config)
    
    mock_tc.get_task.return_value = {"status": "ACTIVE"}
    
    assert strategy.is_channel_ready("DAPI") is False

# --- Tests for LocalFileStrategy ---

def test_local_strategy(tmp_path):
    """
    Verifies simple local filesystem checks.
    """
    strategy = file_io.LocalFileStrategy()
    
    # Create a dummy file
    dapi_path = tmp_path / "dapi.tif"
    dapi_path.touch()
    
    # Test existing
    assert strategy.is_channel_ready("DAPI", str(dapi_path)) is True
    
    # Test missing
    assert strategy.is_channel_ready("CD45", str(tmp_path / "missing.tif")) is False

# --- Tests for Helper Functions ---

@patch("multiplex_pipeline.core_cutting.file_io.create_globus_tc")
def test_list_globus_files_filtering(mock_create_tc, mock_globus_config):
    """
    Verifies that list_globus_files correctly filters for .ome.tif files
    and ignores other junk files.
    """
    mock_tc = MagicMock()
    mock_create_tc.return_value = mock_tc
    
    # Mock directory listing
    mock_tc.operation_ls.return_value = [
        {"name": "image1.ome.tif"},
        {"name": "image2.ome.tiff"}, # check alternate extension
        {"name": "metadata.csv"},     # should be ignored
        {"name": "notes.txt"}         # should be ignored
    ]
    
    files = file_io.list_globus_files(mock_globus_config, "/remote/path")
    
    assert len(files) == 2
    assert "/remote/path/image1.ome.tif" in files[0] # Checks full path construction

@patch("multiplex_pipeline.core_cutting.file_io.imread")
@patch("multiplex_pipeline.core_cutting.file_io.da.from_zarr")
@patch("multiplex_pipeline.core_cutting.file_io.zarr.open")
def test_read_ome_tiff(mock_zarr_open, mock_da, mock_imread):
    """
    Verifies that OME-TIFFs are opened as Zarr stores for lazy loading.
    """
    # Setup mocks
    mock_group = MagicMock()
    mock_group.attrs.asdict.return_value = {
        "multiscales": [{"datasets": [{"path": "0"}]}]
    }
    mock_zarr_open.return_value = mock_group
    
    # Call function
    arr, store = file_io.read_ome_tiff("path.ome.tif")
    
    # Check that it opened with aszarr=True (Efficient reading)
    mock_imread.assert_called_with("path.ome.tif", aszarr=True)
    # Check it accessed the correct Zarr path
    mock_group.__getitem__.assert_called_with("0")