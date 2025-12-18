import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call

# Import module under test
from multiplex_pipeline.core_cutting.controller import CorePreparationController as controller
from multiplex_pipeline.core_cutting.file_io import FileAvailabilityStrategy

# --- Fixtures ---

@pytest.fixture
def mock_metadata():
    """Creates a dummy metadata DataFrame with 2 cores."""
    return pd.DataFrame({
        "core_name": ["Core_01", "Core_02"],
        "geometry": ["poly1", "poly2"] 
    })

@pytest.fixture
def mock_image_paths():
    return {
        "DAPI": "/path/to/dapi.tif",
        "CD45": "/path/to/cd45.tif"
    }

@pytest.fixture
def mock_dependencies():
    """
    Mocks Cutter, Assembler, and File Strategy.
    """
    with patch("multiplex_pipeline.core_cutting.controller.CoreCutter") as MockCutter, \
         patch("multiplex_pipeline.core_cutting.controller.CoreAssembler") as MockAssembler, \
         patch("multiplex_pipeline.core_cutting.controller.read_ome_tiff") as mock_read, \
         patch("multiplex_pipeline.core_cutting.controller.write_temp_tiff") as mock_write, \
         patch("multiplex_pipeline.core_cutting.controller.os.makedirs"):
        
        # Setup specific mock behaviors
        mock_strategy = MagicMock(spec=FileAvailabilityStrategy)
        
        # Setup read_ome_tiff to return a mock array and a mock store (for closing)
        mock_store = MagicMock()
        mock_img = np.zeros((100, 100))
        mock_read.return_value = (mock_img, mock_store)
        
        yield {
            "Cutter": MockCutter,
            "Assembler": MockAssembler,
            "read_ome_tiff": mock_read,
            "store": mock_store,
            "write_temp_tiff": mock_write,
            "strategy": mock_strategy
        }

@pytest.fixture
def test_controller(mock_metadata, mock_image_paths, mock_dependencies):
    """Instantiates the controller with mocked dependencies."""
    return controller(
        metadata_df=mock_metadata,
        image_paths=mock_image_paths,
        temp_dir="/tmp/cores",
        output_dir="/tmp/output",
        file_strategy=mock_dependencies["strategy"]
    )

# --- Tests for Cutting Logic ---

def test_cut_channel_logic(test_controller, mock_dependencies, mock_metadata):
    """
    Verifies that cutting a channel iterates through metadata, 
    extracts cores, and writes temp files.
    """
    # Action
    test_controller.cut_channel("DAPI", "/path/to/dapi.tif")
    
    # 1. Verify Image Loading
    mock_dependencies["read_ome_tiff"].assert_called_with("/path/to/dapi.tif")
    
    # 2. Verify Cutting (called once per core)
    cutter_instance = test_controller.cutter
    assert cutter_instance.extract_core.call_count == len(mock_metadata)
    
    # 3. Verify Writing
    assert mock_dependencies["write_temp_tiff"].call_count == len(mock_metadata)
    
    # 4. Verify Internal State Update
    # Both cores should now mark "DAPI" as ready
    assert "DAPI" in test_controller.ready_cores["Core_01"]
    assert "DAPI" in test_controller.ready_cores["Core_02"]

def test_cut_channel_resource_safety(test_controller, mock_dependencies):
    """
    Ensures file handles are closed even if cutting crashes.
    """
    # Simulate a crash during core extraction
    test_controller.cutter.extract_core.side_effect = RuntimeError("Cutting failed!")
    
    with pytest.raises(RuntimeError):
        test_controller.cut_channel("DAPI", "bad_file.tif")
        
    # The store.close() must still be called
    mock_dependencies["store"].close.assert_called_once()

# --- Tests for Assembly Logic (The Consumer) ---

def test_try_assemble_partial_ready(test_controller, mock_dependencies):
    """
    Scenario: Only 1 of 2 required channels is ready.
    Expected: No assembly should happen.
    """
    # Setup state: Core_01 has DAPI, but needs DAPI + CD45
    test_controller.ready_cores["Core_01"] = {"DAPI"}
    
    test_controller.try_assemble_ready_cores()
    
    # Verify assembler NOT called
    test_controller.assembler.assemble_core.assert_not_called()
    # Verify core remains in waiting list
    assert "Core_01" in test_controller.ready_cores

def test_try_assemble_all_ready(test_controller, mock_dependencies):
    """
    Scenario: All required channels are ready.
    Expected: Assembly triggers and core is removed from waiting list.
    """
    # Setup state: Core_01 has both DAPI and CD45
    test_controller.ready_cores["Core_01"] = {"DAPI", "CD45"}
    
    test_controller.try_assemble_ready_cores()
    
    # Verify assembler CALLED
    test_controller.assembler.assemble_core.assert_called_with("Core_01")
    # Verify core removed from waiting list
    assert "Core_01" not in test_controller.ready_cores

# --- Tests for the Main Run Loop (Integration) ---

@patch("time.sleep") # Prevent actual sleeping
def test_run_loop_workflow(mock_sleep, test_controller, mock_dependencies):
    """
    Simulates a full run where files appear sequentially.
    Iteration 1: DAPI appears -> Cut DAPI.
    Iteration 2: CD45 appears -> Cut CD45 -> Assemble All -> Exit.
    """
    strategy = mock_dependencies["strategy"]
    
    # Mock behavior of is_channel_ready:
    # First call (DAPI) -> True
    # Second call (CD45) -> False (simulate delay)
    # Third call (DAPI) -> Already done (skipped logic)
    # Fourth call (CD45) -> True
    # This requires careful side_effect setup or state management
    
    # We will simulate the loop manually by checking logic flow rather than
    # trusting side_effects which can get messy in loops.
    
    # -- Iteration 1 --
    # DAPI ready, CD45 not
    strategy.is_channel_ready.side_effect = lambda ch, path: ch == "DAPI"
    
    # We can't easily break the infinite loop in `run` without raising an exception 
    # or mocking `all_ready` logic which is internal. 
    # Instead, we force the loop to break by making the file strategy 
    # return True for everything eventually.
    
    # Let's set the strategy to return True for DAPI, then True for CD45 
    # in subsequent calls.
    strategy.is_channel_ready.side_effect = [True, False, True] 
    # Call 1: DAPI (True)
    # Call 2: CD45 (False) -> Loop continues
    # Call 3: CD45 (True - next iteration) -> Loop finishes
    
    # Run the controller
    test_controller.run(poll_interval=0.01)
    
    # Assertions
    
    # 1. Check Cutting Order
    # cut_channel should be called twice (DAPI, then CD45)
    # test controller.cut_channel is not directly mockable,
    assert mock_dependencies["read_ome_tiff"].call_count == 2
    
    # 2. Check Cleanup
    # strategy.cleanup should be called for each path
    assert strategy.cleanup.call_count == 2
    
    # 3. Check Assembly
    # Should happen after CD45 is cut
    assert test_controller.assembler.assemble_core.call_count == 2 # 2 cores assembled
    
    # 4. Check Completed Set
    assert test_controller.completed_channels == {"DAPI", "CD45"}

def test_run_loop_cleanup_trigger(test_controller, mock_dependencies):
    """
    Verifies that the file strategy's cleanup method is called immediately 
    after cutting a channel. This is important for disk space management.
    """
    with patch("time.sleep"):
        strategy = mock_dependencies["strategy"]
        strategy.is_channel_ready.return_value = True # All files ready immediately
        
        test_controller.run(poll_interval=0)
        
        # Verify cleanup called with Path objects
        assert strategy.cleanup.call_count == 2