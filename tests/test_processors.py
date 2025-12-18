import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call

# Import the controller and BaseOp
from multiplex_pipeline.processors.base import BaseOp
from multiplex_pipeline.processors.controller import ResourceBuildingController

# --- Fixtures ---

class MockBuilder(BaseOp):
    """A dummy builder for testing."""
    def __init__(self, output_type="labels"):
        self.output_type = output_type
        # BaseOp requires OUTPUT_TYPE enum; mocking that behavior:
        self.OUTPUT_TYPE = MagicMock()
        self.OUTPUT_TYPE.value = output_type

    def validate_io(self, inputs, outputs):
        # Pass-through validation for simplicity
        return inputs, outputs

    def run(self, *args):
        # Simply return the first input as the 'processed' result
        return args[0]

@pytest.fixture
def mock_sdata():
    """Mocks a SpatialData object."""
    sdata = MagicMock()
    # Behave like a dict for 'in' checks and retrieval
    sdata.__contains__.side_effect = lambda key: key in sdata._elements
    sdata.__getitem__.side_effect = lambda key: sdata._elements[key]
    sdata.__setitem__.side_effect = lambda key, val: sdata._elements.update({key: val})
    sdata.__delitem__.side_effect = lambda key: sdata._elements.pop(key)
    
    # Internal storage for the mock
    sdata._elements = {}
    
    # Mock elements_paths_on_disk returning a list of paths
    sdata.elements_paths_on_disk.return_value = []
    
    return sdata

@pytest.fixture
def controller(mock_sdata):
    """Creates a standard controller instance."""
    builder = MockBuilder(output_type="labels")
    return ResourceBuildingController(
        builder=builder,
        input_names=["input_img"],
        output_names=["output_label"],
        resolution_level=1,  # Non-zero to test scaling
        keep=False,
        overwrite=False
    )

# --- Tests for Validation Logic ---

def test_validate_elements_missing(controller, mock_sdata):
    """Verifies error raised when input element is missing from sdata."""
    # sdata is empty
    with pytest.raises(ValueError, match="Requested source mask 'input_img' not found"):
        controller.validate_elements_present(mock_sdata)

def test_validate_resolution_missing(controller, mock_sdata):
    """Verifies error when requested resolution level doesn't exist."""
    # Create an element with only 1 level (level 0)
    # Controller asks for level 1 (see fixture)
    mock_element = MagicMock()
    mock_element.items.return_value = {"0": "data"} # only 1 item
    mock_sdata._elements["input_img"] = mock_element

    with pytest.raises(ValueError, match="does not have resolution level 1"):
        controller.validate_resolution_present(mock_sdata)

# --- Tests for Overwrite Logic ---

def test_prepare_to_overwrite_error(controller, mock_sdata):
    """Verifies error when output exists and overwrite=False."""
    # Output name is "output_label"
    mock_sdata._elements["output_label"] = "existing_data"
    
    with pytest.raises(ValueError, match="already exists"):
        controller.prepare_to_overwrite(mock_sdata)

def test_prepare_to_overwrite_success(mock_sdata):
    """Verifies deletion when output exists and overwrite=True."""
    controller = ResourceBuildingController(
        builder=MockBuilder(),
        input_names=["in"], output_names=["out"],
        overwrite=True
    )
    
    mock_sdata._elements["out"] = "existing_data"
    mock_sdata.elements_paths_on_disk.return_value = ["path/to/out"]
    
    controller.prepare_to_overwrite(mock_sdata)
    
    # Should be removed from sdata dict
    assert "out" not in mock_sdata._elements
    # Should attempt to delete from disk
    mock_sdata.delete_element_from_disk.assert_called_with("out")

# --- Tests for Resolution & Packing Logic ---

def test_bring_to_max_resolution(controller):
    """
    Verifies upscaling logic.
    Controller configured with resolution_level=1, downscale=2.
    Factor = 2^1 = 2.
    """
    # 10x10 input
    small_arr = np.zeros((10, 10))
    
    upscaled = controller.bring_to_max_resolution(small_arr)
    
    # Should be 20x20
    assert upscaled.shape == (20, 20)

@patch("multiplex_pipeline.processors.controller.Labels2DModel")
def test_pack_into_model_labels(MockLabelsModel, controller):
    """Verifies packing logic for Labels."""
    controller.builder.OUTPUT_TYPE.value = "labels"
    data = np.zeros((100, 100))
    
    controller.pack_into_model(data)
    
    MockLabelsModel.parse.assert_called_once()
    # Check dims argument
    call_kwargs = MockLabelsModel.parse.call_args[1]
    assert call_kwargs["dims"] == ("y", "x")

@patch("multiplex_pipeline.processors.controller.Image2DModel")
def test_pack_into_model_image(MockImageModel, controller):
    """Verifies packing logic for Images (adds 'c' dimension)."""
    controller.builder.OUTPUT_TYPE.value = "image"
    data = np.zeros((100, 100))
    
    controller.pack_into_model(data)
    
    MockImageModel.parse.assert_called_once()
    # Check that data was expanded to have 'c' dim: data[None]
    args = MockImageModel.parse.call_args[1]
    assert args["data"].shape == (1, 100, 100)
    assert args["dims"] == ("c", "y", "x")

# --- Integration Test: The Full Run ---

@patch("multiplex_pipeline.processors.controller.sd.get_pyramid_levels")
@patch("multiplex_pipeline.processors.controller.resize") # Mock resize to save time
@patch("multiplex_pipeline.processors.controller.Labels2DModel")
def test_run_pipeline(MockLabelsModel, mock_resize, mock_get_pyramid, controller, mock_sdata):
    """
    Simulates a full run:
    1. Validation pass.
    2. Data fetching (mocked).
    3. Processing (via MockBuilder).
    4. Upscaling (via mock_resize).
    5. Packing.
    6. Saving to sdata.
    """
    # Setup inputs in sdata
    mock_sdata._elements["input_img"] = MagicMock() # Has levels
    # Mock resolution check to pass (length > 1)
    mock_sdata._elements["input_img"].items.return_value = {"0": "d", "1": "d"}
    
    # Mock Data Fetching
    # get_pyramid_levels returns the data array
    mock_get_pyramid.return_value = np.zeros((1, 50, 50)) # (c, y, x)
    
    # Mock Resize (Upscaling)
    mock_resize.return_value = np.zeros((100, 100))
    
    # Mock Model Packing
    mock_model_obj = MagicMock()
    MockLabelsModel.parse.return_value = mock_model_obj
    
    # Execute
    result_sdata = controller.run(mock_sdata)
    
    # Assertions
    
    # 1. Check Builder Run
    # Builder should receive the squeezed array (50, 50)
    # Our MockBuilder returns input, so result is (50, 50)
    
    # 2. Check Upscaling
    # Controller is set to level 1. It should call resize/bring_to_max_resolution
    mock_resize.assert_called_once()
    
    # 3. Check Saving to Sdata
    # "output_label" should now exist in sdata elements
    assert "output_label" in result_sdata._elements
    assert result_sdata._elements["output_label"] == mock_model_obj
    
    # 4. Check Disk Write (keep=False in fixture, so should NOT be called)
    result_sdata.write_element.assert_not_called()

def test_run_save_to_disk(mock_sdata):
    """Verifies that keep=True triggers write_element."""
    # Setup controller with keep=True
    builder = MockBuilder()
    controller = ResourceBuildingController(
        builder, ["in"], ["out"], keep=True, resolution_level=0
    )
    
    # Minimal setup to pass validation
    mock_sdata._elements["in"] = MagicMock()
    mock_sdata._elements["in"].items.return_value = {"0": "d"}
    
    # Mock data fetch
    with patch("multiplex_pipeline.processors.controller.sd.get_pyramid_levels") as mock_get:
        mock_get.return_value = np.zeros((1, 10, 10))
        
        # Run
        controller.run(mock_sdata)
        
    # Verify write called
    mock_sdata.write_element.assert_called_with("out")