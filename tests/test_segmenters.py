import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Import module under test
# We assume the registry decorator logic works or is tested elsewhere,
# so we import the classes directly.
from multiplex_pipeline.processors.object_segmenters import (
    InstansegSegmenter,
    Cellpose4Segmenter
)

# --- Fixtures for Mocking External Libraries ---

@pytest.fixture
def mock_instanseg_lib():
    """Mocks the 'instanseg' library."""
    with patch.dict("sys.modules", {"instanseg": MagicMock()}):
        mock_model = MagicMock()
        # Mock the return value of InstanSeg constructor
        with patch("instanseg.InstanSeg", return_value=mock_model) as mock_cls:
            yield mock_cls, mock_model

@pytest.fixture
def mock_cellpose_lib():
    """Mocks the 'cellpose' library robustly."""
    # 1. Create the class mock
    mock_cellpose_model_cls = MagicMock()
    
    # 2. Capture the INSTANCE that will be created when the class is called
    # This is the object your test needs to configure (.eval, .run, etc.)
    mock_instance = mock_cellpose_model_cls.return_value

    # 3. Create the module/package structure
    mock_models_module = MagicMock()
    mock_models_module.CellposeModel = mock_cellpose_model_cls
    
    mock_cellpose_pkg = MagicMock()
    mock_cellpose_pkg.models = mock_models_module
    
    # 4. Patch sys.modules
    modules_to_patch = {
        "cellpose": mock_cellpose_pkg,
        "cellpose.models": mock_models_module
    }
    
    with patch.dict("sys.modules", modules_to_patch):
        # YIELD BOTH: The Class (for init checks) and The Instance (for run checks)
        yield mock_cellpose_model_cls, mock_instance

@pytest.fixture
def mock_torch():
    """Mocks torch for cuda cache clearing."""
    with patch("torch.cuda.empty_cache") as mock_empty:
        yield mock_empty

# --- Tests for InstansegSegmenter ---

def test_instanseg_init(mock_instanseg_lib):
    """Verifies initialization parameters are passed to the model."""
    mock_cls, _ = mock_instanseg_lib
    
    cfg = {
        "model": "fluorescence_nuclei_and_cells",
        "resolve_cell_and_nucleus": True
    }
    
    segmenter = InstansegSegmenter(**cfg)
    
    # Check if InstanSeg was initialized with correct model name
    mock_cls.assert_called_with("fluorescence_nuclei_and_cells", verbosity=1)
    
    # Check expected outputs logic
    assert segmenter.EXPECTED_OUTPUTS == 2

def test_instanseg_input_shaping(mock_instanseg_lib):
    """
    Verifies that inputs are standardized to (H, W, C).
    """
    _, mock_model = mock_instanseg_lib
    segmenter = InstansegSegmenter()
    
    # Mock eval return: (labeled_output, info)
    # labeled_output shape: (1, 2, H, W) for nucleus+cell
    mock_out = np.zeros((1, 2, 10, 10))
    mock_model.eval_medium_image.return_value = (mock_out, {})
    
    # Scenario 1: List of 2D arrays (e.g. DAPI, Cytoplasm)
    in1 = np.zeros((10, 10))
    in2 = np.zeros((10, 10))
    
    segmenter.run(in1, in2)
    
    # Check what was passed to eval_medium_image
    args, _ = mock_model.eval_medium_image.call_args
    passed_image = args[0]
    
    # Should be stacked to (10, 10, 2)
    assert passed_image.shape == (10, 10, 2)

def test_instanseg_run_single_channel(mock_instanseg_lib):
    """
    Verifies single channel 2D input handling.
    """
    _, mock_model = mock_instanseg_lib
    segmenter = InstansegSegmenter(resolve_cell_and_nucleus=False)
    
    mock_out = np.zeros((1, 1, 10, 10))
    mock_model.eval_medium_image.return_value = (mock_out, {})
    
    # Input: Single 2D array
    in_img = np.zeros((10, 10))
    
    segmenter.run(in_img)
    
    args, _ = mock_model.eval_medium_image.call_args
    passed_image = args[0]
    
    # Should expand to (10, 10, 1)
    assert passed_image.shape == (10, 10, 1)

def test_instanseg_cuda_cleanup(mock_instanseg_lib, mock_torch):
    """Verifies cuda cache clearing if enabled."""
    _, mock_model = mock_instanseg_lib
    segmenter = InstansegSegmenter(clean_cache=True)
    
    # Mock minimal output
    mock_model.eval_medium_image.return_value = (np.zeros((1, 1, 10, 10)), {})
    
    segmenter.run(np.zeros((10, 10)))
    
    mock_torch.assert_called_once()

# --- Tests for Cellpose4Segmenter ---

def test_cellpose_init(mock_cellpose_lib):
    """Verifies model initialization."""
    mock_cls, _ = mock_cellpose_lib
    
    segmenter = Cellpose4Segmenter(diameter=30)
    
    # Check GPU flag
    mock_cls.assert_called_with(gpu=True)

def test_cellpose_input_shaping(mock_cellpose_lib):
    """
    Verifies input standardization for Cellpose.
    """
    _, mock_model = mock_cellpose_lib
    segmenter = Cellpose4Segmenter()
    
    mock_model.eval.return_value = (np.zeros((10, 10)), None, None, None)
    
    # Scenario: 2 separate arrays passed as args
    in1 = np.zeros((10, 10))
    in2 = np.zeros((10, 10))
    
    segmenter.run(in1, in2)
    
    args, _ = mock_model.eval.call_args
    passed_image = args[0]
    
    # Should be stacked to (2, 10, 10) - (C, H, W) logic in prepare_input
    assert passed_image.shape == (2, 10, 10)

def test_cellpose_input_truncation(mock_cellpose_lib):
    """
    Verifies warning and truncation if >2 channels passed (Cellpose limitation).
    """
    _, mock_model = mock_cellpose_lib
    segmenter = Cellpose4Segmenter()
    mock_model.eval.return_value = (np.zeros((10, 10)), None, None, None)
    
    # Pass 3 images
    in1 = np.zeros((10, 10))
    
    with patch("multiplex_pipeline.processors.object_segmenters.logger") as mock_logger:
        segmenter.run(in1, in1, in1)
        
        # Should warn
        mock_logger.warning.assert_called_with("Cellpose will use only the first two provided channels.")
        
        # Should truncate input to 2 channels
        args, _ = mock_model.eval.call_args
        assert args[0].shape == (2, 10, 10)

def test_cellpose_invalid_input(mock_cellpose_lib):
    """Verifies error on non-2D inputs."""
    segmenter = Cellpose4Segmenter()
    
    # Pass a 3D volume instead of 2D slice
    bad_input = np.zeros((5, 10, 10))
    
    with pytest.raises(ValueError, match="Only 2D arrays are accepted"):
        segmenter.run(bad_input)

# --- Tests for Configuration Validation ---

def test_instanseg_extra_params():
    """Verifies that passing unknown params raises ValidationError."""
    with pytest.raises(Exception): # Pydantic ValidationError
        InstansegSegmenter(unknown_param=123)

def test_cellpose_params_passed(mock_cellpose_lib):
    """Verifies that params like diameter/flow_threshold reach the eval call."""
    _, mock_model = mock_cellpose_lib
    segmenter = Cellpose4Segmenter(diameter=50, flow_threshold=0.8)
    
    mock_model.eval.return_value = (np.zeros((10, 10)), None)
    
    segmenter.run(np.zeros((10, 10)))
    
    _, kwargs = mock_model.eval.call_args
    assert kwargs["diameter"] == 50
    assert kwargs["flow_threshold"] == 0.8