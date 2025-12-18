import pytest
import numpy as np
from unittest.mock import MagicMock, patch, ANY

# Import module under test
from multiplex_pipeline.processors.mask_builders import (
    SubtractionBuilder,
    MultiplicationBuilder,
    RingBuilder,
    BlobBuilder
)

# --- Tests for SubtractionBuilder ---

def test_subtraction_run_logic():
    """
    Verifies that mask_nucleus removes pixels from mask_cell.
    """
    builder = SubtractionBuilder()
    
    # 5x5 Cell (All 1s)
    mask_cell = np.ones((5, 5), dtype=int)
    
    # 3x3 Nucleus in the center
    mask_nucleus = np.zeros((5, 5), dtype=int)
    mask_nucleus[1:4, 1:4] = 1
    
    result = builder.run(mask_cell, mask_nucleus)
    
    # The center (1:4, 1:4) should now be 0
    assert np.all(result[1:4, 1:4] == 0)
    # The border (indices 0 and 4) should still be 1
    assert result[0, 0] == 1
    assert result[4, 4] == 1

def test_subtraction_shape_mismatch():
    """Verifies error if masks have different dimensions."""
    builder = SubtractionBuilder()
    m1 = np.zeros((10, 10))
    m2 = np.zeros((20, 20))
    
    with pytest.raises(ValueError, match="same shape"):
        builder.run(m1, m2)

# --- Tests for MultiplicationBuilder ---

def test_multiplication_run_logic():
    """
    Verifies element-wise multiplication (Intersection).
    """
    builder = MultiplicationBuilder()
    
    # Mask A: Vertical bar
    mask_a = np.zeros((3, 3), dtype=int)
    mask_a[:, 1] = 1  # Middle column
    
    # Mask B: Horizontal bar
    mask_b = np.zeros((3, 3), dtype=int)
    mask_b[1, :] = 1  # Middle row
    
    result = builder.run(mask_a, mask_b)
    
    # Result should be only the center pixel (1,1) where they overlap
    assert result[1, 1] == 1
    assert np.sum(result) == 1

def test_multiplication_shape_mismatch():
    """Verifies error if masks have different dimensions."""
    builder = MultiplicationBuilder()
    m1 = np.zeros((10, 10))
    m2 = np.zeros((10, 11))
    
    with pytest.raises(ValueError, match="same shape"):
        builder.run(m1, m2)

# --- Tests for RingBuilder ---

def test_ring_builder_validation():
    """
    Verifies that outer radius must be strictly greater than inner radius.
    """
    # Case: outer < inner
    with pytest.raises(ValueError, match="strictly greater"):
        RingBuilder(outer=3, inner=5)
        
    # Case: outer == inner
    with pytest.raises(ValueError, match="strictly greater"):
        RingBuilder(outer=5, inner=5)
        
    # Case: Valid
    assert RingBuilder(outer=5, inner=3).params.outer == 5

def test_ring_builder_run_logic():
    """
    Verifies that a hole is created in the mask.
    We use a single pixel label and expand it.
    """
    # outer=2, inner=1
    # Original pixel at (5,5).
    # Expanded (outer=2) covers radius 2.
    # Expanded (inner=1) covers radius 1.
    # Result should have pixels at distance 2, but NOT at distance 0 or 1.
    builder = RingBuilder(outer=2, inner=1)
    
    mask = np.zeros((11, 11), dtype=int)
    mask[5, 5] = 1 # Center pixel
    
    # Using real skimage logic (no mock) to test geometry
    result = builder.run(mask)
    
    # 1. The center (original label) should be hollowed out
    assert result[5, 5] == 0
    
    # 2. A pixel at distance 2 (e.g., 5,7) should be set
    # Distance from (5,5) to (5,7) is 2.
    assert result[5, 7] == 1

# --- Tests for BlobBuilder ---

def test_blob_builder_params():
    """Verifies positive dimensions check."""
    # Invalid negative shape
    with pytest.raises(ValueError, match="positive integers"):
        BlobBuilder(work_shape=(-10, 100))
        
    # Valid
    builder = BlobBuilder(work_shape=(100, 100))
    assert builder.params.work_shape == (100, 100)

@patch("multiplex_pipeline.processors.mask_builders.resize")
@patch("multiplex_pipeline.processors.mask_builders.opening")
@patch("multiplex_pipeline.processors.mask_builders.closing")
def test_blob_builder_workflow(mock_closing, mock_opening, mock_resize):
    """
    Verifies the pipeline sequence:
    Downsample -> Open -> Close -> Upsample.
    """
    builder = BlobBuilder(work_shape=(50, 50), radius=3)
    
    input_mask = np.zeros((100, 100), dtype=int)
    
    # Mock returns
    mock_resize.return_value = np.zeros((50, 50)) # downsampled
    mock_opening.return_value = np.zeros((50, 50))
    mock_closing.return_value = np.zeros((50, 50))
    
    # For the second resize call (upsample), we need it to return correct shape
    # The code calls resize twice.
    # 1. Downsample to work_shape
    # 2. Upsample to orig_shape
    def resize_side_effect(img, shape, **kwargs):
        return np.zeros(shape)
    
    mock_resize.side_effect = resize_side_effect
    
    # Execute
    builder.run(input_mask)
    
    # Assertions
    # 1. Check Downsampling
    mock_resize.assert_any_call(
        ANY, (50, 50), order=1, preserve_range=True
    )
    
    # 2. Check Morphological Ops
    mock_opening.assert_called()
    mock_closing.assert_called()
    
    # 3. Check Upsampling to original shape
    mock_resize.assert_any_call(
        ANY, (50, 50), order=1, preserve_range=True
    )