from unittest.mock import patch

import numpy as np
import pytest

# Import classes under test
from plex_pipe.processors.image_transformers import (
    DenoiseWithMedian,
    MeanOfImages,
    Normalize,
)

# --- Tests for Normalize Transformer ---


def test_normalize_init_defaults():
    """Verifies default parameters (1.0 - 99.0)."""
    norm = Normalize()
    assert norm.params.low == 1.0
    assert norm.params.high == 99.0


def test_normalize_param_validation():
    """Verifies Pydantic validation for invalid percentiles."""
    # Case 1: Low >= High (Logic error)
    with pytest.raises(ValueError, match="must be less than"):
        Normalize(low=50, high=50)

    # Case 2: Out of bounds (0-100)
    with pytest.raises(ValueError, match="Parameters for 'normalize' are not correct:"):
        Normalize(low=-1)
    with pytest.raises(ValueError, match="Parameters for 'normalize' are not correct:"):
        Normalize(high=101)


def test_normalize_run_standard():
    """
    Verifies standard normalization logic.
    Input: [0, 50, 100]
    Percentiles (0-100): Min=0, Max=100
    Output: [0.0, 0.5, 1.0]
    """
    norm = Normalize(low=0, high=100)
    img = np.array([0, 50, 100], dtype=float)

    result = norm.run(img)

    assert np.allclose(result, [0.0, 0.5, 1.0])
    assert result.dtype == np.float32


def test_normalize_run_clipping():
    """
    Verifies that values outside the percentile range are clipped to 0-1.
    """
    # 10th percentile = 10, 90th percentile = 90
    norm = Normalize(low=10, high=90)

    # FIX: Use 11 points so 10% and 90% align exactly with integer indices.
    # [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    img = np.linspace(0, 100, 11).astype(float)

    # Range is 90 - 10 = 80
    result = norm.run(img)

    # 1. Check Low Clipping (Value 0 is below 10 -> should be 0.0)
    assert result[0] == 0.0

    # 2. Check Boundary Low (Value 10 IS the lower bound -> should be 0.0)
    # (10 - 10) / 80 = 0.0
    assert result[1] == 0.0

    # 3. Check Boundary High (Value 90 IS the upper bound -> should be 1.0)
    # (90 - 10) / 80 = 1.0
    assert result[9] == 1.0

    # 4. Check High Clipping (Value 100 is above 90 -> should be 1.0)
    assert result[10] == 1.0


def test_normalize_divide_by_zero_protection():
    """
    CRITICAL TEST: Ensures code doesn't crash or produce NaNs on flat images.
    If an image is solid black (all 0s), p_high - p_low = 0.
    """
    norm = Normalize()
    flat_img = np.zeros((10, 10))

    with pytest.raises(ValueError, match="Normalization skipped"):
        norm.run(flat_img)


def test_normalize_invalid_input():
    """Verifies type check for non-array inputs."""
    norm = Normalize()
    with pytest.raises(TypeError, match="expected a NumPy array"):
        norm.run("not an array")


# --- Tests for DenoiseWithMedian Transformer ---


@patch("skimage.filters.median")
@patch("skimage.morphology.disk")
def test_denoise_median_run(mock_disk, mock_median):
    """
    Verifies that the run method correctly calls skimage functions
    with the configured disk radius.
    """
    # Setup
    transformer = DenoiseWithMedian(disk_radius=5)
    input_img = np.zeros((100, 100))
    expected_output = np.ones((100, 100))

    # Mock returns
    mock_median.return_value = expected_output
    mock_disk.return_value = "disk_struct"

    # Execute
    result = transformer.run(input_img)

    # Assert
    assert result is expected_output
    # Check if disk() was called with correct radius
    mock_disk.assert_called_with(5)
    # Check if median() was called with image and disk structure
    mock_median.assert_called_with(input_img, "disk_struct")


def test_denoise_median_params():
    """Verifies parameter validation (radius > 0)."""
    with pytest.raises(
        ValueError, match="Parameters for 'denoise_with_median' are not correct"
    ):
        DenoiseWithMedian(disk_radius=0)


# --- Tests for MeanOfImages Transformer ---


def test_mean_of_images_run_success():
    """
    Verifies averaging of multiple images.
    Image A: All 0s
    Image B: All 10s
    Image C: All 20s
    Mean should be 10s.
    """
    transformer = MeanOfImages()
    shape = (10, 10)

    img1 = np.zeros(shape)
    img2 = np.full(shape, 10.0)
    img3 = np.full(shape, 20.0)

    result = transformer.run(img1, img2, img3)

    assert result.shape == shape
    assert np.all(result == 10.0)
    assert result.dtype == np.float32


def test_mean_of_images_shape_mismatch():
    """
    Verifies error when input images have different dimensions.
    """
    transformer = MeanOfImages()
    img1 = np.zeros((10, 10))
    img2 = np.zeros((20, 20))  # Mismatch

    with pytest.raises(ValueError, match="must have the same shape"):
        transformer.run(img1, img2)


def test_mean_of_images_empty_input():
    """Verifies error when no images are provided."""
    transformer = MeanOfImages()
    with pytest.raises(ValueError, match="expected at least one image"):
        transformer.run()


def test_mean_of_images_scalar_input():
    """Verifies error when scalar values are passed instead of arrays."""
    transformer = MeanOfImages()
    with pytest.raises(ValueError, match="expected image array"):
        transformer.run(np.array(5))  # 0-d array
