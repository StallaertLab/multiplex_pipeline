from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import module under test
import plex_pipe.utils.im_utils as im_utils

# --- Fixtures ---


@pytest.fixture
def mock_zarr_structure():
    """
    Creates a mock Zarr/OME-Zarr metadata structure.
    """
    return {
        "multiscales": [
            {
                "datasets": [
                    {"path": "0"},  # Level 0 (Full res)
                    {"path": "1"},  # Level 1
                    {"path": "2"},  # Level 2
                ]
            }
        ]
    }


@patch("plex_pipe.utils.im_utils.imread")
@patch("plex_pipe.utils.im_utils.zarr.open")
def test_get_zarr_levels_num(mock_zarr_open, mock_imread, mock_zarr_structure):
    """
    Verifies it correctly counts the number of resolution levels in OME-Zarr metadata.
    """
    # Setup mock Zarr group
    mock_group = MagicMock()
    mock_group.attrs.asdict.return_value = mock_zarr_structure
    mock_zarr_open.return_value = mock_group

    count = im_utils.get_zarr_levels_num("test.zarr")

    # We defined 3 datasets in the fixture
    assert count == 3


@patch("plex_pipe.utils.im_utils.get_small_image")
def test_prepare_rgb_image_math(mock_get_small):
    """
    Verifies the math of contrast stretching and RGB conversion.
    This ensures images aren't just black or white.
    """
    # Create a gradient image 0..100
    # Reshape to 10x10
    input_im = np.arange(100).reshape(10, 10)
    mock_get_small.return_value = input_im

    # Request normalization ignoring the bottom 10% and top 10%
    # 10th percentile of 0..99 is ~9.9
    # 90th percentile of 0..99 is ~89.1
    # Pixels < 10 become 0, Pixels > 89 become 255.
    rgb = im_utils.prepare_rgb_image("dummy", perc_min=10, perc_max=90)

    # 1. Check Output Shape: Must be (H, W, 3)
    assert rgb.shape == (10, 10, 3)

    # 2. Check Data Type: Must be uint8 for plotting/saving
    assert rgb.dtype == np.uint8

    # 3. Check Clipping (Low End)
    # Input 0 should be clipped to 0
    assert np.all(rgb[0, 0] == [0, 0, 0])

    # 4. Check Clipping (High End)
    # Input 99 should be clipped to 255
    assert np.all(rgb[9, 9] == [255, 255, 255])

    # 5. Check Mid tones (Logic check)
    # Value 50 should be roughly middle gray (~127)
    # (50 - 10) / (90 - 10) = 40/80 = 0.5 -> 127.5
    mid_pixel = rgb[5, 0]  # Input value 50
    assert 120 < mid_pixel[0] < 135
