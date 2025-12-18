import pytest
import numpy as np
import pandas as pd
import dask.array as da
from multiplex_pipeline.core_cutting.cutter import CoreCutter

# --- Fixtures ---

@pytest.fixture
def sample_image():
    """Creates a 100x100 grayscale image with specific values."""
    # Create a gradient so we can verify we got the *right* pixels,
    # not just random ones.
    # img[y, x] = y + x
    y, x = np.mgrid[0:100, 0:100]
    return (y + x).astype(np.uint8)

@pytest.fixture
def rect_row():
    """Metadata for a simple 10x10 rectangle in the middle."""
    return pd.Series({
        "row_start": 40,
        "row_stop": 50,
        "column_start": 40,
        "column_stop": 50,
        "poly_type": "rectangle"
    })

@pytest.fixture
def poly_row():
    """Metadata for a triangular polygon."""
    # A triangle with vertices at (10, 10), (20, 10), (10, 20)
    # Bbox should cover this area roughly.
    vertices = np.array([[10, 10], [20, 10], [10, 20]])
    return pd.Series({
        "row_start": 10,
        "row_stop": 20,
        "column_start": 10,
        "column_stop": 20,
        "poly_type": "polygon",
        "polygon_vertices": vertices
    })

# --- Tests ---

def test_extract_rectangle_simple(sample_image, rect_row):
    """Verifies basic slicing without margin."""
    cutter = CoreCutter(margin=0)
    
    cutout = cutter.extract_core(sample_image, rect_row)
    
    # Expected shape: (50-40) x (50-40) = 10x10
    assert cutout.shape == (10, 10)
    # Verify content: Top-left of cutout is (40, 40) in original
    # Value should be 40 + 40 = 80
    assert cutout[0, 0] == 80

def test_extract_rectangle_margin_clipping(sample_image):
    """
    Verifies that margin is applied but clipped at image boundaries.
    """
    cutter = CoreCutter(margin=5)
    
    # Define a box at the very top-left edge (0,0) to (10,10)
    row = pd.Series({
        "row_start": 0, "row_stop": 10,
        "column_start": 0, "column_stop": 10,
        "poly_type": "rectangle"
    })
    
    cutout = cutter.extract_core(sample_image, row)
    
    # Logic:
    # y0 = max(0, 0 - 5) = 0
    # y1 = min(100, 10 + 5) = 15
    # Shape should be 15x15 (not 20x20)
    assert cutout.shape == (15, 15)

def test_extract_polygon_masking(sample_image, poly_row):
    """
    Verifies that pixels outside the polygon are masked to 'mask_value'.
    """
    # Use a weird mask value to be sure
    cutter = CoreCutter(margin=0, mask_value=255)
    
    cutout = cutter.extract_core(sample_image, poly_row)
    
    # The slice is 10x10.
    # The polygon is a triangle filling the top-left half (in local coords).
    # Vertices (global): (10,10), (20,10), (10,20)
    # Shifted by (10,10) -> (0,0), (10,0), (0,10)
    
    # (0,0) is inside (vertex) -> Should preserve image value (10+10=20)
    assert cutout[0, 0] == 20
    
    # (9, 9) is outside the triangle -> Should be masked to 255
    # In the gradient image, (19,19) would be 38, so 255 proves masking worked.
    assert cutout[9, 9] == 255

def test_extract_dask_array(rect_row):
    """
    Verifies that dask arrays are automatically computed for polygons.
    """
    cutter = CoreCutter()
    
    # Create a dask array
    dask_img = da.from_array(np.zeros((100, 100)), chunks=10)
    
    # Use polygon mode to trigger the .compute() call inside the method
    poly_row = rect_row.copy()
    poly_row["poly_type"] = "polygon"
    # A simple square polygon matching the bbox
    poly_row["polygon_vertices"] = np.array([[40, 40], [50, 40], [50, 50], [40, 50]])
    
    result = cutter.extract_core(dask_img, poly_row)
    
    # Result must be a numpy array, not dask
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10)

def test_unknown_poly_type(sample_image, rect_row):
    """Verifies error handling for bad poly_type."""
    cutter = CoreCutter()
    bad_row = rect_row.copy()
    bad_row["poly_type"] = "circle" # Not supported
    
    with pytest.raises(ValueError, match="Unknown poly_type"):
        cutter.extract_core(sample_image, bad_row)

def test_polygon_with_margin(sample_image):
    """
    Verifies coordinate math works when a margin is added.
    Adding a margin changes the 'origin' of the slice, so the 
    relative polygon coordinates must shift correctly.
    """
    cutter = CoreCutter(margin=5)
    
    # 10x10 box at (10,10)
    # Global poly vertex at (10,10) (Top-left of box)
    row = pd.Series({
        "row_start": 10, "row_stop": 20,
        "column_start": 10, "column_stop": 20,
        "poly_type": "polygon",
        "polygon_vertices": np.array([[10, 10], [15, 15], [10, 15]])
    })
    
    cutout = cutter.extract_core(sample_image, row)
    
    # Slice Origin is now (10-5, 10-5) = (5, 5).
    # Global Pixel (10,10) is at local index (5,5).
    # This pixel is a vertex, so it should be INSIDE (value preserved).
    # Value at (10,10) is 20.
    assert cutout[5, 5] == 20
    
    # Pixel at (0,0) local is global (5,5). 
    # This is well outside the polygon. Should be masked (default 0).
    assert cutout[0, 0] == 0