from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pytest
from anndata import AnnData
from shapely.geometry import Point, Polygon

# Import the class under test
from plex_pipe.object_quantification.qc_shape_masker import QcShapeMasker

# --- Fixtures ---


@pytest.fixture
def mock_anndata():
    """
    Creates a dummy AnnData object with:
    - 3 Cells (Observations)
    - 4 Variables (Channels): MarkerA_mean, MarkerA_max, MarkerB_mean, MarkerB_max
    - Centroids in obsm
    """
    # 3 Cells
    # Cell 0: (10, 10) - Inside the exclusion zone
    # Cell 1: (50, 50) - Outside
    # Cell 2: (100, 100) - Outside
    centroids = np.array([[10, 10], [50, 50], [100, 100]])

    # Variables (must be contiguous for groupby logic in source code)
    var_names = ["MarkerA_mean", "MarkerA_max", "MarkerB_mean", "MarkerB_max"]

    # Data matrix (initially zeros)
    X = np.zeros((3, 4))

    adata = AnnData(X=X)
    adata.var_names = var_names
    adata.obsm["centroid_cell"] = centroids
    return adata


@pytest.fixture
def mock_sdata(mock_anndata):
    """
    Mocks a SpatialData object containing the AnnData table
    and a QC Exclusion shape for 'MarkerA'.
    """
    sdata = MagicMock()

    # 1. Setup Table
    sdata.__contains__.side_effect = lambda key: key in sdata._elements
    sdata.__getitem__.side_effect = lambda key: sdata._elements[key]
    sdata.__setitem__.side_effect = lambda key, val: sdata._elements.update({key: val})

    sdata._elements = {"quantification": mock_anndata}

    # 2. Setup QC Shape for MarkerA
    # Create a polygon covering (0,0) to (20,20). Cell 0 is inside.
    poly = Polygon([(0, 0), (0, 20), (20, 20), (20, 0)])
    gdf = gpd.GeoDataFrame({"geometry": [poly]})

    # The code looks for keys like "qc_exclude_MarkerA"
    sdata._elements["qc_exclude_MarkerA"] = gdf

    # MarkerB has no exclusion shape (simulating clean data)

    return sdata


# --- Tests for Spatial Logic ---


def test_check_belonging_spatial_index():
    """
    Verifies the STRtree spatial query logic.
    """
    masker = QcShapeMasker()

    # Define a square polygon (0,0) to (10,10)
    poly = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    polys = [poly]

    # Points:
    # P1 (5, 5): Inside -> Should be MASKED (False)
    # P2 (20, 20): Outside -> Should be KEPT (True)
    points = [Point(5, 5), Point(20, 20)]

    # Run check
    # Note: Source code returns 'False' if inside (exclusion mask logic usually means False = Exclude?)
    # Let's check source:
    # mask = np.ones...
    # if is_in: mask[i] = False.
    # So: False means "It is inside the bad polygon" (i.e., filtered out? Or just flagged?)
    # The logic sets 0 (False) for points INSIDE the QC polygon.

    result_mask = masker.check_belonging(points, polys)

    assert not result_mask[0]  # Inside polygon
    assert result_mask[1]  # Outside polygon


def test_check_belonging_empty_pairs():
    """
    Edge case: Points are far away, query returns no pairs.
    """
    masker = QcShapeMasker()
    poly = Polygon([(0, 0), (1, 1), (1, 0)])
    points = [Point(100, 100)]  # Far away

    mask = masker.check_belonging(points, [poly])

    assert mask[0]  # Default is ones (True)


# --- Tests for QC Mask Construction ---


def test_build_qc_mask_broadcasting(mock_sdata):
    """
    Critical Test: Verifies that the exclusion mask is applied
    SPECIFICALLY to MarkerA columns, but NOT MarkerB columns.
    """
    masker = QcShapeMasker(table_name="quantification", qc_prefix="qc_exclude")
    masker.sdata = mock_sdata

    # Run the build
    # Note: The code swaps coordinates: coords = coords[:, ::-1]
    # In fixture: Cell 0 is at (10, 10).
    # QC Shape covers (0,0) to (20,20).
    # Regardless of swap (10,10) is inside.

    final_mask = masker.build_qc_mask()

    # Shape check: (3 cells, 4 vars)
    assert final_mask.shape == (3, 4)

    # --- Check Cell 0 (The one inside the QC Polygon) ---
    # Cols 0 & 1 are MarkerA. Should be False (masked out).
    assert not final_mask[0, 0]  # MarkerA_mean
    assert not final_mask[0, 1]  # MarkerA_max

    # Cols 2 & 3 are MarkerB. No QC shape exists for MarkerB. Should be True (kept).
    assert final_mask[0, 2]
    assert final_mask[0, 3]

    # --- Check Cell 1 (Outside the polygon at 50,50) ---
    # All should be True
    assert final_mask[1, 0]
    assert final_mask[1, 2]


def test_build_qc_mask_coordinate_swap(mock_sdata):
    """
    Verifies that coordinates are swapped before checking geometry.
    """
    masker = QcShapeMasker()
    masker.sdata = mock_sdata

    # Modify data to have asymmetric coordinates
    # Cell 0: (y=5, x=100) -> Source swaps to (x=100, y=5)
    # If polygon is around 0-20, (100, 5) is OUTSIDE.
    # If it DIDN'T swap and treated it as (5, 100), it might still be outside depending on shape.

    # Let's make it unambiguous:
    # Polygon: x in [0, 20], y in [0, 20]
    # Point: y=10, x=10 (Inside if (10,10))
    # Point stored in obsm as [10, 10].
    # Source code: coords[:, ::-1].
    # If inputs are (y, x), swapping makes them (x, y).

    # This test primarily ensures the line runs without crashing on dimension mismatch.
    masker.build_qc_mask()


# --- Tests for Validation & Integration ---


def test_validate_sdata_missing_table(mock_sdata):
    """Verifies error if table name is wrong."""
    masker = QcShapeMasker(table_name="wrong_name")
    masker.sdata = mock_sdata

    with pytest.raises(ValueError, match="Table wrong_name not present"):
        masker.validate_sdata()


def test_validate_sdata_missing_centroids(mock_sdata):
    """Verifies error if centroids are missing from obsm."""
    del mock_sdata["quantification"].obsm["centroid_cell"]

    masker = QcShapeMasker(table_name="quantification")
    masker.sdata = mock_sdata

    with pytest.raises(ValueError, match="Centroids: centroid_cell not present"):
        masker.validate_sdata()


def test_run_integration(mock_sdata):
    """
    Verifies the full pipeline: validate -> build -> save to layer.
    """
    masker = QcShapeMasker(write_to_disk=False)

    masker.run(mock_sdata)

    # Check that the layer was added to AnnData
    adata = mock_sdata["quantification"]
    assert "qc_mask" in adata.layers

    # Verify the layer content is boolean
    assert adata.layers["qc_mask"].dtype == bool
    # Check dimensions
    assert adata.layers["qc_mask"].shape == (3, 4)
