import numpy as np
import pandas as pd
import pytest

# Import module under test
import plex_pipe.core_definition.roi_utils as roi_utils

# --- Fixtures ---


@pytest.fixture
def mock_image():
    """Creates a 100x100 grayscale image with a bright spot."""
    im = np.zeros((100, 100), dtype=float)
    # Create a bright square at 10,10 to 20,20
    im[10:20, 10:20] = 100.0
    return im


@pytest.fixture
def sample_masks():
    """
    Returns a list of mask dictionaries similar to Segment Anything (SAM) output.
    Format: bbox is [x, y, w, h]
    """
    return [
        {
            "bbox": [10, 10, 10, 10],  # Area 100. Bright spot. High IoU.
            "predicted_iou": 0.95,
            "stability_score": 0.95,
            "id": "good_mask",
        },
        {
            "bbox": [50, 50, 5, 5],  # Area 25. Dark spot.
            "predicted_iou": 0.90,
            "stability_score": 0.90,
            "id": "small_mask",
        },
        {
            "bbox": [80, 80, 10, 10],  # Area 100. Dark spot. Low IoU.
            "predicted_iou": 0.50,  # Below threshold
            "stability_score": 0.90,
            "id": "bad_iou_mask",
        },
    ]


# --- Tests for Filtering Logic (Pre-selection) ---


def test_pre_select_objects_filtering(mock_image, sample_masks):
    """
    Verifies that masks are filtered by Area, IoU, Stability, and Intensity.
    """
    # Define constraints
    filtered = roi_utils.pre_select_objects(
        masks=sample_masks,
        im=mock_image,
        min_area=50,  # Should filter 'small_mask' (area 25)
        max_area=500,
        min_iou=0.80,  # Should filter 'bad_iou_mask' (iou 0.50)
        min_stability=0.80,
        min_int=10.0,  # 'good_mask' is bright (100), others are 0
    )

    # Only 'good_mask' satisfies all conditions
    assert len(filtered) == 1
    assert filtered[0]["id"] == "good_mask"


def test_pre_select_intensity_calculation(mock_image):
    """
    Specific check: ensure it calculates mean intensity correctly from the image slice.
    """
    # 1. ADJUST DATA: Make the bright spot larger to account for the
    # 4-pixel padding (frame=4) applied inside pre_select_objects.
    # The bbox is 10:20, so the slice will be roughly 6:24.
    # We fill 5:25 to be safe.
    mock_image[5:25, 5:25] = 100.0

    # Create a mask over the center of that bright spot
    mask = [{"bbox": [10, 10, 10, 10], "predicted_iou": 1.0, "stability_score": 1.0}]

    # Threshold slightly below 100
    filtered = roi_utils.pre_select_objects(mask, mock_image, 1, 1000, 0, 0, min_int=99)

    assert len(filtered) == 1

    # Threshold slightly above 100
    filtered_high = roi_utils.pre_select_objects(
        mask, mock_image, 1, 1000, 0, 0, min_int=101
    )

    assert len(filtered_high) == 0


# --- Tests for Geometric Overlap (NMS) ---


def test_do_boxes_overlap():
    """
    Verifies the boolean overlap logic.
    """
    box1 = [0, 0, 10, 10]  # x, y, w, h

    # Overlapping box (shifted slightly)
    box2 = [5, 5, 10, 10]
    assert roi_utils.do_boxes_overlap(box1, box2) is True

    # Non-overlapping box (far away)
    box3 = [20, 20, 10, 10]
    assert roi_utils.do_boxes_overlap(box1, box3) is False

    # Touching edges (usually counted as non-overlapping in this logic,
    # depending on strict inequality used in source: <= vs <)
    # Source uses: x1_max <= x2_min. So touching is NOT overlapping.
    box_touching = [10, 0, 10, 10]
    assert roi_utils.do_boxes_overlap(box1, box_touching) is False


def test_remove_overlapping_objects_priority():
    """
    CRITICAL: Verifies that when two objects overlap, the one with
    HIGHER predicted_iou is kept, and the lower one is discarded.
    """
    objects = [
        {"bbox": [0, 0, 10, 10], "predicted_iou": 0.8, "id": "low_qual"},
        {"bbox": [2, 2, 10, 10], "predicted_iou": 0.99, "id": "high_qual"},  # Overlaps
        {
            "bbox": [100, 100, 10, 10],
            "predicted_iou": 0.7,
            "id": "distinct",
        },  # No overlap
    ]

    cleaned = roi_utils.remove_overlapping_objects(objects)

    assert len(cleaned) == 2
    ids = [o["id"] for o in cleaned]

    # "high_qual" should be present (higher IoU)
    assert "high_qual" in ids
    # "low_qual" should be removed (overlapped by high_qual)
    assert "low_qual" not in ids
    # "distinct" should remain (no conflict)
    assert "distinct" in ids


# --- Tests for Polygon Processing & DataFrame ---


def test_prepare_polygons_rescaling_and_clipping():
    """
    Verifies that polygons are:
    1. Rescaled by 2^req_level
    2. Clipped to the original image frame (Crucial for edges)
    """
    # Original image shape: 100x100
    org_shape = (100, 100)
    req_level = 1  # 2^1 = 2x multiplier

    # Polygon 1: Fully inside (10,10) -> Rescaled to (20,20)
    poly_in = [[10, 10], [20, 10], [20, 20], [10, 20]]

    # Polygon 2: Partially outside after rescaling.
    # (40, 40) * 2 = (80, 80) -> Inside
    # (60, 40) * 2 = (120, 80) -> X is 120, Image width is 100. Should clip.
    poly_out = [[40, 40], [60, 40], [60, 60], [40, 60]]

    polys = [poly_in, poly_out]

    results = roi_utils.prepare_polygons(polys, req_level, org_shape)

    # Check Poly 1 (Inside)
    res_p1 = results[0]
    assert np.max(res_p1) <= 100

    # Check Poly 2 (Clipped)
    res_p2 = results[1]
    # The max x-coordinate should be clipped to 100 (image width)
    # Note: shapely/numpy might result in 100.0
    assert np.max(res_p2[:, 0]) <= 100
    assert np.any(res_p2[:, 0] == 100)  # Ensure it actually hit the edge


def test_sort_cores_ordering():
    """
    Verifies that cores are sorted by Row (Y), then Column (X).
    """
    df = pd.DataFrame(
        {
            "row_start": [100000, 10000, 10000],
            "column_start": [10000, 20000, 10000],
            "other_data": ["C", "B", "A"],
        }
    )

    # Expected order:
    # 1. (10, 10) -> A
    # 2. (10, 20) -> B
    # 3. (100, 10) -> C

    sorted_df = roi_utils.sort_cores(df)

    assert sorted_df.iloc[0]["other_data"] == "A"
    assert sorted_df.iloc[1]["other_data"] == "B"
    assert sorted_df.iloc[2]["other_data"] == "C"


def test_prepare_poly_df_for_saving_integration():
    """
    Integration test: Checks the final DataFrame structure.
    """
    poly_data = [[[10, 10], [20, 10], [20, 20], [10, 20]]]
    poly_types = ["manual"]
    req_level = 0
    org_shape = (100, 100)

    df = roi_utils.prepare_poly_df_for_saving(
        poly_data, poly_types, req_level, org_shape
    )

    assert "core_name" in df.columns
    assert df.iloc[0]["core_name"] == "Core_000"
    assert df.iloc[0]["poly_type"] == "manual"
    # Check if bbox was calculated [xmin, xmax, ymin, ymax]
    # Poly x: 10-20, y: 10-20
    assert df.iloc[0]["row_start"] == 10
    assert df.iloc[0]["row_stop"] == 20
