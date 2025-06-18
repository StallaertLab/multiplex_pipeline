import numpy as np
from skimage.draw import polygon as sk_polygon


class CoreCutter:
    """Extract rectangular or polygonal regions from images."""

    def __init__(self, margin: int = 0, mask_value: int = 0) -> None:
        """Create a new cutter.

        Args:
            margin (int, optional): Padding to apply around each core.
            mask_value (int, optional): Value used outside polygon masks.
        """

        self.margin = margin
        self.mask_value = mask_value

    def extract_core(self, array, row):
        """Extract a single core from the given image.

        Args:
            array (numpy.ndarray | dask.array.Array): Source image.
            row (pandas.Series): Metadata describing the core. Required fields
                include ``row_start``, ``row_stop``, ``column_start``,
                ``column_stop`` and ``poly_type``.

        Returns:
            numpy.ndarray: The extracted core image.
        """

        # Read bbox coordinates
        y0 = int(row["row_start"])
        y1 = int(row["row_stop"])
        x0 = int(row["column_start"])
        x1 = int(row["column_stop"])

        # Apply margin & safety clipping
        img_height, img_width = array.shape
        y0m = max(0, y0 - self.margin)
        y1m = min(img_height, y1 + self.margin)
        x0m = max(0, x0 - self.margin)
        x1m = min(img_width, x1 + self.margin)

        # Extract subarray
        subarray = array[y0m:y1m, x0m:x1m]

        # Compute to numpy
        if hasattr(subarray, "compute"):  # Dask array check
            subarray = subarray.compute()

        if row["poly_type"] == "rectangle":
            return subarray

        elif row["poly_type"] == "polygon":
            # Load polygon coordinates and shift to local frame
            polygon = row["polygon_vertices"]  # assuming list of [y, x] pairs

            shifted_polygon = [(y - y0m, x - x0m) for y, x in polygon]
            poly_y, poly_x = zip(*shifted_polygon)

            # Create polygon mask
            rr, cc = sk_polygon(poly_y, poly_x, subarray.shape)
            mask = np.zeros(subarray.shape, dtype=bool)
            mask[rr, cc] = True

            # Apply mask
            masked_array = np.full(
                subarray.shape, self.mask_value, dtype=subarray.dtype
            )
            masked_array[mask] = subarray[mask]

            return masked_array

        else:
            raise ValueError(f"Unknown poly_type: {row['poly_type']}")
