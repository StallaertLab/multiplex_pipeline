import numpy as np
from skimage.draw import polygon as sk_polygon

class CoreCutter:
    def __init__(self, margin=0, mask_value=0):
        self.margin = margin
        self.mask_value = mask_value

    def extract_core(self, array, row):
        """
        array: full image (2D numpy array)
        row: pandas dataframe row containing core information
        """

        # Read bbox coordinates
        y0 = int(row['row_start'])
        y1 = int(row['row_stop'])
        x0 = int(row['column_start'])
        x1 = int(row['column_stop'])

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

        if row['poly_type'] == 'rectangle':
            return subarray

        elif row['poly_type'] == 'polygon':
            # Load polygon coordinates and shift to local frame
            polygon = row['polygon_vertices']  # assuming list of [y, x] pairs

            shifted_polygon = [(y - y0m, x - x0m) for y, x in polygon]
            poly_y, poly_x = zip(*shifted_polygon)

            # Create polygon mask
            rr, cc = sk_polygon(poly_y, poly_x, subarray.shape)
            mask = np.zeros(subarray.shape, dtype=bool)
            mask[rr, cc] = True

            # Apply mask
            masked_array = np.full(subarray.shape, self.mask_value, dtype=subarray.dtype)
            masked_array[mask] = subarray[mask]

            return masked_array

        else:
            raise ValueError(f"Unknown poly_type: {row['poly_type']}")
