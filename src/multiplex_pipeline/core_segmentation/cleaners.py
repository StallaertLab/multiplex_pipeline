import numpy as np
import cv2
from skimage.transform import resize
from skimage.measure import regionprops, label
from skimage.morphology import closing, opening, disk

class BaseCleaner:
    def run(self, mask: np.ndarray) -> np.ndarray:
        """
        Default: calculate the cleaning mask and apply it.
        Subclasses can override if needed.
        """
        self.calculate_cleaning_mask(mask)
        return self.clean_with_mask(mask)

    def calculate_cleaning_mask(self, labeled_mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Cleaners must implement the calculate_cleaning_mask() method.")

    def clean_with_mask(self, labeled_mask: np.ndarray) -> np.ndarray:
        """
        Default label-based cleaning logic.
        """
        if self.cleaning_mask is not None:
            cleaned_mask = np.zeros_like(labeled_mask)
            for region in regionprops(labeled_mask):
                coords = region.coords
                if np.any(self.cleaning_mask[coords[:, 0], coords[:, 1]]):
                    cleaned_mask[coords[:, 0], coords[:, 1]] = region.label
            return cleaned_mask
        else:
            raise RuntimeError("No cleaning mask available. Please run calculate_cleaning_mask() first.")

class DbscanEllipseCleaner(BaseCleaner):
    def __init__(self, target_shape=(1000, 1000), cluster_eps=5, min_samples=20):
        """
        Args:
            target_shape: Shape (height, width) for resizing masks during ellipse fitting.
            cluster_eps: DBSCAN 'eps' parameter for clustering.
            min_samples: DBSCAN 'min_samples' parameter.
        """
        self.target_shape = target_shape
        self.cluster_eps = cluster_eps
        self.min_samples = min_samples
        self.last_ellipse_params = None  # (cx, cy, major, minor, angle)
        self.cleaning_mask = None

    def fit_ellipse(self, xs: np.ndarray, ys: np.ndarray) -> tuple:
        """
        Fit an ellipse robustly to (xs, ys) using DBSCAN, convex hull, and cv2.fitEllipse.
        Returns (cx, cy, major, minor, angle) in resized coordinates.
        """
        from sklearn.cluster import DBSCAN
        from scipy.spatial import ConvexHull

        coords = np.column_stack((xs, ys))
        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.min_samples).fit(coords)
        labels, counts = np.unique(clustering.labels_[clustering.labels_ >= 0], return_counts=True)
        if len(counts) == 0:
            raise RuntimeError("No dense core cluster found!")
        main_label = labels[np.argmax(counts)]
        core_points = coords[clustering.labels_ == main_label]
        if len(core_points) < 5:
            raise RuntimeError("Not enough points in main cluster to fit an ellipse.")
        hull = ConvexHull(core_points)
        hull_points = core_points[hull.vertices]
        if len(hull_points) >= 5:
            ellipse = cv2.fitEllipse(hull_points.astype(np.int32))
            (cx, cy), (major, minor), angle = ellipse
            return cx, cy, major, minor, angle
        else:
            raise RuntimeError("Not enough points on hull to fit ellipse.")

    def ellipse_mask(self, shape: tuple, cx: float, cy: float, major: float, minor: float, angle: float) -> np.ndarray:
        """
        Generate a boolean mask of an ellipse with given params in the given shape.
        """
        Y, X = np.ogrid[:shape[0], :shape[1]]
        cos_a = np.cos(np.deg2rad(angle))
        sin_a = np.sin(np.deg2rad(angle))
        x_shift = X - cx
        y_shift = Y - cy
        ellipse_eq = (((x_shift * cos_a + y_shift * sin_a)**2) / (major/2)**2 +
                      ((-x_shift * sin_a + y_shift * cos_a)**2) / (minor/2)**2)
        return ellipse_eq <= 1

    def calculate_cleaning_mask(self, labeled_mask: np.ndarray) -> np.ndarray:
        """
        Compute a robust ellipse mask from the binary/labeled mask.

        Returns:
            Ellipse mask at original scale (boolean array, same shape as mask)
        """
        orig_shape = labeled_mask.shape

        # Downsample mask for ellipse detection
        resized_mask = resize(labeled_mask.astype(int), self.target_shape, order=0, preserve_range=True) > 0

        ys, xs = np.nonzero(resized_mask)
        cx, cy, major, minor, angle = self.fit_ellipse(xs, ys)
        ellipse_small = self.ellipse_mask(self.target_shape, cx, cy, major, minor, angle)
        ellipse_mask_orig = resize(ellipse_small.astype(float), orig_shape, order=0, preserve_range=True) > 0

        # Store for inspection/QC if needed
        self.last_ellipse_params = (
            cx * (orig_shape[1] / self.target_shape[1]),
            cy * (orig_shape[0] / self.target_shape[0]),
            major * (orig_shape[1] / self.target_shape[1]),
            minor * (orig_shape[0] / self.target_shape[0]),
            angle,
        )
        self.cleaning_mask = ellipse_mask_orig



class BlobCleaner(BaseCleaner):
    def __init__(self, target_shape=(250, 250), morph_radius=5):
        """
        Cleaner that uses morphological operations on a resized mask to remove small blobs/artifacts.

        Args:
            target_shape: tuple, shape to which the mask is resized for cleaning.
            morph_radius: radius for disk-shaped structuring element in opening/closing.
        """
        self.target_shape = target_shape
        self.morph_radius = morph_radius
        self.cleaning_mask = None

    def calculate_cleaning_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Perform morphological cleaning on a resized version of the mask and return the upsampled cleaning mask.

        Args:
            mask: 2D numpy array (binary or labeled).

        Returns:
            cleaning_mask: boolean array, same shape as input mask.
        """
        orig_shape = mask.shape
        labeled_mask = label(mask > 0)

        # Downsample mask for robust morphological cleaning
        resized_mask = resize(labeled_mask.astype(int), self.target_shape, order=1, preserve_range=True) > 0

        # Morphological opening & closing
        selem = disk(self.morph_radius)
        clean_mask = opening(resized_mask, selem)
        clean_mask = closing(clean_mask, selem)

        # Upsample to original shape
        cleaning_mask = resize(clean_mask.astype(float), orig_shape, order=1, preserve_range=True) > 0
        self.cleaning_mask = cleaning_mask