from __future__ import annotations

from typing import Any, Mapping, Sequence

from loguru import logger
from skimage.morphology import closing, disk, opening
from skimage.transform import resize

from multiplex_pipeline.processors.base import BaseOp, OutputType
from multiplex_pipeline.processors.registry import register

################################################################################
# Mask Builders
################################################################################


@register("mask_builder", "subtract")
class SubtractionBuilder(BaseOp):

    EXPECTED_INPUTS = 2
    EXPECTED_OUTPUTS = 1
    OUTPUT_TYPE = OutputType.LABELS

    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        if cfg:
            raise ValueError("SubtractionBuilder takes no parameters.")

    def run(self, mask_cell, mask_nucleus):
        if mask_cell.shape != mask_nucleus.shape:
            raise ValueError(
                "Source masks must have the same shape for subtraction."
            )
        result = mask_cell.copy()
        result[mask_nucleus > 0] = (
            0  # zero out regions where mask_nucleus is present
        )
        return result


@register("mask_builder", "multiply")
class MultiplicationBuilder(BaseOp):

    EXPECTED_INPUTS = 2
    EXPECTED_OUTPUTS = 1
    OUTPUT_TYPE = OutputType.LABELS

    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        if cfg:
            raise ValueError("SubtractionBuilder takes no parameters.")

    def run(self, mask1, mask2):
        if mask1.shape != mask2.shape:
            raise ValueError("Source masks must have the same shape.")
        result = mask1 * mask2
        return result


@register("mask_builder", "ring")
class RingBuilder(BaseOp):

    EXPECTED_INPUTS = 1
    EXPECTED_OUTPUTS = 1
    OUTPUT_TYPE = OutputType.LABELS

    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        if (
            not isinstance(cfg, dict)
            or "outer" not in cfg
            or "inner" not in cfg
        ):
            raise ValueError(
                "RingBuilder requires parameters: outer and inner radius."
            )
        if not isinstance(cfg["outer"], int) or cfg["outer"] <= 0:
            raise ValueError("'outer' radius must be a positive integer")
        if not isinstance(cfg["inner"], int) or cfg["inner"] < 0:
            raise ValueError("'inner' radius must be a non-negative integer.")

    def run(self, mask):
        from skimage.segmentation import expand_labels

        mask_big = expand_labels(mask, self.cfg["outer"])
        mask_small = expand_labels(mask, self.cfg["inner"])
        result = mask_big - mask_small
        return result


@register("mask_builder", "blob")
class BlobBuilder(BaseOp):

    EXPECTED_INPUTS = 1
    EXPECTED_OUTPUTS = 1
    OUTPUT_TYPE = OutputType.LABELS

    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        # Defaults
        work_shape = cfg.get("work_shape", (250, 250))
        radius = cfg.get("radius", 5)

        # Validate work_shape
        if (
            not isinstance(work_shape, Sequence)
            or len(work_shape) != 2
            or not all(isinstance(x, int) for x in work_shape)
            or not all(x > 0 for x in work_shape)
        ):
            raise ValueError(
                "Parameter 'work_shape' must be a tuple/list of two positive integers, "
                f"e.g. (250, 250). Got: {work_shape!r}"
            )

        # Validate radius
        if not isinstance(radius, int) or radius <= 0:
            raise ValueError(
                f"Parameter 'radius' has to be a positive integer. Got: {radius!r}"
            )

        # Store canonicalized config
        self.cfg = {"work_shape": tuple(work_shape), "radius": int(radius)}

        # Log if defaults were used
        if "work_shape" not in cfg or "radius" not in cfg:
            logger.warning(
                "%s: using defaults work_shape=%s, radius=%s.",
                self.__class__.__name__,
                self.cfg["work_shape"],
                self.cfg["radius"],
            )

    def run(self, source):

        orig_shape = source.shape
        binary_mask = source > 0

        # Downsample mask for robust morphological cleaning
        resized_mask = (
            resize(
                binary_mask.astype(int),
                self.cfg["work_shape"],
                order=1,
                preserve_range=True,
            )
            > 0
        )

        # Morphological opening & closing
        selem = disk(self.cfg["radius"])
        blob_mask = opening(resized_mask, selem)
        blob_mask = closing(blob_mask, selem)

        # Upsample to original shape
        blob_mask = (
            resize(
                blob_mask.astype(float),
                orig_shape,
                order=1,
                preserve_range=True,
            )
            > 0
        )
        return blob_mask.astype("int")



from typing import Any, Mapping, Sequence, Tuple, List
import numpy as np
import cv2

from skimage.transform import resize
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN


@register("mask_cleaner", "ellipse_dbscan")
class DbscanEllipseCleaner(BaseOp):
    """
    Clean a binary/labeled mask by fitting a robust ellipse to the dominant region.

    Steps:
      1) Downsample mask to `work_shape` for stability.
      2) Cluster foreground pixels with DBSCAN; pick the largest (core) cluster.
      3) Fit an ellipse on the convex hull of core points (OpenCV `fitEllipse`).
      4) Rasterize the ellipse and upsample back to original resolution.
    """

    EXPECTED_INPUTS = 1
    EXPECTED_OUTPUTS = 1
    OUTPUT_TYPE = OutputType.LABELS

    # --- Public QC fields (optional): stored after run() ---
    last_ellipse_params: tuple | None = None  # (cx, cy, major, minor, angle) in original-scale coords

    # --- Config validation ---
    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        work_shape = cfg.get("work_shape", (1000, 1000))
        cluster_eps = cfg.get("cluster_eps", 5)
        min_samples = cfg.get("min_samples", 20)

        # work_shape
        if (
            not isinstance(work_shape, Sequence)
            or len(work_shape) != 2
            or not all(isinstance(x, int) for x in work_shape)
            or not all(x > 0 for x in work_shape)
        ):
            raise ValueError(
                "Parameter 'work_shape' must be a tuple/list of two positive ints, e.g. (1000, 1000). "
                f"Got: {work_shape!r}"
            )

        # cluster_eps
        if not (isinstance(cluster_eps, (int, float)) and cluster_eps > 0):
            raise ValueError(
                f"'cluster_eps' must be a positive number. Got: {cluster_eps!r}"
            )

        # min_samples
        if not (isinstance(min_samples, int) and min_samples > 0):
            raise ValueError(
                f"'min_samples' must be a positive integer. Got: {min_samples!r}"
            )

        self.cfg = {
            "work_shape": tuple(work_shape),
            "cluster_eps": float(cluster_eps),
            "min_samples": int(min_samples),
        }

    # --- Core operation ---
    def run(self, source: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        source : np.ndarray
            Binary or labeled mask (nonzero = foreground). 2D.

        Returns
        -------
        np.ndarray
            Cleaned mask as int array (0/1), same shape as input.
        """
        orig_shape = tuple(source.shape)
        binary = source > 0

        # Downsample for robust fitting
        small = (
            resize(
                binary.astype(int),
                self.cfg["work_shape"],
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )
            > 0
        )

        ys, xs = np.nonzero(small)
        cx, cy, major, minor, angle = self._fit_ellipse(xs, ys)

        # Rasterize ellipse at small scale, then upsample
        ell_small = self._ellipse_mask(self.cfg["work_shape"], cx, cy, major, minor, angle)
        ell_big = (
            resize(
                ell_small.astype(float),
                orig_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )
            > 0
        )

        # Store original-scale ellipse params for QC
        sx = orig_shape[1] / self.cfg["work_shape"][1]
        sy = orig_shape[0] / self.cfg["work_shape"][0]
        self.last_ellipse_params = (cx * sx, cy * sy, major * sx, minor * sy, angle)

        return ell_big.astype("int")

    # --- Helpers ---
    def _fit_ellipse(self, xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float, float]:
        """
        Fit an ellipse robustly to (xs, ys) using DBSCAN, convex hull, and cv2.fitEllipse.
        Returns (cx, cy, major, minor, angle) in small/work coordinates.
        """
        if xs.size == 0:
            raise ValueError("No foreground pixels found to fit an ellipse.")

        coords = np.column_stack((xs, ys))
        clustering = DBSCAN(
            eps=self.cfg["cluster_eps"],
            min_samples=self.cfg["min_samples"],
        ).fit(coords)

        labels = clustering.labels_
        mask = labels >= 0
        if not np.any(mask):
            raise ValueError("DBSCAN found no core cluster to fit an ellipse.")

        core_label, counts = np.unique(labels[mask], return_counts=True)
        main_label = core_label[np.argmax(counts)]
        core_points = coords[labels == main_label]
        if len(core_points) < 5:
            raise ValueError("Not enough points in main cluster to fit an ellipse (need ≥5).")

        hull = ConvexHull(core_points)
        hull_pts = core_points[hull.vertices]
        if len(hull_pts) < 5:
            raise ValueError("Not enough convex-hull points to fit an ellipse (need ≥5).")

        (cx, cy), (major, minor), angle = cv2.fitEllipse(hull_pts.astype(np.int32))
        return float(cx), float(cy), float(major), float(minor), float(angle)

    @staticmethod
    def _ellipse_mask(
        shape: tuple[int, int],
        cx: float,
        cy: float,
        major: float,
        minor: float,
        angle: float,
    ) -> np.ndarray:
        """Boolean rasterization of the ellipse inside `shape`."""
        h, w = shape
        Y, X = np.ogrid[:h, :w]

        a = np.deg2rad(angle)
        ca, sa = np.cos(a), np.sin(a)

        x = X - cx
        y = Y - cy

        # Rotate coordinates
        xr = x * ca + y * sa
        yr = -x * sa + y * ca

        # Ellipse equation: (xr/(major/2))^2 + (yr/(minor/2))^2 <= 1
        a2 = (major / 2.0) ** 2
        b2 = (minor / 2.0) ** 2
        return (xr * xr) / a2 + (yr * yr) / b2 <= 1.0
