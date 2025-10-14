from __future__ import annotations
from typing import Any, Callable, Dict, Mapping, Sequence, Type
from abc import ABC, abstractmethod
from multiplex_pipeline.processors.registry  import register
from multiplex_pipeline.processors.base  import BaseOp, OutputType
import numpy as np
from loguru import logger
from skimage.measure import label
from skimage.morphology import closing, disk, opening
from skimage.transform import resize

################################################################################
# Mask Cleaners
################################################################################

class BaseMaskCleaner(BaseOp):
    """
    Base class for cleaners that derive a boolean cleaning mask and apply it.
    """

    EXPECTED_INPUTS = 1
    EXPECTED_OUTPUTS = 1
    OUTPUT_TYPE = OutputType.LABELS  # cleaners generally output labels

    @abstractmethod
    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        ...

    @abstractmethod
    def make_cleaning_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Return a boolean array (True = keep) of the same shape as `mask`.
        """
        ...

    # ---- reusable defaults ----
    def clean_with_mask(self, labeled_mask: np.ndarray) -> np.ndarray:
        if self.cleaning_mask is None:
            raise RuntimeError("No cleaning mask available. Run calculate_cleaning_mask() first.")

        if self.cleaning_mask.shape != labeled_mask.shape:
            raise ValueError("cleaning_mask and labeled_mask must have the same shape.")

        # Boolean mask input: just AND
        if labeled_mask.dtype == bool:
            return np.logical_and(labeled_mask, self.cleaning_mask)

        # Integer-labeled image:
        # 1) Which labels touch the cleaning mask?
        touching = np.unique(labeled_mask[self.cleaning_mask])
        # 2) Build a boolean lookup: keep[label] = True if it touches
        max_label = int(labeled_mask.max()) if labeled_mask.size else 0
        keep = np.zeros(max_label + 1, dtype=bool)
        keep[touching] = True  # background 0 stays False (drops correctly)
        # 3) Zero out labels not in `keep`
        out = labeled_mask.copy()
        out[~keep[out]] = 0
        return out

    def run(self, mask):
        if not hasattr(mask, "__array__"):
            raise TypeError(
                f"{self.__class__.__name__}.run() expected a NumPy array–like object, got {type(mask).__name__}."
            )
        arr = np.asarray(mask)
        if arr.ndim != 2:
            raise ValueError(f"{self.__class__.__name__}.run() expects a 2D mask, got shape {arr.shape}.")
        if arr.size == 0:
            raise ValueError("Input mask is empty.")

        cleaning_mask = self.make_cleaning_mask(arr)
        out = self.clean_with_mask(arr, cleaning_mask)
        logger.info(f"{self.__class__.__name__} applied cleaning mask.")
        return out


@register("mask_cleaner", "blob")
class BlobCleaner(BaseOp):
        
    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        # Defaults
        work_shape = cfg.get("work_shape", (250, 250))
        radius = cfg.get("radius", 5)

        # Validate work_shape
        if (not isinstance(work_shape, Sequence) or len(work_shape) != 2 or
            not all(isinstance(x, int) for x in work_shape) or
            not all(x > 0 for x in work_shape)):
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
                self.__class__.__name__, self.cfg["work_shape"], self.cfg["radius"]
            )
    
    def make_cleaning_mask(self, input):
        
        orig_shape = input.shape
        labeled_mask = label(input > 0)

        # Downsample mask for robust morphological cleaning
        resized_mask = (
            resize(
                labeled_mask.astype(int),
                self.target_shape,
                order=1,
                preserve_range=True,
            )
            > 0
        )

        # Morphological opening & closing
        selem = disk(self.morph_radius)
        clean_mask = opening(resized_mask, selem)
        clean_mask = closing(clean_mask, selem)

        # Upsample to original shape
        cleaning_mask = (
            resize(
                clean_mask.astype(float),
                orig_shape,
                order=1,
                preserve_range=True,
            )
            > 0
        )
        self.cleaning_mask = cleaning_mask
