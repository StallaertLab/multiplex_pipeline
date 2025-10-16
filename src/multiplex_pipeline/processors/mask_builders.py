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
            raise ValueError("Source masks must have the same shape for subtraction.")
        result = mask_cell.copy()
        result[mask_nucleus > 0] = 0  # zero out regions where mask_nucleus is present
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
        if not isinstance(cfg, dict) or "outer" not in cfg or "inner" not in cfg:
            raise ValueError("RingBuilder requires parameters: outer and inner radius.")
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
