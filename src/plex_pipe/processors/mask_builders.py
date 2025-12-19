from __future__ import annotations

from typing import Tuple

from pydantic import Field, field_validator, model_validator
from skimage.morphology import closing, disk, opening
from skimage.transform import resize

from plex_pipe.processors.base import (
    BaseOp,
    OutputType,
    ProcessorParamsBase,
)
from plex_pipe.processors.registry import register

################################################################################
# Mask Builders
################################################################################


@register("mask_builder", "subtract")
class SubtractionBuilder(BaseOp):

    EXPECTED_INPUTS = 2
    EXPECTED_OUTPUTS = 1
    OUTPUT_TYPE = OutputType.LABELS

    def run(self, mask_cell, mask_nucleus):
        if mask_cell.shape != mask_nucleus.shape:
            raise ValueError("Source masks must have the same shape for subtraction.")
        result = mask_cell.copy()
        result[mask_nucleus > 0] = 0  # zero out regions where mask_nucleus is present
        return result


###############################################################################
###############################################################################


@register("mask_builder", "multiply")
class MultiplicationBuilder(BaseOp):

    EXPECTED_INPUTS = 2
    EXPECTED_OUTPUTS = 1
    OUTPUT_TYPE = OutputType.LABELS

    def run(self, mask1, mask2):
        if mask1.shape != mask2.shape:
            raise ValueError("Source masks must have the same shape.")
        result = mask1 * mask2
        return result


###############################################################################
###############################################################################
@register("mask_builder", "ring")
class RingBuilder(BaseOp):

    EXPECTED_INPUTS = 1
    EXPECTED_OUTPUTS = 1
    OUTPUT_TYPE = OutputType.LABELS

    ######################################################
    # current implementation is super slow and not memory efficient
    ######################################################

    class Params(ProcessorParamsBase):
        """Parameters for creating a ring mask."""

        outer: int = Field(
            7, gt=0, description="The outer radius of the ring in pixels."
        )
        inner: int = Field(
            2, ge=0, description="The inner radius of the ring in pixels."
        )

        @model_validator(mode="after")
        def check_radii_are_valid(self):
            """Ensures the outer radius is strictly greater than the inner radius."""
            if self.outer <= self.inner:
                raise ValueError(
                    f"outer radius ({self.outer}) must be strictly greater than inner radius ({self.inner})"
                )
            return self

    def run(self, mask):
        from skimage.segmentation import expand_labels

        mask_big = expand_labels(mask, self.params.outer)
        mask_small = expand_labels(mask, self.params.inner)
        mask_big[mask_small > 0] = 0

        return mask_big


###############################################################################
###############################################################################
@register("mask_builder", "blob")
class BlobBuilder(BaseOp):

    EXPECTED_INPUTS = 1
    EXPECTED_OUTPUTS = 1
    OUTPUT_TYPE = OutputType.LABELS

    class Params(ProcessorParamsBase):
        """Parameters for the blob creation operation."""

        work_shape: Tuple[int, int] = (250, 250)
        radius: int = Field(5, gt=0)

        @field_validator("work_shape")
        def check_positive_dimensions(cls, v: Tuple[int, int]) -> Tuple[int, int]:
            """Ensures both dimensions in work_shape are positive integers."""
            if not (v[0] > 0 and v[1] > 0):
                raise ValueError(
                    "Both dimensions in 'work_shape' must be positive integers."
                )
            return v

    def run(self, source):

        orig_shape = source.shape
        binary_mask = source > 0

        # Downsample mask for robust morphological cleaning
        resized_mask = (
            resize(
                binary_mask.astype(int),
                self.params.work_shape,
                order=1,
                preserve_range=True,
            )
            > 0
        )

        # Morphological opening & closing
        selem = disk(self.params.radius)
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
