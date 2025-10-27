from __future__ import annotations

import numpy as np
from loguru import logger
from pydantic import (
    Field,
    ValidationInfo,
    model_validator,
)

from multiplex_pipeline.processors.base import (
    BaseOp,
    OutputType,
    ProcessorParamsBase,
)
from multiplex_pipeline.processors.registry import register

################################################################################
# Image Transformers
################################################################################


@register("image_transformer", "normalize")
class Normalize(BaseOp):

    EXPECTED_INPUTS = 1
    EXPECTED_OUTPUTS = 1
    OUTPUT_TYPE = OutputType.IMAGE

    class Params(ProcessorParamsBase):
        """Parameters for percentile normalization."""

        low: float = Field(
            1.0, ge=0.0, le=100.0, description="Lower percentile bound (0-100)."
        )
        high: float = Field(
            99.0, ge=0.0, le=100.0, description="Upper percentile bound (0-100)."
        )

        @model_validator(mode="after")
        def check_low_less_than_high(self, info: ValidationInfo):
            """Ensures the lower bound is strictly less than the upper bound."""
            if self.low >= self.high:
                raise ValueError(
                    f"'low' ({self.low}) must be less than 'high' ({self.high})."
                )

            return self

    def run(self, img):
        # Must be array-like
        if not hasattr(img, "__array__"):
            raise TypeError(
                f"{self.__class__.__name__}.run() expected a NumPy array–like object, "
                f"got {type(img).__name__}."
            )

        arr = np.asarray(img)

        low, high = self.params.low, self.params.high
        p_low = np.percentile(arr, low)
        p_high = np.percentile(arr, high)

        denom = p_high - p_low
        if denom <= 0 or not np.isfinite(denom):
            message = f"Normalization skipped: invalid percentiles (low={p_low}, high={p_high}, Δ={denom})"
            logger.error(message)
            raise ValueError(message)

        out = (arr - p_low) / denom
        out = np.clip(out, 0, 1).astype(np.float32, copy=False)

        logger.info(
            f"Applied normalization (percentiles {low}–{high}) → [{p_low}, {p_high}]",
            low,
            high,
            p_low,
            p_high,
        )
        return out


@register("image_transformer", "denoise_with_median")
class DenoiseWithMedian(BaseOp):

    EXPECTED_INPUTS = 1
    EXPECTED_OUTPUTS = 1
    OUTPUT_TYPE = OutputType.IMAGE

    class Params(ProcessorParamsBase):
        """Parameters for the median denoising operation."""

        # This handles the default, type hint, and positive integer check.
        disk_radius: int = Field(
            3,
            gt=0,
            description="The radius of the disk-shaped kernel for the median filter.",
        )

    def run(self, img):
        # Must be array-like
        if not hasattr(img, "__array__"):
            raise TypeError(
                f"{self.__class__.__name__}.run() expected a NumPy array–like object, "
                f"got {type(img).__name__}."
            )

        from skimage.filters import median
        from skimage.morphology import disk

        med = median(img, disk(self.params.disk_radius))

        return med


@register("image_transformer", "mean_of_images")
class MeanOfImages(BaseOp):
    """Compute the mean of multiple image arrays."""

    EXPECTED_INPUTS = None  # allow any number of inputs
    EXPECTED_OUTPUTS = 1
    OUTPUT_TYPE = OutputType.IMAGE  # produces a single averaged image

    def run(self, *images):
        """Compute elementwise mean over all provided images."""

        # --- validation ---
        if len(images) == 0:
            raise ValueError(
                f"{self.__class__.__name__}.run() expected at least one image."
            )

        arrays = []
        for i, img in enumerate(images):
            if not hasattr(img, "__array__"):
                raise TypeError(
                    f"{self.__class__.__name__}.run(): input #{i} is not array-like "
                    f"(got {type(img).__name__})."
                )
            arr = np.asarray(img)
            if arr.ndim == 0:
                raise ValueError(f"Input #{i} is scalar; expected image array.")
            arrays.append(arr)

        # --- shape consistency ---
        ref_shape = arrays[0].shape
        if any(a.shape != ref_shape for a in arrays[1:]):
            raise ValueError(
                f"All input images must have the same shape; got {[a.shape for a in arrays]}."
            )

        # --- compute mean ---
        stack = np.stack(arrays, axis=0)
        mean_img = np.mean(stack, axis=0).astype(np.float32, copy=False)

        logger.info(
            f"{self.__class__.__name__}: computed mean of {len(arrays)} images "
            f"with shape {ref_shape}."
        )
        return mean_img
