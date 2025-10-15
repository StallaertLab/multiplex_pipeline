from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from loguru import logger

from multiplex_pipeline.processors.base import BaseOp, OutputType
from multiplex_pipeline.processors.registry import register

################################################################################
# Image Transformers
################################################################################


@register("image_transformer", "normalize")
class Normalize(BaseOp):

    EXPECTED_INPUTS = 1
    EXPECTED_OUTPUTS = 1
    OUTPUT_TYPE = OutputType.IMAGE

    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        # If config missing or empty, set defaults and log.
        if not cfg:
            logger.warning(
                "No parameters provided for normalization; using defaults low=1, high=99."
            )
            cfg = {"low": 1, "high": 99}

        low = cfg.get("low")
        high = cfg.get("high")

        # Coerce to floats
        low = float(low)
        high = float(high)

        # Validate numeric range
        if not (0.0 <= low <= 100.0):
            raise ValueError(
                f"Parameter 'low' must be between 0 and 100 (got {low})."
            )
        if not (0.0 <= high <= 100.0):
            raise ValueError(
                f"Parameter 'high' must be between 0 and 100 (got {high})."
            )
        if low >= high:
            raise ValueError(
                f"'low' must be < 'high' (got low={low}, high={high})."
            )

        # Store canonicalized config
        self.cfg = {"low": low, "high": high}

    def run(self, img):
        # Must be array-like
        if not hasattr(img, "__array__"):
            raise TypeError(
                f"{self.__class__.__name__}.run() expected a NumPy array–like object, "
                f"got {type(img).__name__}."
            )

        arr = np.asarray(img)

        low, high = self.cfg["low"], self.cfg["high"]
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

    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        if not cfg:
            cfg["disk_radius"] = 3
            logger.warning(
                f"No kernel size for denoising with median kernel, using disk_radius = {cfg['disk_radius']}."
            )

        if "disk_radius" in cfg and (
            not isinstance(cfg["disk_radius"], int)
            or (cfg["disk_radius"] <= 0)
        ):
            message = "Parameter 'disk_radius' has to be a positive integer."
            logger.error(message)
            raise ValueError(message)

    def run(self, img):
        # Must be array-like
        if not hasattr(img, "__array__"):
            raise TypeError(
                f"{self.__class__.__name__}.run() expected a NumPy array–like object, "
                f"got {type(img).__name__}."
            )

        from skimage.filters import median
        from skimage.morphology import disk

        med = median(img, disk(self.cfg["disk_radius"]))

        return med


@register("image_transformer", "mean_of_images")
class MeanOfImages(BaseOp):
    """Compute the mean of multiple image arrays."""

    EXPECTED_INPUTS = None  # allow any number of inputs
    EXPECTED_OUTPUTS = 1
    OUTPUT_TYPE = OutputType.IMAGE  # produces a single averaged image

    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        if cfg:
            raise ValueError("MeanOfImages takes no parameters.")

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
                raise ValueError(
                    f"Input #{i} is scalar; expected image array."
                )
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
