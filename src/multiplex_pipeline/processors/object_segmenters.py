from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from loguru import logger

from multiplex_pipeline.processors.base import BaseOp, OutputType
from multiplex_pipeline.processors.registry import register

################################################################################
# Object Segmenters
################################################################################


@register("object_segmenter", "instanseg")
class InstansegSegmenter(BaseOp):

    OUTPUT_TYPE = OutputType.LABELS

    def validate_config(self, cfg: Mapping[str, Any]) -> None:

        # requires model as a paramter
        if not isinstance(cfg, dict) or "model" not in cfg:
            message = f"Instanseg requires specified model, instead got {cfg}"
            logger.error(message)
            raise ValueError(message)

        # checks if other parameters are instanseg specific
        # logs warning if not
        defined_parameters = [
            "pixel_size",
            "resolve_cell_and_nucleus",
            "cleanup_fragments",
            "clean_cache",
            "model"
        ]
        for param in cfg:
            if param not in defined_parameters:
                message = f"Instanseg does not accept parameter {param}. Check instanseg documentation for details."
                logger.warning(message)

        # specify number of outputs depending on selected model option
        if (
            "resolve_cell_and_nucleus" in cfg
            and self.cfg["resolve_cell_and_nucleus"]
        ):
            self.EXPECTED_OUTPUTS = 2
        else:
            self.EXPECTED_OUTPUTS = 1

    def initialize(self):

        from instanseg import InstanSeg

        self.model = InstanSeg(self.cfg["model"], verbosity=1)

    def prepare_input(self, in_image):

        if isinstance(in_image, (tuple,list)):
            in_image = np.stack(in_image, axis=-1)  # (H, W, C)
        elif isinstance(in_image, np.ndarray):
            if in_image.ndim == 2:
                in_image = in_image[..., np.newaxis]
            # 3D but has to transposed
            elif in_image.ndim == 3 and in_image.shape[0] < min(
                in_image.shape[1:]
            ):  # (C, H, W)
                in_image = np.moveaxis(in_image, 0, -1)

        else:
            raise ValueError(
                "Input must be a 2D/3D numpy array or a list of 2D arrays."
            )

        return in_image

    def run(self, *in_image):

        # standardize input
        in_image = self.prepare_input(in_image)

        # Call InstanSeg
        labeled_output, _ = self.model.eval_medium_image(in_image, **self.cfg)

        # extract result
        segm_arrays = [
            np.array(x).astype(int) for x in labeled_output[0, :, :, :]
        ]

        # clean cuda cache
        if self.cfg.get('clean_cache',False):
            import torch
            torch.cuda.empty_cache()

        return segm_arrays


@register("object_segmenter", "cellpose")
class Cellpose4Segmenter(BaseOp):

    EXPECTED_OUTPUTS = 1
    OUTPUT_TYPE = OutputType.LABELS

    def validate_config(self, cfg: Mapping[str, Any]) -> None:

        # checks if other parameters are instanseg specific
        # logs warning if not
        if isinstance(cfg, dict):
            defined_parameters = [
                "diameter",
                "flow_threshold",
                "cellprob_threshold",
                "niter",
            ]
            for param in cfg:
                if param not in defined_parameters:
                    message = f"Cellpose does not accept parameter {param}. Check Cellpose documentation for details."
                    logger.warning(message)

        # add additional checks

    def initialize(self):

        from cellpose import models

        self.model = models.CellposeModel(gpu=True)

    def prepare_input(self, in_image):

        message = None
        if isinstance(in_image, (tuple, list)):
            for im in in_image:
                if im.ndim != 2:
                    message = "Only 2D arrays are accepted."
            if len(in_image) > 2:
                logger.warning(
                    "Cellpose will use only the first two provided channels."
                )
            in_image = np.stack(in_image[:2], axis=0)  # (C, H, W)

        elif isinstance(in_image, np.ndarray):
            if in_image.ndim != 2:
                message = "Only 2D arrays are accepted."

        if message:
            logger.error(message)
            raise ValueError(message)

        return in_image

    def run(self, *in_image) -> np.ndarray:
        """
        Run segmentation using Cellpose.
        It can accept a single image or a pair of images for nucleus and cytoplasm in any order.
        """

        # prepare input
        in_image = self.prepare_input(in_image)

        # run segmentation
        mask, *_ = self.model.eval(in_image, **self.cfg)

        return mask
