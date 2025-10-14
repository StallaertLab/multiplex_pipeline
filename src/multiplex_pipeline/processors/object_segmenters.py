from __future__ import annotations
from typing import Any, Callable, Dict, Mapping, Sequence, Type
from multiplex_pipeline.processors.registry  import register
from multiplex_pipeline.processors.base  import BaseOp, OutputType
import numpy as np

################################################################################
# Object Segmenters
################################################################################

@register("object_segmenter", "instanseg")
class InstansegSegmenter(BaseOp):

    EXPECTED_OUTPUTS = 2
    OUTPUT_TYPE = OutputType.LABELS


    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        
        if not isinstance(cfg, dict) or "model" not in cfg:
            raise ValueError("Instanseg requires parameters: model.")
        
        # add additional checks

        import torch
        from instanseg import InstanSeg
        self.model = InstanSeg(cfg['model'], verbosity=1)
    
    def run(self, image):
        
        # Input normalization
        if isinstance(image, list):
            image = np.stack(image, axis=-1)  # (H, W, C)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = image[..., np.newaxis]
            # 3D but has to transposed
            elif image.ndim == 3 and image.shape[0] < min(
                image.shape[1:]
            ):  # (C, H, W)
                image = np.moveaxis(image, 0, -1)

        else:
            raise ValueError(
                "Input must be a 2D/3D numpy array or a list of 2D arrays."
            )
        
        # Call InstanSeg
        labeled_output, _ = self.model.eval_medium_image(
            image, **self.cfg
        )

        segm_arrays = [
            np.array(x).astype(int) for x in labeled_output[0, :, :, :]
        ]

        # Clear CUDA cache (optional, can be disabled by kwarg)
        import torch
        torch.cuda.empty_cache()

        return segm_arrays
