from typing import List

import numpy as np
import spatialdata as sd
from loguru import logger
from scipy.ndimage import median_filter
from spatialdata import SpatialData
from spatialdata.models import Image2DModel


class PreSegmentationProcessor:
    def __init__(
        self, mix=None, denoise=None, normalize=None, output_name=None
    ):
        """
        mix: 'sum', 'mean', or 'none'
        denoise: None or string like 'median'
        normalize: a
        output_name: only used when mix != 'none'
        """
        self.mix_mode = mix
        self.denoise = denoise
        self.normalize = normalize
        self.output_name = output_name

    def run(self, sdata: SpatialData, channels: List[str]) -> SpatialData:
        """
        Processes channels and adds result(s) to sdata.images.
        Returns the modified sdata.
        """

        if self.mix_mode:
            images = [
                np.array(sd.get_pyramid_levels(sdata[ch], n=0)).squeeze()
                for ch in channels
            ]
            if self.mix_mode == "sum":
                img = sum(images)
            elif self.mix_mode == "mean":
                img = sum(images) / len(images)
            else:
                raise ValueError(f"Unknown mix mode: {self.mix_mode}")

            img_list = [img]
            name_list = [self.output_name]

        else:
            img_list = [
                np.array(sd.get_pyramid_levels(sdata[ch], n=0)).squeeze()
                for ch in channels
            ]
            name_list = [f"{ch}_preseg" for ch in channels]

        # Process each image in the list
        for img, name in zip(img_list, name_list):

            img = self.post_process(img)
            sdata[name] = Image2DModel.parse(
                img[None],
                dims=("c", "y", "x"),
                scale_factors=[2] * (len(sdata[channels[0]].children) - 1),
            )
            # sdata.write_element(name)

        return sdata

    def run_denoise(self, img):

        # denoise with a median filter
        if self.config.get("denoise") == "median":

            img = median_filter(img, size=3)
            logger.info("Applied median filter for denoising.")

        # denoise with Noise2Void
        if self.config.get("denoise") == "Noise2Void":
            # to be implemented
            img = img
            logger.info("Applied Noise2Void for denoising. To be implemented.")

        return img

    def run_normalize(self, img):
        if self.normalize:
            # check that these are 2 values
            if (
                isinstance(self.normalize, (list, tuple))
                and len(self.normalize) == 2
            ):
                img_min = np.percentile(img, self.normalize[0])
                img_max = np.percentile(img, self.normalize[1])
                img = (img - img_min) / (img_max - img_min)
                img = np.clip(img, 0, 1)
                # img = (img * 255).astype(np.uint8)  # Convert to uint8
                logger.info(
                    f"Applied normalization with percentiles {self.normalize}."
                )
            else:
                logger.warning(
                    "Normalization requires a list of two percentiles, e.g. [1, 99]. Using default [0, 100]."
                )
        return img

    def post_process(self, img):
        if self.denoise:
            img = self.run_denoise(img)
        if self.normalize:
            img = self.run_normalize(img)
        return img
