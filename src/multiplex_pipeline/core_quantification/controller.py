import numpy as np
import pandas as pd
import anndata as ad
from skimage.measure import regionprops_table
import spatialdata as sd
from typing import Dict, List, Optional, Sequence
from multiplex_pipeline.im_utils import calculate_median
from functools import reduce
from loguru import logger

class QuantificationController:
    def __init__(
        self,
        spatial_data,
        mask_keys: Dict[str, str],  # e.g. {'cell': 'cell_mask', 'nuc': 'nucleus_mask'}
        table_name: str = 'quantification',
        channels: Optional[List[str]] = None,
        derivative_components: Optional[Sequence[str]] = None,  # e.g. ('cell_mask', 'nucleus_mask')
        derivative_mask_name: str = 'cyto',
    ):
        """
        mask_keys: dict mapping mask suffix (e.g. 'cell') to sdata.labels key (e.g. 'cell_mask')
        channels: list of channels to quantify
        derivative_components: tuple/list of two sdata.labels keys for mask subtraction
        derivative_mask_name: the suffix to use for the derivative mask in outputs (default: 'cyto')
        """
        self.sdata = spatial_data
        self.mask_keys = mask_keys.copy()
        self.channels = channels or list(spatial_data.images.keys())
        self.derivative_components = derivative_components
        self.derivative_mask_name = derivative_mask_name
        self.table_name = table_name

    def prepare_masks(self):
        # Load all user-requested masks
        self.masks = {suffix: self.get_mask(mask_key) for suffix, mask_key in self.mask_keys.items()}
        # Optionally create derivative mask from two provided label keys
        if self.derivative_components:
            if len(self.derivative_components) != 2:
                raise ValueError("derivative_components must have exactly two SpatialData label keys")
            key_a, key_b = self.derivative_components
            mask_a = self.get_mask(key_a)
            mask_b = self.get_mask(key_b)
            derivative_mask = np.where((mask_a > 0) & (mask_b == 0), mask_a, 0)
            self.masks[self.derivative_mask_name] = derivative_mask
            logger.info(f"Derivative mask '{self.derivative_mask_name}' created as {key_a} minus {key_b}.")

    def get_mask(self, mask_key: str) -> np.ndarray:
        mask = np.array(sd.get_pyramid_levels(self.sdata[mask_key], n=0)).squeeze()
        return mask

    def get_channel(self, channel_key: str) -> np.ndarray:
        
        img = np.array(sd.get_pyramid_levels(self.sdata[channel_key], n=0)).squeeze()
        
        if img.ndim > 2:
            # warning if more than 2D will take the mean across channels
            logger.warning(f"Channel '{channel_key}' has more than 2 dimensions. Taking mean across channels.")
            img = np.mean(img, axis=0)

        return img
    

    def run(self) -> ad.AnnData:
        
        self.prepare_masks()
        logger.info('Prepared masks for quantification.')
        
        # Precompute all morphology features per mask 
        morph_dfs = []
        for mask_suffix, mask in self.masks.items():
            logger.info(f"Quantifying morphology features for mask '{mask_suffix}'")
            morph_props = regionprops_table(mask, properties=["label", "area", "eccentricity", "solidity", "perimeter", "centroid", "euler_number"])
            morph_df = pd.DataFrame(morph_props)
            morph_df = morph_df.rename(columns={c: f"{c}_{mask_suffix}" for c in morph_df.columns if c != "label"})
            morph_dfs.append(morph_df.set_index("label"))

        # For each mask: one row per object; For each channel: quantification columns
        quant_dfs = []
        for ch in self.channels:
            img = self.get_channel(ch)
            for mask_suffix, mask in self.masks.items():
                logger.info(f"Quantifying channel '{ch}' with mask '{mask_suffix}'")
                props = regionprops_table(
                    mask, intensity_image=img,
                    properties=["label", "mean_intensity"],
                    extra_properties=[calculate_median]
                )
                df = pd.DataFrame(props)
                df = df.rename(columns={
                    c: f"{ch}_{c}_{mask_suffix}" for c in df.columns if c != "label"
                })
                quant_dfs.append(df.set_index("label"))

        # arrange AnnData object
        logger.info("Combining morphology and quantification data into AnnData object.")
        obs = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), morph_dfs)
        quant_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), quant_dfs)
        X = quant_df.to_numpy()
        var = pd.DataFrame(index=quant_df.columns)  

        adata = ad.AnnData(
            X=X,
            obs=obs,
            var=var
        )

        try:
            self.sdata[self.table_name] = adata
            self.sdata.write_element(self.table_name)
            logger.success(f"Quantification complete. Table '{self.table_name}' written to {self.sdata.path}")
        except Exception as e:
            logger.error(f"Failed to write table '{self.table_name}': {e}")
            raise
