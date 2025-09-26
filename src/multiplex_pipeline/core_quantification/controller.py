import numpy as np
import pandas as pd
import anndata as ad
from skimage.measure import regionprops_table
import spatialdata as sd
from spatialdata.models import TableModel
from typing import Dict, List, Optional, Sequence
from multiplex_pipeline.im_utils import calculate_median
from functools import reduce
from loguru import logger

class QuantificationController:
    def __init__(
        self,
        mask_keys: Dict[str, str],  # e.g. {'cell': 'cell_mask', 'nuc': 'nucleus_mask'}
        table_name: str = 'quantification',
        channels: Optional[List[str]] = None,
        cytoplasm_components: Optional[Sequence[str]] = None,  # e.g. ('cell_mask', 'nucleus_mask')
        cytoplasm_mask_name: str = 'cyto',
    ) -> None:
        """
        mask_keys: dict mapping mask suffix (e.g. 'cell') to sdata.labels key (e.g. 'cell_mask')
        channels: list of channels to quantify
        cytoplasm_components: tuple/list of two sdata.labels keys for mask subtraction
        cytoplasm_mask_name: the suffix to use for the cytoplasm mask in outputs (default: 'cyto')
        """

        self.mask_keys = mask_keys.copy()
        self.channels = channels 
        self.cytoplasm_components = cytoplasm_components
        self.cytoplasm_mask_name = cytoplasm_mask_name
        self.table_name = table_name

    def prepare_masks(self):
        # Load all user-requested masks
        self.masks = {suffix: self.get_mask(mask_key) for suffix, mask_key in self.mask_keys.items()}
        # Optionally create cytoplasm mask from two provided label keys
        if self.cytoplasm_components:
            if len(self.cytoplasm_components) != 2:
                raise ValueError("Cytoplasm_components must have exactly two SpatialData label keys")
            key_a, key_b = self.cytoplasm_components
            mask_a = self.get_mask(key_a)
            mask_b = self.get_mask(key_b)
            cytoplasm_mask = np.where((mask_a > 0) & (mask_b == 0), mask_a, 0)
            self.masks[self.cytoplasm_mask_name] = cytoplasm_mask
            logger.info(f"Cytoplasm mask '{self.cytoplasm_mask_name}' created as {key_a} minus {key_b}.")

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
    
    def run(self,
            spatial_data: sd.SpatialData,
            ) -> None:

        # set data
        self.sdata = spatial_data
        self.channels = self.channels or list(spatial_data.images.keys())

        # prepare masks
        self.prepare_masks()
        logger.info('Prepared masks for quantification.')

        # deal with the table of the same name already existing
        if self.table_name in self.sdata:
            logger.warning(f"Table '{self.table_name}' already exists in sdata. It will be overwritten.")
            del self.sdata[self.table_name]
            self.sdata.delete_element_from_disk(self.table_name)
            logger.info(f"Deleted existing table '{self.table_name}' from disk.")

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
                    c: f"{ch}_{c.replace('mean_intensity', 'mean').replace('calculate_median', 'median')}_{mask_suffix}" for c in df.columns if c != "label"
                })
                quant_dfs.append(df.set_index("label"))

        # arrange AnnData object
        logger.info("Combining morphology and quantification data into AnnData object.")
        obs = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), morph_dfs)
        quant_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), quant_dfs)
        X = quant_df.to_numpy()
        var = pd.DataFrame(index=quant_df.columns)  

        # add for connectivity
        obs['region'] = 'instanseg_cell'
        obs['cell'] = obs.index.astype(int)

        adata = ad.AnnData(
            X=X,
            obs=obs,
            var=var
        )

        # connect to the cell layer for napari-spatialdata compatibility

        adata.uns["spatialdata"] = {
            "region": ["instanseg_cell"],        # the element(s) this table annotates
            "region_key": "region",         # column in obs that points to the region
            "instance_key": "cell"   # column in obs that points to the object ID
        }

        adata = TableModel.parse(
            adata,
            region=["instanseg_cell"],
            region_key="region",      # << name of the column in obs
            instance_key="cell",     # << name of the column in obs with instance ids
            overwrite_metadata=True,
        )

        try:
            self.sdata[self.table_name] = adata
            self.sdata.write_element(self.table_name)
            logger.success(f"Quantification complete. Table '{self.table_name}' written to {self.sdata.path}")
        except Exception as e:
            logger.error(f"Failed to write table '{self.table_name}': {e}")
            raise
