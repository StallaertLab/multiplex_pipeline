from functools import reduce
from typing import Dict, List, Optional, Sequence

import re
import anndata as ad
import numpy as np
import pandas as pd
import spatialdata as sd
from loguru import logger
from skimage.measure import regionprops_table
from spatialdata.models import TableModel

from multiplex_pipeline.utils.im_utils import calculate_median


class QuantificationController:
    def __init__(
        self,
        mask_keys: Dict[
            str, str
        ],  # e.g. {'cell': 'cell_mask', 'nuc': 'nucleus_mask'}
        table_name: str = "quantification",
        connect_to_mask: Optional[str] = None,
        input: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        mask_keys: dict mapping mask suffix (e.g. 'cell') to sdata.labels key (e.g. 'cell_mask')
        channels: list of channels to quantify
        """

        if (connect_to_mask) and (connect_to_mask not in mask_keys.values()):
            raise ValueError(
                f"connect_to_mask '{connect_to_mask}' must be one of the provided mask_keys: {list(mask_keys.keys())}"
            )

        self.mask_keys = mask_keys.copy()
        self.connect_to_mask = connect_to_mask
        self.channels = input
        self.table_name = table_name
        self.overwrite = overwrite

    def prepare_masks(self):
        # Load all user-requested masks
        self.masks = {
            suffix: self.get_mask(mask_key)
            for suffix, mask_key in self.mask_keys.items()
        }

    def get_mask(self, mask_key: str) -> np.ndarray:
        mask = np.array(
            sd.get_pyramid_levels(self.sdata[mask_key], n=0)
        ).squeeze()
        return mask

    def get_channel(self, channel_key: str) -> np.ndarray:

        img = np.array(
            sd.get_pyramid_levels(self.sdata[channel_key], n=0)
        ).squeeze()

        if img.ndim > 2:
            # warning if more than 2D will take the mean across channels
            logger.warning(
                f"Channel '{channel_key}' has more than 2 dimensions. Taking mean across channels."
            )
            img = np.mean(img, axis=0)

        return img

    def build_obs(self):
        morph_dfs = []
        for mask_suffix, mask in self.masks.items():
            logger.info(
                f"Quantifying morphology features for mask '{mask_suffix}'"
            )
            morph_props = regionprops_table(
                mask,
                properties=[
                    "label",
                    "area",
                    "eccentricity",
                    "solidity",
                    "perimeter",
                    "centroid",
                    "euler_number",
                ],
            )
            morph_df = pd.DataFrame(morph_props)
            morph_df = morph_df.rename(
                columns={
                    c: f"{c}_{mask_suffix}"
                    for c in morph_df.columns
                    if c != "label"
                }
            )
            morph_dfs.append(morph_df.set_index("label"))

        # create obs object
        obs = reduce(
            lambda left, right: pd.merge(
                left, right, left_index=True, right_index=True, how="outer"
            ),
            morph_dfs,
        )

        return obs

    def find_ndims_columns(self, names: List[str]) -> List[str]:
        """
        Identify columns in the provided list that represent multi-dimensional data.
        """
        _ndims_regex = re.compile(r'^(?P<base>[^-]+)-(?P<dim>\d+)(?P<suffix>.*)?$') # first '-' is expected to precede dimension number

        ndims_buckets = {}  # (property:str) -> List[tuple[dim:int, col:str]]
        for col in names:
            m = _ndims_regex.match(col)
            if m:
                prop = ''.join([m.group("base"),m.group("suffix")])
                dim = int(m.group("dim"))
                ndims_buckets.setdefault(prop, []).append((dim, col))

        for prop, dim_cols in ndims_buckets.items():
            # --- Check uniqueness of dimension indices ---
            dims = [d for d, _ in dim_cols]
            dup_dims = [d for d in set(dims) if dims.count(d) > 1]
            if dup_dims:
                raise ValueError(
                    f"Duplicate dimension indices found for '{prop}': {dup_dims}. "
                    "Each dimension (e.g., -0, -1, ...) must be unique."
                )
            if len(dims) <= 1:
                # remove from ndims_buckets
                logger.warning(
                    f"Property '{prop}' has only a single dimension. Skipping addition to obsm."
                )
                ndims_buckets[prop] = []

        return ndims_buckets
    
    def build_obsm(self, obs, ndims_buckets):
        obsm = {}
        cols_to_drop = []
        for prop, dim_cols in ndims_buckets.items():

            # --- Sort by dimension index ---
            dim_cols_sorted = sorted(dim_cols, key=lambda t: t[0])
            dims_sorted, cols_sorted = zip(*dim_cols_sorted) if dim_cols_sorted else ([], [])

            # --- Build array ---
            arr = np.column_stack([obs[c].to_numpy() for c in cols_sorted])
            obsm[prop] = arr
            cols_to_drop.extend(cols_sorted)

        return obsm, cols_to_drop

    def build_X_and_var(self):

        # For each mask: one row per object; For each channel: quantification columns
        quant_dfs = []
        for ch in self.channels:
            img = self.get_channel(ch)
            for mask_suffix, mask in self.masks.items():
                logger.info(
                    f"Quantifying channel '{ch}' with mask '{mask_suffix}'"
                )
                props = regionprops_table(
                    mask,
                    intensity_image=img,
                    properties=["label", "mean_intensity"],
                    extra_properties=[calculate_median],
                )
                df = pd.DataFrame(props)
                df = df.rename(
                    columns={
                        c: f"{ch}_{c.replace('mean_intensity', 'mean').replace('calculate_median', 'median')}_{mask_suffix}"
                        for c in df.columns
                        if c != "label"
                    }
                )
                quant_dfs.append(df.set_index("label"))
        
        # create X
        quant_df = reduce(
            lambda left, right: pd.merge(
                left, right, left_index=True, right_index=True, how="outer"
            ),
            quant_dfs,
        )
        X = quant_df.to_numpy()
        var = pd.DataFrame(index=quant_df.columns)

        return X, var

    def prepare_to_overwrite(self):
        
        if self.table_name in self.sdata:
            if not self.overwrite:
                message = f"Table '{self.table_name}' already exists in sdata. Please provide a unique table name."
                logger.error(message)
                raise ValueError(message)
            else:
                logger.warning(
                    f"Table '{self.table_name}' already exists and will be overwritten."
                )
                del self.sdata[self.table_name]
                self.sdata.delete_element_from_disk(self.table_name)
                logger.info(f"Existing table '{self.table_name}' deleted from sdata.")

    def validate_sdata_as_input(self):

        # validate masks
        for mask in self.mask_keys.values():
            if mask not in self.sdata.labels.keys():
                    message = f"Mask '{ch}' not found in sdata. Masks present: {list(self.sdata.labels.keys())}"
                    logger.error(message)
                    raise ValueError(message)

        if self.connect_to_mask not in self.sdata.labels.keys():
            message = f"Cannot connect the table to {self.connect_to_mask}, not present in sdata."
            logger.error(message)
            raise ValueError(message)          


        # validate channels
        if self.channels:
            # check channels exist
            for ch in self.channels:    
                if ch not in self.sdata:
                    message = f"Channel '{ch}' not found in sdata. Channels present: {list(self.sdata.images.keys())}"
                    logger.error(message)
                    raise ValueError(message)
            logger.info(f"Quantifying {len(self.channels)} user-specified channels: {self.channels}.")
        else:
            self.channels = list(self.sdata.images.keys())
            logger.info(f"Channels not specified. Quantifying all existing channels ({len(self.channels)}).")
            
    def run(
        self,
        spatial_data: sd.SpatialData,
    ) -> None:

        ########################################################################
        # Validate inputs and prepare data
        ########################################################################

        # set data
        self.sdata = spatial_data
        
        # Validate masks and channels
        self.validate_sdata_as_input()

        # Handle overwiting
        self.prepare_to_overwrite()

        # prepare masks
        self.prepare_masks()
        logger.info("Prepared masks for quantification.")

        ########################################################################
        # Compute obs (morphology, positions, etc.)
        ########################################################################

        # create obs
        obs = self.build_obs()

        # if matching pairs present, create obsm
        ndims_buckets = self.find_ndims_columns(list(obs.columns))
        if ndims_buckets:
            logger.info(
                f"Found {len(ndims_buckets)} columns with multiple dimensions: {list(ndims_buckets.keys())}. Creating obsm table."
            )
            obsm, cols_to_drop = self.build_obsm(obs,ndims_buckets)
            obs = obs.drop(columns=cols_to_drop)
        else:
            logger.info("No multi-dimensional columns found.")
            obsm = None

        ########################################################################
        # Compute X (intensities)
        ########################################################################

        X, var = self.build_X_and_var()

        ########################################################################
        # Create and save AnnData object
        ########################################################################

        # create AnnData object
        adata = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm)
        logger.info(
            f"Quantification complete. Resulting AnnData has {adata.n_obs} observations and {adata.n_vars} variables."
        )

        # add for connectivity
        if self.connect_to_mask:
            adata.obs["region"] = self.connect_to_mask
            adata.obs["cell"] = adata.obs.index.astype(int)

            # connect to the cell layer for napari-spatialdata compatibility
            adata.uns["spatialdata"] = {
                "region": [self.connect_to_mask],  # the element(s) this table annotates
                "region_key": "region",  # column in obs that points to the region
                "instance_key": "cell",  # column in obs that points to the object ID
            }


        adata = TableModel.parse(
            adata,
            region=self.connect_to_mask,
            region_key="region",  # << name of the column in obs
            instance_key="cell",  # << name of the column in obs with instance ids
            overwrite_metadata=True,
        )

        try:
            self.sdata[self.table_name] = adata
            self.sdata.write_element(self.table_name)
            logger.success(
                f"Quantification complete. Table '{self.table_name}' written to {self.sdata.path}"
            )
        except Exception as e:
            logger.error(f"Failed to write table '{self.table_name}': {e}")
            raise
