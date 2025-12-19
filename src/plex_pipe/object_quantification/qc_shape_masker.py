from itertools import groupby
from typing import Optional

import numpy as np
import spatialdata as sd
from loguru import logger
from shapely import Point
from shapely.strtree import STRtree


class QcShapeMasker:

    def __init__(
        self,
        table_name: str = "quantification",
        qc_prefix: str = "qc_exclude",
        object_name: Optional[str] = "cell",
        layer_name: Optional[str] = "qc_mask",
        write_to_disk=False,
    ) -> None:

        self.table_name = table_name
        self.qc_prefix = qc_prefix
        self.object_name = object_name
        self.layer_name = layer_name
        self.write_to_disk = write_to_disk

    def validate_sdata(self):

        # check the table present in sdata
        if self.table_name in self.sdata:
            logger.info(f"Table {self.table_name} present in the spatialdata object.")
        else:
            msg = f"Table {self.table_name} not present in the spatialdata object."
            logger.error(msg)
            raise ValueError(msg)

        # check that the centroids of the selected object are present in obsm
        expected_name = f"centroid_{self.object_name}"
        if expected_name in list(self.sdata[self.table_name].obsm.keys()):
            logger.info(
                f"Centroids: {expected_name} present in the anndata table {self.table_name}."
            )
        else:
            msg = f"Centroids: {expected_name} not present in the anndata table {self.table_name}."
            logger.error(msg)
            raise ValueError(msg)

    def rewrite_table(self):

        # it's not dask backed so standard overwrite should work

        try:
            self.sdata.delete_element_from_disk(self.table_name)
            self.sdata.write_element(self.table_name, overwrite=True)
            logger.success(f"Table '{self.table_name}' written to {self.sdata.path}.")
        except Exception as e:
            logger.error(f"Failed to write table '{self.table_name}': {e}")
            raise

    def check_belonging(self, points, polys):

        # Build spatial index over polygons
        tree = STRtree(polys)

        # Candidate polygon indices for each point
        # query returns (geom_idx_in_points, geom_idx_in_polys) pairs
        pairs = tree.query(points).T

        # Initialize as all True
        mask = np.ones(len(points), dtype=bool)

        # Group candidate polygons per point to run exact test only on candidates
        # pairs is an (k, 2) ndarray; safe even if empty
        if len(pairs):
            p_idx, poly_idx = pairs[:, 0], pairs[:, 1]
            # Optional: prefetch polygon objects
            candidate_polys = [polys[j] for j in poly_idx]
            candidate_points = [points[i] for i in p_idx]

            exact = [
                poly.covers(pt) for pt, poly in zip(candidate_points, candidate_polys)
            ]

            # Any True among candidates marks the point as inside some polygon
            for i, is_in in zip(p_idx, exact):
                if is_in:
                    mask[i] = False

        return mask

    def build_qc_mask(self):

        coords = np.asarray(
            self.sdata[self.table_name].obsm[f"centroid_{self.object_name}"],
            dtype=float,
        )
        coords = coords[:, ::-1]  # swap to match Interactive
        points = [Point(xy) for xy in coords]

        mask = np.ones(self.sdata[self.table_name].X.shape, dtype="bool")

        markers = [
            x.split("_")[0] for x in self.sdata[self.table_name].var.index.tolist()
        ]

        # group by marker
        for marker, group in groupby(enumerate(markers), key=lambda t: t[1]):
            # collect the contiguous indices for this marker
            cols = [i for i, _ in group]
            start, end = cols[0], cols[-1] + 1

            shapes_key = f"{self.qc_prefix}_{marker}"
            if shapes_key not in self.sdata:
                # no QC shapes for this marker
                continue

            shapes_gdf = self.sdata[shapes_key]
            polys = list(shapes_gdf.geometry.values)
            if not polys:
                continue

            # compute per-observation pass mask once for the marker
            mask_column = self.check_belonging(points, polys)

            # broadcast to the whole contiguous block of columns
            mask[:, start:end] = mask_column[:, None]

        return mask

    def run(
        self,
        sdata: sd.SpatialData,
    ) -> None:

        self.sdata = sdata

        # validate input
        self.validate_sdata()

        # add qc_mask
        mask = self.build_qc_mask()

        # add mask to anndata
        self.sdata[self.table_name].layers[self.layer_name] = mask

        # save to disk if requested
        if self.write_to_disk:
            self.rewrite_table()
