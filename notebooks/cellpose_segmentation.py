import argparse
from pathlib import Path
import zarr
import dask.array as da
from tifffile import imread
from skimage.io import imsave
from cellpose import models
from loguru import logger

def parse_args():
    p = argparse.ArgumentParser(description="Load level 0 from a multiscale Zarr-backed image")
    p.add_argument("--input", type=Path, help="Path to the TIFF file.")
    p.add_argument("--output", "-o", type=Path, required=True, help="Path to save the mask.")
    p.add_argument("--level", type=str, default="0", help="Resolution level to load (default: 0)")
    return p.parse_args()

def main():
    args = parse_args()
    file_path = args.input
    resolution_level = args.level
    output_path = args.output

    # read in the image
    store = imread(file_path, aszarr=True)
    root = zarr.open(store, mode="r") 
    im = da.from_zarr(root[resolution_level])

    # run cellpose segmentation
    model = models.CellposeModel(gpu=True)

    flow_threshold = 0.4
    cellprob_threshold = -0.2
    tile_norm_blocksize = 0

    im_small = im[17000:19000, 45000:47000]

    o1, _, _ = model.eval(im_small, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                    normalize={"tile_norm_blocksize": tile_norm_blocksize})
    

    # save results
    imsave(output_path, o1.astype("int"))
    logger.info(f"Saved segmentation mask to {output_path}.")


if __name__ == "__main__":
    main()