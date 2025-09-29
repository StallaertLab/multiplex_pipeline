"""Image utilities for reading multiscale images and preparing RGB previews."""

import dask.array as da
import numpy as np
import zarr
from skimage.transform import rescale
from tifffile import imread


def get_org_im_shape(im_path):
    """
    Function to get the size of the original image.

    Args:
        im_path (str): Path to the image.
    Returns:
        tuple: The size of the image.
    """
    store = imread(im_path, aszarr=True)
    d = da.from_zarr(store, 0)

    return d.shape


def get_zarr_levels_num(path):
    """
    Get the number of multiscale levels in a Zarr file.

    Args:
        path (str): The path to the Zarr file.
    Returns:
        int: The number of multiscale levels.
    """
    store = imread(path, aszarr=True)
    group = zarr.open(store, mode="r")
    zattrs = group.attrs.asdict()

    return len(zattrs["multiscales"][0]["datasets"])


def get_small_image(im_path, req_level):
    """
    Function to get a small image from the zarr file.

    Args:
        im_path (str): Path to the zarr file.
        req_level (int): Requested level of the image.
    Returns:
        np.array: The image of requested level.
    """
    store = imread(im_path, aszarr=True)
    group = zarr.open(store, mode="r")
    zattrs = group.attrs.asdict()

    if store.is_multiscales:
        level_num = np.min([req_level, get_zarr_levels_num(im_path)])
    else:
        level_num = 0

    path = zattrs["multiscales"][0]["datasets"][level_num]["path"]
    im = da.from_zarr(group[path]).compute()

    # additional resizing if requested level was not present
    if level_num > req_level:
        im = rescale(im, 1 / (2 ** (req_level - level_num)), anti_alias=True)

    return im


def prepare_rgb_image(im_path, perc_min=0.01, perc_max=99, req_level=0):
    """
    Function to change an image into RGB image of stretched intensity.

    Args:
        im_path (str): Path to the image.
        perc_min (float): Lower percentile for stretching.
        perc_max (float): Upper percentile for stretching.
    Returns:
        np.array: The RGB image.
    """

    # get the image of requested resolution
    im = get_small_image(im_path, req_level)

    # rescale between given percentiles
    im_min, im_max = np.percentile(im, [perc_min, perc_max])
    im = (im - im_min) / (im_max - im_min)
    im = np.clip(im, 0, 1)

    im = (im * 255).astype(np.uint8)  # change to 8 bit

    # convert to RGB
    im_rgb = np.stack((im,) * 3, axis=-1)

    return im_rgb


def calculate_median(mask, im):
    # extra property for regionprops
    return np.median(im[mask > 0])
