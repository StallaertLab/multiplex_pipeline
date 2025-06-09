import os
import zarr
from tifffile import imread, imwrite
import dask.array as da


def write_temp_tiff(array, core_id: str, channel: str, temp_dir: str):
    """
    Save a NumPy array as TIFF in temp/core_id/channel.tiff
    """
    core_path = os.path.join(temp_dir, core_id)
    os.makedirs(core_path, exist_ok=True)
    fname = os.path.join(core_path, f"{channel}.tiff")
    imwrite(fname, array)


def read_ome_tiff(path: str,level_num: int = 0) -> da.Array:
    """
    Return a Dask array from an OME-TIFF file.
    Assumes channels are along axis 0.
    """
    store = imread(path,aszarr=True)
    group = zarr.open(store, mode='r')
    zattrs = group.attrs.asdict()
    path = zattrs['multiscales'][0]['datasets'][level_num]['path']
    
    return da.from_zarr(group[path])
