import pytest
import os
from pathlib import Path
import tifffile
import numpy as np
import spatialdata as sd
from multiplex_pipeline.core_cutting.assembler import CoreAssembler

def _write_tiff(path: Path, shape=(8, 8), dtype=np.uint16, value=1):
    arr = (np.ones(shape, dtype=dtype) * value)
    tifffile.imwrite(str(path), arr)

def test_assemble_core_raises_on_missing_folder(tmp_path):
    """
    Verifies: assembler fails fast when the expected per-core temp folder is absent.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    asm = CoreAssembler(temp_dir=str(tmp_path / "temp"), output_dir=str(out_dir))

    with pytest.raises(FileNotFoundError):
        asm.assemble_core("Core_001") # temp folder does not exist

def test_assemble_core_writes_zarr_and_elements(tmp_path):
    """
    Verifies: valid per-channel TIFFs are packed into SpatialData elements on disk.
    """
    temp_dir = tmp_path / "temp"
    core_dir = temp_dir / "Core_000"
    core_dir.mkdir(parents=True)

    # Two simple channels
    _write_tiff(core_dir / "DAPI.tiff", value=10)
    _write_tiff(core_dir / "CD3.tiff", value=20)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    asm = CoreAssembler(
        temp_dir=str(temp_dir),
        output_dir=str(out_dir),
        max_pyramid_levels=1, 
        cleanup=True,
    )
    zarr_path = asm.assemble_core("Core_000")

    # Assert zarr directory exists
    assert os.path.isdir(zarr_path)

    # Read back with SpatialData and check element presence
    s = sd.read_zarr(zarr_path)
    assert "DAPI" in s
    assert "CD3" in s

    # test cleanup
    assert not (core_dir / "DAPI.tiff").exists()
    assert not (core_dir / "CD3.tiff").exists()