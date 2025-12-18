import pytest
from pathlib import Path
import spatialdata as sd
import shutil
from multiplex_pipeline.object_quantification.controller import QuantificationController 

@pytest.fixture(scope="session")
def example_sdata_copy(tmp_path_factory):
    """
    Create a single session-level copy of the example .zarr dataset.
    """
    project_root = Path(__file__).parent 
    src = project_root / "example_data" / "Core_000.zarr"
    dst = tmp_path_factory.mktemp("example_sdata") / "Core_000.zarr"
    shutil.copytree(src, dst)
    return dst

@pytest.fixture
def sdata_read(example_sdata_copy):
    """
    Load the copied dataset freshly for each test.
    """
    return sd.read_zarr(example_sdata_copy)


def test_validate_inputs_raises_on_missing_mask_or_channel(sdata_read):
    """
    Verifies input validation for essential pipeline contracts:
    - Fails fast if a required mask or channel is absent.
    """

    sdata = sdata_read

    qc = QuantificationController(mask_keys={"cell": "instanseg_cell"}, to_quantify=["DAPI"])
    qc.sdata = sdata
    qc.validate_sdata_as_input()

    ch = 'ch1'
    qc = QuantificationController(mask_keys={"cell": "instanseg_cell"}, to_quantify=[ch])
    qc.sdata = sdata
    with pytest.raises(ValueError) as er:
        qc.validate_sdata_as_input()
    assert f"Channel '{ch}' not found in sdata" in str(er.value)

    mask = 'x'
    qc = QuantificationController(mask_keys={"cell": mask}, to_quantify=[ch])
    qc.sdata = sdata
    with pytest.raises(ValueError) as er:
        qc.validate_sdata_as_input()
    assert f"Mask '{mask}' not found in sdata. Masks present" in str(er.value)

def test_overwrite_semantics_respected_on_existing_table(sdata_read):
    """
    If a table with the same name already exists in the dataset:
    - overwrite=False => raises,
    - overwrite=True  => replaces without error.
    Protects previous results by default, while allowing explicit refresh.
    """
    sdata = sdata_read

    table_name = 'instanseg_table'
    mask = 'instanseg_cell'
    ch = 'DAPI'

    # overwrite=False -> error
    qc_no_over = QuantificationController(
        mask_keys={"cell": mask},
        to_quantify=[ch],
        table_name=table_name,
        overwrite=False,
    )
    with pytest.raises(ValueError) as er:
        qc_no_over.run(sdata)
    assert f"Table '{table_name}' already exists in sdata." in str(er.value)

    # overwrite=True -> should succeed
    qc_over = QuantificationController(
        mask_keys={"cell": mask},
        to_quantify=[ch],
        table_name=table_name,
        overwrite=True,
    )
    qc_over.run(sdata)

def test_run_writes_new_table_and_aligns_vars_obs(sdata_read):
    """
    Runs controller on a temp copy and asserts:
    - a new table is attached,
    - it has observations (labels),
    - and expected feature names (mean/median) for the chosen channel exist in var.
    This confirms end-to-end stitching of morphology+intensity on real data.
    """
    sdata = sdata_read
    mask_key, ch_key = "instanseg_cell", "DAPI"

    new_table = "qc_test_table"  
    assert new_table not in getattr(sdata, "tables", {})

    qc = QuantificationController(
        mask_keys={"cell": mask_key},
        to_quantify=[ch_key],
        table_name=new_table,
        connect_to_mask=mask_key,
        overwrite=False,
    )
    qc.run(sdata)

    # Table attached
    assert new_table in sdata
    adata = sdata[new_table]
    assert adata.n_obs > 0

    # Expected feature names exist (controller-specific naming: mean/median for channel)
    var_names = list(adata.var.index)
    # Use a relaxed check: presence of "mean" and "median" features related to the channel
    assert any(ch_key in v and "mean" in v for v in var_names)
    assert any(ch_key in v and "median" in v for v in var_names)
    assert 'area_cell' in adata.obs.columns
