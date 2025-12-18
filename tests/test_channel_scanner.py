import pytest

import plex_pipe.core_cutting.channel_scanner as channel_scanner
from plex_pipe.core_cutting.channel_scanner import (
    scan_channels_from_list,
    discover_channels,
    build_transfer_map,
)

def test_scan_picks_latest_per_marker_and_prefers_001_dapi():
    """
    Verifies: channel parsing + selection policy.
    - Latest round is chosen for non-DAPI markers.
    - DAPI is normalized to key "DAPI" and prefers 001_DAPI when available.
    Why: Ensures deterministic channel set used downstream for cutting/quant.
    """
    files = [
        "p_001.0.4_R000_DAPI_x.ome.tif",           
        "p_002.0.4_R000_dye_CD3_x.ome.tif",
        "p_003.0.4_R000_dye_CD3_x.ome.tif",  
        "p_004.0.4_R000_dye_CK7_x.ome.tif",
        "p_004.0.4_R000_DAPI_x.ome.tif",  
    ]
    out = scan_channels_from_list(files)
    assert out.keys() == {"DAPI", "CD3", "CK7"}
    # latest round for CD3 is kept
    assert out["CD3"] == "p_003.0.4_R000_dye_CD3_x.ome.tif"
    # DAPI key is normalized
    assert out["DAPI"] == "p_001.0.4_R000_DAPI_x.ome.tif"

def test_scan_include_channels_overrides_grouping():
    """
    Verifies: explicit include list bypasses grouping and keeps the exact
    channel IDs requested.
    """
    files = [
        "p_001.0.4_R000_DAPI_xxx.ome.tif",
        "p_002.0.4_R000_dye_CD3-01_x.ome.tif",
        "p_003.0.4_R000_dye_CD3-02_x.ome.tif",
    ]
    # include exact channel name "003_CD3" to keep that one verbatim
    out = scan_channels_from_list(files, include_channels=["002_CD3"])
    assert out["CD3"] == "p_002.0.4_R000_dye_CD3-01_x.ome.tif"

def test_scan_exclude_channels_overrides_grouping():
    """
    Verifies: explicit exclude list bypasses grouping.
    """
    files = [
        "p_001.0.4_R000_DAPI_xxx.ome.tif",
        "p_002.0.4_R000_dye_CD3-01_x.ome.tif",
        "p_003.0.4_R000_dye_CD3-02_x.ome.tif",
    ]
    # include exact channel name "003_CD3" to keep that one verbatim
    out = scan_channels_from_list(files, exclude_channels=["003_CD3"])
    assert out["CD3"] == "p_002.0.4_R000_dye_CD3-01_x.ome.tif"

def test_scan_include_exclude_channels_overrides_grouping():
    """
    Verifies: includes overwrite excludes.
    """
    files = [
        "p_001.0.4_R000_DAPI_xxx.ome.tif",
        "p_002.0.4_R000_dye_CD3-01_x.ome.tif",
        "p_003.0.4_R000_dye_CD3-02_x.ome.tif",
    ]
    # include exact channel name "003_CD3" to keep that one verbatim
    out = scan_channels_from_list(files, exclude_channels=["003_CD3"], include_channels=["003_CD3"])
    assert out["CD3"] == "p_003.0.4_R000_dye_CD3-02_x.ome.tif"

def test_scan_ignore_markers_overrides_grouping():
    """
    Verifies: includes overwrite excludes.
    """
    files = [
        "p_001.0.4_R000_DAPI_xxx.ome.tif",
        "p_002.0.4_R000_dye_CD3-01_x.ome.tif",
        "p_003.0.4_R000_dye_CD3-02_x.ome.tif",
    ]
    # include exact channel name "003_CD3" to keep that one verbatim
    out = scan_channels_from_list(files, ignore_markers=["CD3"])
    assert "CD3" not in out

def test_scan_use_markers_overrides_grouping():
    """
    Verifies: includes overwrite excludes.
    """
    files = [
        "p_001.0.4_R000_DAPI_xxx.ome.tif",
        "p_002.0.4_R000_dye_CD3-01_x.ome.tif",
        "p_003.0.4_R000_dye_CD3-02_x.ome.tif",
    ]
    # include exact channel name "003_CD3" to keep that one verbatim
    out = scan_channels_from_list(files, use_markers=["DAPI"])
    assert "CD3" not in out

def test_build_transfer_map_creates_posix_destinations_from_local_dir(tmp_path):
    """
    Verifies: local destination mapping is POSIX-format and file-named.
    Why: Globus expects POSIX-style targets regardless of the OS running tests.
    """
    remote = {
        "CD3": "/remote/dir/CD3-02.ome.tif",
        "DAPI": "/remote/dir/DAPI.ome.tif",
    }
    # Use a POSIX-ish base; function converts to POSIX for Globus anyway.
    base = tmp_path / "work" / "local"
    base.mkdir(parents=True, exist_ok=True)

    m = build_transfer_map(remote, str(base))
    # remote side preserved as POSIX
    assert m["CD3"][0].endswith("/remote/dir/CD3-02.ome.tif")
    # local side is base + filename, POSIX-formatted
    assert m["CD3"][1].endswith("/work/local/CD3-02.ome.tif")
    assert m["DAPI"][1].endswith("/work/local/DAPI.ome.tif")

def test_discover_channels_uses_globus_listing(monkeypatch):
    calls = {}

    def fake_list_globus(gc, path):
        calls["hit"] = True
        return [
            "p_001.0.4_R000_DAPI_xxx.ome.tif",
            "p_002.0.4_R000 dye CK7-01.ome.tif".replace(" ", "_"),
        ]

    # Patch the **local binding** inside channel_scanner
    monkeypatch.setattr(channel_scanner, "list_globus_files", fake_list_globus)

    out = channel_scanner.discover_channels("/remote/path", gc=object())
    assert calls.get("hit")
    assert set(out) == {"DAPI", "CK7"}