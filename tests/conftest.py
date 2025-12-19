import sys
from unittest.mock import MagicMock
import pytest

# =============================================================================
# GLOBAL PATCH (Runs at import time, BEFORE collection)
# =============================================================================

# 1. Check if we are in a headless environment (GitHub Actions)
#    (Or just always mock if you want consistent behavior)
try:
    import napari
except ImportError:
    # Napari is missing, so we MUST mock it immediately
    print("Headless environment detected: Mocking napari and qtpy...")

    # --- MOCK NAPARI ---
    mock_napari = MagicMock()
    mock_napari.__path__ = [] # Vital: makes it look like a package
    
    sys.modules["napari"] = mock_napari
    sys.modules["napari.layers"] = MagicMock()
    sys.modules["napari.viewer"] = MagicMock()
    sys.modules["napari.utils"] = MagicMock()
    sys.modules["napari.utils.theme"] = MagicMock()
    sys.modules["napari.resources"] = MagicMock()
    sys.modules["napari.types"] = MagicMock()
    sys.modules["napari._pydantic_compat"] = MagicMock()

    # --- MOCK QTPY ---
    mock_qtpy = MagicMock()
    mock_qtpy.QtWidgets.QFileDialog.getOpenFileName.return_value = ("test.csv", "")
    mock_qtpy.QtWidgets.QFileDialog.getSaveFileName.return_value = ("test.pkl", "")
    
    sys.modules["qtpy"] = mock_qtpy
    sys.modules["qtpy.QtWidgets"] = mock_qtpy.QtWidgets
    sys.modules["PyQt5"] = MagicMock() # Prevent backend search