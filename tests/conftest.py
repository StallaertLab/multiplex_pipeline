# tests/conftest.py
import sys
from unittest.mock import MagicMock
import pytest

@pytest.fixture(scope="session", autouse=True)
def mock_gui_dependencies():
    """
    Mock napari and qtpy globally for the entire test session.
    This prevents 'napari is not a package' errors when npe2 
    or other utilities try to import submodules.
    """
    # 1. Mock Napari (The critical part)
    mock_napari = MagicMock()
    # THIS LINE SAVES YOU: It tells Python "I am a package"
    mock_napari.__path__ = [] 
    
    # 2. Register mock submodules to satisfy imports like 'from napari.resources...'
    sys.modules["napari"] = mock_napari
    sys.modules["napari.resources"] = MagicMock()
    sys.modules["napari.utils"] = MagicMock()
    sys.modules["napari.utils.theme"] = MagicMock()
    sys.modules["napari.layers"] = MagicMock()
    sys.modules["napari.viewer"] = MagicMock()
    sys.modules["napari._pydantic_compat"] = MagicMock() # Handles the Color validator import

    # 3. Mock QtPy
    mock_qtpy = MagicMock()
    mock_qtpy.QtWidgets.QFileDialog.getOpenFileName.return_value = ("test.csv", "")
    mock_qtpy.QtWidgets.QFileDialog.getSaveFileName.return_value = ("test.pkl", "")
    
    sys.modules["qtpy"] = mock_qtpy
    sys.modules["qtpy.QtWidgets"] = mock_qtpy.QtWidgets

    yield