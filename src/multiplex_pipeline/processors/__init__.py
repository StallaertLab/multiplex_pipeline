from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules

# 1) Auto-discover processor modules so @register runs on import
_pkg_path = Path(__file__).parent
for m in iter_modules([str(_pkg_path)]):
    if not m.ispkg and m.name not in {"__init__", "registry", "base"}:
        import_module(f"{__name__}.{m.name}")

# 2) Re-export the public API from registry for a nice import surface
from .registry import (  # noqa: E402
    REGISTRY,
    build_processor,
    register,
)

__all__ = [
    "build_processor",
    "register",
    "REGISTRY",
]
