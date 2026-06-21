"""Trading system package."""

from __future__ import annotations

import os

from .paths import project_root


_mpl_cache_dir = project_root() / ".cache" / "matplotlib"
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

__all__ = ["__version__"]

__version__ = "0.1.0"
