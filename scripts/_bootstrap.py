from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MPL_CACHE = ROOT / ".cache" / "matplotlib"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
