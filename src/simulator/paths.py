"""Where the simulated specimens are: <repo>/data, or STEMNAV_DATA."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("STEMNAV_DATA", Path(__file__).resolve().parents[2] / "data"))
