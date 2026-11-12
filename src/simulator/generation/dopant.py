"""A finite monolayer MoS2 specimen with exactly one W dopant (substituting Mo).

The simulated image (data/dopant, simulator/generation/simulate_dopant.py) is periodic. Each episode rolls it so
that the W lands at a random position at least `margin` px from every edge; the roll is seamless.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from simulator.paths import DATA_DIR
from simulator.generation.geometry import PointTargets

DATA = DATA_DIR / "dopant"


class DopantSpecimen:
    def __init__(self, margin: int, data_dir: str | Path = DATA):
        data_dir = Path(data_dir)
        meta = json.loads((data_dir / "meta.json").read_text())
        self._simulated = np.load(data_dir / meta["image_file"])
        self._w_rc = (meta["dopant"]["pixel_row"], meta["dopant"]["pixel_col"])

        px = meta["pixel_size_A"]
        self.pixel_size_A = 0.5 * (px["x"] + px["y"])
        self.window = (float(self._simulated.min()), float(self._simulated.max()))  # black, the W column
        self.H, self.W = self._simulated.shape
        self.margin = margin

        self.image = self._simulated
        self.dopant_rc = self._w_rc
        self.targets = PointTargets([self.dopant_rc])

    def build(self, rng: np.random.Generator) -> None:
        m = self.margin
        tr, tc = int(rng.integers(m, self.H - m)), int(rng.integers(m, self.W - m))
        shift = (int(round(tr - self._w_rc[0])), int(round(tc - self._w_rc[1])))

        self.image = np.roll(self._simulated, shift, axis=(0, 1))
        self.dopant_rc = (self._w_rc[0] + shift[0], self._w_rc[1] + shift[1])
        self.targets = PointTargets([self.dopant_rc])

    def markers(self) -> list[dict]:
        return [{"kind": "target", "type": "W", "rc": tuple(self.dopant_rc)}]

    def episode_info(self) -> dict:
        return {}
