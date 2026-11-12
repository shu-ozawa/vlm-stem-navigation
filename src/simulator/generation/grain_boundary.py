"""A finite monolayer MoS2 specimen with one straight grain boundary at a random angle.

The simulated bicrystal (data/grain_boundary, simulator/generation/simulate_gb.py) is periodic and holds two
boundaries of different core structure, running along its rows: "main" in the middle and "wrap" at the
edge (rolling the image by half its width puts it in the middle). Each episode:

1. picks one of the two boundaries,
2. repeats the image along the boundary and cuts a square around it, with the boundary offset from the
   centre at random (it must cross the region >= BMARGIN px inside the episode image),
3. rotates the square by a uniform random angle in [0, 180) deg (cubic splines), crops ALONG x ACROSS px
   (~27 x 32 nm) and mirrors it at random.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from simulator.paths import DATA_DIR
from simulator.generation.geometry import LineTargets, clip_segment, seg_hits_box

DATA = DATA_DIR / "grain_boundary"


class GrainBoundarySpecimen:
    ALONG_TILES = 2      # the simulated period is repeated this many times along the boundary
    ACROSS = 2560        # episode image width across the boundary (before rotation) [px]
    BMARGIN = 640        # the boundary crosses the region >= this many px inside the episode image

    def __init__(self, data_dir: str | Path = DATA):
        data_dir = Path(data_dir)
        meta = json.loads((data_dir / "meta.json").read_text())
        image = np.load(data_dir / meta["image_file"])
        self._col = meta["boundary_cols"]["main"]
        self._sources = {"main": image, "wrap": np.roll(image, image.shape[1] // 2, axis=1)}

        px = meta["pixel_size_A"]
        self.pixel_size_A = 0.5 * (px["x"] + px["y"])
        self.window = (float(image.min()), float(image.max()))  # black, the brightest atomic column
        self.image = np.zeros((0, 0), np.float32)
        self.H = self.W = 0
        self.segment = ((0.0, 0.0), (0.0, 0.0))
        self.targets = LineTargets([])

    def build(self, rng: np.random.Generator) -> None:
        names = sorted(self._sources)
        self.variant = names[rng.integers(len(names))]
        source = self._sources[self.variant]
        period = source.shape[0]
        along, across = self.ALONG_TILES * period, self.ACROSS

        # a random rotation and mirroring; out = Co + R^T (square - Cs) maps the square to the episode
        S = int(np.ceil(np.hypot(along, across))) + 4          # square side: the episode's diagonal
        theta = float(rng.uniform(0.0, 180.0))
        flip = bool(rng.integers(2))
        t = np.deg2rad(theta)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        Cs, Co = np.array([S / 2, S / 2]), np.array([along / 2, across / 2])

        def to_episode(p):
            return Co + R.T @ (np.asarray(p, float) - Cs)

        # the boundary's offset from the centre of the square: it must cross the inner region
        inner = (self.BMARGIN, along - self.BMARGIN, self.BMARGIN, across - self.BMARGIN)
        dmax = min(along, across) / 2 - self.BMARGIN / 2
        while True:
            d = float(rng.uniform(-dmax, dmax))
            if seg_hits_box(to_episode((0, S / 2 + d)), to_episode((S, S / 2 + d)), *inner):
                break

        # cut the square (repeated along the boundary, rolled at random) and rotate it into the episode
        crop0 = int(round(self._col - (S / 2 + d)))
        d = self._col - crop0 - S / 2                            # exact offset after rounding the crop
        roll = int(rng.integers(period))
        square = np.tile(source[:, crop0:crop0 + S], (-(-S // period) + 1, 1))
        square = np.roll(square, roll, axis=0)[:S]
        image = ndimage.affine_transform(square, R, offset=Cs - R @ Co, output_shape=(along, across),
                                         order=3, mode="nearest").astype(np.float32)

        # the boundary in the episode, mirrored with the image
        a, b = to_episode((-1e4, S / 2 + d)), to_episode((S + 1e4, S / 2 + d))
        if flip:
            image = np.ascontiguousarray(image[:, ::-1])
            a, b = (a[0], across - 1 - a[1]), (b[0], across - 1 - b[1])
        a, b = clip_segment(tuple(a), tuple(b), along, across)

        self.image, (self.H, self.W) = image, image.shape
        self.segment = (tuple(map(float, a)), tuple(map(float, b)))
        self.targets = LineTargets([self.segment])
        self.orient = {"angle_deg": theta, "fliplr": flip, "offset_px": d}

    def markers(self) -> list[dict]:
        ab = clip_segment(*self.segment, self.H, self.W)
        return [] if ab is None else [{"kind": "target", "type": "grain_boundary", "line": ab}]

    def episode_info(self) -> dict:
        return {"orientation": self.orient, "variant": self.variant, "line": list(self.segment)}
