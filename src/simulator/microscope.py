
"""Simulated microscope: stage movement, magnification, and image observation."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from simulator.actions import NAMED, parse_move


ZOOM_ACTIONS = ("zoom_in", "zoom_out")
REF_PEAK = 0.8


@dataclass
class MicroscopeConfig:
    fov: int = 256
    zoom_levels: tuple[float, ...] = (1, 2, 3)
    move_frac: float = 1 / 16
    upscale: int = 2
    dose: float | None = None


class Microscope:
    def __init__(self, cfg: MicroscopeConfig):
        self.cfg = cfg

        # Specimen
        self.image = np.zeros((0, 0), np.float32)
        self.H = self.W = 0
        self.pixel_size_A = 1.0
        self.lo, self.hi = 0.0, 1.0

        # Episode
        self.zi = 0
        self.center = (0, 0)
        self.start_center = (0, 0)
        self.steps = 0
        self.episode_seed = None

    # Specimen and episode

    def load(
        self,
        image: np.ndarray,
        pixel_size_A: float,
        window: tuple[float, float],
    ) -> None:
        """Load a specimen image and its intensity window."""
        self.image = image
        self.H, self.W = image.shape
        self.pixel_size_A = pixel_size_A
        self.lo, self.hi = window

    def start(self, center: tuple[int, int], zi: int, seed: int | None) -> None:
        """Start a new episode."""
        self.zi = zi
        self.center = center
        self.start_center = center
        self.steps = 0
        self.episode_seed = seed

    # Geometry

    def F_at(self, zi: int) -> int:
        """Field-of-view width in specimen pixels."""
        return int(round(self.cfg.fov * self.cfg.zoom_levels[zi]))

    @property
    def F(self) -> int:
        return self.F_at(self.zi)

    @property
    def n_zoom(self) -> int:
        return len(self.cfg.zoom_levels)

    def clamp(
        self, cy: float, cx: float, F: int | None = None
    ) -> tuple[int, int]:
        """Keep the FOV centre within the specimen."""
        fov = self.F if F is None else F

        min_center = fov // 2
        max_row = self.H - (fov - min_center)
        max_col = self.W - (fov - min_center)

        row = min(max(int(round(cy)), min_center), max_row)
        col = min(max(int(round(cx)), min_center), max_col)

        return row, col

    @property
    def fov_rc(self) -> tuple[int, int]:
        """Top-left corner of the current FOV (row, col)."""
        row, col = self.center
        return row - self.F // 2, col - self.F // 2

    def at_limit(self) -> tuple[str, ...]:
        """Directions in which a named move would be blocked."""
        step_size = self.F * self.cfg.move_frac
        row, col = self.center

        blocked_directions = []

        for direction, (dr, dc) in NAMED.items():
            new_center = self.clamp(
                row + dr * step_size,
                col + dc * step_size,
            )
            if new_center == self.center:
                blocked_directions.append(direction)

        return tuple(blocked_directions)

    def operator_state(self) -> dict:
        """Return the microscope state available to the operator."""
        start_row, start_col = self.start_center
        row, col = self.center
        pixel_size = self.pixel_size_A

        return {
            "zoom_index": self.zi,
            "n_zoom": self.n_zoom,
            "fov_A": self.F * pixel_size,
            "pos_A": (
                (col - start_col) * pixel_size,
                (row - start_row) * pixel_size,
            ),
            "at_limit": self.at_limit(),
        }

    # Actions

    def act(self, action: str) -> bool:
        """Apply an action. Return True if it was blocked."""
        if action in ZOOM_ACTIONS:
            new_zoom = self.zi - 1 if action == "zoom_in" else self.zi + 1

            if not 0 <= new_zoom < self.n_zoom:
                return True

            self.zi = new_zoom
            self.center = self.clamp(*self.center)
            return False

        step_size = self.F * self.cfg.move_frac
        dy, dx = parse_move(action)

        new_center = self.clamp(
            self.center[0] + dy * step_size,
            self.center[1] + dx * step_size,
        )

        blocked = new_center == self.center
        self.center = new_center
        return blocked

    # Observation

    def fov_array(self) -> np.ndarray:
        """Extract the current FOV and optionally apply Poisson noise."""
        row, col = self.fov_rc
        fov = self.F
        crop = self.image[row:row + fov, col:col + fov]

        if self.cfg.dose:
            seed = [
                0 if self.episode_seed is None else int(self.episode_seed),
                self.steps,
                row,
                col,
                fov,
            ]
            electrons_per_pixel = self.cfg.dose * self.pixel_size_A ** 2

            crop = np.random.default_rng(seed).poisson(
                np.clip(crop, 0, None) * electrons_per_pixel
            ) / electrons_per_pixel

        return crop

    def to_uint8(self, arr: np.ndarray) -> np.ndarray:
        """Map specimen intensities to 8-bit grayscale."""
        width = (self.hi - self.lo) / REF_PEAK
        mid = self.lo + width / 2
        lo, hi = mid - width / 2, mid + width / 2

        return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

    def observe(self) -> bytes:
        """Return the current view as an 8-bit grayscale PNG."""
        image = Image.fromarray(self.to_uint8(self.fov_array()), mode="L")

        output_size = self.cfg.fov * self.cfg.upscale
        if image.size[0] != output_size:
            resampling = (
                Image.LANCZOS if image.size[0] > output_size
                else Image.BILINEAR
            )
            image = image.resize(
                (output_size, output_size),
                resampling,
            )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
