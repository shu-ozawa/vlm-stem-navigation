
"""Navigation environment: specimen, microscope, and episode rules."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from simulator.microscope import Microscope, MicroscopeConfig


@dataclass
class TaskRules:
    start_mode: str = "hidden"
    start_zoom: int = 0
    center_frac: float = 0.2
    visible_margin: int = 12
    start_clearance: int = 24
    max_steps: int = 60
    builds: int = 1
    start_tries: int = 20000


class NavigationEnv:
    def __init__(
        self,
        name: str,
        specimen,
        microscope: MicroscopeConfig,
        rules: TaskRules,
    ):
        self.name = name
        self.specimen = specimen
        self.rules = rules
        self.microscope = Microscope(microscope)

        self.steps = 0
        self.done = True
        self.history = []

    # Task rules

    def visible(
        self,
        center=None,
        zi: int | None = None,
        margin: float | None = None,
    ) -> bool:
        """Whether a target is visible inside the FOV margin."""
        microscope = self.microscope

        if center is None:
            center = microscope.center
        if zi is None:
            zi = microscope.zi
        if margin is None:
            margin = self.rules.visible_margin * microscope.cfg.zoom_levels[zi]

        half_width = microscope.F_at(zi) / 2 - margin
        return self.specimen.targets.visible_box(center, half_width)

    def in_center(self, center=None, half: float | None = None) -> bool:
        """Whether a target is inside the success square."""
        if center is None:
            center = self.microscope.center
        if half is None:
            half = self.rules.center_frac * self.microscope.cfg.fov / 2

        return self.specimen.targets.success_box(center, half)

    def nearest(self, center=None) -> tuple[float, float]:
        """Offset (row, col) to the nearest target."""
        if center is None:
            center = self.microscope.center
        return self.specimen.targets.nearest(center)

    def _start_ok(self, center) -> bool:
        """Check whether a starting position satisfies the task rules."""
        rules = self.rules
        microscope = self.microscope

        if rules.start_mode == "visible":
            return (
                self.visible(center, rules.start_zoom)
                and not self.in_center(center)
            )

        # Hidden: no target near the starting FOV.
        start_half_width = (
            microscope.F_at(rules.start_zoom) / 2
            + rules.start_clearance
        )
        if self.specimen.targets.visible_box(center, start_half_width):
            return False

        # But a target must become visible at the lowest magnification.
        lowest_zoom = microscope.n_zoom - 1
        lowest_center = microscope.clamp(
            *center, F=microscope.F_at(lowest_zoom)
        )

        return self.visible(lowest_center, lowest_zoom)

    # Episode

    def reset(self, seed: int | None = None) -> tuple[bytes, dict]:
        """Build a specimen and find a valid starting position."""
        rules = self.rules
        microscope = self.microscope
        rng = np.random.default_rng(seed)

        start_center = None

        for _ in range(rules.builds):
            self.specimen.build(rng)
            microscope.load(
                self.specimen.image,
                self.specimen.pixel_size_A,
                self.specimen.window,
            )

            start_fov = microscope.F_at(rules.start_zoom)

            for _ in range(rules.start_tries):
                candidate = microscope.clamp(
                    rng.integers(microscope.H),
                    rng.integers(microscope.W),
                    F=start_fov,
                )

                if self._start_ok(candidate):
                    start_center = candidate
                    break

            if start_center is not None:
                break

        if start_center is None:
            raise RuntimeError(
                f"no start satisfies start_mode={rules.start_mode!r}"
            )

        microscope.start(start_center, rules.start_zoom, seed)

        self.steps = 0
        self.done = False
        self.history = []

        info = self.info(action=None, blocked=False)
        self.history.append(info)

        return microscope.observe(), info

    def step(self, action: str) -> tuple[bytes, float, bool, dict]:
        """Execute one action and return observation, reward, done, info."""
        microscope = self.microscope

        self.steps += 1
        microscope.steps = self.steps

        blocked = False
        success = False

        if action == "found":
            success = microscope.zi == 0 and self.in_center()
            self.done = True
        else:
            blocked = microscope.act(action)

        truncated = not self.done and self.steps >= self.rules.max_steps
        if truncated:
            self.done = True

        info = self.info(
            action=action,
            blocked=blocked,
            success=success,
            truncated=truncated,
        )
        self.history.append(info)

        reward = 1.0 if success else 0.0
        return microscope.observe(), reward, self.done, info

    # Observation and evaluation

    def info(
        self,
        action: str | None,
        blocked: bool,
        success: bool = False,
        truncated: bool = False,
    ) -> dict:
        """Return episode information, including ground truth."""
        microscope = self.microscope

        row, col = microscope.fov_rc
        dr, dc = self.nearest()
        distance = math.hypot(dr, dc)

        return {
            "step": self.steps,
            "action": action,
            "blocked": blocked,
            "fov_row": row,
            "fov_col": col,
            "fov_size": microscope.F,
            "center_row": microscope.center[0],
            "center_col": microscope.center[1],
            "zoom": microscope.cfg.zoom_levels[microscope.zi],
            "zoom_index": microscope.zi,
            "target_visible": self.visible(),
            "target_in_center": self.in_center(),
            "distance_px": distance,
            "distance_A": distance * microscope.pixel_size_A,
            "target_offset_rc": [dr, dc],
            "success": success,
            "truncated": truncated,
            "done": self.done,
            "task": self.name,
            **self.specimen.episode_info(),
        }

    def operator_state(self) -> dict:
        """Return the microscope state available to the agent."""
        return self.microscope.operator_state()

    def truth_markers(self) -> list[dict]:
        """Return ground-truth markers for visualization and evaluation."""
        return self.specimen.markers()
