
"""Task configurations and environment factory."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Callable

from simulator.microscope import MicroscopeConfig
from simulator.generation.dopant import DopantSpecimen
from simulator.generation.grain_boundary import GrainBoundarySpecimen
from simulator.env import NavigationEnv, TaskRules


# Point target: 32 / 64 / 96 Å FOV, 512 px observations
POINT_MICROSCOPE = MicroscopeConfig(fov=256, zoom_levels=(1, 2, 3), upscale=2)
POINT_RULES = TaskRules(visible_margin=12, start_clearance=24)

# Line target: 64 / 96 / 128 Å FOV, 512 px observations
LINE_MICROSCOPE = MicroscopeConfig(fov=512, zoom_levels=(1, 1.5, 2), upscale=1)
LINE_RULES = TaskRules(visible_margin=32, start_clearance=16, builds=50, start_tries=4000)


def target_margin(microscope: MicroscopeConfig, rules: TaskRules) -> int:
    """Minimum target distance from specimen edges, in pixels."""
    max_fov = microscope.fov * max(microscope.zoom_levels)
    return int(math.ceil(max_fov / 2 + rules.center_frac * max_fov / 2))


@dataclass(frozen=True)
class Task:
    name: str
    specimen: Callable[[MicroscopeConfig, TaskRules], object]
    microscope: MicroscopeConfig = field(default_factory=MicroscopeConfig)
    rules: TaskRules = field(default_factory=TaskRules)


TASKS = {
    "dopant_zoom": Task(
        name="dopant_zoom",
        specimen=lambda microscope, rules: DopantSpecimen(
            margin=target_margin(microscope, rules)
        ),
        microscope=POINT_MICROSCOPE,
        rules=POINT_RULES,
    ),
    "grain_boundary": Task(
        name="grain_boundary",
        specimen=lambda microscope, rules: GrainBoundarySpecimen(),
        microscope=LINE_MICROSCOPE,
        rules=LINE_RULES,
    ),
}


def make(name: str, **overrides) -> NavigationEnv:
    """Create a task environment with optional configuration overrides."""
    task = TASKS[name]

    microscope_fields = {item.name for item in fields(MicroscopeConfig)}
    microscope_overrides = {}
    rule_overrides = {}

    for key, value in overrides.items():
        if key in microscope_fields:
            microscope_overrides[key] = value
        else:
            rule_overrides[key] = value

    microscope = replace(task.microscope, **microscope_overrides)
    rules = replace(task.rules, **rule_overrides)
    specimen = task.specimen(microscope, rules)

    return NavigationEnv(name, specimen, microscope, rules)
