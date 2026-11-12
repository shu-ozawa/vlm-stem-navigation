
"""Ground-truth navigation baseline for SPL evaluation. Never used by the agent."""

from simulator.actions import format_move
from simulator.env import NavigationEnv


def oracle_action(env: NavigationEnv) -> str:
    """Choose the next action using the true target position."""
    microscope = env.microscope
    visible = env.visible(margin=0)

    if not visible and microscope.zi < microscope.n_zoom - 1:
        return "zoom_out"

    half_width = (
        env.rules.center_frac
        * microscope.cfg.fov
        * microscope.cfg.zoom_levels[microscope.zi]
        / 2
    )

    if visible and env.in_center(half=half_width):
        return "found" if microscope.zi == 0 else "zoom_in"

    row_offset, col_offset = env.nearest()
    return format_move(col_offset, row_offset)


def shortest_path(env: NavigationEnv, seed: int) -> int:
    """Count oracle actions until the episode ends."""
    env.reset(seed)
    steps = 0

    while not env.done:
        env.step(oracle_action(env))
        steps += 1

    return steps
