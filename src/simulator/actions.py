
"""Convert microscope move actions between strings and direction vectors."""

import math

# Named directions in image coordinates (row, column).
NAMED = {
    "up": (-1.0, 0.0),
    "down": (1.0, 0.0),
    "left": (0.0, -1.0),
    "right": (0.0, 1.0),
}

PREFIX = "move:"


def format_move(dx: float, dy: float) -> str:
    """Convert (dx, dy) into a normalized move action string."""
    length = math.hypot(dx, dy)
    return f"{PREFIX}{dx / length:+.3f},{dy / length:+.3f}"


def parse_move(action: str) -> tuple[float, float] | None:
    """Return the unit direction (d_row, d_col), or None if not a move."""
    if action in NAMED:
        return NAMED[action]

    if not action.startswith(PREFIX):
        return None

    dx_text, dy_text = action[len(PREFIX):].split(",")
    dx = float(dx_text)
    dy = float(dy_text)

    length = math.hypot(dx, dy)
    return dy / length, dx / length


def compass(action: str) -> str:
    """Return the nearest compass direction for logging."""
    direction = parse_move(action)

    if direction is None or action in NAMED:
        return action

    dr, dc = direction
    angle = math.degrees(math.atan2(-dr, dc)) % 360

    names = (
        "right", "up-right", "up", "up-left",
        "left", "down-left", "down", "down-right",
    )
    return names[int((angle + 22.5) // 45) % 8]
