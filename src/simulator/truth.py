
"""Ground-truth labels for evaluating model decisions. Never used by the agent."""

def is_correct(truth, choice: str) -> bool | None:
    """Check whether a choice matches any accepted label."""
    if truth is None:
        return None
    if isinstance(truth, list):
        return choice in truth
    return choice == truth


def question_truth(env, info: dict) -> dict:
    """Compute accepted labels from the specimen's ground truth."""
    center = (int(info["center_row"]), int(info["center_col"]))
    half_width = env.rules.center_frac * float(info["fov_size"]) / 2

    if env.in_center(center, half=half_width):
        direction = ["inside"]
    else:
        row_offset, col_offset = env.nearest(center)
        direction = []

        if abs(col_offset) >= half_width:
            direction.append("left" if col_offset < 0 else "right")

        if abs(row_offset) >= half_width:
            direction.append("up" if row_offset < 0 else "down")

    visible = info["target_visible"]

    return {
        "target_visible": "yes" if visible else "no",
        "direction": direction,
        "locate": direction if visible else "none",
    }
