
"""Evaluate single decisions at controlled distances from the target (views) or with the target placed in one of three
groups of the view (group_views)."""

import math
import time

import numpy as np

from agent.agent import Agent
from simulator.truth import is_correct, question_truth

GROUPS = ("inside", "outside", "not_in_view")


def group_of(truth: dict) -> str:
    """The group of a view, from its truth: inside the green square, outside it but in view, or not in view."""
    if truth["target_visible"] != "yes":
        return "not_in_view"
    return "inside" if "inside" in truth["locate"] else "outside"


def views(env, seed: int, offsets_A: list[float], zoom_indices: list[int]):
    """Generate reproducible probe views for one specimen."""
    env.reset(seed)

    rng = np.random.default_rng(seed)
    microscope = env.microscope
    targets = env.specimen.targets

    if hasattr(targets, "rcs"):  # Point target
        reference = targets.rcs[rng.integers(len(targets.rcs))]
        angles = [float(rng.uniform(0, 360)) for _ in offsets_A]

    else:  # Line target
        (r0, c0), (r1, c1) = targets.segments[0]

        fraction = rng.uniform(0.25, 0.75)
        reference = (
            r0 + fraction * (r1 - r0),
            c0 + fraction * (c1 - c0),
        )

        line_angle = math.degrees(math.atan2(r1 - r0, c1 - c0))
        side = 90.0 if rng.integers(2) else -90.0
        angles = [line_angle + side for _ in offsets_A]

    for offset_A, angle in zip(offsets_A, angles):
        distance_px = offset_A / microscope.pixel_size_A
        angle_rad = math.radians(angle)

        center = (
            reference[0] + distance_px * math.sin(angle_rad),
            reference[1] + distance_px * math.cos(angle_rad),
        )

        for zoom_index in zoom_indices:
            valid_center = microscope.clamp(
                *center, F=microscope.F_at(zoom_index)
            )
            yield zoom_index, offset_A, angle, valid_center


def clip(start, end, box) -> tuple[float, float] | None:
    """The part [t0, t1] of [0, 1] for which start + t (end - start) lies in box (row_lo, row_hi, col_lo, col_hi)."""
    t0, t1 = 0.0, 1.0
    for s, e, lo, hi in ((start[0], end[0], box[0], box[1]), (start[1], end[1], box[2], box[3])):
        d = e - s
        if abs(d) < 1e-12:
            if not lo <= s <= hi:
                return None
        else:
            a, b = sorted(((lo - s) / d, (hi - s) / d))
            t0, t1 = max(t0, a), min(t1, b)
    return (t0, t1) if t0 <= t1 else None


def centres(targets, group: str, size: float, point) -> list[tuple]:
    """The view centres that put the target at this size from the centre (inside: through this point of the green
    square), as segments (start, end) in specimen pixels: a square ring around a point target, lines along a line
    target (the target's line moved by the offset)."""
    if hasattr(targets, "rcs"):  # Point target: centre = target - its offset from the centre
        segments = []
        for r, c in targets.rcs:
            if group == "inside":
                segments.append(((r - point[0], c - point[1]),) * 2)
            else:
                k = size  # the four sides of the ring, offsets (row, col) from the centre
                for (ar, ac), (br, bc) in (((-k, -k), (-k, k)), ((-k, k), (k, k)), ((k, k), (k, -k)), ((k, -k), (-k, -k))):
                    segments.append(((r - ar, c - ac), (r - br, c - bc)))
        return segments

    segments = []  # Line target: centre = a point of the line - the offset of the line from the centre
    for (r0, c0), (r1, c1) in targets.segments:
        if group == "inside":
            shifts = [(-point[0], -point[1])]
        else:
            length = math.hypot(r1 - r0, c1 - c0)
            normal = (-(c1 - c0) / length, (r1 - r0) / length)
            distance = size * (abs(normal[0]) + abs(normal[1]))
            shifts = [(sign * distance * normal[0], sign * distance * normal[1]) for sign in (-1, 1)]
        for dr, dc in shifts:
            segments.append(((r0 + dr, c0 + dc), (r1 + dr, c1 + dc)))
    return segments


def group_views(
    env,
    seed: int,
    groups: list[str],
    zoom_indices: list[int],
    not_in_view_max_fov: float,
    max_tries: int = 10000,
):
    """Generate reproducible probe views with the target placed by group, one per group and magnification.

    Sizes are those of centred squares: a point at (row, col) from the view centre lies on the square of half-size
    max(|row|, |col|); a line at distance s with unit normal n touches the square of half-size |s| / (|n_row| + |n_col|).
      inside       the target at a uniform random point of the green square (the line through it)
      outside      in view, outside the green square: size uniform from the green square's half-width to the visible
                   region's (half the FOV less the visibility margin, the margin of the truth)
      not_in_view  size uniform from half the FOV plus the margin to not_in_view_max_fov FOVs
    Between the two, a band of the margin on each side of the FOV edge holds no target. For a drawn size, the centre is
    uniform along the centres that give it (centres()) and that the specimen allows (clamp's bounds); a size no centre
    gives is drawn again, so the sizes are uniform over those the specimen allows. A view whose truth is not the
    group's (rounding the centre to a pixel at the green square's edge) is drawn again. Each (seed, magnification,
    group) has its own random stream. Yields (zoom_index, centre, placement), placement: group, size_A (as drawn),
    attempts (draws)."""
    env.reset(seed)

    microscope = env.microscope
    targets = env.specimen.targets
    px = microscope.pixel_size_A

    for zoom_index in zoom_indices:
        fov = microscope.F_at(zoom_index)
        green = env.rules.center_frac * fov / 2
        margin = env.rules.visible_margin * microscope.cfg.zoom_levels[zoom_index]
        sizes = {
            "outside": (green, fov / 2 - margin),
            "not_in_view": (fov / 2 + margin, not_in_view_max_fov * fov),
        }
        low = fov // 2  # the centres microscope.clamp leaves in place
        box = (low, microscope.H - (fov - low), low, microscope.W - (fov - low))

        for group in groups:
            rng = np.random.default_rng([seed, zoom_index, GROUPS.index(group)])

            for attempt in range(1, max_tries + 1):
                point = rng.uniform(-green, green, size=2) if group == "inside" else None
                size = float(np.abs(point).max()) if group == "inside" else float(rng.uniform(*sizes[group]))

                spans = []  # (start, end, t0, t1, length) of the allowed part of each segment of centres
                for start, end in centres(targets, group, size, point):
                    part = clip(start, end, box)
                    if part is not None:
                        length = (part[1] - part[0]) * math.hypot(end[0] - start[0], end[1] - start[1])
                        spans.append((start, end, *part, length))
                if not spans:
                    continue

                lengths = np.array([s[-1] for s in spans])
                weights = lengths / lengths.sum() if lengths.sum() > 0 else np.full(len(spans), 1 / len(spans))
                start, end, t0, t1, _ = spans[rng.choice(len(spans), p=weights)]
                t = rng.uniform(t0, t1)
                center = (start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1]))
                rounded = (int(round(center[0])), int(round(center[1])))

                microscope.start(rounded, zoom_index, seed)
                info = env.info(action=None, blocked=False)
                if group_of(question_truth(env, info)) == group:
                    break
            else:
                raise RuntimeError(f"no {group} view in {max_tries} draws: seed {seed}, zoom index {zoom_index}")

            yield zoom_index, rounded, {"group": group, "size_A": size * px, "attempts": attempt}


def probe_view(
    agent: Agent,
    env,
    seed: int,
    zi: int,
    centre,
    target: str,
    placement: dict,
) -> dict:
    """Evaluate one model decision and return its record; placement (how the view was placed: offset_A and angle_deg,
    or group and size_A) goes into the record, with the target's offset from the centre for a group."""
    microscope = env.microscope
    microscope.start(centre, zi, seed)

    obs = microscope.observe()
    info = env.info(action=None, blocked=False)
    truth = question_truth(env, info)
    if "group" in placement:
        dr, dc = env.nearest()
        placement = {**placement, "target_rc_A": [dr * microscope.pixel_size_A, dc * microscope.pixel_size_A]}

    agent.reset(target)

    start_time = time.perf_counter()
    decision = agent.act(obs, env.operator_state())
    decision_s = time.perf_counter() - start_time

    answers = []

    for stage in decision["trace"]["stages"]:
        question = stage["id"]
        expected = truth.get(question)

        answer = {
            "id": question,
            "choice": stage["choice"],
            "probs": stage["probs"],
            "truth": expected,
            "correct": is_correct(expected, stage["choice"]),
        }

        for key, value in stage["parts"][0].items():
            if key not in ("view", "choice", "probs"):
                answer[key] = value

        answers.append(answer)

    return {
        "task": env.name,
        "reader": agent.reader.name,
        "model": agent.reader.model,
        "seed": seed,
        "zoom_index": zi,
        "fov_A": microscope.F * microscope.pixel_size_A,
        **placement,
        "distance_A": info["distance_A"],
        "truth": truth,
        "answers": answers,
        "action": decision["choice"],
        "branch": decision["trace"]["branch"],
        "decision_s": decision_s,
    }
