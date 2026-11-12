
"""Navigation policy: convert model responses into microscope actions."""

from __future__ import annotations

from simulator.actions import compass, format_move
from agent.prompts.build import labels, place_key, state_text
from agent.prompts.image import draw_box
from agent.readers import Reader, Request


STUCK = "stuck"
LOST = "lost"
NO_ANSWER = "no_answer"
ENDS = (STUCK, LOST, NO_ANSWER)

NO_ANSWER_BRANCH = (
    "no usable answer to locate "
    "(max_tokens reached, or not the requested JSON)"
)


# Unavailable labels

def failed_labels(history: list[dict], state: dict) -> dict[str, str]:
    """Find previous zoom-ins that lost the target at the current position."""
    current_place = place_key(state["fov_A"], state["pos_A"])
    failed = {}

    for index, entry in enumerate(history):
        if (
            place_key(entry["fov_A"], entry["pos_A"]) == current_place
            and entry["action"] == "zoom_in"
            and entry["where"] == "inside"
            and index + 1 < len(history)
            and history[index + 1]["seen"] is False
        ):
            failed["inside"] = "zooming in from here lost the target"

    return failed


def unavailable_labels(history: list[dict], state: dict) -> dict[str, str]:
    """Labels excluded because of past failures or stage limits."""
    unavailable = failed_labels(history, state)

    for direction in state["at_limit"]:
        unavailable[direction] = "the stage is at its limit in this direction"

    return unavailable


# Action selection

def _mask(
    probs: dict[str, float],
    unavailable: dict[str, str],
) -> tuple[dict[str, float], str, list[str]]:
    """Remove unavailable labels and renormalize their remaining probabilities."""
    removed = [label for label in probs if label in unavailable]

    if not removed:
        return probs, max(probs, key=probs.get), []

    remaining = {
        label: prob
        for label, prob in probs.items()
        if label not in unavailable
    }

    total = sum(remaining.values())
    if not remaining or total <= 0:
        return remaining, STUCK, removed

    remaining = {
        label: prob / total
        for label, prob in remaining.items()
    }

    return remaining, max(remaining, key=remaining.get), removed


def _act_on_direction(
    probs: dict[str, float],
    unavailable: dict[str, str],
    zoom_index: int,
    stage: dict,
    trace: dict,
) -> tuple[str, str, list[str]]:
    """Select an action for a visible target."""
    probs, choice, removed = _mask(probs, unavailable)
    suffix = f" (ruled out here: {', '.join(removed)})" if removed else ""

    if removed:
        stage["removed"] = removed
        stage["probs_after_removal"] = probs

    if choice == STUCK:
        reason = f"stuck (ruled out here: {', '.join(removed)}; nothing else possible)"
        return STUCK, reason, removed

    if choice == "inside":
        if zoom_index == 0:
            return "found", "inside at highest magnification -> found" + suffix, removed
        return "zoom_in", "inside -> zoom in" + suffix, removed

    dx = probs.get("right", 0.0) - probs.get("left", 0.0)
    dy = probs.get("down", 0.0) - probs.get("up", 0.0)

    trace["vector"] = {
        "dx": dx,
        "dy": dy,
        "rule": "(P(right) - P(left), P(down) - P(up)), normalized",
    }

    if dx == 0 and dy == 0:
        return choice, "weighted vector is zero -> most probable direction" + suffix, removed

    action = format_move(dx, dy)
    reason = f"move along the probability-weighted direction ({compass(action)})"
    return action, reason + suffix, removed


# Main decision

def decide(
    obs_png: bytes,
    target: str,
    state: dict,
    history: list[dict],
    reader: Reader,
    inspect: bool = False,
) -> dict:
    """Ask where the target is and return the next microscope action."""
    unavailable = unavailable_labels(history, state)

    views = {"plain": obs_png}
    stages = []
    trace = {
        "zoom_index": state["zoom_index"],
        "n_zoom": state["n_zoom"],
        "stages": stages,
        "failed_here": unavailable,
    }

    def ask(exclude: tuple[str, ...] = ()) -> dict:
        """Ask the locate question and record the model response."""
        if "boxed" not in views:
            views["boxed"] = draw_box(obs_png)

        request = Request(
            question="locate",
            labels=tuple(
                label for label in labels("locate")
                if label not in exclude
            ),
            state=state_text(),
            target=target,
            image=views["boxed"],
        )

        answer = reader.answer(request, inspect)

        part = {
            "view": "boxed",
            **answer.record,
            "choice": answer.choice,
            "probs": answer.probs,
            "latency_s": answer.latency_s,
        }
        stage = {
            "id": "locate",
            "question": "locate",
            "choice": answer.choice,
            "probs": answer.probs,
            "parts": [part],
        }
        stages.append(stage)
        return stage

    # 1. Locate the target.
    stage = ask()

    if stage["choice"] is None:
        action, branch = NO_ANSWER, NO_ANSWER_BRANCH

    elif stage["choice"] == "none":
        if state["zoom_index"] < state["n_zoom"] - 1:
            action, branch = "zoom_out", "not visible -> zoom out"
        else:
            action, branch = LOST, "not visible at the lowest magnification -> lost"

    else:
        # 2. Remove 'none' and condition on the target being visible.
        probs = {
            label: prob
            for label, prob in stage["probs"].items()
            if label != "none"
        }
        total = sum(probs.values())
        probs = {label: prob / total for label, prob in probs.items()}

        # 3. Select an action.
        action, branch, removed = _act_on_direction(
            probs, unavailable, state["zoom_index"], stage, trace
        )

        # 4. Retry once if unavailable labels left no usable action.
        if action == STUCK and reader.supports_label_exclusion and removed:
            available = [
                label for label in stage["probs"]
                if label not in unavailable and label != "none"
            ]

            if available:
                stage = ask((*removed, "none"))

                if stage["choice"] is None:
                    action, branch = NO_ANSWER, NO_ANSWER_BRANCH
                else:
                    action, branch, _ = _act_on_direction(
                        stage["probs"],
                        unavailable,
                        state["zoom_index"],
                        stage,
                        trace,
                    )
                    branch = "answer ruled out, asked again without it -> " + branch

    # 5. Return the decision and trace.
    trace["branch"] = branch
    trace["action"] = action

    result = {
        "choice": action,
        "label": " > ".join(f"{stage['id']}:{stage['choice']}" for stage in stages),
        "latency_s": sum(stage["parts"][0]["latency_s"] for stage in stages),
        "reader": reader.name,
        "model": reader.model,
        "trace": trace,
    }

    if inspect:
        result["images"] = dict(views)

    return result
