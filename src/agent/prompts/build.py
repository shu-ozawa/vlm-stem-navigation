"""What every reader sends for a question: the state, the target description, the question with its labels, and
(images.py) the image. The texts are in texts.yaml; this module only puts them together, and the request is laid
out as the decision server lays it out (layout.py), for every reader.

  state      the conventions
  target     the task's description (the user turn); none: the model is not told what the target is
  question   locate (QUESTIONS): where the target is relative to the green square, or none.

The model gets no log of the previous steps (the policy keeps it for its rules: log_entry, place_key).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agent.prompts.layout import questions_of, system_text, user_content

TEXTS_PATH = Path(__file__).with_name("texts.yaml")
TEXTS = yaml.safe_load(TEXTS_PATH.read_text())

QUESTIONS = {q: {"type": "choice", **spec} for q, spec in TEXTS["questions"].items()}  # System One questions
TARGETS = TEXTS["targets"]  # task -> description name -> text ("reference": the default)


def target_description(task: str, name: str = "reference") -> str:
    """The target description `name` of `task` (texts.yaml targets)."""
    if name not in TARGETS[task]:
        raise KeyError(f"no description {name!r} for {task}; choose from {list(TARGETS[task])}")
    return TARGETS[task][name]


def labels(question: str) -> tuple[str, ...]:
    """The labels of `question`, in order."""
    return tuple(QUESTIONS[question]["criteria"])


# ---------------------------------------------------------------- the log of the previous steps
def log_entry(state: dict, action: str, blocked: bool, decision: dict | None = None) -> dict:
    """One step of the log. `state`: env.operator_state() BEFORE the action (field of view in A, stage position in A
    relative to the start, zoom index); `decision`: policy.decide() output, or None for a manual action. Only the
    model's own answers are recorded (never the ground truth)."""
    seen = where = None
    if decision is not None:
        for st in decision["trace"]["stages"]:
            if st["id"] == "locate":
                seen = st["choice"] != "none"
                where = st["choice"] if seen else None

    return {"fov_A": state["fov_A"], "pos_A": list(state["pos_A"]), "zoom_index": state["zoom_index"],
            "n_zoom": state["n_zoom"], "seen": seen, "where": where, "action": action, "blocked": bool(blocked)}


def place_key(fov_A: float, pos_A) -> tuple:
    """Field of view and stage position, rounded to 1 A: two log entries with the same key are the same place."""
    return round(fov_A), round(pos_A[0]), round(pos_A[1])


# ---------------------------------------------------------------- the state
def state_text() -> str:
    """The state of a request: the conventions (the same at every step)."""
    return TEXTS["conventions"]


# ---------------------------------------------------------------- the request
def system_one_state(state: str, target: str) -> dict:
    """The System One state of a request: the state (system turn) and the target description (user turn)."""
    return {"state": state, "target": target}


def chat_messages(state: str, target: str, image_uri: str, question: str, choices: tuple[str, ...]) -> list[dict]:
    """A request of `question` with the labels `choices`, as chat messages, laid out as the decision server lays out
    a typed read (layout.py)."""
    q = QUESTIONS[question]
    qs = questions_of({question: {**q, "criteria": {l: d for l, d in q["criteria"].items() if l in choices}}})
    st = system_one_state(state, target)
    return [{"role": "system", "content": system_text(st, qs)},
            {"role": "user", "content": user_content(st, [{"type": "image_url", "image_url": {"url": image_uri}}], qs)}]
