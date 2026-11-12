# SPDX-License-Identifier: Apache-2.0
"""How a request is laid out for the model, for every reader: the decision server lays out the typed reads with it
(serving/decision_server/vllm_engine.py), the generated readers through build.py.

  system     how to answer, "State:" and the state
  user       "Target:" and the target description, the image(s), then the questions with their labels
  assistant  typed reads: prefilled up to "<id>:" and the next token is read

The System One `state` is either a string (the state) or {"state": ..., "target": ...}. Choice questions only.

Adapted from openjev (github.com/razorback16/openjev, commit a0ddd7d, openjev/engine.py; Apache-2.0, see
serving/decision_server/LICENSE): its system text with the questions and their labels.
"""

from __future__ import annotations

import json

SYSTEM = ("Answer a fixed set of questions about the state the user provides. Each question lists its allowed answers; "
          "reply with exactly one label per question.")


def text_of(value) -> str:
    """Descriptions and instructions may be strings, objects or arrays."""
    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def questions_of(questions: dict) -> list[dict]:
    """System One choice questions -> [{"id", "instructions", "choices": [(label, description)], "labels"}]."""
    qs = []
    for qid, q in questions.items():
        if q["type"] != "choice":
            raise ValueError(f"question {qid}: only choice questions are read here, not {q['type']!r}")
        choices = [(name, text_of(desc)) for name, desc in q["criteria"].items()]
        qs.append({"id": qid, "instructions": text_of(q.get("instructions")), "choices": choices,
                   "labels": [name for name, _ in choices]})
    return qs


def split_state(state) -> tuple[str, str]:
    """(the state, the target description) of a System One state."""
    if isinstance(state, dict) and set(state) <= {"state", "target"}:
        return text_of(state.get("state")), text_of(state.get("target"))
    return text_of(state), ""


def question_text(qs: list[dict]) -> str:
    out = []
    for q in qs:
        labels = "\n".join(f"{label}: {desc}" if desc else label for label, desc in q["choices"])
        out.append(f"Question {q['id']}: {q['instructions'] or 'Answer about the state.'}\n\nLabels:\n\n{labels}")
    return "\n\n".join(out)


def system_text(state, qs: list[dict]) -> str:
    text, _ = split_state(state)
    return SYSTEM + (f"\n\nState:\n\n{text}" if text else "")


def user_content(state, images: list[dict] | None, qs: list[dict]):
    """The user turn: "Target:" and the description (if any), the images (OpenAI-style image parts), the questions."""
    _, target = split_state(state)
    before = [{"type": "text", "text": f"Target:\n\n{target}"}] if target else []
    after = [{"type": "text", "text": question_text(qs)}]
    if not images:
        return "\n\n".join(p["text"] for p in before + after)
    return before + list(images) + after


def shown(content) -> str:
    """The user turn as text, its images as "[image]" (for the record)."""
    if isinstance(content, str):
        return content
    return "\n\n".join(p["text"] if p["type"] == "text" else "[image]" for p in content)
