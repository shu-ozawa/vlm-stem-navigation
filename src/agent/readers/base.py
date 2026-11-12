
"""Common input, output, and interface for model readers."""

from __future__ import annotations

from dataclasses import dataclass, field

import requests


SESSION = requests.Session()


@dataclass(frozen=True)
class Request:
    question: str
    labels: tuple[str, ...]
    state: str
    target: str
    image: bytes


@dataclass
class Answer:
    choice: str | None
    probs: dict[str, float]
    latency_s: float
    record: dict = field(default_factory=dict)


class Reader:
    name = ""
    supports_label_exclusion = False

    def __init__(self, model: str):
        self.model = model

    def answer(self, request: Request, inspect: bool = False) -> Answer:
        raise NotImplementedError
