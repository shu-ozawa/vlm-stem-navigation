"""Read label probabilities from the Decision Server."""

from __future__ import annotations

import time

from agent import models
from agent.prompts.build import QUESTIONS, system_one_state
from agent.prompts.image import png_data_uri
from agent.readers.base import SESSION, Answer, Reader, Request


class TypedReader(Reader):
    name = "typed"

    def __init__(self, model: str):
        super().__init__(model)
        self.spec = models.get(model)

    def answer(self, request: Request, inspect: bool = False) -> Answer:
        expected_labels = tuple(QUESTIONS[request.question]["criteria"])
        if request.labels != expected_labels:
            raise ValueError("Typed reader requires all labels")

        start_time = time.perf_counter()

        payload = {
            "model": self.spec.checkpoint,
            "state": system_one_state(request.state, request.target),
            "images": [png_data_uri(request.image)],
            "questions": {request.question: QUESTIONS[request.question]},
            "return_messages": inspect,
        }

        response = SESSION.post(
            f"{self.spec.decision_url}/v1/systemone",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        latency_s = time.perf_counter() - start_time

        result = response.json()
        answer = result["answers"][request.question]

        record = {"usage": result["usage"]}
        if inspect:
            record["messages"] = result.get("messages")

        if answer is None:
            return Answer(None, {}, latency_s, record)

        record["confidence"] = answer.get("confidence")

        return Answer(
            choice=answer["choice"],
            probs=answer["probabilities"],
            latency_s=latency_s,
            record=record,
        )