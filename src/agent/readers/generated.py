"""Read generated JSON labels from the model's chat completions API."""

from __future__ import annotations

import json
import math
import re
import time

from agent import models
from agent.prompts.build import chat_messages
from agent.prompts.image import png_data_uri
from agent.prompts.layout import shown
from agent.readers.base import SESSION, Answer, Reader, Request


TOP_LOGPROBS = 20


class GeneratedReader(Reader):
    supports_label_exclusion = True

    def __init__(self, model: str, thinking: bool = False):
        super().__init__(model)
        self.spec = models.get(model)
        self.thinking = thinking
        self.name = "generated_thinking" if thinking else "generated"

    def answer(self, request: Request, inspect: bool = False) -> Answer:
        question = request.question
        labels = request.labels
        settings = self.spec.ask
        mode = settings["with_thinking" if self.thinking else "without_thinking"]

        messages = chat_messages(
            request.state,
            request.target,
            png_data_uri(request.image),
            question,
            labels,
        )

        record = {}
        if inspect:
            record["messages"] = [{
                "system": messages[0]["content"],
                "user": shown(messages[1]["content"]),
                "answer": '{"%s": ■}' % question,
            }]

        if mode.get("instruction"):
            messages[0] = {
                **messages[0],
                "content": system_content(messages[0]["content"], mode),
            }

        schema = {
            "type": "object",
            "properties": {
                question: {"type": "string", "enum": list(labels)}
            },
            "required": [question],
            "additionalProperties": False,
        }

        payload = {
            "model": self.spec.checkpoint,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": schema,
                },
            },
            "max_tokens": models.MAX_TOKENS,
            **settings["sampling"],
        }

        if "template_kwargs" in mode:
            payload["chat_template_kwargs"] = mode["template_kwargs"]

        payload["logprobs"] = True
        payload["top_logprobs"] = TOP_LOGPROBS

        start_time = time.perf_counter()
        response = SESSION.post(
            f"{self.spec.base_url}/v1/chat/completions",
            json=payload,
            timeout=3600,
        )
        response.raise_for_status()
        latency_s = time.perf_counter() - start_time

        result = response.json()
        completion = result["choices"][0]
        message = completion["message"]

        record["reasoning"] = (
            message.get("reasoning") or message.get("reasoning_content")
        )
        record["usage"] = result.get("usage", {})

        if message["content"] is None:
            return Answer(None, {}, latency_s, record)

        try:
            choice = json.loads(message["content"])[question]
        except (ValueError, KeyError, TypeError):
            choice = None

        if choice not in labels:
            record["usage"] = {
                **record["usage"],
                "unusable_answer": message["content"],
            }
            return Answer(None, {}, latency_s, record)

        record["confidence"] = 1.0

        tokens = (completion.get("logprobs") or {}).get("content")
        if tokens:
            record.update(answer_token_probs(tokens, question, labels))

        probs = {label: float(label == choice) for label in labels}
        return Answer(choice, probs, latency_s, record)


def system_content(system: str, mode: dict) -> str | list[dict]:
    """Append the model-specific thinking instruction."""
    text = system + "\n\n" + mode["instruction"]

    if not (mode.get("instruction_as_chunks") and "[THINK]" in text):
        return text

    before, rest = text.split("[THINK]", 1)
    thinking, after = rest.split("[/THINK]", 1)

    return [
        {"type": "text", "text": before},
        {"type": "thinking", "thinking": thinking, "closed": True},
        {"type": "text", "text": after},
    ]


def answer_token_probs(
    tokens: list[dict],
    question: str,
    labels: tuple[str, ...],
) -> dict:
    """Estimate label probabilities from the generated answer token."""
    token_starts = []
    text = ""

    for token in tokens:
        token_starts.append(len(text))
        text += token["token"]

    match = None
    pattern = r'"%s"\s*:\s*"' % re.escape(question)
    for candidate in re.finditer(pattern, text):
        match = candidate

    if match is None:
        return {}

    label_start = match.end()
    token_index = max(
        index
        for index, start in enumerate(token_starts)
        if start <= label_start
    )
    offset = label_start - token_starts[token_index]
    top_candidates = tokens[token_index]["top_logprobs"]

    masses = dict.fromkeys(labels, 0.0)

    for candidate in top_candidates:
        if candidate["token"][:offset] != tokens[token_index]["token"][:offset]:
            continue

        remaining = candidate["token"][offset:]
        matching_labels = [
            label for label in labels
            if remaining and (
                label.startswith(remaining) or remaining.startswith(label)
            )
        ]

        if len(matching_labels) == 1:
            masses[matching_labels[0]] += math.exp(candidate["logprob"])

    floor = math.exp(min(c["logprob"] for c in top_candidates) - 5.0)
    masses = {label: mass or floor for label, mass in masses.items()}

    total = sum(masses.values())

    return {
        "token_probs": {
            label: mass / total for label, mass in masses.items()
        },
        "token_top": [
            [candidate["token"], candidate["logprob"]]
            for candidate in top_candidates
        ],
    }