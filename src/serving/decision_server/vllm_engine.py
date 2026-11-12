# SPDX-License-Identifier: Apache-2.0
"""System One choice decisions using next-token logprobs from vLLM.

Label probability calculation adapted from openjev
(github.com/razorback16/openjev, commit a0ddd7d, Apache-2.0).
See the accompanying LICENSE.
"""

import math

import httpx

from agent.prompts.layout import questions_of, shown, system_text, user_content


TOPK = 20


def make_answer(question: dict, logprobs: dict[int, float], token_ids: list[int]) -> dict:
    """Convert next-token logprobs into a System One choice answer."""

    # Extract label logprobs and normalize them.
    floor = min(logprobs.values()) - 5.0
    scores = [logprobs.get(token_id, floor) for token_id in token_ids]

    max_score = max(scores)
    weights = [math.exp(score - max_score) for score in scores]
    probabilities = [weight / sum(weights) for weight in weights]

    # Entropy-based confidence.
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    confidence = 1.0 - entropy / math.log(len(probabilities))
    confidence = max(0.0, min(1.0, confidence))

    # Most probable label.
    labels = question["labels"]
    best_index = probabilities.index(max(probabilities))

    return {
        "type": "choice",
        "choice": labels[best_index],
        "probabilities": dict(zip(labels, probabilities)),
        "confidence": confidence,
    }


class VllmEngine:
    def __init__(self, base_url: str, model: str, chat_template_kwargs: dict):
        self.model = model
        self.chat_template_kwargs = chat_template_kwargs or None
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(120.0, connect=5.0),
        )
        self.label_cache = {}

    async def label_ids(self, prefix: str, labels: list[str]) -> list[int]:
        """Get and cache the single-token ID of each label."""

        key = (prefix, tuple(labels))
        if key in self.label_cache:
            return self.label_cache[key]

        async def tokenize(text: str) -> list[int]:
            response = await self.client.post("/tokenize", json={
                "model": self.model,
                "prompt": text,
                "add_special_tokens": False,
            })
            response.raise_for_status()
            return response.json()["tokens"]

        base = await tokenize(prefix)
        ids = []

        for label in labels:
            tokens = await tokenize(f"{prefix} {label}")

            if len(tokens) != len(base) + 1 or tokens[:-1] != base:
                raise ValueError(
                    f"Label {label!r} is not one token after {prefix!r}"
                )

            ids.append(tokens[-1])

        self.label_cache[key] = ids
        return ids

    async def next_token(
        self,
        system: str,
        user,
        prefix: str,
        token_ids: list[int],
    ) -> tuple[dict[int, float], int]:
        """Read next-token logprobs and prompt-token usage from vLLM."""

        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": prefix},
            ],
            "add_generation_prompt": False,
            "continue_final_message": True,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": TOPK,
            "logprob_token_ids": token_ids,
            "return_tokens_as_token_ids": True,
            "chat_template_kwargs": self.chat_template_kwargs,
        }

        response = await self.client.post("/v1/chat/completions", json=request)
        response.raise_for_status()
        data = response.json()

        top_logprobs = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        logprobs = {
            int(item["token"].split(":")[1]): item["logprob"]
            for item in top_logprobs
        }

        return logprobs, data["usage"]["prompt_tokens"]

    async def decide(
        self,
        questions: dict,
        state,
        images: list[dict] | None = None,
    ) -> tuple[dict, dict, list[dict]]:
        """Answer each question using next-token label probabilities."""

        qs = questions_of(questions)
        system = system_text(state, qs)
        user = user_content(state, images, qs)

        answers = {}
        previous = ""
        input_tokens = 0

        for question in qs:
            prefix = f"{previous}{question['id']}:"

            # Get candidate token IDs.
            token_ids = await self.label_ids(prefix, question["labels"])

            # Read token logprobs.
            logprobs, tokens = await self.next_token(
                system, user, prefix, token_ids
            )

            # Build the System One answer.
            answer = make_answer(question, logprobs, token_ids)
            answers[question["id"]] = answer

            input_tokens += tokens
            previous = f"{prefix} {answer['choice']}\n"

        messages = [{
            "system": system,
            "user": shown(user),
            "answer": "\n".join(f"{q['id']}: ■" for q in qs),
        }]

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": 0,
        }

        return answers, usage, messages