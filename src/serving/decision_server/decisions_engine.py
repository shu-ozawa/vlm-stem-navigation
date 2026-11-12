"""Reads on OpenAI's Decisions API (POST /v1/decisions, gpt-6-luna): typed answers with label probabilities from the
API itself, the System One request translated to its shapes.

    noul    -> predicate   {"probability": p}                              -> {"noul": p}
    choice  -> choice      {"choices": [{"value", "description"}]}         -> {"choice", "probabilities", "confidence"}
    score   -> score       {"levels": [{"label", "description"}]}          -> {"score", "legend", "probabilities", "confidence"}
A refusal answers None (policy.py: no answer). The state and the images go into one user message, the state as
text first (the API takes user messages only, so there is no system turn); the questions go unchanged otherwise. The API rounds its probabilities to two decimals.

A request that is rate-limited (429), fails on the API's side (5xx, or 499) or does not reach it is sent again
after a wait (the API's retry-after, else doubling from BACKOFF_S), up to MAX_TRIES times in all.
"""

from __future__ import annotations

import asyncio
import os

import httpx

from agent.prompts.layout import text_of

RETRY_STATUS = (429, 499)  # a rate limit; an internal error reported as 499 (seen on TypeSafe's API); and every 5xx
MAX_TRIES = 8              # requests per read: again after a rate limit or a server error
BACKOFF_S = 1.0            # the wait before the second request, doubled before each further one


class DecisionsEngine:
    def __init__(self, base_url: str, model: str):
        self.model = model
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(120.0, connect=5.0),
                                        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})

    async def decide(self, questions: dict, state, images: list[dict] | None = None) -> tuple[dict, dict, list[dict]]:
        """Answer a request (ar_engine.ArEngine.decide's arguments). Returns
        (answers keyed by question id, None for a refusal; usage: the API's input and output tokens and the number
        of requests sent; the request as sent, its images as "[image]")."""
        state_text = text_of(state)
        content = [{"type": "input_text", "text": state_text}] + [
            {"type": "input_image", "image_url": image["image_url"]["url"]} for image in images or []]
        body = {"model": self.model, "input": [{"role": "user", "content": content}],
                "questions": [to_question(k, q) for k, q in questions.items()]}

        d, tries = await self.post(body)
        if d["model"] != self.model:
            raise RuntimeError(f"answered by {d['model']!r}, not the requested {self.model!r}")
        usage = {"input_tokens": d["usage"]["input_tokens"], "output_tokens": d["usage"]["output_tokens"],
                 "tries": tries}
        answers = {a["name"]: from_answer(a) for a in d["answers"]}

        sent = "\n".join(f"{q['name']} ({q['type']}): {q['instructions']}" + "".join(
            f"\n  {o.get('value', o.get('label'))}: {o.get('description', '')}" for o in q.get("choices", q.get("levels", [])))
            for q in body["questions"])
        messages = [{"system": f"{self.model} (Decisions API: user messages only, no system turn)",
                     "user": "[image]\n" * len(images or []) + state_text + "\n\n" + sent, "answer": "■"}]
        return {k: answers[k] for k in questions}, usage, messages

    async def post(self, body: dict) -> tuple[dict, int]:
        """The API's response to `body`, and the number of requests it took."""
        for tries in range(1, MAX_TRIES + 1):
            wait = None
            try:
                r = await self.client.post("/v1/decisions", json=body)
            except httpx.TransportError as e:  # did not reach the API, or no response in time
                error = repr(e)
            else:
                if r.status_code not in RETRY_STATUS and r.status_code < 500:
                    if r.is_error:  # the request itself is wrong: sending it again does not help
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                    return r.json(), tries
                error, wait = f"HTTP {r.status_code}: {r.text[:300]}", r.headers.get("retry-after")
            if tries < MAX_TRIES:
                await asyncio.sleep(float(wait) if wait and wait.isdigit() else BACKOFF_S * 2 ** (tries - 1))
        raise RuntimeError(f"no answer in {MAX_TRIES} requests (last: {error})")


def to_question(key: str, q: dict) -> dict:
    """A System One question as a Decisions question named `key`."""
    out = {"name": key, "instructions": text_of(q.get("instructions"))}
    if q["type"] == "noul":
        criteria = q.get("criteria") or {}
        definitions = [f"{v}: {text_of(criteria.get(v))}" for v in ("true", "false") if criteria.get(v)]
        if definitions:
            out["instructions"] += "\n" + "\n".join(definitions)
        return {"type": "predicate", **out}
    if q["type"] == "choice":
        return {"type": "choice", **out, "choices": [{"value": label, "description": text_of(description)}
                                                     for label, description in q["criteria"].items()]}
    return {"type": "score", **out, "levels": [{"label": str(i), "description": text_of(level)}
                                               for i, level in enumerate(q["criteria"])]}


def from_answer(a: dict) -> dict | None:
    """A Decisions answer in the System One API's shape; None for a refusal."""
    if a["type"] == "predicate":
        return {"type": "noul", "noul": a["probability"]}
    if a["type"] == "choice":
        return {"type": "choice", "choice": a["choice"],
                "probabilities": {p["value"]: p["probability"] for p in a["probabilities"]},
                "confidence": a["confidence"]}
    if a["type"] == "score":
        return {"type": "score", "score": a["score"], "legend": {str(p["value"]): p["label"] for p in a["probabilities"]},
                "probabilities": {str(p["value"]): p["probability"] for p in a["probabilities"]},
                "confidence": a["confidence"]}
    return None  # "refusal"
