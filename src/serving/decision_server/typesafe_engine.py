"""Reads on TypeSafe's Jev (hosted, System One API at api.typesafe.ai): the request is passed on as it is.

Jev answers the System One API itself, label probabilities included, so nothing is read here: the questions go
unchanged, and the images go into the state, the place the API gives an image that every question is about
(docs.typesafe.ai/images): each image is an asset under a fresh UUID reference, and the state becomes
{"image": <reference>, "context": <the state>} ("images": [...] for several). The API lays out the prompt
itself. The answer must come from the
pinned release that was asked for (models.yaml), not from whatever an alias points to.

A request that is rate-limited (429), overloaded (529), fails on the API's side (5xx, or 499 with error_type
"internal_error") or does not reach it is
sent again after a wait (the API's retry-after, else doubling from BACKOFF_S), up to MAX_TRIES times in all.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid

import httpx

IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}  # images are off by default (a preview feature)
FORMATS = ("png", "jpg", "jpeg")  # the image formats the API takes
RETRY_STATUS = (429, 499)         # a rate limit; the API's own "internal_error" (seen as 499); and every 5xx
MAX_TRIES = 8                     # requests per read: again after a rate limit, an overload or a server error
BACKOFF_S = 1.0                   # the wait before the second request, doubled before each further one


class TypeSafeEngine:
    def __init__(self, base_url: str, model: str):
        self.model = model
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(120.0, connect=5.0),
                                        headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"})

    async def decide(self, questions: dict, state, images: list[dict] | None = None) -> tuple[dict, dict, list[dict]]:
        """Answer a request (vllm_engine.VllmEngine.decide's arguments). Returns
        (the API's answers, usage: its input and output tokens and the number of requests sent, the request as
        sent, its images as "[image]")."""
        refs = [str(uuid.uuid4()) for _ in images or []]
        body = {"model": self.model, "state": with_images(state, refs),
                "questions": {k: {f: v for f, v in q.items() if v is not None} for k, q in questions.items()}}
        if refs:
            body["assets"] = {ref: image_asset(image["image_url"]["url"]) for ref, image in zip(refs, images)}

        d, tries = await self.post(body, IMAGES_HEADER if refs else {})
        if d["model"] != self.model:
            raise RuntimeError(f"answered by {d['model']!r}, not the requested {self.model!r}")
        usage = {"input_tokens": d["usage"]["input_tokens"], "output_tokens": d["usage"]["output_tokens"],
                 "tries": tries}

        sent = json.dumps({"state": body["state"], "questions": body["questions"]}, ensure_ascii=False, indent=1)
        for ref in refs:
            sent = sent.replace(ref, "[image]")
        messages = [{"system": f"{self.model} (System One API: the API lays out the prompt)", "user": sent,
                     "answer": "■"}]
        return {k: d["answers"][k] for k in questions}, usage, messages

    async def post(self, body: dict, headers: dict) -> tuple[dict, int]:
        """The API's response to `body`, and the number of requests it took."""
        for tries in range(1, MAX_TRIES + 1):
            wait = None
            try:
                r = await self.client.post("/v1/systemone", json=body, headers=headers)
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


def with_images(state, refs: list[str]):
    """The state with the image references in it (the state unchanged if there are none)."""
    if not refs:
        return state
    return {**({"image": refs[0]} if len(refs) == 1 else {"images": refs}), "context": state}


def image_asset(url: str) -> dict:
    """A data:image/<format>;base64,... URL as an image asset of the System One API."""
    head, data = url.split(",", 1)
    fmt = head.removeprefix("data:image/").removesuffix(";base64")
    if fmt not in FORMATS:
        raise ValueError(f"the API takes {', '.join(FORMATS)} images, not {head!r}")
    return {"type": "image", "format": fmt, "data_base64": data}
