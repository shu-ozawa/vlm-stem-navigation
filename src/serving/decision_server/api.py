# SPDX-License-Identifier: Apache-2.0
"""The HTTP API of the decision server (POST /v1/systemone, GET /health) and its request models.

Adapted from openjev (github.com/razorback16/openjev, commit a0ddd7d, openjev/api.py; Apache-2.0, see the LICENSE in
this directory): the request models of POST /v1/systemone. Changed: only these two routes; no authentication, body
limits, model routing, request ids or image validation; the engine (vllm, typesafe, decisions) and the server
behind it are command-line arguments (__main__.py).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field
from serving.decision_server.vllm_engine import VllmEngine
from serving.decision_server.decisions_engine import DecisionsEngine
from serving.decision_server.typesafe_engine import TypeSafeEngine

JSONContent = Union[str, dict[str, Any], list[Any]]
Described = Union[str, dict[str, Any], list[Any], None]


class NoulCriteria(BaseModel):
    true: Described = None
    false: Described = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Described = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Described = None
    criteria: dict[str, Described]


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Described = None
    criteria: list[JSONContent] = Field(min_length=1)


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str
    questions: dict[str, Question] = Field(min_length=1)
    images: list[str] | None = None  # data:image/...;base64,... URLs
    return_messages: bool = False


def create_app(vllm_url: str, served_model: str, engine: str, chat_template_kwargs: dict | None = None) -> FastAPI:
    """The server for the model `served_model` of the vLLM server at `vllm_url`.
    engine "vllm" (vllm_engine.py; `chat_template_kwargs` go with every request), "typesafe"
    (typesafe_engine.py; `vllm_url` is https://api.typesafe.ai, and the key is TYPESAFE_API_KEY) or "decisions"
    (decisions_engine.py; `vllm_url` is https://api.openai.com, and the key is OPENAI_API_KEY)."""

    @asynccontextmanager
    async def lifespan(app):
        if engine == "typesafe":
            app.state.engine = TypeSafeEngine(vllm_url, served_model)

        elif engine == "decisions":
            app.state.engine = DecisionsEngine(vllm_url, served_model)

        else:
            app.state.engine = VllmEngine(vllm_url, served_model, chat_template_kwargs or {})

        yield

        await app.state.engine.client.aclose()

    app = FastAPI(title="decision server", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": served_model}

    @app.post("/v1/systemone")
    async def systemone(req: SystemOneRequest):
        questions = {k: q.model_dump() for k, q in req.questions.items()}
        images = [{"type": "image_url", "image_url": {"url": u}} for u in req.images or []]
        answers, usage, messages = await app.state.engine.decide(questions, req.state, images or None)
        out = {"model": served_model, "answers": answers, "usage": usage}
        if req.return_messages:
            out["messages"] = messages
        return out

    return app
