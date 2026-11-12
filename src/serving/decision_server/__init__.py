"""The decision server: typed answers with label probabilities (TypeSafe's System One API) from a model served by
vLLM, or from a hosted API (typesafe_engine.py: TypeSafe's Jev, which answers the same API itself;
decisions_engine.py: OpenAI's Decisions API). See api.py for the request and response shapes, and vllm_engine.py for
how a vLLM model is read. api.py and vllm_engine.py are adapted from openjev and stay under Apache-2.0 (LICENSE in
this directory); the other files here are MIT, as the rest of the repository. Of this repository it uses only the
layout of a request (agent/prompts/layout.py), which the generated readers share; the agent calls it over HTTP.
"""
