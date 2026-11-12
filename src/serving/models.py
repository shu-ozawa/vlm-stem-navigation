"""How each model is served: models.yaml (serve, api) made into the arguments of vLLM and of the decision server.

A model is served under its checkpoint's name (vLLM's default), which every request names. Every model is served on
the same ports: vLLM on 8000 (STEMNAV_PORT, or a full URL with STEMNAV_BASE_URL_<NAME>) and the decision server on
8080 (STEMNAV_DECISION_PORT). A hosted model has the decision server alone, no vLLM. serve_model.sh asks for
the arguments:

    python -m serving.models list | serve-args <name> [<reader>] | decision-args <name>
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

TABLE = yaml.safe_load((Path(__file__).resolve().parents[1] / "models.yaml").read_text())
PORT = int(os.environ.get("STEMNAV_PORT", "8000"))                    # vLLM
DECISION_PORT = int(os.environ.get("STEMNAV_DECISION_PORT", "8080"))  # the decision server
GPU_MEM_UTIL = float(os.environ.get("STEMNAV_GPU_MEM_UTIL", "0.9"))   # the share of the GPU's memory vLLM may use


@dataclass(frozen=True)
class ServedModel:
    name: str
    checkpoint: str            # the Hugging Face checkpoint; a hosted model: the API's model name
    vllm: dict                 # "typed" / "generated" -> the model's own vLLM flags
    typed_template_kwargs: dict
    engine: str                # the decision server's: vllm, or typesafe or decisions (hosted)
    api_base: str              # hosted only
    api_key_env: str           # hosted only: the variable (.env) with the API key

    @property
    def hosted(self) -> bool:
        return bool(self.api_base)

    @property
    def vllm_served(self) -> bool:
        """Served by vLLM on a GPU, the decision server in front; else the decision server alone (hosted)."""
        return self.engine == "vllm"

    @property
    def kind(self) -> str:
        """vllm or hosted (serve_model.sh starts the servers by it)."""
        return "vllm" if self.vllm_served else "hosted"

    @property
    def base_url(self) -> str:
        return os.environ.get(f"STEMNAV_BASE_URL_{self.name.upper()}") or self.api_base or f"http://localhost:{PORT}"

    @property
    def decision_url(self) -> str:
        return os.environ.get("STEMNAV_DECISION_URL") or f"http://localhost:{DECISION_PORT}"


def _model(name: str, e: dict) -> ServedModel:
    serve, api = e.get("serve", {}), e.get("api", {})
    vllm = {r: tuple(serve[r]) for r in ("typed", "generated") if r in serve}
    for flags in vllm.values():
        both = {f for f in flags if f.startswith("--")} & set(TABLE["vllm_common"])
        if both:
            raise ValueError(f"models.yaml, {name}: {sorted(both)} is in vllm_common already")
    return ServedModel(name, e["checkpoint"], vllm, serve.get("typed_template_kwargs", {}),
                       serve.get("engine", "vllm"), api.get("base", ""), api.get("key_env", ""))


MODELS = {name: _model(name, e) for name, e in TABLE["models"].items()}


def get(name: str) -> ServedModel:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {list(MODELS)}")
    return MODELS[name]


def serve_flags(name: str, reader: str = "typed") -> tuple[str, ...]:
    """The `vllm serve` flags of model `name` for `reader` (the generated readers share one server)."""
    own = get(name).vllm["typed" if reader == "typed" else "generated"]
    return (*TABLE["vllm_common"], *own, "--gpu-memory-utilization", str(GPU_MEM_UTIL))


def serve_argv(name: str, reader: str = "typed", port: int | None = None) -> list[str]:
    """`vllm` argv (without the executable) that serves model `name` for `reader`."""
    m = get(name)
    if not m.vllm_served:
        raise ValueError(f"{name} is not served by vLLM (engine {m.engine})")
    return ["serve", m.checkpoint, "--port", str(port or PORT), *serve_flags(name, reader)]


def decision_argv(name: str) -> list[str]:
    """`python -m serving.decision_server` arguments for model `name` (its vLLM server must be running; for a hosted
    model, --vllm-url is the API)."""
    m = get(name)
    argv = ["--vllm-url", m.base_url, "--served-model", m.checkpoint, "--engine", m.engine,
            "--port", str(DECISION_PORT)]
    if not m.vllm_served:
        return argv
    return argv + ["--chat-template-kwargs", json.dumps(m.typed_template_kwargs)]


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "serve-args":
        print("\n".join(serve_argv(sys.argv[2], *sys.argv[3:4])))
    elif len(sys.argv) >= 3 and sys.argv[1] == "decision-args":
        print("\n".join(decision_argv(sys.argv[2])))
    elif len(sys.argv) >= 2 and sys.argv[1] == "list":
        for n, m in MODELS.items():
            print(f"{n:22s} {m.kind:7s} {m.checkpoint}")
    else:
        sys.exit("usage: python -m serving.models list | serve-args <name> [<reader>] | decision-args <name>")
