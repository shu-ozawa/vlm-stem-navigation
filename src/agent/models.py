"""Model configuration and server endpoints for agent requests."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


CONFIG = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "models.yaml").read_text()
)

PORT = int(os.environ.get("STEMNAV_PORT", "8000"))
DECISION_PORT = int(os.environ.get("STEMNAV_DECISION_PORT", "8080"))

MAX_TOKENS = CONFIG["max_tokens"]
GENERATED_READERS = ("generated", "generated_thinking")


@dataclass(frozen=True)
class AskedModel:
    name: str
    checkpoint: str
    api_base: str
    ask: dict | None

    @property
    def hosted(self) -> bool:
        return bool(self.api_base)

    @property
    def readers(self) -> tuple[str, ...]:
        if self.ask:
            return ("typed", *GENERATED_READERS)
        return ("typed",)

    @property
    def base_url(self) -> str:
        override = os.environ.get(f"STEMNAV_BASE_URL_{self.name.upper()}")
        return override or self.api_base or f"http://localhost:{PORT}"

    @property
    def decision_url(self) -> str:
        override = os.environ.get("STEMNAV_DECISION_URL")
        return override or f"http://localhost:{DECISION_PORT}"


MODELS = {}

for name, config in CONFIG["models"].items():
    MODELS[name] = AskedModel(
        name=name,
        checkpoint=config["checkpoint"],
        api_base=config.get("api", {}).get("base", ""),
        ask=config.get("ask"),
    )


def get(name: str) -> AskedModel:
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}. Available: {list(MODELS)}")
    return MODELS[name]