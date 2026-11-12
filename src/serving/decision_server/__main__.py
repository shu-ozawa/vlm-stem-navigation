"""Start the decision server in front of a running vLLM server:

    uv run python -m serving.decision_server --vllm-url http://localhost:8000 \\
        --served-model nvidia/Gemma-4-26B-A4B-NVFP4 --engine vllm \\
        --chat-template-kwargs '{"enable_thinking": false}'
    TYPESAFE_API_KEY=... uv run python -m serving.decision_server --vllm-url https://api.typesafe.ai \\
        --served-model jev-1.13.0 --engine typesafe
    OPENAI_API_KEY=... uv run python -m serving.decision_server --vllm-url https://api.openai.com \\
        --served-model gpt-6-luna --engine decisions
"""

import argparse
import json

import uvicorn

from serving.decision_server.api import create_app

ap = argparse.ArgumentParser()
ap.add_argument("--vllm-url", required=True)
ap.add_argument("--served-model", required=True)
ap.add_argument("--engine", required=True, choices=("vllm", "typesafe", "decisions"))
ap.add_argument("--chat-template-kwargs", default="{}", help="vllm: JSON, sent with every request")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8080)
a = ap.parse_args()

app = create_app(a.vllm_url, a.served_model, a.engine, json.loads(a.chat_template_kwargs))
uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
