"""Browser viewer: watch a VLM navigate a task step by step, with its full decision trace.

Serves the page (viewer.html, viewer.css, viewer.js) and a small JSON API that drives a task
environment and asks the model (through the same policy as the experiments) for each move:

  GET  /api/meta    the tasks and the green square (once, when the page opens)
  GET  /api/model   which model is being served (vLLM) and whether its decision server is up
  POST /api/reset   start an episode with the model being served
  POST /api/step    one decision of the model and one step of the environment

The model only ever sees the observation images; the overview, trajectory, ground-truth markers and the
per-question truth checks are for the human viewer only. Every episode is logged to
runs/<timestamp>_<n>_<task>_seed<seed>/ (config.json, steps.jsonl with the decision trace, and the
exact images sent to the model at each step).

Usage: uv run python -m viewer.app [--port 8090] [--host 0.0.0.0]
The viewer has no authentication: with the default host it is reachable from any machine that can
reach this one (use --host 127.0.0.1 and an SSH tunnel on untrusted networks).
Needs a model served with src/serving/serve_model.sh <model> (scripts/viewer.sh <model> starts both).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

from agent import models, policy
from agent.agent import Agent
from agent.prompts.build import target_description
from agent.prompts.image import BOX_FRAC
from agent.readers import make_reader
from simulator.actions import compass
from simulator.tasks import TASKS, make
from simulator.truth import is_correct, question_truth

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RUNS = REPO_ROOT / "runs"
OVERVIEW_PX = 600      # longest side of the overview image
TRACE_THUMB_PX = 176   # side of the per-question image thumbnails
DECISION_KEYS = ("label", "latency_s", "trace", "reader", "model")  # what steps.jsonl keeps of a decision

app = FastAPI()
_lock = threading.Lock()
_state: dict = {"env": None}


class ResetReq(BaseModel):
    task: str = "dopant_zoom"
    seed: int = 0
    start_mode: str = "hidden"
    dose: float | None = None      # None = noiseless
    reader: str = "typed"          # readers.READERS


# ---------------------------------------------------------------- images
def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _b64png(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _overview_png(env) -> bytes:
    m = env.microscope
    img = Image.fromarray(m.to_uint8(m.image), mode="L")
    s = OVERVIEW_PX / max(m.H, m.W)
    return _png(img.resize((round(m.W * s), round(m.H * s)), Image.BILINEAR))


def _thumb(png: bytes) -> str:
    img = Image.open(io.BytesIO(png))
    img.thumbnail((TRACE_THUMB_PX, TRACE_THUMB_PX), Image.LANCZOS)
    return _b64png(_png(img))


# ---------------------------------------------------------------- the model being served
def served_model() -> dict:
    """{"model": the models.yaml name served by the decision server, else by vLLM (or None), "decision_server": its
    decision server is up, "message"}. All local models share one port; the first one served there is used. A
    hosted API model has only the decision server."""
    try:
        served = requests.get(f"{models.get('gemma4').decision_url}/health", timeout=3).json().get("model")
    except requests.RequestException:
        served = None
    if served:
        name = next(n for n, m in models.MODELS.items() if m.checkpoint == served)
        return {"model": name, "decision_server": True, "message": f"{name} ready"}

    try:
        served = [m["id"] for m in requests.get(f"{models.get('gemma4').base_url}/v1/models", timeout=3).json()["data"]]
    except requests.RequestException:
        return {"model": None, "decision_server": False,
                "message": "No model server (src/serving/serve_model.sh <model>)"}

    name = next(n for n, m in models.MODELS.items() if m.checkpoint in served)
    try:
        decision = requests.get(f"{models.get(name).decision_url}/health", timeout=3).ok
    except requests.RequestException:
        decision = False
    return {"model": name, "decision_server": decision,
            "message": f"{name} ready" if decision else f"{name}: no decision server (generated readers only)"}


def _public_state(env, obs: bytes) -> dict:
    return {
        "obs": _b64png(obs),  # exactly what the model sees for the plain-image questions
        "info": env.history[-1],
        "history": [{k: h[k] for k in ("step", "fov_size", "center_row", "center_col")} for h in env.history],
        "operator": env.operator_state(),  # what the model is told: zoom index, field of view [A], stage limits
        "max_steps": env.rules.max_steps,
        "done": env.done,
    }


# ---------------------------------------------------------------- API
@app.get("/")
def index():
    return FileResponse(HERE / "viewer.html")


@app.get("/viewer.{ext}")
def static(ext: str):
    return FileResponse(HERE / f"viewer.{ext}")


@app.get("/api/meta")
def meta():
    return {"tasks": list(TASKS), "default_task": "dopant_zoom",
            "box_frac": BOX_FRAC}  # green square shown to the model: this fraction of the image at every zoom


@app.get("/api/model")
def model():
    return served_model()


@app.post("/api/reset")
def reset(req: ResetReq):
    with _lock:
        name = served_model()["model"]
        env = make(req.task, start_mode=req.start_mode, dose=req.dose)
        try:
            obs, _ = env.reset(req.seed)
        except RuntimeError as e:  # e.g. no start position found
            _state["env"] = None
            return JSONResponse({"error": f"{e}: choose another start for {req.task}"}, status_code=400)

        stamp, n = time.strftime("%Y%m%d_%H%M%S"), 0
        while (run_dir := RUNS / f"{stamp}_{n:02d}_{req.task}_seed{req.seed}").exists():
            n += 1
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps({**req.model_dump(), "model": name}, indent=1))
        (run_dir / "step_000_obs.png").write_bytes(obs)

        agent = Agent(make_reader(req.reader, name))
        agent.reset(target_description(req.task))
        _state.update(env=env, run_dir=run_dir, obs=obs, req=req, model=name, agent=agent)
        return {**_public_state(env, obs), "run_dir": str(run_dir.relative_to(REPO_ROOT)), "model": name,
                "overview": _b64png(_overview_png(env)), "H": env.microscope.H, "W": env.microscope.W,
                "markers": env.truth_markers()}  # viewer-only ground truth


@app.post("/api/step")
def step():
    with _lock:
        if _state["env"] is None:  # no episode yet, or the last reset failed
            return JSONResponse({"error": "no episode: reset first"}, status_code=409)
        env, agent, run_dir = _state["env"], _state["agent"], _state["run_dir"]
        truth = question_truth(env, env.history[-1])  # of the view being judged
        state = env.operator_state()  # before the action, for the log
        try:
            decision = agent.act(_state["obs"], state, inspect=True)
        except Exception as e:  # no fallback: show the error and do not move
            return JSONResponse({"error": f"model error: {e}"}, status_code=502)

        # the truth of each question (viewer only), the thumbnails, and the exact images sent
        for st in decision["trace"]["stages"]:
            st["truth"] = truth.get(st["id"])
            st["correct"] = is_correct(st["truth"], st["choice"])
        images = decision.pop("images")
        decision["trace_images"] = {k: _thumb(v) for k, v in images.items()}
        for k, v in images.items():
            (run_dir / f"step_{len(env.history):03d}_input_{k}.png").write_bytes(v)

        action = decision["choice"]
        rec = {"step": len(env.history), "action": action, "decision": {k: decision[k] for k in DECISION_KEYS}}
        if action in policy.ENDS:  # stuck or no answer: the controller cannot act (a failure in run.py)
            with open(run_dir / "steps.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            return JSONResponse({"error": f"{action}: {decision['trace']['branch']}"}, status_code=409)

        obs, _, _, info = env.step(action)
        agent.record(state, decision, info["blocked"])
        _state["obs"] = obs
        (run_dir / f"step_{info['step']:03d}_obs.png").write_bytes(obs)
        with open(run_dir / "steps.jsonl", "a") as f:
            f.write(json.dumps({**rec, "info": info}) + "\n")

        return {**_public_state(env, obs), "action": action, "action_name": compass(action), "decision": decision}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0",
                    help="listen address (default: all interfaces; use 127.0.0.1 for local-only access)")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
