"""Check that a model's typed reads work through the decision server, before it enters a benchmark.

Run against the model's running servers (src/serving/serve_model.sh <name>):
    uv run python scripts/check_model.py <name> [--out report.json]

Checks (each prints PASS / FAIL; the exit code is 1 if any check fails):
  questions   every question of prompts/texts.yaml is answered (the decision server refuses a question whose
              labels are not single tokens)
  image       the image reaches the model: the input grows by > 50 tokens with the image
  sanity      locate on start views of dopant_zoom: more answers other than none on visible than on hidden starts
Also reports how much identical requests differ (vLLM is not deterministic with images) and the latency
of one request.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import requests

from agent import models
from agent.prompts.build import QUESTIONS, state_text, system_one_state, target_description
from agent.prompts.image import draw_box, png_data_uri
from serving import models as serving_models
from simulator.tasks import make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=10, help="start views per start mode for the sanity check")
    a = ap.parse_args()

    spec = models.get(a.model)
    env = make("dopant_zoom", start_mode="visible")
    png, _ = env.reset(1000)
    plain, boxed = [png_data_uri(png)], [png_data_uri(draw_box(png))]
    rep, ok = {"model": a.model, "hf_id": serving_models.get(a.model).checkpoint}, True

    def ask(question, images, state=None, target=target_description("dopant_zoom")):
        """(the whole response, latency) of one question."""
        t0 = time.perf_counter()
        r = requests.post(f"{spec.decision_url}/v1/systemone", timeout=300, json={
            "model": spec.name, "images": images, "questions": {question: QUESTIONS[question]},
            "state": system_one_state(state or state_text(), target)})
        r.raise_for_status()
        return r.json(), time.perf_counter() - t0

    def result(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        rep[name] = {"pass": bool(passed), **detail}
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {json.dumps(detail)[:300]}", flush=True)

    # questions: every question (an error here stops the check)
    answered = {q: ask(q, boxed)[0]["answers"][q]["choice"] for q in QUESTIONS}
    result("questions", True, {"answers": answered})

    # image: the input tokens with and without the image
    n_tokens = {name: ask("locate", images)[0]["usage"]["input_tokens"]
                for name, images in (("image", plain), ("none", []))}
    result("image", n_tokens["image"] - n_tokens["none"] > 50, {"image_tokens": n_tokens["image"] - n_tokens["none"]})

    # repeatability (reported, not a check): the same locate request three times
    runs = [ask("locate", boxed) for _ in range(3)]
    probs = [r[0]["answers"]["locate"]["probabilities"] for r in runs]
    diff = max(abs(p[k] - probs[0][k]) for p in probs for k in probs[0])
    rep["repeatability"] = {"max_prob_difference": diff, "choices": [r[0]["answers"]["locate"]["choice"] for r in runs]}
    print(f"[info] repeatability: max probability difference {diff:.3f} over 3 identical requests", flush=True)
    lat = [r[1] for r in runs]

    # sanity: locate on start views
    seen = {}
    for mode in ("visible", "hidden"):
        e = make("dopant_zoom", start_mode=mode)
        answers = []
        for s in range(a.n):
            p, _ = e.reset(1000 + s)
            d, latency = ask("locate", [png_data_uri(draw_box(p))])
            lat.append(latency)
            answers.append(d["answers"]["locate"]["choice"] != "none")
        seen[mode] = sum(answers) / len(answers)
    result("sanity", seen["visible"] > seen["hidden"], {"seen_rate": seen})

    rep["latency_s"] = {"median": statistics.median(lat), "max": max(lat)}
    print(f"latency median {statistics.median(lat) * 1000:.0f} ms", flush=True)
    if a.out:
        open(a.out, "w").write(json.dumps(rep, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
