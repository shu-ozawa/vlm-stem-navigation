"""One episode: the agent and the environment in turn until the episode ends, with the truth of every question."""

from __future__ import annotations

import time

from agent import policy
from agent.agent import Agent
from simulator.oracle import shortest_path
from simulator.truth import is_correct, question_truth


def run_episode(agent: Agent, env, oracle_env, seed: int, target: str) -> tuple[dict, list[dict]]:
    """(the episode's line of episodes.jsonl, its lines of steps.jsonl). `target`: the target description given to
    the model.

    result: success | wrong_found (found off target) | truncated (max_steps without found) | stuck | lost (not
    visible at the lowest magnification) | no_answer"""
    obs, info = env.reset(seed)
    agent.reset(target)
    oracle_steps = shortest_path(oracle_env, seed)  # ground truth, for SPL only
    actions, steps, ended = [], [], False
    t_episode = time.perf_counter()

    while not env.done:
        truth = question_truth(env, info)  # of the view the model judges
        state = env.operator_state()
        t0 = time.perf_counter()
        d = agent.act(obs, state)
        decision_s = time.perf_counter() - t0

        for st in d["trace"]["stages"]:
            st["truth"], st["correct"] = truth.get(st["id"]), is_correct(truth.get(st["id"]), st["choice"])

        ended = d["choice"] in policy.ENDS
        if not ended:
            obs, _, _, info = env.step(d["choice"])
            actions.append(d["choice"])
            agent.record(state, d, info["blocked"])

        steps.append({"seed": seed, "step": len(steps) + 1, "action": d["choice"], "decision_s": decision_s,
                      "model_latency_s": d["latency_s"], "zoom": info["zoom"], "distance_A": info["distance_A"],
                      "truth": truth, "trace": d["trace"]})
        if ended:
            break

    result = (d["choice"] if ended else "success" if info["success"] else "truncated" if info["truncated"]
              else "wrong_found")
    episode = {
        "seed": seed, "result": result, "success": result == "success", "steps": len(actions),
        "oracle_steps": oracle_steps,
        "spl": oracle_steps / max(oracle_steps, len(actions)) if result == "success" else 0.0,
        "episode_s": time.perf_counter() - t_episode,
        "decision_s_mean": sum(s["decision_s"] for s in steps) / len(steps),
        "questions": sum(len(s["trace"]["stages"]) for s in steps),
        "final_distance_A": info["distance_A"],
        "blocked_actions": sum(h["blocked"] for h in agent.history),
        "actions": actions,
    }
    return episode, steps
