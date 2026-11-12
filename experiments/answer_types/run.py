
"""Run configured episode and probe experiments.

Servers are shared by jobs with the same model and serving settings.
Completed experiments are skipped. Interrupted episode sets restart;
probes resume by skipping previously recorded views.
"""

import argparse
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from agent import models as agent_models
from agent.agent import Agent
from agent.prompts.build import target_description
from agent.readers import make_reader
from episode import run_episode
from probe import group_views, probe_view, views
from records import HOST, now_utc, run_record
from serving import models as serving_models
from serving.servers import Servers
from simulator.tasks import make


HERE = Path(__file__).resolve().parent


def jobs(cfg: dict, args, output: Path) -> list[dict]:
    """Build the episode and probe jobs selected by the config and CLI."""
    planned = []

    for reader, model in itertools.product(cfg["readers"], cfg["models"]):
        if args.reader and reader not in args.reader:
            continue
        if args.model and model not in args.model:
            continue

        model_spec = agent_models.get(model)
        if reader not in model_spec.readers:
            continue

        episodes = []
        probes = []

        if "episodes" in cfg and args.only != "probes":
            settings = cfg["episodes"]
            doses = settings["doses"] if "doses" in settings else [settings.get("dose")]
            descriptions = (
                settings["descriptions"] if "descriptions" in settings
                else [cfg.get("description", "reference")]
            )

            for description, dose, task, start_mode in itertools.product(
                descriptions, doses, settings["tasks"], settings["start_modes"]
            ):
                if args.task and task not in args.task:
                    continue
                if args.start_mode and start_mode not in args.start_mode:
                    continue

                folder = output / "episodes" / reader / model
                if "descriptions" in settings:
                    folder = folder / f"description={description}"
                if "doses" in settings:
                    folder = folder / f"dose={dose}"

                episodes.append({
                    "task": task,
                    "start_mode": start_mode,
                    "dose": dose,
                    "description": description,
                    "out_dir": folder / f"{task}_{start_mode}",
                })

        if "probes" in cfg and args.only != "episodes":
            for task in cfg["probes"]["tasks"]:
                if args.task and task not in args.task:
                    continue

                probes.append({
                    "task": task,
                    "out": output / "probes" / task / f"{reader}_{model}.jsonl",
                })

        if not episodes and not probes:
            continue

        served = serving_models.get(model)
        serving = serving_models.serve_flags(model, reader) if served.vllm_served else ()
        planned.append({
            "reader": reader,
            "model": model,
            "episodes": episodes,
            "probes": probes,
            "serving": serving,
        })

    return planned


def serving_record(model: str, reader: str) -> dict:
    """How the model is served: vLLM's argv (None for a hosted model)."""
    served = serving_models.get(model)
    return {"serve_argv": serving_models.serve_argv(model, reader) if served.vllm_served else None}


def lines(path: Path) -> int:
    """Count lines in an existing JSONL file."""
    if not path.exists():
        return 0

    with path.open() as file:
        return sum(1 for _ in file)


def run_episodes(agent: Agent, exp: dict, cfg: dict, count: int, session: dict) -> None:
    """Run and save all episodes of one experiment."""
    settings = cfg["episodes"]
    seed_start = settings["seeds"]["start"]
    seeds = range(seed_start, seed_start + count)

    env_kwargs = {
        "start_mode": exp["start_mode"],
        "dose": exp["dose"],
        "max_steps": settings["max_steps"],
    }

    env = make(exp["task"], **env_kwargs)
    oracle_env = make(exp["task"], **env_kwargs)

    model = agent.reader.model
    description = exp["description"]
    target = target_description(exp["task"], description)

    meta = {
        "reader": agent.reader.name,
        "label_exclusion": agent.reader.supports_label_exclusion,
        "model": model,
        "task": exp["task"],
        "start_mode": exp["start_mode"],
        "seeds": [seeds.start, seeds.stop - 1],
        "env_kwargs": env_kwargs,
        "description": description,
        "target_description": target,
        "model_spec": serving_models.TABLE["models"][model],
        **serving_record(model, agent.reader.name),
        **session,
        "started_utc": now_utc(),
    }

    folder = exp["out_dir"]
    folder.mkdir(parents=True, exist_ok=True)

    with (folder / "episodes.jsonl").open("w") as episode_file, \
         (folder / "steps.jsonl").open("w") as step_file:

        for seed in seeds:
            episode, steps = run_episode(agent, env, oracle_env, seed, target)

            episode_file.write(json.dumps(episode) + "\n")
            step_file.writelines(json.dumps(step) + "\n" for step in steps)

            episode_file.flush()
            step_file.flush()

            print(
                f"  seed {seed}: {episode['result']:11s} "
                f"steps {episode['steps']:2d} "
                f"(oracle {episode['oracle_steps']:2d}) "
                f"{episode['episode_s']:6.1f} s",
                flush=True,
            )

    meta["finished_utc"] = now_utc()
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))


def run_probe(agent: Agent, probe: dict, cfg: dict, count: int, session: dict) -> None:
    """Run and append missing decision probes: the views at offsets from the target (offsets_A), or with the target
    placed by group (groups)."""
    settings = cfg["probes"]
    seed_start = settings["seeds"]["start"]
    seeds = range(seed_start, seed_start + count)
    by_group = "groups" in settings
    placed_by = "group" if by_group else "offset_A"

    output = probe["out"]
    output.parent.mkdir(parents=True, exist_ok=True)

    completed = set()
    if output.exists():
        with output.open() as file:
            for line in file:
                row = json.loads(line)
                completed.add((row["seed"], row["zoom_index"], row[placed_by]))

    description = cfg.get("description", "reference")
    target = target_description(probe["task"], description)

    meta = {
        "reader": agent.reader.name,
        "label_exclusion": agent.reader.supports_label_exclusion,
        "model": agent.reader.model,
        "task": probe["task"],
        "seeds": [seeds.start, seeds.stop - 1],
        **{key: settings[key] for key in ("offsets_A", "groups", "not_in_view_max_fov") if key in settings},
        "zoom_indices": settings["zoom_indices"],
        **serving_record(agent.reader.model, agent.reader.name),
        **session,
        "started_utc": now_utc(),
        "description": description,
        "target_description": target,
    }

    env = make(probe["task"], start_mode="visible")

    def placements(seed: int):
        """(zoom index, centre, placement) of every view of one specimen."""
        if by_group:
            yield from group_views(
                env, seed, settings["groups"], settings["zoom_indices"], settings["not_in_view_max_fov"]
            )
        else:
            for zoom, offset, angle, center in views(
                env, seed, settings["offsets_A"], settings["zoom_indices"]
            ):
                yield zoom, center, {"offset_A": offset, "angle_deg": angle}

    with output.open("a") as file:
        for seed in seeds:
            for zoom, center, placement in placements(seed):
                if (seed, zoom, placement[placed_by]) in completed:
                    continue

                row = probe_view(agent, env, seed, zoom, center, target, placement)

                file.write(json.dumps(row) + "\n")
                file.flush()

                answer = row["answers"][0]
                correct = "" if answer["correct"] else " (x)"
                where = f"{row['group']:11s}" if by_group else f"offset {row['offset_A']:4.0f} A"

                print(
                    f"  seed {seed} zoom {zoom} {where}: "
                    f"{answer['choice']}{correct} -> {row['action']} "
                    f"{row['decision_s']:.2f} s",
                    flush=True,
                )

    meta["finished_utc"] = now_utc()
    meta_path = output.parent / f"meta_{agent.reader.name}_{agent.reader.model}.json"
    meta_path.write_text(json.dumps(meta, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reader", action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--task", action="append")
    parser.add_argument("--start-mode", action="append")
    parser.add_argument("--only", choices=("episodes", "probes"))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--probe-seeds", type=int)
    parser.add_argument("--use-running-server", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    with open(args.config) as file:
        cfg = yaml.safe_load(file)

    run_info = run_record()
    changed_files = run_info["git"]["changed_files"]

    if changed_files and not args.allow_dirty:
        raise SystemExit(
            "uncommitted changes: " + ", ".join(changed_files)
        )

    output = Path(os.environ.get("STEMNAV_RESULTS", HERE / "results")) / cfg["run"]

    episode_count = args.episodes or cfg.get("episodes", {}).get("seeds", {}).get("count")
    probe_count = args.probe_seeds or cfg.get("probes", {}).get("seeds", {}).get("count")

    # Select experiments that still need to run.
    pending = []

    for job in jobs(cfg, args, output):
        remaining_episodes = []

        for exp in job["episodes"]:
            folder = exp["out_dir"]
            complete = (
                lines(folder / "episodes.jsonl") >= episode_count
                and (folder / "meta.json").exists()
            )
            if args.force or not complete:
                remaining_episodes.append(exp)

        remaining_probes = []

        if job["probes"]:
            per_view = cfg["probes"].get("groups") or cfg["probes"]["offsets_A"]
            expected = probe_count * len(per_view) * len(cfg["probes"]["zoom_indices"])

            for probe in job["probes"]:
                if lines(probe["out"]) < expected:
                    remaining_probes.append(probe)

        job["episodes"] = remaining_episodes
        job["probes"] = remaining_probes

        if remaining_episodes or remaining_probes:
            pending.append(job)

    # Group readers that can share the same model server.
    model_order = list(dict.fromkeys(job["model"] for job in pending))
    pending.sort(key=lambda job: (model_order.index(job["model"]), job["serving"]))

    grouped = itertools.groupby(
        pending,
        key=lambda job: (job["model"], job["serving"]),
    )

    for (model, _), group_items in grouped:
        group = list(group_items)
        readers = ", ".join(job["reader"] for job in group)
        print(f"=== {model}: readers {readers}", flush=True)

        session = {**run_info, "server_managed": not args.use_running_server}
        server = None

        try:
            if not args.use_running_server:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                reader = group[0]["reader"]
                log_path = output / "servers" / f"{model}_{reader}_{HOST}_{stamp}.log"

                server = Servers(model, reader, log_path, cfg["server"])
                session["server_startup_s"] = server.start()
                session["server_log"] = str(server.log_path.relative_to(output))

            for job in group:
                agent = Agent(make_reader(job["reader"], model))

                for exp in job["episodes"]:
                    print(
                        f"--- episodes {job['reader']} / {model} / "
                        f"{exp['task']} / {exp['start_mode']}",
                        flush=True,
                    )
                    run_episodes(agent, exp, cfg, episode_count, session)

                for probe in job["probes"]:
                    print(
                        f"--- probes {job['reader']} / {model} / {probe['task']}",
                        flush=True,
                    )
                    run_probe(agent, probe, cfg, probe_count, session)

        finally:
            if server is not None:
                server.stop()


if __name__ == "__main__":
    main()
