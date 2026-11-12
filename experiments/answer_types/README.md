# answer_types

The question: in the simulated microscope, does it help the agent to let the model think before it answers? The
agent is the same in every run (prompts, policy); only the reader differs:

| reader | how the answer is obtained |
|---|---|
| `typed` | one pass, the probability of every label read off the model (decision server) |
| `generated` | the model writes its label as JSON, thinking off |
| `generated_thinking` | the same request, thinking on |

- `run.py`: the episodes and decision probes of a config (see its docstring for the outputs); with it, `episode.py`
  (one episode), `probe.py` (the probe views and one decision on each) and `records.py` (what a run records).
- `configs/`: `smoke.yaml` (a short run to try it) and the paper's runs: `answer_types_v2.yaml` (the comparison of
  the readers), `answer_types_v2_noise.yaml` (under shot noise), `answer_types_v2_descriptions.yaml` (the target
  described in other words) and `answer_types_v2_probe_groups.yaml` (the probe views placed by group). Each config's
  header says how it is run.
- `run.py` starts the model's servers itself (vLLM and the decision server) and records the commit it runs; it refuses
  uncommitted changes to tracked files (`--allow-dirty` to run anyway).
- `results/`: outside git.

```
uv run python experiments/answer_types/run.py --config experiments/answer_types/configs/smoke.yaml
```
