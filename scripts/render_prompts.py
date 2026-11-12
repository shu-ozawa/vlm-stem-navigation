"""Write out the prompts as the models receive them, for every reader and question, to read and compare when the texts
(src/agent/prompts/texts.yaml) are revised. No model is needed: the text is built by the same code as in the runs
(typed: the decision server's layout; generated: prompts/build.py), the image is written next to it.

Cases, per task:
  locate            the first step of an episode (visible start)
  locate_reask      locate asked again without inside and none (generated readers, policy.py step 4)

    uv run python scripts/render_prompts.py [--out docs/prompts]

Writes <out>/<task>/<case>_<reader>.txt and <out>/<task>/<case>.png (the image sent), and <out>/README.md
(the commit it was written from).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from agent import models
from agent.prompts.build import chat_messages, labels, state_text, target_description
from agent.prompts.image import draw_box
from agent.prompts.layout import shown
from agent.readers.generated import system_content
from simulator.tasks import make

REPO = Path(__file__).resolve().parents[1]
SEED = 2000
# the models whose generated system turn differs (a thinking instruction); the others get the plain one
THINKING_INSTRUCTION_MODELS = ["cosmos_reason2_32b", "magistral_small_2509"]


def render(state: str, target: str, question: str, choices: tuple[str, ...], reader: str) -> str:
    """The request of `reader` as text: the same layout for every reader (src/agent/prompts/layout.py); what differs
    is the answer (typed: the next token after the prefilled "<id>:"; generated: JSON, structured output) and, with
    thinking, a model's thinking instruction in the system turn."""
    messages = chat_messages(state, target, "<image>", question, choices)
    system, user = messages[0]["content"], shown(messages[1]["content"])
    out = [f"=== SYSTEM\n\n{system}"]
    if reader == "generated_thinking":
        for model in THINKING_INSTRUCTION_MODELS:
            content = system_content(system, models.get(model).ask["with_thinking"])
            text = content if isinstance(content, str) else "".join(
                c.get("text") or f"[THINK]{c['thinking']}[/THINK]" for c in content)
            out.append(f"=== SYSTEM ({model}, which needs a thinking instruction: added at the end)\n\n{text}")
    out.append(f"=== USER\n\n{user}")
    if reader == "typed":
        out.append(f"=== ASSISTANT (prefilled; the next token is read)\n\n{question}:")
    else:
        schema = "{" + f'"{question}": one of {list(choices)}' + "}"
        out.append(f"=== ASSISTANT\n\n{'(thinks first) ' if reader == 'generated_thinking' else ''}JSON {schema} "
                   "(structured output)")
    return "\n\n".join(out)


def cases(task: str):
    """(case, question, image PNG, state text, target description, labels offered)."""
    env = make(task, start_mode="visible")
    obs, _ = env.reset(SEED)
    target = target_description(task)
    yield "locate", "locate", draw_box(obs), state_text(), target, labels("locate")
    yield ("locate_reask", "locate", draw_box(obs), state_text(), target,
           tuple(l for l in labels("locate") if l not in ("inside", "none")))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "docs" / "prompts"))
    a = ap.parse_args()
    out = Path(a.out)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()

    for task in ["dopant_zoom", "grain_boundary"]:
        (out / task).mkdir(parents=True, exist_ok=True)
        for case, question, png, state, target, choices in cases(task):
            (out / task / f"{case}.png").write_bytes(png)
            readers = ["generated", "generated_thinking"] if case == "locate_reask" else [  # typed: no second request
                "typed", "generated", "generated_thinking"]
            for reader in readers:
                text = render(state, target, question, choices, reader)
                (out / task / f"{case}_{reader}.txt").write_text(text + "\n")
                print(out / task / f"{case}_{reader}.txt")

    (out / "README.md").write_text(
        "# The prompts as the models receive them\n\n"
        "Written by `scripts/render_prompts.py` from `src/agent/prompts/texts.yaml` "
        f"(commit `{commit}`), task seed {SEED}. Per task: `locate` (the first step), `locate_reask` "
        "(asked again without inside and none: the generated readers' fallback), each with the image sent (`.png`) and one file per reader.\n")


if __name__ == "__main__":
    main()
