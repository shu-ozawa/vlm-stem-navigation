"""The agent: from an image and the microscope state to an action. Only the reader differs between the answer types
compared; the prompts and the policy are the same for all of them.

  agent.py    the agent of an episode: one decision per step (act), and the log of its actions (record)
  policy.py   the fixed rules that turn the model's answer into an action
  prompts/    everything the model sees: the texts (texts.yaml), their layout (layout.py), the green square (image.py)
  readers/    how the answer is obtained: typed (the decision server), generated (vLLM directly)
  models.py   where each model's servers listen, and how the generated readers ask it (models.yaml, ask)
"""
