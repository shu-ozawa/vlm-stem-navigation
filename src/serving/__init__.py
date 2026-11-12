"""The model side: the servers the agent sends its questions to.

  serve_model.sh     starts a model's vLLM server and, in front of it, the decision server (serve_model.sh <model>
                     [<reader>]; a hosted model: the decision server alone); servers.py does the same from
                     Python, for the experiments
  decision_server/   typed reads: the System One API, label probabilities from vLLM (or a hosted API)
  models.py          each model's vLLM flags and decision server arguments (models.yaml, serve)
The generated readers send their requests to vLLM directly.
"""
