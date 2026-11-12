#!/usr/bin/env bash
# Serve a model and open the viewer on it, in one command:
#
#   scripts/viewer.sh <name>     start the model's servers (src/serving/serve_model.sh), then the viewer
#   scripts/viewer.sh            only the viewer, on the model already being served (with no model being
#                                served, or with -h / --help: this help and the model names)
#
# The viewer listens on http://<this machine>:8090 (STEMNAV_VIEWER_PORT) on all interfaces, without
# authentication; on an untrusted network use an SSH tunnel instead (ssh -L 8090:localhost:8090 <host>).
# Ctrl+C stops the viewer and the servers it started.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || { echo "run 'uv sync' in ${ROOT} first" >&2; exit 1; }
cd "${ROOT}"

VLLM_PORT="${STEMNAV_PORT:-8000}"
DECISION_PORT="${STEMNAV_DECISION_PORT:-8080}"
VIEWER_PORT="${STEMNAV_VIEWER_PORT:-8090}"

usage() {
  echo "usage: scripts/viewer.sh <model>   serve the model, then open the viewer" >&2
  echo "       scripts/viewer.sh           open the viewer on the model already being served" >&2
  echo "models:" >&2
  "${PY}" -m serving.models list | sed 's/^/  /' >&2
}
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -eq 0 ]] && ! curl -sf -o /dev/null "http://localhost:${VLLM_PORT}/health" \
    && ! curl -sf -o /dev/null "http://localhost:${DECISION_PORT}/health"; then
  echo "no model is being served" >&2
  usage
  exit 1
fi

trap 'kill $(jobs -p) 2>/dev/null; wait' EXIT

if [[ $# -ge 1 ]]; then
  src/serving/serve_model.sh "$1" &
  serve_pid=$!
  echo "waiting for $1 (vLLM, then the decision server); the first start takes several minutes" >&2
  until curl -sf -o /dev/null "http://localhost:${DECISION_PORT}/health"; do
    kill -0 "${serve_pid}" 2>/dev/null || { echo "the model servers stopped during startup" >&2; exit 1; }
    sleep 5
  done
fi

echo "viewer: http://$(hostname):${VIEWER_PORT}" >&2
"${PY}" -m viewer.app --port "${VIEWER_PORT}" &
wait -n
