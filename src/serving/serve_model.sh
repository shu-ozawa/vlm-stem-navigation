#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Configuration
# ============================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
VLLM="${ROOT}/.venv/bin/vllm"

VLLM_PORT="${STEMNAV_PORT:-8000}"
DECISION_PORT="${STEMNAV_DECISION_PORT:-8080}"

cd "$ROOT"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found. Run 'uv sync' first." >&2
    exit 1
fi


# ============================================================
# Help
# ============================================================

show_help() {
    echo "Usage: serve_model.sh <model> [reader] [vllm options]"
    echo
    echo "Available models:"
    "$PYTHON" -m serving.models list
}


# ============================================================
# Parse arguments
# ============================================================

if [[ $# -eq 0 ]]; then
    show_help
    exit 1
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

MODEL="$1"
shift

READER="typed"

if [[ $# -gt 0 && "$1" != -* ]]; then
    READER="$1"
    shift
fi

# Remaining arguments ($@) are passed directly to vLLM.


# ============================================================
# Models without vLLM: hosted APIs
# ============================================================

MODEL_TYPE=$("$PYTHON" -m serving.models list |
    awk -v name="$MODEL" '$1 == name {print $2}')

mapfile -t DECISION_ARGS < <(
    "$PYTHON" -m serving.models decision-args "$MODEL"
)

if [[ "$MODEL_TYPE" != "vllm" ]]; then
    if [[ "$MODEL_TYPE" == "hosted" ]]; then
        # The API key.
        set -a
        source "$ROOT/.env"
        set +a
    fi

    exec "$PYTHON" -m serving.decision_server "${DECISION_ARGS[@]}"
fi


# ============================================================
# Local vLLM models
# ============================================================

# Configure CUDA and FlashInfer.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"

# Get model-specific vLLM arguments.
mapfile -t VLLM_ARGS < <(
    "$PYTHON" -m serving.models serve-args "$MODEL" "$READER"
)


# ============================================================
# Server management
# ============================================================

check_ports() {
    for port in "$VLLM_PORT" "$DECISION_PORT"; do
        if curl -s -o /dev/null "http://localhost:$port/health"; then
            echo "Port $port is already in use." >&2
            exit 1
        fi
    done
}


stop_servers() {
    echo "Stopping servers..."

    for pid in $(jobs -p); do
        kill "$pid" 2>/dev/null || true
    done

    wait || true
}

trap stop_servers EXIT


start_vllm() {
    echo "Starting vLLM server on port $VLLM_PORT..."

    "$VLLM" "${VLLM_ARGS[@]}" "$@" &
    VLLM_PID=$!

    echo "Waiting for vLLM to become ready..."

    until curl -sf -o /dev/null "http://localhost:$VLLM_PORT/health"; do

        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "vLLM failed to start." >&2
            exit 1
        fi

        sleep 5
    done

    echo "vLLM is ready."
}


start_decision_server() {
    echo "Starting Decision Server on port $DECISION_PORT..."

    "$PYTHON" -m serving.decision_server "${DECISION_ARGS[@]}" &
}


# ============================================================
# Main
# ============================================================

check_ports

start_vllm "$@"
start_decision_server

echo "Both servers are running."

wait -n

echo "A server exited. Shutting down..."
exit 1