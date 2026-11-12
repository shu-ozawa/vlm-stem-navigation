
"""Record Git, library versions, host, GPU, and timestamps."""

import importlib.metadata
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from serving.servers import gpu_name


REPO_ROOT = Path(__file__).resolve().parents[2]
HOST = socket.gethostname()


def git_state() -> dict:
    """Return the commit and tracked changes of the running code."""
    snapshot_commit = os.environ.get("STEMNAV_COMMIT")

    if snapshot_commit:
        return {
            "commit": snapshot_commit,
            "snapshot": True,
            "changed_files": [],
        }

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=no")

    return {
        "commit": git("rev-parse", "HEAD"),
        "snapshot": False,
        "changed_files": [
            line[3:] for line in status.splitlines()
        ],
    }


def versions() -> dict:
    """Return installed library versions."""
    installed = {}

    for package in ("vllm", "torch", "transformers"):
        try:
            installed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed[package] = None

    return installed


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_record() -> dict:
    """Return reproducibility metadata for a run."""
    return {
        "git": git_state(),
        "versions": versions(),
        "host": HOST,
        "gpu": gpu_name(),
    }
