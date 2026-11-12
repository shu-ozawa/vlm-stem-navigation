"""Start and stop model servers for experiments."""

import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from serving import models


ROOT = Path(__file__).resolve().parents[2]
SERVE_SCRIPT = Path(__file__).with_name("serve_model.sh")


def healthy(url: str) -> bool:
    """Check whether a server responds to /health."""
    try:
        return urllib.request.urlopen(f"{url}/health", timeout=3).status == 200
    except OSError:
        return False


def wait_until(condition, timeout: float, description: str) -> float:
    """Wait until a condition becomes true, returning elapsed seconds."""
    start = time.monotonic()

    while not condition():
        if time.monotonic() - start >= timeout:
            raise TimeoutError(f"Timed out waiting for {description}")
        time.sleep(5)

    return time.monotonic() - start


def visible_gpus() -> list[str]:
    """Restrict nvidia-smi to GPUs assigned by Slurm."""
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    return ["-i", devices] if devices else []


def gpu_apps() -> list[str]:
    """Return processes currently using the assigned GPUs."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
            *visible_gpus(),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def gpu_name() -> str | None:
    """Return GPU model, memory and driver information."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
                *visible_gpus(),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except FileNotFoundError:
        return None


class Servers:
    def __init__(self, model: str, reader: str, log_path: Path, cfg: dict):
        self.model = model
        self.reader = reader
        self.log_path = log_path
        self.cfg = cfg
        self.spec = models.get(model)
        self.proc = None

    def start(self) -> float:
        """Start the servers and wait until the decision server is ready."""

        # Check GPU availability.
        if self.spec.vllm_served and self.cfg["require_idle_gpu"]:
            running = gpu_apps()
            if running:
                raise RuntimeError(
                    "GPU is already in use:\n" + "\n".join(running)
                )

        # Start serve_model.sh and save its output.
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        with self.log_path.open("a") as log:
            self.proc = subprocess.Popen(
                [str(SERVE_SCRIPT), self.model, self.reader],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        # Wait for the decision server.
        def ready():
            if self.proc.poll() is not None:
                raise RuntimeError(f"Server exited. See {self.log_path}")
            return healthy(self.spec.decision_url)

        try:
            return wait_until(
                ready,
                self.cfg["startup_timeout_s"],
                f"{self.model} server",
            )
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        """Stop the servers and wait for GPU resources to be released."""

        if self.proc is None:
            return

        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)

            try:
                self.proc.wait(timeout=self.cfg["shutdown_timeout_s"])
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()

        self.proc = None

        if self.spec.vllm_served:
            wait_until(
                lambda: not gpu_apps(),
                self.cfg["shutdown_timeout_s"],
                "GPU to become idle",
            )