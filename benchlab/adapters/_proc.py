"""Run a harness subprocess with logging, Ctrl-C forwarding and the budget guard."""

from __future__ import annotations

import signal
import subprocess
import time
from pathlib import Path

from benchlab.redact import redact
from benchlab.usage import BudgetExceeded, BudgetGuard


def run_logged(
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    budget: BudgetGuard | None = None,
    cwd: str | Path | None = None,
    stop_signal: int = signal.SIGINT,
) -> None:
    """Run ``cmd`` appending output to ``log_path``. On budget trip or Ctrl-C the
    child gets ``stop_signal`` (SIGINT lets harnesses checkpoint) before we raise."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out = open(log_path, "ab")
    proc = subprocess.Popen(
        cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=str(cwd) if cwd else None
    )
    tripped: list[BudgetExceeded] = []

    def on_trip(e: BudgetExceeded) -> None:
        tripped.append(e)
        if proc.poll() is None:
            proc.send_signal(stop_signal)

    if budget is not None:
        budget.watch(interval=5.0, on_trip=on_trip)
    try:
        while proc.poll() is None:
            time.sleep(1.0)
    except KeyboardInterrupt:
        proc.send_signal(stop_signal)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    finally:
        if budget is not None:
            budget.stop()
        out.close()
    if tripped:
        raise tripped[0]
    if proc.returncode != 0:
        tail = log_path.read_text(errors="replace")[-3000:]
        raise RuntimeError(
            f"{Path(cmd[0]).name} exited with {proc.returncode}; tail of {log_path.name}:\n{redact(tail)}"
        )
