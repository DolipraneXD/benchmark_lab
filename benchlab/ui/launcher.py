"""``python -m benchlab.ui.launcher <job.json>``: run the job's command (or, for container
jobs, its steps one after another: one container per benchmark family) with output in
``<id>.log``, and write the exit code to ``<id>.exit`` when it ends.

The launcher is the leader of the job's process group. It ignores SIGINT itself (a
Stop is sent to the whole group) so it stays alive to record the CLI's exit code. Every
command it starts gets SIGINT back (an ignored signal is otherwise inherited), so a Stop
reaches each step, the second container of a job as much as the first.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path


def main(job_json: str) -> int:
    jpath = Path(job_json)
    job = json.loads(jpath.read_text())
    steps = job.get("steps") or [job["argv"]]
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        final = _run_steps(steps, jpath.with_suffix(".log"))
    finally:
        signal.signal(signal.SIGINT, previous)
    jpath.with_suffix(".exit").write_text(str(final))
    return 0


def _default_sigint() -> None:  # runs in the child, before exec
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def _run_steps(steps: list[list[str]], log: Path) -> int:
    final = 0
    with open(log, "ab", buffering=0) as out:
        for argv in steps:
            p = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                preexec_fn=_default_sigint,
            )
            rc = p.wait()
            rc = rc if rc >= 0 else 128 - rc
            if rc == 130:  # stopped: don't start the next family
                return 130
            final = final or rc  # the first failure decides, like `benchlab run`
    return final


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1]))
