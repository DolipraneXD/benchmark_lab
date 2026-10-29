"""``python -m benchlab.ui.launcher <job.json>``: run the job's command (or, for container
jobs, its steps one after another: one container per benchmark family) with output in
``<id>.log``, and write the exit code to ``<id>.exit`` when it ends.

The launcher is the leader of the job's process group. It ignores SIGINT itself (a
Stop is sent to the whole group) so it stays alive to record the CLI's exit code.
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
    log = jpath.with_suffix(".log")
    steps = job.get("steps") or [job["argv"]]
    final = 0
    with open(log, "ab", buffering=0) as out:
        for i, argv in enumerate(steps):
            p = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT
            )
            if i == 0:  # after the first spawn: children keep SIGINT, the launcher ignores it
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            rc = p.wait()
            rc = rc if rc >= 0 else 128 - rc
            if rc == 130:  # stopped: don't start the next family
                final = 130
                break
            final = final or rc  # the first failure decides, like `benchlab run`
    jpath.with_suffix(".exit").write_text(str(final))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1]))
