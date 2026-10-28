"""``python -m benchlab.ui.launcher <job.json>``: run the job's command with output in
``<id>.log`` and write its exit code to ``<id>.exit`` when it ends.

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
    with open(log, "ab", buffering=0) as out:
        p = subprocess.Popen(
            job["argv"], stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT
        )
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # after spawning: the child keeps SIGINT
        rc = p.wait()
    jpath.with_suffix(".exit").write_text(str(rc if rc >= 0 else 128 - rc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1]))
