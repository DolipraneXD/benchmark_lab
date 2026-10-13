"""RULER (NVIDIA/RULER) - deliberately NOT integrated; this adapter explains why.

Checked 2026-10-13:

* The repository's README marks its original pipeline (``scripts/run.sh``,
  ``pred/call_api.py``) as **deprecated** and points to the ``rulerv1-ns`` branch,
  whose pipeline is NeMo-Skills (``ns prepare_data ruler`` / ``ns eval``) driven
  by a cluster config and a model *server* it launches itself (vLLM/SGLang/TRT-LLM
  on GPUs). It is not written for an arbitrary OpenAI-compatible endpoint.
* The legacy OpenAI client (``scripts/pred/client_wrappers.py``) hardcodes
  ``AZURE_API_ID/SECRET/ENDPOINT`` environment variables and a fixed
  ``model2length`` table, so it cannot talk to our proxy without editing the
  harness - which would no longer be the official harness.

Options, in order of fidelity: (1) run NeMo-Skills on the GPU box that serves
Laguna / MiMo and feed it through its own vLLM server (official, heavy);
(2) use NeMo-Skills' OpenAI-server mode against our proxy if a local cluster
config is acceptable; (3) accept a RULER re-implementation (e.g. another
framework's port) - that violates the "official harness" rule. Decide, then this
adapter can be replaced. Until then ``prepare()`` fails with this message and
the run is skipped, not faked.
"""

from __future__ import annotations

from typing import Any

from benchlab.adapters.base import BenchAdapter, HarnessNotReady, RunPlan

REASON = (
    "RULER skipped: NVIDIA/RULER's current pipeline is NeMo-Skills (cluster config + its own vLLM/SGLang "
    "server, GPU-only) and the legacy OpenAI client hardcodes Azure env vars and a fixed model table, so it "
    "cannot target an OpenAI-compatible endpoint without modifying the harness. See benchlab/adapters/ruler.py."
)


class RulerAdapter(BenchAdapter):
    key = "ruler"
    display = "RULER (skipped)"
    stage = 4
    needs_proxy = False

    def prepare(self) -> dict[str, Any]:
        raise HarnessNotReady(REASON)

    def plan(self, model, subset, trials, smoke) -> RunPlan:
        return RunPlan(0, 1, None, 0, 0, ["skipped: " + REASON])

    def run(self, model, run_dir, subset, trials, smoke, budget=None):
        raise HarnessNotReady(REASON)

    def parse(self, run_dir):
        raise HarnessNotReady(REASON)
