"""Benchmark registry: keys, stages, suites, tracker keys and (when built) adapters.

Rough token profiles are only for ``--dry-run`` cost estimates; the real cost
comes from the usage logs.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

from benchlab.config import ModelSpec


@dataclass(frozen=True)
class BenchInfo:
    key: str
    display: str
    stage: int
    suites: tuple[str, ...]  # core | coder | extras | retrieval | serving
    tracker_key: str  # <bench> part of <modelId>__<bench>
    harness: str
    adapter: str | None = None  # dotted path "module:Class"; None = not integrated yet
    only_models: tuple[str, ...] = ()  # extras: which model ids run it
    n_tasks_default: int = 0
    tokens_in_per_task: int = 2000  # rough, for dry-run only
    tokens_out_per_task: int = 500
    needs_docker: bool = False
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)
    trials_default: int = 1  # used when --trials is not given (τ²: 3 for pass^k)


BENCHES: dict[str, BenchInfo] = {}


def _reg(b: BenchInfo) -> None:
    BENCHES[b.key] = b


# ---- core suite ---------------------------------------------------------------
_reg(
    BenchInfo(
        "gpqa",
        "GPQA Diamond",
        1,
        ("core",),
        "gpqa",
        "inspect_evals/gpqa_diamond",
        adapter="benchlab.adapters.inspect_bench:GPQAAdapter",
        n_tasks_default=198,
        tokens_in_per_task=700,
        tokens_out_per_task=1500,
    )
)
_reg(
    BenchInfo(
        "bfcl",
        "BFCL",
        1,
        ("core",),
        "bfcl",
        "inspect_evals/bfcl",
        adapter="benchlab.adapters.inspect_bench:BFCLAdapter",
        n_tasks_default=3641,
        tokens_in_per_task=1500,
        tokens_out_per_task=200,
    )
)
_reg(
    BenchInfo(
        "t2air",
        "τ²-bench airline",
        2,
        ("core",),
        "t2air",
        "sierra-research/tau2-bench",
        adapter="benchlab.adapters.tau2:Tau2Airline",
        n_tasks_default=20,
        trials_default=3,
        tokens_in_per_task=60000,
        tokens_out_per_task=3000,
        needs_docker=False,
    )
)
_reg(
    BenchInfo(
        "t2ret",
        "τ²-bench retail",
        2,
        ("core",),
        "t2ret",
        "sierra-research/tau2-bench",
        adapter="benchlab.adapters.tau2:Tau2Retail",
        n_tasks_default=20,
        trials_default=3,
        tokens_in_per_task=60000,
        tokens_out_per_task=3000,
    )
)
_reg(
    BenchInfo(
        "t2tel",
        "τ²-bench telecom",
        2,
        ("core",),
        "t2tel",
        "sierra-research/tau2-bench",
        adapter="benchlab.adapters.tau2:Tau2Telecom",
        n_tasks_default=20,
        trials_default=3,
        tokens_in_per_task=60000,
        tokens_out_per_task=3000,
    )
)
_reg(
    BenchInfo(
        "tb",
        "Terminal-Bench 2.x",
        3,
        ("core", "coder"),
        "tb",
        "harbor/terminal-bench",
        adapter="benchlab.adapters.harbor:TerminalBenchAdapter",
        n_tasks_default=50,
        tokens_in_per_task=400000,
        tokens_out_per_task=15000,
        needs_docker=True,
    )
)
_reg(
    BenchInfo(
        "swe",
        "SWE-bench Verified",
        3,
        ("core", "coder"),
        "swe",
        "harbor/swebench-verified",
        adapter="benchlab.adapters.harbor:SWEBenchAdapter",
        n_tasks_default=50,
        tokens_in_per_task=500000,
        tokens_out_per_task=15000,
        needs_docker=True,
    )
)
_reg(
    BenchInfo(
        "gaia",
        "GAIA (subset)",
        4,
        ("core",),
        "gaia",
        "inspect_evals/gaia",
        adapter="benchlab.adapters.inspect_bench:GAIAAdapter",
        n_tasks_default=50,
        tokens_in_per_task=50000,
        tokens_out_per_task=3000,
        needs_docker=True,
    )
)
_reg(
    BenchInfo(
        "ttft",
        "TTFT probe",
        1,
        ("core", "coder", "serving"),
        "ttft",
        "benchlab (fixed prompt set)",
        adapter="benchlab.adapters.ttft:TTFTAdapter",
        n_tasks_default=10,
        tokens_in_per_task=200,
        tokens_out_per_task=64,
    )
)

_reg(
    BenchInfo(
        "bcomp",
        "BrowseComp (subset, alt. to GAIA)",
        4,
        (),
        "gaia",
        "inspect_evals/browse_comp",
        adapter="benchlab.adapters.inspect_bench:BrowseCompAdapter",
        n_tasks_default=100,
        tokens_in_per_task=80000,
        tokens_out_per_task=4000,
        needs_docker=True,
        notes="run with --bench bcomp; needs TAVILY_API_KEY and Docker",
    )
)

# ---- extras ---------------------------------------------------------------------
_reg(
    BenchInfo(
        "lag_tb21",
        "Terminal-Bench 2.1 (Laguna)",
        4,
        ("extras",),
        "lag_tb21",
        "harbor/terminal-bench@2.1",
        adapter="benchlab.adapters.harbor:TerminalBench21Adapter",
        only_models=("laguna",),
        needs_docker=True,
        tokens_in_per_task=400000,
        tokens_out_per_task=15000,
    )
)
_reg(
    BenchInfo(
        "lag_ruler",
        "RULER (Laguna)",
        4,
        ("extras",),
        "lag_ruler",
        "NVIDIA/RULER",
        adapter="benchlab.adapters.ruler:RulerAdapter",
        only_models=("laguna",),
        tokens_in_per_task=64000,
        tokens_out_per_task=100,
    )
)
_reg(
    BenchInfo(
        "mimo_long",
        "RULER (MiMo)",
        4,
        ("extras",),
        "mimo_long",
        "NVIDIA/RULER",
        adapter="benchlab.adapters.ruler:RulerAdapter",
        only_models=("mimo",),
        tokens_in_per_task=64000,
        tokens_out_per_task=100,
    )
)
_reg(
    BenchInfo(
        "mimo_tool",
        "Toolathlon (MiMo)",
        4,
        ("extras",),
        "mimo_tool",
        "Toolathlon",
        adapter=None,
        only_models=("mimo",),
        notes="build order step 9",
    )
)
_reg(
    BenchInfo(
        "mimo_osw",
        "OSWorld-Verified (MiMo)",
        4,
        ("extras",),
        "mimo_osw",
        "OSWorld",
        adapter=None,
        only_models=("mimo",),
        notes="optional, needs VMs",
    )
)
_reg(
    BenchInfo(
        "s5_api",
        "Step 5 core via API",
        4,
        ("extras",),
        "s5_api",
        "core suite",
        adapter=None,
        only_models=("step5",),
        notes="aggregate of core on step5",
    )
)
_reg(
    BenchInfo(
        "s5_self",
        "Step 5 core self-hosted",
        4,
        ("extras",),
        "s5_self",
        "core suite",
        adapter=None,
        only_models=("step5self",),
        notes="aggregate of core on step5self",
    )
)
_reg(
    BenchInfo(
        "s5_tok",
        "Step 5 output tokens/task",
        4,
        ("extras",),
        "s5_tok",
        "usage logs",
        adapter=None,
        only_models=("step5", "step5self"),
    )
)
_reg(
    BenchInfo(
        "ds_auto",
        "AutomationBench (DeepSeek)",
        4,
        ("extras",),
        "ds_auto",
        "AutomationBench",
        adapter=None,
        only_models=("dsflash",),
        notes="verify repo is public first",
    )
)
_reg(
    BenchInfo(
        "ds_cps",
        "cost per solved task (DeepSeek)",
        4,
        ("extras",),
        "ds_cps",
        "usage logs",
        adapter=None,
        only_models=("dsflash",),
    )
)
_reg(
    BenchInfo(
        "ds_lps",
        "latency per solved task (DeepSeek)",
        4,
        ("extras",),
        "ds_lps",
        "usage logs",
        adapter=None,
        only_models=("dsflash",),
    )
)
_reg(
    BenchInfo(
        "glm_aider",
        "Aider Polyglot (GLM)",
        4,
        ("extras",),
        "glm_aider",
        "aider benchmark (docker)",
        adapter="benchlab.adapters.aider:AiderGLM",
        only_models=("glm",),
        needs_docker=True,
        n_tasks_default=225,
        tokens_in_per_task=8000,
        tokens_out_per_task=2000,
    )
)
_reg(
    BenchInfo(
        "ml4_aider",
        "Aider Polyglot (Mistral)",
        4,
        ("extras",),
        "ml4_aider",
        "aider benchmark (docker)",
        adapter="benchlab.adapters.aider:AiderML4",
        only_models=("ml4",),
        needs_docker=True,
        n_tasks_default=225,
        tokens_in_per_task=8000,
        tokens_out_per_task=2000,
    )
)
_reg(
    BenchInfo(
        "q27_lcb",
        "LiveCodeBench (Qwen3.8-27B)",
        4,
        ("extras",),
        "q27_lcb",
        "LiveCodeBench",
        adapter="benchlab.adapters.livecodebench:LCBQwen27",
        only_models=("qwen27",),
        tokens_in_per_task=1500,
        tokens_out_per_task=2500,
    )
)
_reg(
    BenchInfo(
        "gpt_sanity",
        "gpt-oss-120b sanity vs published",
        4,
        ("extras",),
        "gpt_sanity",
        "core suite",
        adapter=None,
        only_models=("gptoss",),
        notes="computed from core results",
    )
)

# ---- coder suite (shares tb/swe above) ------------------------------------------
_reg(
    BenchInfo(
        "lcb",
        "LiveCodeBench",
        4,
        ("coder",),
        "lcb",
        "LiveCodeBench",
        adapter="benchlab.adapters.livecodebench:LCBAdapter",
        tokens_in_per_task=1500,
        tokens_out_per_task=2500,
    )
)
_reg(
    BenchInfo(
        "aider",
        "Aider Polyglot",
        4,
        ("coder",),
        "aider",
        "aider benchmark (docker)",
        adapter="benchlab.adapters.aider:AiderAdapter",
        needs_docker=True,
        n_tasks_default=225,
        tokens_in_per_task=8000,
        tokens_out_per_task=2000,
    )
)

# ---- retrieval & serving (separate pipelines, listed for --dry-run completeness) -----
for k, d in [
    ("mteb_ret", "MTEB retrieval"),
    ("mteb_rr", "MTEB reranking"),
    ("beir", "BEIR nDCG@10"),
    ("beir_rr", "BEIR nDCG@10 + reranker"),
    ("own_rec", "own data recall@10"),
    ("own_rec_rr", "own data recall@10 + rr"),
    ("own_ndcg", "own data nDCG@10"),
    ("own_ndcg_rr", "own data nDCG@10 + rr"),
]:
    _reg(
        BenchInfo(
            k,
            d,
            4,
            ("retrieval",),
            k,
            "mteb / beir / benchlab",
            adapter=None,
            only_models=("qemb",),
        )
    )
for k, d in [
    ("srv_quant", "quantization"),
    ("srv_gpus", "GPUs"),
    ("srv_mem", "GPU memory"),
    ("srv_ttft", "serving TTFT"),
    ("srv_tps", "tokens/s per request"),
    ("srv_c1", "throughput c=1"),
    ("srv_c8", "throughput c=8"),
    ("srv_c32", "throughput c=32"),
    ("srv_match", "self-hosted vs API score match"),
]:
    _reg(BenchInfo(k, d, 4, ("serving",), k, "vllm bench serve", adapter=None))

SUITES = ("core", "coder", "extras", "retrieval", "serving")
STAGE_BENCHES = {
    1: ("gpqa", "bfcl"),
    2: ("t2air", "t2ret", "t2tel"),
    3: ("tb", "swe"),
    4: tuple(k for k, b in BENCHES.items() if b.stage == 4),
}


def get(key: str) -> BenchInfo:
    if key in BENCHES:
        return BENCHES[key]
    for b in BENCHES.values():
        if key in b.aliases:
            return b
    raise KeyError(f"unknown bench {key!r}; known: {sorted(BENCHES)}")


def applies_to(b: BenchInfo, m: ModelSpec) -> bool:
    if b.only_models:
        return m.id in b.only_models or b.key in m.extras
    if b.key in m.extras:
        return True
    return any(s in (m.suites or []) for s in b.suites)


def select_benches(
    bench: str | None = None, suite: str | None = None, stage: int | None = None
) -> list[BenchInfo]:
    """Intersect the given selectors (any may be None)."""
    keys: set[str] | None = None

    def narrow(s: set[str]) -> None:
        nonlocal keys
        keys = s if keys is None else keys & s

    if bench:
        narrow({get(k.strip()).key for k in bench.split(",") if k.strip()})
    if suite:
        if suite not in SUITES:
            raise KeyError(f"unknown suite {suite!r}; known: {SUITES}")
        narrow({k for k, b in BENCHES.items() if suite in b.suites})
    if stage is not None:
        if stage not in STAGE_BENCHES:
            raise KeyError(f"unknown stage {stage}; known: {sorted(STAGE_BENCHES)}")
        narrow(set(STAGE_BENCHES[stage]))
    if keys is None:
        raise ValueError("give at least one of --bench, --suite, --stage")
    order = list(BENCHES)
    return [BENCHES[k] for k in sorted(keys, key=order.index)]


def load_adapter(b: BenchInfo):
    """Import the adapter class; returns None when not integrated yet, raises
    ImportError if the module is registered but missing (a bug, not a stage gap)."""
    if not b.adapter:
        return None
    mod, _, cls = b.adapter.partition(":")
    try:
        module = importlib.import_module(mod)
    except ModuleNotFoundError as e:
        if e.name == mod:
            return None  # planned adapter, not built yet
        raise
    return getattr(module, cls)
