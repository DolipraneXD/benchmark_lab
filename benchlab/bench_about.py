"""One-line descriptions and project links for each benchmark, shown when you hover a
benchmark in the control panel. Model-specific extras reuse their base benchmark's entry."""

from __future__ import annotations

from typing import Any

from benchlab import registry

T2 = (
    "τ²-bench: the model is a customer-support agent with tools and a written policy, "
    "talking to a simulated customer. It passes when the conversation leaves the "
    "database in the right state."
)
TB = (
    "Terminal-Bench: real tasks in a Linux terminal inside Docker (build, debug, "
    "configure, data work). Each task is checked by tests."
)
AIDER = (
    "Aider Polyglot: 225 hard Exercism exercises in C++, Go, Java, JavaScript, Python "
    "and Rust. The model edits the code until the tests pass, with two tries."
)
LCB = (
    "LiveCodeBench: competitive-programming problems released after the model's "
    "knowledge cutoff, so they can't have been memorized."
)
RULER = (
    "RULER: long-context tests. Finding, tracking and combining information spread over "
    "inputs of up to 128k tokens."
)
MTEB = "MTEB: the standard embedding benchmark; here its English retrieval and reranking tasks."

GH = "https://github.com/"
ABOUT: dict[str, tuple[str, str | None]] = {
    "gpqa": (
        "GPQA Diamond: 198 graduate-level multiple-choice questions in biology, physics "
        "and chemistry, written to be hard to look up. Measures expert reasoning.",
        GH + "idavidrein/gpqa",
    ),
    "bfcl": (
        "Berkeley Function Calling Leaderboard: does the model call the right function "
        "with correct arguments, from single calls to parallel and multiple ones.",
        GH + "ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard",
    ),
    "t2air": (
        T2 + " Domain: airline (bookings, changes, cancellations).",
        GH + "sierra-research/tau2-bench",
    ),
    "t2ret": (
        T2 + " Domain: retail (orders, returns, exchanges).",
        GH + "sierra-research/tau2-bench",
    ),
    "t2tel": (
        T2 + " Domain: telecom, where the customer must also act on their own device.",
        GH + "sierra-research/tau2-bench",
    ),
    "tb": (TB, GH + "laude-institute/terminal-bench"),
    "lag_tb21": (TB, GH + "laude-institute/terminal-bench"),
    "swe": (
        "SWE-bench Verified: 500 real GitHub issues from Python projects, checked by "
        "people. The model writes a patch and the project's own tests decide.",
        GH + "SWE-bench/SWE-bench",
    ),
    "gaia": (
        "GAIA: real-world assistant questions that need web browsing, files and several "
        "steps of reasoning, each with one short correct answer.",
        "https://huggingface.co/datasets/gaia-benchmark/GAIA",
    ),
    "bcomp": (
        "BrowseComp: hard-to-find facts that take persistent web browsing. Short answers, "
        "checked by a grader model.",
        GH + "openai/simple-evals",
    ),
    "ttft": (
        "Time to first token: how long the model takes to start answering a fixed set of "
        "streamed prompts. BenchLab's own probe.",
        None,
    ),
    "lag_ruler": (RULER, GH + "NVIDIA/RULER"),
    "mimo_long": (RULER, GH + "NVIDIA/RULER"),
    "mimo_tool": (
        "Toolathlon: long, multi-step tasks across real apps and tools (email, Canvas, "
        "Kubernetes and more), run in the official containers.",
        GH + "hkust-nlp/Toolathlon",
    ),
    "mimo_osw": (
        "OSWorld-Verified: computer-use tasks in a real desktop virtual machine, driven "
        "from screenshots.",
        GH + "xlang-ai/OSWorld",
    ),
    "ds_auto": (
        "AutomationBench: business automation tasks over simulated SaaS tools (CRM, email, "
        "tickets and more).",
        GH + "zapier/AutomationBench",
    ),
    "aider": (AIDER, GH + "Aider-AI/polyglot-benchmark"),
    "glm_aider": (AIDER, GH + "Aider-AI/polyglot-benchmark"),
    "ml4_aider": (AIDER, GH + "Aider-AI/polyglot-benchmark"),
    "lcb": (LCB, GH + "LiveCodeBench/LiveCodeBench"),
    "q27_lcb": (LCB, GH + "LiveCodeBench/LiveCodeBench"),
    "mteb_ret": (MTEB, GH + "embeddings-benchmark/mteb"),
    "mteb_rr": (MTEB, GH + "embeddings-benchmark/mteb"),
    "beir": (
        "BEIR: zero-shot retrieval over 18 datasets from many domains, scored by nDCG@10.",
        GH + "beir-cellar/beir",
    ),
    "own": (
        "Your own documents and questions in retrieval/: does the right document come back "
        "in the top 10?",
        None,
    ),
    "serving": (
        "Load test of a self-hosted endpoint: time to first token, tokens per second and "
        "throughput at 1, 8 and 32 parallel users, plus GPU memory.",
        GH + "vllm-project/guidellm",
    ),
    "redteam": (
        "Red team: attack prompts from an outside lab's report (promptfoo). The score is "
        "the share of attacks the model resisted.",
        GH + "promptfoo/promptfoo",
    ),
}


def about(key: str) -> dict[str, Any]:
    """Name, description and project link for a bench key (or alias). Unknown keys and
    computed scores fall back to the registry's own notes."""
    try:
        b = registry.get(key)
    except KeyError:
        return {"key": key, "name": key, "text": "", "url": None}
    text, url = ABOUT.get(b.key, (b.notes, None))
    return {"key": b.key, "name": b.display, "text": text, "url": url}
