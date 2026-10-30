"""Per-test detail for a finished run: which tests passed or failed, and what the model saw
and said. Read from each harness's own raw output (nothing is re-scored here), cached in
``<run>/samples.json``.

Every benchmark gives the same record (``Sample``): id, status (pass / fail / error),
score, category, the system prompt, the user prompt, the model's answer, the expected
answer, the grader's reason and the conversation. Fields a harness doesn't record stay
empty. Benchmarks without an extractor fall back to the ``per_task`` list in result.json.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchlab.redact import redact
from benchlab.runs import RunDir

MAX_TEXT = 6000  # characters kept per field / message, so pages stay fast
CACHE = "samples.json"
CACHE_VERSION = 2  # bump when an extractor changes, so cached runs are read again
INSPECT = ("gpqa", "bfcl", "gaia", "bcomp")
TASK_MARK, SCREEN_MARK = "Task Description:", "Current terminal state:"  # Terminus prompt


@dataclass
class Sample:
    id: str
    status: str  # pass | fail | error
    score: float | None = None
    category: str = ""
    system: str = ""
    prompt: str = ""
    response: str = ""
    expected: str = ""
    reason: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)  # role, content, tool_calls


def _text(x: Any, limit: int = MAX_TEXT) -> str:
    """Content of any shape (str, list of parts, dict) as plain, redacted, trimmed text."""
    if x is None:
        return ""
    if isinstance(x, list):
        x = "\n".join(_text(p, limit) for p in x)
    elif isinstance(x, dict):
        x = x.get("text") or x.get("content") or json.dumps(x, ensure_ascii=False, default=str)
    s = redact(str(x))
    return s if len(s) <= limit else s[:limit] + f"… [{len(s) - limit} more characters]"


def _calls(calls: Any) -> list[str]:
    out = []
    for c in calls or []:
        fn = c.get("function")
        name = fn.get("name") if isinstance(fn, dict) else fn or c.get("name")
        args = c.get("arguments")
        if args is None and isinstance(fn, dict):
            args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args or {}, ensure_ascii=False, default=str)
        out.append(_text(f"{name}({args})", 1500))
    return out


def _msg(role: str, content: Any, calls: Any = None) -> dict[str, Any]:
    m: dict[str, Any] = {"role": role or "?", "content": _text(content)}
    tc = _calls(calls)
    if tc:
        m["tool_calls"] = tc
    return m


# --------------------------------------------------------------------------- #
# per-harness extractors (each returns None when its raw output is missing)
# --------------------------------------------------------------------------- #
def _inspect_status(value: Any) -> tuple[str, float | None]:
    if value in ("C", "c", True):
        return "pass", 1.0
    if value in ("I", "i", False, "N", "P"):
        return ("fail", 0.0) if value != "P" else ("fail", 0.5)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "fail", None
    return ("pass" if v >= 1.0 else "fail"), v


def _inspect(rd: RunDir) -> list[Sample] | None:
    p = rd.path / "raw" / "log_dump.json"
    if not p.exists():
        return None
    log = json.loads(p.read_text())
    sys_default = ((log.get("plan") or {}).get("config") or {}).get("system_message") or ""
    out = []
    for s in log.get("samples") or []:
        inp = s.get("input")
        msgs_in = inp if isinstance(inp, list) else [{"role": "user", "content": inp}]
        system = "\n".join(_text(m.get("content")) for m in msgs_in if m.get("role") == "system")
        prompt = "\n".join(_text(m.get("content")) for m in msgs_in if m.get("role") == "user")
        if s.get("choices"):
            prompt += "\n\n" + "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(s["choices"]))
        scores = s.get("scores") or {}
        sc = next(iter(scores.values()), {}) if scores else {}
        status, score = _inspect_status(sc.get("value"))
        if s.get("error"):
            status = "error"
        output = s.get("output") or {}
        response = output.get("completion") or ""
        conv = [
            _msg(m.get("role"), m.get("content"), m.get("tool_calls"))
            for m in s.get("messages") or []
        ]
        if not response:
            last = next((m for m in reversed(conv) if m["role"] == "assistant"), None)
            if last:
                response = last["content"] or "\n".join(last.get("tool_calls", []))
        target = s.get("target")
        meta = s.get("metadata") or {}
        epoch = int(s.get("epoch") or 1)
        out.append(
            Sample(
                id=str(s.get("id")) + (f" (try {epoch})" if epoch > 1 else ""),
                status=status,
                score=score,
                category=str(
                    meta.get("category_name") or meta.get("subject") or meta.get("level") or ""
                ),
                system=system or _text(sys_default),
                prompt=prompt.strip(),
                response=_text(response),
                expected=_text(", ".join(target) if isinstance(target, list) else target),
                reason=_text(sc.get("explanation") or sc.get("answer") or s.get("error") or ""),
                messages=conv,
            )
        )
    return out


def _tau2(rd: RunDir) -> list[Sample] | None:
    hits = sorted((rd.path / "raw").glob("tau2_*/results.json"))
    if not hits:
        return None
    r = json.loads(hits[-1].read_text())
    info = r.get("info") or {}
    policy = _text(((info.get("environment_info") or {}).get("policy")) or "")
    tasks = {str(t.get("id")): t for t in r.get("tasks") or []}
    out = []
    for s in r.get("simulations") or []:
        tid = str(s.get("task_id"))
        task = tasks.get(tid, {})
        scenario = (task.get("user_scenario") or {}).get("instructions") or task.get("description")
        reward = (s.get("reward_info") or {}).get("reward")
        term = str(s.get("termination_reason") or "")
        status = (
            "error" if "error" in term else ("pass" if reward and float(reward) >= 1.0 else "fail")
        )
        conv = [
            _msg(m.get("role"), m.get("content"), m.get("tool_calls"))
            for m in s.get("messages") or []
        ]
        info_r = s.get("reward_info") or {}
        why = [f"ended by: {term}"] if term else []
        for k in ("db_check", "action_checks", "communicate_checks", "nl_assertions"):
            v = info_r.get(k)
            if v:
                why.append(f"{k}: {_text(v, 800)}")
        out.append(
            Sample(
                id=f"{tid}" + (f" (try {int(s.get('trial')) + 1})" if s.get("trial") else ""),
                status=status,
                score=None if reward is None else float(reward),
                category=str((info.get("environment_info") or {}).get("domain_name") or ""),
                system=policy,
                prompt=_text(scenario),
                response=next(
                    (m["content"] for m in reversed(conv) if m["role"] == "assistant"), ""
                ),
                expected=_text((task.get("evaluation_criteria") or {}), 3000),
                reason="\n".join(why),
                messages=conv,
            )
        )
    return out


def _harbor_conversation(trial: Path) -> tuple[str, str, str, list[dict[str, Any]]]:
    """(agent instructions, task, final answer, conversation) from the agent's trajectory."""
    p = trial / "agent" / "trajectory.json"
    if not p.exists():
        return "", "", "", []
    steps = json.loads(p.read_text()).get("steps") or []
    system = prompt = response = ""
    conv = []
    for i, st in enumerate(steps):
        text = st.get("message") or ""
        if st.get("source") == "user":
            if i == 0 and TASK_MARK in text:  # Terminus puts its instructions and the task in one
                head, _, task = text.partition(TASK_MARK)
                system, prompt = head.strip(), task.split(SCREEN_MARK, 1)[0].strip()
                conv.append(_msg("user", prompt))
            else:
                conv.append(_msg("user", text))
            continue
        calls = []
        for c in st.get("tool_calls") or []:
            args = c.get("arguments") or {}  # shell commands: show just the keystrokes
            calls.append(
                {"name": c.get("function_name"), "arguments": args.get("keystrokes") or args}
            )
        conv.append(_msg("assistant", text, calls))
        if text:
            response = text
        seen = [r.get("content") for r in (st.get("observation") or {}).get("results") or []]
        if any(seen):
            conv.append(_msg("tool", "\n".join(str(x) for x in seen if x)))
    return _text(system), _text(prompt), _text(response), conv


def _test_failures(stdout: str) -> str:
    """The FAIL / ERROR blocks of a unittest or pytest log, else its last lines."""
    lines = stdout.splitlines()
    blocks, keep = [], False
    for ln in lines:
        if ln.startswith(("FAIL:", "ERROR:", "_____")):  # unittest / pytest failure headers
            keep = True
        elif ln.startswith(("Ran ", "=====")):
            keep = False
        if keep:
            blocks.append(ln)
    return "\n".join(blocks) if blocks else "\n".join(lines[-40:])


def _harbor_verdict(trial: Path, failed: bool) -> tuple[str, str]:
    """(what had to pass, why it passed or failed) from the verifier's own reports."""
    v = trial / "verifier"
    expected, reason = [], []
    report = v / "report.json"
    if report.exists():  # SWE-bench: tests the fix must make pass, and must keep passing
        r = next(iter(json.loads(report.read_text()).values()), {})
        ts = r.get("tests_status") or {}
        f2p, p2p = ts.get("FAIL_TO_PASS") or {}, ts.get("PASS_TO_PASS") or {}
        must = (f2p.get("success") or []) + (f2p.get("failure") or [])
        keep = (p2p.get("success") or []) + (p2p.get("failure") or [])
        expected.append("Tests the fix must make pass:\n" + "\n".join(f"- {t}" for t in must))
        if keep:
            expected.append(f"and {len(keep)} test(s) that already passed must keep passing.")
        if not r.get("patch_exists"):
            reason.append("The agent changed nothing, so there was no fix to test.")
        elif not r.get("patch_successfully_applied"):
            reason.append("The agent's changes could not be applied to the repository.")
        else:
            reason.append(
                f"Tests the fix had to make pass: {len(f2p.get('success') or [])} of {len(must)} pass."
            )
            reason += [f"  still failing: {t}" for t in f2p.get("failure") or []]
            reason.append(
                f"Tests that already passed: {len(p2p.get('success') or [])} of {len(keep)} still pass."
            )
            reason += [f"  now broken: {t}" for t in p2p.get("failure") or []]
    ctrf = v / "ctrf.json"
    if ctrf.exists():  # Terminal-Bench: pytest results of the task's checks
        tests = (json.loads(ctrf.read_text()).get("results") or {}).get("tests") or []
        expected.append("Checks that must pass:\n" + "\n".join(f"- {t.get('name')}" for t in tests))
        passed = sum(1 for t in tests if t.get("status") == "passed")
        reason.append(f"Checks passed: {passed} of {len(tests)}.")
        for t in tests:
            if t.get("status") != "passed":
                msg = (t.get("message") or t.get("trace") or "").strip()
                reason.append(f"  {t.get('status')}: {t.get('name')}" + (f"\n{msg}" if msg else ""))
    out = v / "test-stdout.txt"
    if failed and out.exists():
        reason.append("\nTest output:\n" + _test_failures(out.read_text(errors="replace")))
    return _text("\n".join(expected)), _text("\n".join(reason))


def _harbor(rd: RunDir) -> list[Sample] | None:
    files = sorted((rd.path / "raw").glob("**/result.json"))
    trials = []
    for p in files:
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "task_name" in d:
            trials.append((p.parent, d))
    if not trials:
        return None
    out = []
    for trial, t in trials:
        rewards = (t.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward", next(iter(rewards.values()), None)) if rewards else None
        exc = t.get("exception_info") or {}
        status = (
            "error"
            if exc and reward is None
            else ("pass" if reward and float(reward) >= 1 else "fail")
        )
        system, prompt, response, conv = _harbor_conversation(trial)
        name = str(t.get("task_name"))
        repo = name.split("__")[0] if "__" in name else ""  # SWE-bench ids: <repo>__<repo>-<n>
        expected, reason = _harbor_verdict(trial, failed=status != "pass")
        if exc:
            why = f"{exc.get('exception_type', '')}: {exc.get('exception_message', '')}"
            reason = why + ("\n\n" + reason if reason else "")
        out.append(
            Sample(
                id=name,
                status=status,
                score=None if reward is None else float(reward),
                category=repo,
                system=system,
                prompt=prompt or _text(t.get("instruction") or ""),
                response=response,
                expected=expected,
                reason=_text(reason),
                messages=conv,
            )
        )
    return out


def _autobench(rd: RunDir) -> list[Sample] | None:
    p = rd.path / "raw" / "results.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    out = []
    for t in d.get("tasks") or []:
        msgs = t.get("messages") or []
        conv = [_msg(m.get("role"), m.get("content"), m.get("tool_calls")) for m in msgs]
        failed = [a for a in t.get("assertion_results") or [] if not a.get("passed")]
        out.append(
            Sample(
                id=str(t.get("name") or t.get("id")),
                status="pass" if t.get("passed") else "fail",
                score=t.get("score"),
                category=str(t.get("id", "")).split(".")[0],
                system=next((m["content"] for m in conv if m["role"] == "system"), ""),
                prompt=next((m["content"] for m in conv if m["role"] == "user"), ""),
                response=next(
                    (m["content"] for m in reversed(conv) if m["role"] == "assistant"), ""
                ),
                reason=_text(
                    f"{t.get('assertions_passed')}/{t.get('assertions_total')} checks passed"
                    + "".join(f"\nfailed: {_text(a, 500)}" for a in failed[:10])
                ),
                messages=conv,
            )
        )
    return out


def _ttft(rd: RunDir) -> list[Sample] | None:
    from benchlab.adapters.ttft import PROMPTS

    p = rd.path / "raw" / "samples.jsonl"
    if not p.exists():
        return None
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ok = bool(r.get("ok")) and r.get("ttft_s") is not None
        i = r.get("prompt")
        out.append(
            Sample(
                id=str(r.get("id")),
                status="pass" if ok else "error",
                score=r.get("ttft_s"),
                prompt=_text(PROMPTS[i]) if isinstance(i, int) and i < len(PROMPTS) else "",
                reason=(
                    f"first token after {r['ttft_s']:.2f} s, full answer after {r.get('latency_s') or 0:.2f} s"
                    if ok
                    else _text(r.get("error") or r.get("status"))
                ),
            )
        )
    return out


def _redteam(rd: RunDir) -> list[Sample] | None:
    from benchlab.adapters.redteam import verdict

    files = sorted((rd.path / "raw").glob("*.csv"))
    if not files:
        return None
    with open(files[-1], newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    status_of = {"pass": "pass", "fail": "fail", "error": "error"}
    return [
        Sample(
            id=str(r.get("Test ID") or i),
            status=status_of.get(verdict(r), "fail"),
            score=_float(r.get("Score")),
            category=f"{r.get('Plugin ID', '')} · {r.get('Strategy', '')}".strip(" ·"),
            prompt=_text(r.get("Prompt")),
            response=_text(r.get("Response")),
            reason=_text(r.get("Reason")),
        )
        for i, r in enumerate(rows)
    ]


def _apptasks(rd: RunDir) -> list[Sample] | None:
    p = rd.path / "raw" / "cases.jsonl"
    if not p.exists():
        return None
    return [Sample(**json.loads(line)) for line in p.read_text().splitlines() if line.strip()]


def _float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _from_per_task(result: dict[str, Any]) -> list[Sample]:
    """Fallback: the adapter's own per-task summary in result.json (no prompts)."""
    out = []
    for t in result.get("per_task") or []:
        if "passed" in t:
            status = "pass" if t["passed"] else "fail"
        elif t.get("pass@1") is not None:
            status = "pass" if float(t["pass@1"]) >= 0.5 else "fail"
        elif t.get("reward") is not None:
            status = "pass" if float(t["reward"]) >= 1 else "fail"
        else:
            status = "error" if t.get("error") else "fail"
        extra = {k: v for k, v in t.items() if k not in ("id", "passed")}
        out.append(
            Sample(
                id=str(t.get("id")),
                status=status,
                category=str(t.get("language") or t.get("difficulty") or ""),
                reason=_text(json.dumps(extra, ensure_ascii=False, default=str), 1500),
            )
        )
    return out


EXTRACTORS = {
    **{b: _inspect for b in INSPECT},
    "t2air": _tau2,
    "t2ret": _tau2,
    "t2tel": _tau2,
    "tb": _harbor,
    "swe": _harbor,
    "autobench": _autobench,
    "ttft": _ttft,
    "redteam": _redteam,
    "apptasks": _apptasks,
}


def extract(rd: RunDir, bench: str, result: dict[str, Any]) -> tuple[list[Sample], str]:
    """(samples, source) where source says how much detail there is."""
    fn = EXTRACTORS.get(bench)
    if fn is not None:
        try:
            got = fn(rd)
        except Exception as e:  # noqa: BLE001 - a broken raw file must not break the page
            got = None
            result = {**result, "warnings": [*(result.get("warnings") or []), f"detail: {e}"]}
        if got is not None:
            return got, "harness"
    return _from_per_task(result), "summary"


def load(rd: RunDir) -> dict[str, Any]:
    """Per-test detail for a finished run, cached next to result.json."""
    cache = rd.path / CACHE
    res_path = rd.path / RunDir.RESULT
    if cache.exists() and res_path.exists() and cache.stat().st_mtime >= res_path.stat().st_mtime:
        try:
            cached = json.loads(cache.read_text())
            if cached.get("version") == CACHE_VERSION:
                return cached
        except json.JSONDecodeError:
            pass
    result = rd.read_result() if res_path.exists() else {}
    bench = result.get("bench") or rd.read_config().get("bench", "")
    samples, source = extract(rd, bench, result)
    counts = {k: sum(1 for s in samples if s.status == k) for k in ("pass", "fail", "error")}
    cats: dict[str, dict[str, int]] = {}
    for s in samples:
        if s.category:
            c = cats.setdefault(s.category, {"pass": 0, "fail": 0, "error": 0})
            c[s.status] = c.get(s.status, 0) + 1
    data = {
        "version": CACHE_VERSION,
        "source": source,
        "counts": counts,
        "categories": dict(sorted(cats.items())),
        "samples": [asdict(s) for s in samples],
    }
    if res_path.exists():
        try:
            cache.write_text(json.dumps(data, ensure_ascii=False))
        except OSError:
            pass
    return data
