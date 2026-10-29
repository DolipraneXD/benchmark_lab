"""App tasks: your own test cases, written in YAML under ``evals/``, run straight against
the model. Each case is a system prompt and a user prompt (or a conversation), optional
documents for RAG and tools, and checks on the answer. A case passes when every check
passes; with ``--trials N`` each case runs N times and a case only counts as consistent
when all N pass.

Checks (all optional, combine freely):
  contains / contains_any / not_contains   text in the answer (case-insensitive)
  regex                                     pattern that must match
  equals                                    the whole answer (trimmed, case-insensitive)
  max_words                                 length limit
  json: true                                answer is JSON (``` fences allowed)
  json_includes: {key: value}               JSON contains these values
  json_keys: [k1, k2]                       JSON has these keys
  tool_call: name / tool_args: {k: v}       the model called this tool with these args
  no_tool_call: true                        the model answered without calling a tool
  judge: "rubric"                           a grader model decides (grader_model or baseline;
                                            skipped, with a warning, when that is the model
                                            under test)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from benchlab.adapters.base import BenchAdapter, HarnessNotReady, RunPlan
from benchlab.config import MissingKeyError, ModelSpec
from benchlab.harness_env import building_image
from benchlab.llm import LLMClient
from benchlab.redact import redact
from benchlab.runs import ErrorCounts, NormalizedResult, RunDir
from benchlab.samples import Sample, _text
from benchlab.usage import BudgetGuard

CASES_FILE = "cases.jsonl"
CHECKS = {
    "contains",
    "contains_any",
    "not_contains",
    "regex",
    "equals",
    "max_words",
    "json",
    "json_includes",
    "json_keys",
    "tool_call",
    "tool_args",
    "no_tool_call",
    "judge",
}
SMOKE_CASES = 4


def load_cases(folder: Path) -> list[dict[str, Any]]:
    """All cases from ``folder/*.yaml``, ids unique, in file then list order."""
    cases: list[dict[str, Any]] = []
    for p in sorted(folder.glob("*.y*ml")):
        doc = yaml.safe_load(p.read_text(encoding="utf-8")) or []
        items = doc.get("cases", []) if isinstance(doc, dict) else doc
        for c in items:
            if not isinstance(c, dict) or "id" not in c:
                raise ValueError(f"{p.name}: every case needs an id")
            if "prompt" not in c and "messages" not in c:
                raise ValueError(f"{p.name}: case {c['id']} needs a prompt or messages")
            unknown = set(c.get("checks") or {}) - CHECKS
            if unknown:
                raise ValueError(f"{p.name}: case {c['id']} has unknown checks {sorted(unknown)}")
            if not c.get("checks"):
                raise ValueError(f"{p.name}: case {c['id']} has no checks")
            cases.append({**c, "id": str(c["id"]), "_file": p.name})
    ids = [c["id"] for c in cases]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"duplicate case ids: {dup}")
    return cases


def messages_for(case: dict[str, Any]) -> list[dict[str, Any]]:
    system = case.get("system") or ""
    if case.get("documents"):
        docs = "\n\n".join(
            f"[{i + 1}] {d}"
            if isinstance(d, str)
            else f"[{i + 1}] {d.get('title', '')}\n{d.get('text', '')}"
            for i, d in enumerate(case["documents"])
        )
        system = (system + "\n\n" if system else "") + "Documents:\n" + docs
    msgs = [{"role": "system", "content": system}] if system else []
    if case.get("messages"):
        msgs += [dict(m) for m in case["messages"]]
    else:
        msgs.append({"role": "user", "content": case["prompt"]})
    return msgs


def _json_of(text: str) -> Any:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    return json.loads(t)


def _matches(want: Any, got: Any) -> bool:
    if isinstance(want, dict):
        return isinstance(got, dict) and all(
            k in got and _matches(v, got[k]) for k, v in want.items()
        )
    if isinstance(want, str) and isinstance(got, str):
        return want.strip().lower() == got.strip().lower()
    if (
        isinstance(want, (int, float))
        and isinstance(got, (int, float))
        and not isinstance(got, bool)
    ):
        return abs(float(want) - float(got)) < 1e-6
    return want == got


def _as_list(v: Any) -> list[str]:
    return [str(x) for x in (v if isinstance(v, list) else [v])]


def check(case: dict[str, Any], answer: str, calls: list[Any]) -> list[str]:
    """Failed checks as sentences (empty list = pass). ``judge`` is handled separately."""
    c = case.get("checks") or {}
    low = answer.lower()
    fails = []
    for s in _as_list(c["contains"]) if "contains" in c else []:
        if s.lower() not in low:
            fails.append(f"answer should contain {s!r}")
    if "contains_any" in c and not any(s.lower() in low for s in _as_list(c["contains_any"])):
        fails.append(f"answer should contain one of {_as_list(c['contains_any'])}")
    for s in _as_list(c["not_contains"]) if "not_contains" in c else []:
        if s.lower() in low:
            fails.append(f"answer must not contain {s!r}")
    if "regex" in c and not re.search(c["regex"], answer, re.S | re.I):
        fails.append(f"answer should match /{c['regex']}/")
    if "equals" in c and answer.strip().lower() != str(c["equals"]).strip().lower():
        fails.append(f"answer should be exactly {c['equals']!r}")
    if "max_words" in c and len(answer.split()) > int(c["max_words"]):
        fails.append(f"answer has {len(answer.split())} words, limit {c['max_words']}")
    if c.get("json") or "json_includes" in c or "json_keys" in c:
        try:
            data = _json_of(answer)
        except (json.JSONDecodeError, ValueError):
            fails.append("answer should be valid JSON")
        else:
            if "json_includes" in c and not _matches(c["json_includes"], data):
                fails.append(
                    f"JSON should include {json.dumps(c['json_includes'], ensure_ascii=False)}"
                )
            missing = [
                k for k in c.get("json_keys") or [] if not (isinstance(data, dict) and k in data)
            ]
            if missing:
                fails.append(f"JSON is missing keys {missing}")
    if "tool_call" in c:
        hit = [t for t in calls if t.name == c["tool_call"]]
        if not hit:
            got = [t.name for t in calls] or "no tool"
            fails.append(f"should call {c['tool_call']}, called {got}")
        elif "tool_args" in c and not any(_matches(c["tool_args"], t.arguments or {}) for t in hit):
            fails.append(
                f"{c['tool_call']} arguments should include {json.dumps(c['tool_args'], ensure_ascii=False)},"
                f" got {hit[0].arguments_raw}"
            )
    if c.get("no_tool_call") and calls:
        fails.append(f"should answer without tools, called {[t.name for t in calls]}")
    return fails


JUDGE_PROMPT = """You grade an AI assistant's answer against a rubric.
Rubric: {rubric}

Conversation given to the assistant:
{conversation}

Assistant's answer:
{answer}

Reply with PASS or FAIL on the first line, then one sentence explaining why."""


class AppTasksAdapter(BenchAdapter):
    key = "apptasks"
    display = "App tasks"
    stage = 1
    needs_proxy = False  # direct LiteLLM calls, logged to usage.jsonl
    tokens_in_per_task = 1500
    tokens_out_per_task = 400

    @property
    def folder(self) -> Path:
        return Path(self.cfg.settings.apptasks_dir)

    def cases(self, smoke: bool = False) -> list[dict[str, Any]]:
        cases = load_cases(self.folder) if self.folder.exists() else []
        return cases[:SMOKE_CASES] if smoke else cases

    def prepare(self) -> dict[str, Any]:
        if building_image():  # nothing to install; the cases are mounted at run time
            return {"name": "benchlab-apptasks"}
        try:
            cases = self.cases()
        except (ValueError, yaml.YAMLError) as e:
            raise HarnessNotReady(f"app tasks: {e}") from e
        if not cases:
            raise HarnessNotReady(f"app tasks: no cases in {self.folder}/*.yaml")
        files = sorted({c["_file"] for c in cases})
        digest = json.dumps(
            [{k: v for k, v in c.items() if k != "_file"} for c in cases], sort_keys=True
        )
        import hashlib

        return {
            "name": "benchlab-apptasks",
            "folder": str(self.folder),
            "files": files,
            "cases": len(cases),
            "cases_sha256": hashlib.sha256(digest.encode()).hexdigest()[:16],
        }

    def plan(self, model: ModelSpec, subset: str | None, trials: int, smoke: bool) -> RunPlan:
        try:
            cases = self.cases(smoke)
        except (ValueError, yaml.YAMLError) as e:
            return RunPlan(0, trials, None, 0, 0, [f"cases unreadable: {e}"])
        cats = sorted({c.get("category", "general") for c in cases})
        return RunPlan(
            len(cases),
            trials,
            [c["id"] for c in cases],
            self.tokens_in_per_task,
            self.tokens_out_per_task,
            [f"{len(cases)} cases from {self.folder}/ ({', '.join(cats)})"],
        )

    def grader(self, model: ModelSpec) -> ModelSpec | None:
        gid = self.cfg.settings.grader_model or self.cfg.settings.baseline_model
        if gid == model.id:
            return None
        try:
            return self.cfg.get(gid)
        except KeyError:
            return None

    def run(self, model, run_dir, subset, trials, smoke, budget: BudgetGuard | None = None) -> Path:
        try:
            model.api_key()
        except MissingKeyError as e:
            raise HarnessNotReady(str(e)) from e
        cases = self.cases(smoke)
        client = LLMClient(self.cfg.settings, usage_log=run_dir.usage_log)
        grader = self.grader(model)
        done = run_dir.completed_ids()
        out = run_dir.raw_dir / CASES_FILE
        lock = asyncio.Lock()

        async def one(case: dict[str, Any], attempt: int) -> None:
            sid = case["id"] + (f" (try {attempt + 1})" if trials > 1 else "")
            if sid in done:
                return
            msgs = messages_for(case)
            res = await client.acomplete(  # bounded by the model's max_concurrency
                model,
                msgs,
                tools=case.get("tools"),
                call_type="apptask",
                tags={"case": sid},
                **({"max_tokens": int(case["max_tokens"])} if case.get("max_tokens") else {}),
            )
            answer = res.content or ""
            calls = res.tool_calls
            if not res.ok and res.status != "malformed_tool_call":
                status, reasons = (
                    "error",
                    [f"{res.status}: {res.error_msg or res.error_type or ''}"],
                )
            else:
                reasons = check(case, answer, calls)
                rubric = (case.get("checks") or {}).get("judge")
                if rubric and not reasons and grader is not None:  # no grader: see parse()
                    verdict = await self.judge(client, grader, rubric, msgs, answer, calls)
                    if verdict:
                        reasons.append(verdict)
                status = "fail" if reasons else "pass"
            shown = answer or "\n".join(f"{t.name}({t.arguments_raw})" for t in calls)
            conv = [{"role": m["role"], "content": _text(m.get("content"))} for m in msgs]
            conv.append(
                {
                    "role": "assistant",
                    "content": _text(answer),
                    **(
                        {"tool_calls": [f"{t.name}({t.arguments_raw})" for t in calls]}
                        if calls
                        else {}
                    ),
                }
            )
            sample = Sample(
                id=sid,
                status=status,
                score=1.0 if status == "pass" else 0.0,
                category=str(case.get("category") or "general"),
                system=_text(next((m["content"] for m in msgs if m["role"] == "system"), "")),
                prompt=_text(
                    case.get("prompt") or json.dumps(case.get("messages"), ensure_ascii=False)
                ),
                response=_text(shown),
                expected=_text(
                    yaml.safe_dump(case.get("checks") or {}, allow_unicode=True, sort_keys=False)
                ),
                reason="\n".join(redact(r) for r in reasons),
                messages=conv,
            )
            async with lock:
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
                run_dir.mark_done(sid, ok=status != "error")
                if budget is not None:
                    budget.add(res.cost_usd)

        async def go():
            await asyncio.gather(*(one(c, k) for c in cases for k in range(max(1, trials))))

        asyncio.run(go())
        return run_dir.raw_dir

    async def judge(self, client, grader, rubric, msgs, answer, calls) -> str | None:
        """None when the grader says PASS, else its reason."""
        convo = "\n".join(f"{m['role']}: {m.get('content')}" for m in msgs)
        shown = answer or "; ".join(f"{t.name}({t.arguments_raw})" for t in calls)
        res = await client.acomplete(
            grader,
            [
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(rubric=rubric, conversation=convo, answer=shown),
                }
            ],
            call_type="judge",
            max_tokens=200,
            temperature=0.0,
        )
        text = (res.content or "").strip()
        if not res.ok:
            return f"judge failed: {res.status}"
        first = text.splitlines()[0].upper() if text else ""
        if first.startswith("PASS"):
            return None
        return "judge: " + (text.split("\n", 1)[1].strip() if "\n" in text else text or "FAIL")

    def parse(self, run_dir: RunDir) -> NormalizedResult:
        cfg = run_dir.read_config()
        p = run_dir.raw_dir / CASES_FILE
        if not p.exists():
            raise FileNotFoundError(f"no app task results at {p}")
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        trials = int(cfg.get("trials") or 1)
        attempts = len(rows)
        passed = sum(r["status"] == "pass" for r in rows)
        errors = ErrorCounts(api_errors=sum(r["status"] == "error" for r in rows))
        by_cat: dict[str, list[bool]] = {}
        by_case: dict[str, list[bool]] = {}
        for r in rows:
            by_cat.setdefault(r["category"], []).append(r["status"] == "pass")
            by_case.setdefault(r["id"].split(" (try ")[0], []).append(r["status"] == "pass")
        metrics: dict[str, float | None] = {
            "pass_rate": round(100.0 * passed / attempts, 2) if attempts else None,
        }
        if trials > 1:
            metrics["consistent_cases"] = round(
                100.0 * sum(all(v) for v in by_case.values()) / len(by_case), 2
            )
        for cat, v in sorted(by_cat.items()):
            metrics[f"pass_rate_{cat}"] = round(100.0 * sum(v) / len(v), 2)
        usage = run_dir.usage_log.summarize(cfg["model_id"])
        warnings = []
        try:
            spec = self.cfg.get(cfg["model_id"])
            if self.grader(spec) is None and any(
                "judge" in (r.get("expected") or "") for r in rows
            ):
                warnings.append(
                    "judge checks were skipped: the grader would be the model under test "
                    "(set grader_model to another model)"
                )
        except KeyError:
            spec = None
        tid = spec.tracker if spec else cfg["model_id"]
        h = cfg.get("harness") or {}
        note = (
            f"benchlab app tasks: {len(by_case)} cases from {h.get('folder', 'evals')}/ "
            f"(sha {h.get('cases_sha256', '?')}), {trials} trial(s), pass rate over attempts"
        )
        if cfg.get("smoke"):
            note += "; SMOKE"
        return NormalizedResult(
            bench=self.key,
            model_id=cfg["model_id"],
            metrics=metrics,
            primary_metric="pass_rate",
            n_tasks=len(by_case),
            trials=trials,
            errors=errors,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
            cost_usd=usage.cost_usd if (usage.n_requests and usage.cost_known) else None,
            wall_time_s=run_dir.state().get("elapsed_s", 0.0)
            or time.time() - run_dir.state().get("started_at", time.time()),
            harness=h,
            note=note,
            tracker={f"{tid}__apptasks": {"v": f"{metrics['pass_rate'] or 0:.1f}", "note": note}},
            warnings=warnings,
        )
