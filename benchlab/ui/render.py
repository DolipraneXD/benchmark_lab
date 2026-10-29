"""Template set-up shared by the panel and the exported report.html, so both use the same
filters, the same benchmark hover cards and the same look."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from benchlab.bench_about import about
from benchlab.planning import fmt_duration
from benchlab.providers import PROVIDERS
from benchlab.redact import redact

HERE = Path(__file__).parent
TEMPLATES = HERE / "templates"
STATIC = HERE / "static"


def _usd(v: float | None) -> str:
    if v is None:
        return "-"
    return f"${v:.4f}" if v < 0.01 else f"${v:.2f}"


def install(env: Environment) -> Environment:
    env.filters["redact"] = lambda s: redact(str(s))
    env.filters["usd"] = _usd
    env.filters["num"] = lambda v, nd=1: "-" if v is None else f"{float(v):.{nd}f}"
    env.filters["ts"] = lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
    env.filters["dur"] = fmt_duration
    env.globals["about"] = about  # benchmark hover cards
    env.globals["providers"] = PROVIDERS
    return env


def standalone_env() -> Environment:
    """For files written outside the panel (report.html)."""
    return install(
        Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    )
