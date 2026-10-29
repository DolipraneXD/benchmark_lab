"""Add / edit / delete models in a models.yaml without losing its comments.

The file is round-tripped with ruamel.yaml (comments, key order, quoting kept).
Every change is validated with BenchLab's own ``BenchConfig`` before it is written,
a ``.bak`` copy of the previous file is kept, and the write is atomic.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from benchlab.config import BenchConfig

# fields the form edits; anything else on a model (published, serving, tracker_id, ...) is kept
EDITABLE = (
    "id",
    "name",
    "provider",
    "base_url",
    "model",
    "api_version",
    "region",
    "api_key_env",
    "kind",
    "self_hosted",
    "price_in_per_mtok",
    "price_out_per_mtok",
    "max_concurrency",
    "suites",
    "extras",
    "extra_params",
    "knowledge_cutoff",
    "notes",
)


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096
    y.indent(mapping=2, sequence=4, offset=2)
    y.representer.add_representer(  # keep `key: null` as written, not `key:`
        type(None), lambda r, _: r.represent_scalar("tag:yaml.org,2002:null", "null")
    )
    return y


def _load(path: Path) -> Any:
    return _yaml().load(path.read_text(encoding="utf-8"))


def _dump(doc: Any) -> str:
    buf = io.StringIO()
    _yaml().dump(doc, buf)
    return buf.getvalue()


def _validate(text: str) -> None:
    raw = yaml.safe_load(text) or {}
    try:
        BenchConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(_plain(e)) from None


def _plain(e: ValidationError) -> str:
    """Pydantic's report as sentences: the message, plus the field it is about."""
    out = []
    for err in e.errors():
        msg = str(err.get("msg", "")).removeprefix("Value error, ")
        field = [str(x) for x in err.get("loc", ())[2:]]  # ("models", 3, "price_in_per_mtok")
        out.append(f"{' '.join(field)}: {msg}" if field else msg)
    return "; ".join(out)


# settings that name a model; a model holding one of these can't be deleted or renamed
ROLES = {
    "baseline_model": "baseline (every score is compared to it)",
    "grader_model": "BrowseComp grader",
    "tau2_user_model": "τ²-bench user simulator",
}


def uses(doc: Any, model_id: str) -> list[str]:
    """What in the file points at ``model_id``: settings roles and other models' compare_to."""
    settings = doc.get("settings") or {}
    out = [label for key, label in ROLES.items() if settings.get(key) == model_id]
    out += [
        f"comparison target of {m.get('id')} (compare_to)"
        for m in doc.get("models") or []
        if m.get("compare_to") == model_id and m.get("id") != model_id
    ]
    return out


def _in_use(doc: Any, model_id: str, action: str) -> None:
    u = uses(doc, model_id)
    if u:
        raise ValueError(
            f"{model_id} can't be {action}: it is the {' and the '.join(u)}. "
            "Choose another model for that role first, then try again."
        )


def _write(path: Path, text: str) -> None:
    shutil.copy2(path, path.with_name(path.name + ".bak"))
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        shutil.copymode(path, tmp)  # mkstemp files are 0600; keep the original permissions
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _flow(v: Any) -> Any:
    """Small dicts/lists in flow style ({a: 1}, [x, y]) like the hand-written file."""
    if isinstance(v, dict):
        m = CommentedMap(v)
        m.fa.set_flow_style()
        return m
    if isinstance(v, list):
        from ruamel.yaml.comments import CommentedSeq

        s = CommentedSeq(v)
        s.fa.set_flow_style()
        return s
    return v


def save_model(path: str | Path, data: dict[str, Any], original_id: str | None = None) -> None:
    """Create (``original_id`` None) or update a model. ``data`` holds EDITABLE keys;
    a value of None removes the key."""
    path = Path(path)
    doc = _load(path)
    models = doc["models"]
    ids = [m.get("id") for m in models]
    if original_id is None:
        if data.get("id") in ids:
            raise ValueError(f"model id {data.get('id')!r} already exists")
        entry = CommentedMap()
        models.append(entry)
    else:
        if original_id not in ids:
            raise ValueError(f"unknown model id {original_id!r}")
        if data.get("id") != original_id and data.get("id") in ids:
            raise ValueError(f"model id {data.get('id')!r} already exists")
        if "id" in data and data["id"] != original_id:
            _in_use(doc, original_id, "renamed")
        entry = models[ids.index(original_id)]
    for k in EDITABLE:
        if k not in data:
            continue
        v = data[k]
        if v is None:
            entry.pop(k, None)
        else:
            entry[k] = _flow(v)
    text = _dump(doc)
    _validate(text)
    _write(path, text)


def delete_model(path: str | Path, model_id: str) -> None:
    path = Path(path)
    doc = _load(path)
    models = doc["models"]
    ids = [m.get("id") for m in models]
    if model_id not in ids:
        raise ValueError(f"unknown model id {model_id!r}")
    _in_use(doc, model_id, "deleted")
    del models[ids.index(model_id)]
    text = _dump(doc)
    _validate(text)
    _write(path, text)


def save_roles(path: str | Path, baseline: str, grader: str | None, user_model: str) -> None:
    """Set the models that play the baseline, grader and user-simulator roles.
    ``grader`` None removes grader_model, so the baseline grades (BenchLab's default)."""
    path = Path(path)
    doc = _load(path)
    if doc.get("settings") is None:
        doc.insert(0, "settings", CommentedMap())
    s = doc["settings"]
    s["baseline_model"] = baseline
    s["tau2_user_model"] = user_model
    if grader is None:
        s.pop("grader_model", None)
    else:
        s["grader_model"] = grader
    text = _dump(doc)
    _validate(text)
    _write(path, text)


def raw_model(path: str | Path, model_id: str) -> dict[str, Any]:
    """The model entry as written in the file (not the defaults pydantic fills in)."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for m in doc.get("models", []):
        if m.get("id") == model_id:
            return m
    raise KeyError(model_id)
