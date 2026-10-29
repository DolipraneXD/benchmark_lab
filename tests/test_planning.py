"""Run plan: cost and time estimates, and the switch from a typical pace to the pace a
model measured on its last run of the same benchmark."""

from benchlab import registry
from benchlab.config import load_config
from benchlab.planning import SETUP_SECONDS, TYPICAL_SECONDS, fmt_duration, plan_rows
from benchlab.runs import ErrorCounts, NormalizedResult, RunConfig, RunDir


def _finished_run(cfg, model_id, bench, n, wall):
    rd = RunDir.create(cfg.settings.runs_dir, model_id, bench)
    rd.write_config(
        RunConfig(
            model_id=model_id,
            bench=bench,
            model={},
            request_params={},
            per_model_exceptions={},
            harness={},
            subset={},
            trials=1,
            smoke=False,
        )
    )
    rd.write_result(
        NormalizedResult(
            bench=bench,
            model_id=model_id,
            metrics={"accuracy": 50.0},
            primary_metric="accuracy",
            n_tasks=n,
            trials=1,
            errors=ErrorCounts(),
            wall_time_s=wall,
        )
    )


def test_time_estimate_typical_then_measured(models_yaml):
    cfg = load_config(models_yaml)
    mock = cfg.get("mock")  # max_concurrency 4
    gpqa = registry.get("gpqa")
    (row,) = plan_rows(cfg, [mock], [gpqa], None, None, smoke=True)
    attempts = row.n * row.trials
    want = SETUP_SECONDS + TYPICAL_SECONDS["gpqa"] * attempts / min(4, attempts)
    assert row.time_source == "typical" and row.est_s == want

    _finished_run(cfg, "mock", "gpqa", n=10, wall=200.0)  # 20 s per question
    (row,) = plan_rows(cfg, [mock], [gpqa], None, None, smoke=True)
    assert row.time_source == "last run" and row.est_s == SETUP_SECONDS + 20.0 * attempts


def test_fmt_duration():
    assert fmt_duration(None) == "-" and fmt_duration(45) == "45s"
    assert fmt_duration(750) == "12m30s" and fmt_duration(3900) == "1h05m"
