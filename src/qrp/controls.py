"""Negative controls: experiments whose correct answer is known in advance.

If the evaluation machinery is sound, destroying the information in the data must destroy the
measured signal, a deliberately leaky feature must be caught, and an evaluation that ignores
label overlap must be visibly fooled where the purged one is not. Each control has a fixed
pass criterion and the whole set is reported next to the real result.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from qrp.backtest import BookTimeline, run_backtest, threshold_target
from qrp.experiment import (
    ExperimentConfig,
    Prepared,
    block_len,
    design_matrix,
    load_run_config,
    prepare,
    select_model,
    strategy_thresholds,
    usable_rows,
)
from qrp.features import available_time, build_features
from qrp.models import ModelSpec, fit
from qrp.splits import walk_forward
from qrp.stats import evaluate, rank_ic

NS = 1_000_000_000


def _test_ic(
    cfg: ExperimentConfig,
    dev: pl.DataFrame,
    test: pl.DataFrame,
    names: list[str],
    forced: ModelSpec | None = None,
) -> tuple[str, Any]:
    """Test-period rank IC of the model walk-forward selects, or of ``forced`` refit on dev."""
    chosen = forced or select_model(dev, cfg, names)[0]
    model = fit(chosen, *design_matrix(dev, chosen))
    pred = model.predict(design_matrix(test, chosen)[0])
    y = test["fwd_ret"].to_numpy()
    return chosen.name, evaluate(pred, y, block_len(cfg), cfg.evaluation.n_boot, cfg.seed).rank_ic


def _covers_zero(ci: Any) -> bool:
    return bool(ci.low <= 0 <= ci.high)


def _permuted(df: pl.DataFrame, cols: list[str], rng: np.random.Generator) -> pl.DataFrame:
    order = rng.permutation(df.height)
    return df.with_columns(df.select(cols)[order.tolist()])


def truncation_violations(
    prep: Prepared,
    cfg: ExperimentConfig,
    extra: Callable[[pl.DataFrame], pl.DataFrame] = lambda f: f,
    cuts: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> list[str]:
    """Columns whose values at t <= T change when every event *available* after T is deleted.

    Truncation uses the availability clock (receive time plus configured latency, or the
    assumed latency), so a leak shorter than the latency cannot hide in the gap between the
    raw timestamp and the research clock. Several cut points within the first session.
    """
    session = prep.features["session"].min()
    caps = prep.manifest.capabilities
    events = prep.events.with_columns(available_time(caps, cfg.timing))
    full = extra(prep.features.filter(pl.col("session") == session))
    columns = [c for c in full.columns if c not in ("fwd_ret", "label_end_ns")]
    flagged: set[str] = set()
    for fraction in cuts:
        cut = int(full["t_ns"][int(full.height * fraction)])
        kept = events.filter(pl.col("available_time_ns") <= cut).drop("available_time_ns")
        truncated = extra(build_features(kept, caps, cfg.features, cfg.timing))
        head = full.head(truncated.height)
        flagged |= {c for c in columns if not head[c].equals(truncated[c])}
    return sorted(flagged)


def run_controls(run_dir: Path) -> list[dict[str, Any]]:
    cfg = load_run_config(run_dir)
    prep = prepare(cfg)
    dev, test = usable_rows(prep, cfg)
    names = prep.feature_names
    rng = np.random.default_rng(cfg.seed)
    results: list[dict[str, Any]] = []

    def record(name: str, expectation: str, observed: str, passed: bool) -> None:
        results.append(
            {"name": name, "expectation": expectation, "observed": observed, "passed": passed}
        )

    # 1-3. Information-destroying transformations. Selection should fall back to the zero
    #      forecast, and the run's own model refit on the destroyed data must show a test IC
    #      whose 95% CI contains 0 (this exercises the interval, not just the selection).
    spec = ModelSpec.model_validate(json.loads((run_dir / "model.json").read_text())["spec"])
    forced = spec if spec.features else ModelSpec(name="ridge", kind="ridge", features=names)
    transforms: dict[str, Callable[[pl.DataFrame], pl.DataFrame]] = {
        "shuffled_labels": lambda d: _permuted(d, ["fwd_ret"], rng),
        "shuffled_features": lambda d: _permuted(d, names, rng),
        "time_rotated_labels": lambda d: d.with_columns(
            pl.Series("fwd_ret", np.roll(d["fwd_ret"].to_numpy(), d.height // 2))
        ),
    }
    for name, transform in transforms.items():
        d, t_ = transform(dev), transform(test)
        selected, sel_ic = _test_ic(cfg, d, t_, names)
        _, ic = _test_ic(cfg, d, t_, names, forced)
        record(
            name,
            "selection returns no signal; forced model's test IC CI contains 0",
            f"selected {selected} (IC {sel_ic.estimate:.4f}); {forced.name} IC "
            f"{ic.estimate:.4f} [{ic.low:.4f}, {ic.high:.4f}]",
            _covers_zero(sel_ic) and _covers_zero(ic),
        )

    # 4. Leakage canary: a feature built from the future label must fail truncation invariance,
    #    and no real feature may.
    real = truncation_violations(prep, cfg)
    canary = truncation_violations(
        prep, cfg, lambda f: f.with_columns(leak_canary=pl.col("fwd_ret"))
    )
    leak_dev = dev.with_columns(leak_canary=pl.col("fwd_ret"))
    leak_test = test.with_columns(leak_canary=pl.col("fwd_ret"))
    _, leak_ic = _test_ic(cfg, leak_dev, leak_test, [*names, "leak_canary"])
    record(
        "leakage_canary",
        "truncation test flags only the canary",
        f"flagged real={real}, canary run={canary}; canary test IC {leak_ic.estimate:.3f}",
        real == [] and canary == ["leak_canary"],
    )

    # 5. Label overlap: a nearest-in-time "model" scores well under shuffled K-fold without
    #    purging (neighbours share 90% of their label window) and not under purged walk-forward.
    t = dev["t_ns"].to_numpy()
    y = dev["fwd_ret"].to_numpy()
    order = rng.permutation(len(t))
    kfold_pred = np.empty(len(t))
    for part in np.array_split(order, 5):
        train = np.setdiff1d(np.arange(len(t)), part)
        nearest = np.clip(np.searchsorted(t[train], t[part]), 1, len(train) - 1)
        left_closer = (t[part] - t[train][nearest - 1]) <= (t[train][nearest] - t[part])
        kfold_pred[part] = y[train][np.where(left_closer, nearest - 1, nearest)]
    folds = walk_forward(
        t, dev["label_end_ns"].to_numpy(), cfg.evaluation.n_folds, cfg.evaluation.embargo_s * NS
    )
    wf_pred, wf_y = [], []
    for fold in folds:
        # Walk-forward trains only on the past, so the nearest training sample in time is the
        # last admissible one, at least horizon + embargo before the test block.
        train_t = t[fold.train]
        nearest = np.clip(np.searchsorted(train_t, t[fold.test]), 1, len(train_t)) - 1
        wf_pred.append(y[fold.train][nearest])
        wf_y.append(y[fold.test])
    wf_pred_all, wf_y_all = np.concatenate(wf_pred), np.concatenate(wf_y)
    kfold_ic, wf_ic = rank_ic(kfold_pred, y), rank_ic(wf_pred_all, wf_y_all)
    record(
        "label_overlap",
        "shuffled K-fold IC >> purged walk-forward IC",
        f"nearest-in-time model: shuffled 5-fold IC {kfold_ic:.3f}, purged walk-forward IC "
        f"{wf_ic:.3f}",
        kfold_ic > 0.3 and abs(wf_ic) < kfold_ic / 3,
    )

    # 6-7. Execution controls on the test period.
    preds = pl.read_parquet(run_dir / "predictions.parquet")
    signals = (
        pl.read_parquet(run_dir / "signals.parquet")
        .filter(pl.col("split") == "test")
        .select("t_ns", "session", "pred")
    )
    t = signals["t_ns"].to_numpy()
    lo, hi = int(t.min()), int(t.max())
    window = pl.col("available_time_ns").is_between(lo - 3600 * NS, hi + 3600 * NS)
    book = BookTimeline.from_quotes(prep.quotes.filter(window))
    rules = strategy_thresholds(cfg, preds.filter(pl.col("split") == "dev_oos"))
    target = threshold_target(min(rules.values()), cfg.backtest.position_size)
    shuffled = rng.permutation(signals["pred"].to_numpy())
    random_signals = signals.with_columns(pl.Series("pred", shuffled))
    rand = run_backtest(book, random_signals, cfg.backtest, target, cfg.seed).summary
    costs = rand["spread_cost"] + rand["latency_cost"] + rand["fees"]
    record(
        "random_signal_backtest",
        "net P&L < 0 (pays costs, no edge)",
        f"edge {rand['edge_at_decision']:.2f}, costs {costs:.2f}, net {rand['pnl_total']:.2f}",
        rand["fills"] > 0 and rand["pnl_total"] < 0,
    )
    null = run_backtest(book, signals, cfg.backtest, lambda _: 0.0, cfg.seed).summary
    record(
        "null_strategy",
        "no fills, P&L exactly 0",
        f"fills {null['fills']}, P&L {null['pnl_total']}",
        null["fills"] == 0 and null["pnl_total"] == 0.0,
    )
    (run_dir / "controls.json").write_text(json.dumps(results, indent=2) + "\n")
    return results
