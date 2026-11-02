"""Config-driven experiments.

``run`` is the only place that touches the holdout test period, and it does so once, after the
model and every hyper-parameter have been chosen by walk-forward validation on the development
period. Each run gets its own directory; earlier runs of the same experiment are counted and
reported so that repeated peeking at the test set is visible.

Stages, each restartable from the run directory: run (select + predict) -> evaluate -> backtest.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import platform
import subprocess
import time
import tomllib
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel

from qrp.backtest import BacktestConfig, BookTimeline, run_backtest, threshold_target
from qrp.book import BookStats, trades_inside_stale_quote
from qrp.catalog import Catalog
from qrp.features import (
    FeatureConfig,
    TimingConfig,
    available_time,
    book_quotes,
    build_features,
    feature_docs,
    select_features,
)
from qrp.models import FittedModel, ModelSpec, fit
from qrp.schemas import TRADE, DatasetManifest
from qrp.splits import admissible, walk_forward
from qrp.stats import calibration_table, circular_shift_pvalue, evaluate, rank_ic

log = logging.getLogger(__name__)
NS = 1_000_000_000


class DataConfig(BaseModel):
    catalog: Path = Path("data/catalog")
    dataset: str
    version: str = "latest"
    symbol: str
    dev_dates: tuple[str, str]
    test_dates: tuple[str, str]
    # Knowledge time (ISO-8601, UTC): with version = "latest", use the newest version that had
    # been imported by then. Lets an experiment be replayed with only the data known on a date.
    as_of: str | None = None


class EvalConfig(BaseModel):
    n_folds: int = 5
    embargo_s: int = 30
    n_boot: int = 500
    block_len: int | None = None  # samples; None = 2 x max(horizon, longest window)
    n_null: int = 200
    regime_feature: str = "rv_30s"
    significance: float = 0.01  # null p-value required (with a CI excluding 0) to call a signal


class ModelGrid(BaseModel):
    name: str
    kind: Literal["zero", "mean", "ridge"]
    features: list[str] | Literal["all"] = []
    alphas: list[float] = [0.0]

    def specs(self, available: list[str]) -> list[ModelSpec]:
        feats = available if self.features == "all" else self.features
        if self.kind != "ridge":
            return [ModelSpec(name=self.name, kind=self.kind)]
        return [
            ModelSpec(
                name=self.name if len(self.alphas) == 1 else f"{self.name}[alpha={a:g}]",
                kind="ridge",
                features=list(feats),
                alpha=a,
            )
            for a in self.alphas
        ]


class StrategyConfig(BaseModel):
    """Threshold rules for turning forecasts into targets; all set from development data."""

    threshold: float | None = None  # fixed forecast threshold; None = cost rule
    threshold_cost_multiple: float = 1.0  # cost rule: multiple * round-trip cost estimate
    # Diagnostic second strategy: trade the top (1 - q) of |forecast| regardless of cost, to
    # show how much edge exists before costs and where it goes. None disables it.
    diagnostic_quantile: float | None = 0.8


class ExperimentConfig(BaseModel):
    name: str
    seed: int
    description: str = ""
    data: DataConfig
    timing: TimingConfig = TimingConfig()
    features: FeatureConfig = FeatureConfig()
    evaluation: EvalConfig = EvalConfig()
    models: list[ModelGrid]
    strategy: StrategyConfig = StrategyConfig()
    backtest: BacktestConfig = BacktestConfig()


def load_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(tomllib.loads(path.read_text()))


@dataclass
class Prepared:
    manifest: DatasetManifest
    events: pl.DataFrame
    quotes: pl.DataFrame
    features: pl.DataFrame
    feature_names: list[str]
    book_stats: BookStats
    quote_checks: dict[str, int]  # trades that printed inside the quote we held (stale book)


def prepare(cfg: ExperimentConfig) -> Prepared:
    """Load the pinned dataset version, rebuild the book and compute features."""
    catalog = Catalog(cfg.data.catalog)
    as_of_ns = None
    if cfg.data.as_of is not None:
        as_of = dt.datetime.fromisoformat(cfg.data.as_of)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=dt.UTC)
        as_of_ns = int(as_of.timestamp() * 1e9)
    manifest = catalog.resolve(cfg.data.dataset, cfg.data.version, as_of_ns)
    dates = sorted([*cfg.data.dev_dates, *cfg.data.test_dates])
    events = (
        catalog.scan(cfg.data.dataset, manifest.version, [cfg.data.symbol], (dates[0], dates[-1]))
        .drop("date")
        .collect()
        .sort("stream", "seq", maintain_order=True)
    )
    if events.is_empty():
        raise ValueError(f"no events for {cfg.data.symbol} in {dates[0]}..{dates[-1]}")
    caps = manifest.capabilities
    quotes, stats = book_quotes(events, caps, cfg.timing, cfg.features.depth_levels)
    features = build_features(events, caps, cfg.features, cfg.timing, quotes=quotes)
    names = select_features(cfg.features, caps)
    trades = (
        events.filter(pl.col("kind") == TRADE)
        .with_columns(available_time(caps, cfg.timing))
        .filter(pl.col("available_time_ns").is_not_null())
        .sort("available_time_ns", maintain_order=True)
    )
    checks = trades_inside_stale_quote(quotes, trades) if trades.height else {}
    return Prepared(manifest, events, quotes, features, names, stats, checks)


def block_len(cfg: ExperimentConfig) -> int:
    if cfg.evaluation.block_len is not None:
        return cfg.evaluation.block_len
    longest = max(cfg.features.horizon_s, *cfg.features.windows_s)
    return 2 * longest * 1000 // cfg.features.interval_ms


def _between(dates: tuple[str, str]) -> pl.Expr:
    lo, hi = (pl.lit(d).str.to_date() for d in dates)
    return pl.col("session").is_between(lo, hi)


def complete_rows(prep: Prepared, cfg: ExperimentConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Development and test rows with a valid book and every feature present, labelled or not.

    These are the rows a live strategy could act on; the backtest uses them, so whether a
    signal exists never depends on the future (a label needs the book to stay valid to t+h).
    """
    complete = prep.features.filter(
        pl.col("valid")
        & pl.all_horizontal(
            pl.col(c).is_not_null() & pl.col(c).is_finite() for c in prep.feature_names
        )
    )
    return complete.filter(_between(cfg.data.dev_dates)), complete.filter(
        _between(cfg.data.test_dates)
    )


def usable_rows(prep: Prepared, cfg: ExperimentConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Complete rows that also have a label: what models are fitted and scored on."""
    labelled = pl.col("fwd_ret").is_not_null() & pl.col("fwd_ret").is_finite()
    dev, test = (frame.filter(labelled) for frame in complete_rows(prep, cfg))
    if dev.is_empty() or test.is_empty():
        raise ValueError("development or test period has no usable rows")
    dev_t, test_t = dev["t_ns"].to_numpy(), test["t_ns"].to_numpy()
    if test_t.min() <= dev_t.max():
        raise ValueError("test period must start after the development period ends")
    test_start = int(test_t.min())
    embargo = cfg.evaluation.embargo_s * NS
    keep = admissible(dev["label_end_ns"].to_numpy(), test_start, embargo)
    return dev.filter(pl.Series(keep)), test


def design_matrix(df: pl.DataFrame, spec: ModelSpec) -> tuple[np.ndarray, np.ndarray]:
    x = df.select(spec.features).to_numpy() if spec.features else np.empty((df.height, 0))
    return x.astype(np.float64), df["fwd_ret"].to_numpy().astype(np.float64)


def select_model(
    dev: pl.DataFrame, cfg: ExperimentConfig, feature_names: list[str]
) -> tuple[ModelSpec, pl.DataFrame, pl.DataFrame]:
    """Walk-forward on development data only. Returns the chosen spec, the per-fold score table
    and the chosen model's out-of-sample predictions."""
    folds = walk_forward(
        dev["t_ns"].to_numpy(),
        dev["label_end_ns"].to_numpy(),
        cfg.evaluation.n_folds,
        cfg.evaluation.embargo_s * NS,
    )
    specs = [s for grid in cfg.models for s in grid.specs(feature_names)]
    rows, oos = [], {}
    for spec in specs:
        preds = []
        x, y = design_matrix(dev, spec)
        for k, fold in enumerate(folds):
            model = fit(spec, x[fold.train], y[fold.train])
            pred = model.predict(x[fold.test])
            preds.append(pred)
            rows.append(
                {
                    "model": spec.name,
                    "fold": k,
                    "n_train": len(fold.train),
                    "n_test": len(fold.test),
                    "purged": fold.purged,
                    "mse": float(((y[fold.test] - pred) ** 2).mean()),
                    "zero_mse": float((y[fold.test] ** 2).mean()),
                    "rank_ic": rank_ic(pred, y[fold.test]),
                }
            )
        oos[spec.name] = np.concatenate(preds)
    table = pl.DataFrame(rows)
    ranking = (
        table.group_by("model", maintain_order=True)
        .agg(pl.col("mse").mean().alias("mean_mse"))
        .with_row_index("order")
        .sort("mean_mse", "order")
    )
    chosen = next(s for s in specs if s.name == ranking["model"][0])
    tested = np.concatenate([f.test for f in folds])
    oos_frame = dev[tested.tolist()].with_columns(pred=pl.Series(oos[chosen.name]))
    return chosen, table, oos_frame


def walk_forward_signals(
    dev: pl.DataFrame, dev_complete: pl.DataFrame, spec: ModelSpec, cfg: ExperimentConfig
) -> pl.DataFrame:
    """Out-of-sample forecasts for every complete development row from the first test block on.

    Fold k's model (trained on admissible labelled rows before block k) forecasts all complete
    rows from block k's start until block k+1's start, labelled or not.
    """
    folds = walk_forward(
        dev["t_ns"].to_numpy(),
        dev["label_end_ns"].to_numpy(),
        cfg.evaluation.n_folds,
        cfg.evaluation.embargo_s * NS,
    )
    x, y = design_matrix(dev, spec)
    starts = [f.test_start_ns for f in folds] + [2**62]
    parts = []
    for k, fold in enumerate(folds):
        model = fit(spec, x[fold.train], y[fold.train])
        rows = dev_complete.filter(pl.col("t_ns").is_between(starts[k], starts[k + 1], "left"))
        parts.append(rows.with_columns(pred=pl.Series(model.predict(design_matrix(rows, spec)[0]))))
    return pl.concat(parts)


class RunIntegrityError(RuntimeError):
    """A run directory's config no longer matches what the run was executed with."""


def load_run_config(run_dir: Path) -> ExperimentConfig:
    """The run's config, refusing one edited after the run (which would let a re-scored stage
    quietly change thresholds, costs or statistics without a new test evaluation)."""
    text = (run_dir / "config.toml").read_text()
    recorded = json.loads((run_dir / "manifest.json").read_text())["config_sha256"]
    if hashlib.sha256(text.encode()).hexdigest() != recorded:
        raise RunIntegrityError(f"{run_dir / 'config.toml'} was modified after the run")
    cfg = ExperimentConfig.model_validate(tomllib.loads(text))
    cfg.data.version = json.loads((run_dir / "manifest.json").read_text())["dataset"]["version"]
    return cfg


def record_test_access(
    cfg: ExperimentConfig, manifest: DatasetManifest, run_dir: Path, config_sha256: str
) -> int:
    """Append to the dataset's test-access ledger; return earlier accesses to these dates.

    The ledger lives in the catalog next to the data, so renaming the experiment or using a
    different runs directory does not reset the count.
    """
    ledger = Path(cfg.data.catalog) / cfg.data.dataset / "test_access.jsonl"
    lo, hi = cfg.data.test_dates
    entries = (
        [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    )
    prior = sum(1 for e in entries if e["test_dates"][0] <= hi and lo <= e["test_dates"][1])
    entry = {
        "at_ns": time.time_ns(),
        "experiment": cfg.name,
        "run": str(run_dir),
        "config_sha256": config_sha256,
        "dataset_version": manifest.version,
        "test_dates": [lo, hi],
    }
    with ledger.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return prior


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def provenance() -> dict[str, Any]:
    src = Path(__file__).parent
    code = hashlib.sha256()
    for path in sorted(src.rglob("*.py")):
        code.update(path.read_bytes())
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "source_sha256": code.hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            p: metadata.version(p)
            for p in ("qrp", "numpy", "polars", "pyarrow", "duckdb", "pydantic")
        },
    }


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def run(cfg: ExperimentConfig, config_text: str, runs_root: Path = Path("runs")) -> Path:
    """Select on development data, fit once, predict the test period once, then evaluate and
    backtest. Returns the run directory."""
    started = time.time_ns()
    exp_dir = runs_root / cfg.name
    cfg_hash = hashlib.sha256(config_text.encode()).hexdigest()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(started / 1e9))
    run_dir = exp_dir / f"{stamp}.{started // 1_000_000 % 1000:03d}Z-{cfg_hash[:8]}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.toml").write_text(config_text)

    prep = prepare(cfg)
    dev, test = usable_rows(prep, cfg)
    dev_complete, test_complete = complete_rows(prep, cfg)
    chosen, table, dev_oos = select_model(dev, cfg, prep.feature_names)
    x_dev, y_dev = design_matrix(dev, chosen)
    final: FittedModel = fit(chosen, x_dev, y_dev)
    # Everything above used development data only. From here on the test period is read.
    prior = record_test_access(cfg, prep.manifest, run_dir, cfg_hash)
    test_pred = test.with_columns(pred=pl.Series(final.predict(design_matrix(test, chosen)[0])))
    test_signals = test_complete.with_columns(
        pred=pl.Series(final.predict(design_matrix(test_complete, chosen)[0]))
    )
    dev_signals = walk_forward_signals(dev, dev_complete, chosen, cfg)
    pl.concat(
        [
            dev_signals.select("t_ns", "session", "pred").with_columns(split=pl.lit("dev_oos")),
            test_signals.select("t_ns", "session", "pred").with_columns(split=pl.lit("test")),
        ]
    ).write_parquet(run_dir / "signals.parquet")

    keep = [
        "symbol",
        "session",
        "t_ns",
        "mid",
        "bid_px_0",
        "ask_px_0",
        "spread_bps",
        "fwd_ret",
        "pred",
    ]
    regime = cfg.evaluation.regime_feature
    keep += [regime] if regime in prep.feature_names and regime not in keep else []
    pl.concat(
        [
            dev_oos.select(keep).with_columns(split=pl.lit("dev_oos")),
            test_pred.select(keep).with_columns(split=pl.lit("test")),
        ]
    ).write_parquet(run_dir / "predictions.parquet")
    table.write_csv(run_dir / "selection.csv")
    _write_json(
        run_dir / "model.json",
        {
            "spec": chosen.model_dump(),
            "intercept": final.intercept,
            "coef": dict(zip(chosen.features, final.coef.tolist(), strict=True)),
        },
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment": cfg.name,
            "run_id": run_dir.name,
            "started_at_ns": started,
            "config_sha256": cfg_hash,
            "seed": cfg.seed,
            "dataset": {
                "name": prep.manifest.name,
                "version": prep.manifest.version,
                "source": prep.manifest.source.model_dump(),
                "capabilities": prep.manifest.capabilities.model_dump(),
                "validation": prep.manifest.validation,
                "duplicates_removed": prep.manifest.duplicates_removed,
            },
            "book": {**prep.book_stats.model_dump(), **prep.quote_checks},
            "features": {n: feature_docs(cfg.features)[n] for n in prep.feature_names},
            "rows": {
                "grid": prep.features.height,
                "valid": int(prep.features["valid"].sum()),
                "dev": dev.height,
                "test": test.height,
            },
            "selected_model": chosen.name,
            "test_evaluations_before_this_run": prior,
            "provenance": provenance(),
        },
    )
    log.info("model selected", extra={"run": run_dir.name, "model": chosen.name})
    evaluate_run(run_dir)
    backtest_run(run_dir, prep)
    return run_dir


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    cfg = load_run_config(run_dir)
    preds = pl.read_parquet(run_dir / "predictions.parquet")
    blen, ev = block_len(cfg), cfg.evaluation
    out: dict[str, Any] = {"block_len": blen}
    dev = preds.filter(pl.col("split") == "dev_oos")
    for split, df in (("dev_oos", dev), ("test", preds.filter(pl.col("split") == "test"))):
        pred, y = df["pred"].to_numpy(), df["fwd_ret"].to_numpy()
        metrics = evaluate(pred, y, blen, ev.n_boot, cfg.seed)
        p_value, _ = circular_shift_pvalue(pred, y, ev.n_null, blen, cfg.seed)
        entry: dict[str, Any] = {
            **metrics.model_dump(),
            "null_p_value": p_value,
            "calibration": calibration_table(pred, y).to_dicts(),
        }
        if ev.regime_feature in df.columns:
            # Tercile cut points come from development data only, then apply to both splits.
            cuts = dev[ev.regime_feature].quantile(1 / 3), dev[ev.regime_feature].quantile(2 / 3)
            labelled = df.with_columns(
                regime=pl.when(pl.col(ev.regime_feature) <= cuts[0])
                .then(pl.lit("low"))
                .when(pl.col(ev.regime_feature) <= cuts[1])
                .then(pl.lit("mid"))
                .otherwise(pl.lit("high"))
            )
            entry["regimes"] = {
                name: {
                    "n": part.height,
                    "mean_abs_ret": float(np.abs(part["fwd_ret"].to_numpy()).mean()),
                    **evaluate(
                        part["pred"].to_numpy(),
                        part["fwd_ret"].to_numpy(),
                        blen,
                        ev.n_boot,
                        cfg.seed,
                    ).model_dump(include={"rank_ic", "oos_r2"}),
                }
                for name in ("low", "mid", "high")
                if (part := labelled.filter(pl.col("regime") == name)).height > 2 * blen
            }
        out[split] = entry
    _write_json(run_dir / "metrics.json", out)
    return out


def cost_threshold(cfg: ExperimentConfig, dev: pl.DataFrame) -> float:
    """Round-trip cost estimate in log-return units, from development data only."""
    half_spread = float(np.median(dev["spread_bps"].to_numpy())) / 2 / 1e4
    fee = cfg.backtest.costs.taker_fee_bps / 1e4
    return cfg.strategy.threshold_cost_multiple * 2 * (half_spread + fee)


def strategy_thresholds(cfg: ExperimentConfig, dev: pl.DataFrame) -> dict[str, float]:
    rules = {
        "cost_rule": cfg.strategy.threshold
        if cfg.strategy.threshold is not None
        else cost_threshold(cfg, dev)
    }
    q = cfg.strategy.diagnostic_quantile
    if q is not None:
        top = np.quantile(np.abs(dev["pred"].to_numpy()), q)
        rules[f"diagnostic_top{round((1 - q) * 100)}pct"] = float(top)
    return rules


def backtest_run(run_dir: Path, prep: Prepared | None = None) -> dict[str, Any]:
    cfg = load_run_config(run_dir)  # pins the dataset version the run used
    if prep is None:
        prep = prepare(cfg)
    preds = pl.read_parquet(run_dir / "predictions.parquet")
    all_signals = pl.read_parquet(run_dir / "signals.parquet")
    rules = strategy_thresholds(cfg, preds.filter(pl.col("split") == "dev_oos"))
    out: dict[str, Any] = {"thresholds": rules, "results": {}}
    for split in ("dev_oos", "test"):
        signals = all_signals.filter(pl.col("split") == split).select("t_ns", "session", "pred")
        t = signals["t_ns"].to_numpy()
        window = pl.col("available_time_ns").is_between(
            int(t.min()) - 3600 * NS, int(t.max()) + 3600 * NS
        )
        book = BookTimeline.from_quotes(prep.quotes.filter(window))
        out["results"][split] = {}
        for rule, threshold in rules.items():
            target = threshold_target(threshold, cfg.backtest.position_size)
            result = run_backtest(book, signals, cfg.backtest, target, cfg.seed)
            result.fills.write_parquet(run_dir / f"fills_{split}_{rule}.parquet")
            result.equity.write_parquet(run_dir / f"equity_{split}_{rule}.parquet")
            out["results"][split][rule] = {
                **result.summary,
                "max_abs_forecast": float(np.abs(signals["pred"].to_numpy()).max()),
            }
    _write_json(run_dir / "backtest.json", out)
    return out
