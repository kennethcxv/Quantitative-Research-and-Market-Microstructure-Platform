"""Config-driven experiments.

``run`` is the only place that touches the holdout test period, and it does so once, after the
model and every hyper-parameter have been chosen by walk-forward validation on the development
period. Each run gets its own directory; earlier runs of the same experiment are counted and
reported so that repeated peeking at the test set is visible.

Stages, each restartable from the run directory: run (select + predict) -> evaluate -> backtest.
"""

from __future__ import annotations

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


class EvalConfig(BaseModel):
    n_folds: int = 5
    embargo_s: int = 30
    n_boot: int = 500
    block_len: int | None = None  # samples; None = 2 x max(horizon, longest window)
    n_null: int = 200
    regime_feature: str = "rv_30s"


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
    manifest = catalog.resolve(cfg.data.dataset, cfg.data.version)
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


def usable_rows(prep: Prepared, cfg: ExperimentConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Development and test rows with a valid book, a label and every feature present."""
    cols = [*prep.feature_names, "fwd_ret"]
    complete = prep.features.filter(
        pl.col("valid")
        & pl.all_horizontal(pl.col(c).is_not_null() & pl.col(c).is_finite() for c in cols)
    )

    def between(dates: tuple[str, str]) -> pl.Expr:
        lo, hi = (pl.lit(d).str.to_date() for d in dates)
        return pl.col("session").is_between(lo, hi)

    dev = complete.filter(between(cfg.data.dev_dates))
    test = complete.filter(between(cfg.data.test_dates))
    if dev.is_empty() or test.is_empty():
        raise ValueError("development or test period has no usable rows")
    if test["t_ns"].min() <= dev["t_ns"].max():  # type: ignore[operator]
        raise ValueError("test period must start after the development period ends")
    test_start = int(test["t_ns"].min())  # type: ignore[arg-type]
    embargo = cfg.evaluation.embargo_s * NS
    keep = admissible(dev["label_end_ns"].to_numpy(), test_start, embargo)
    return dev.filter(pl.Series(keep)), test


def _xy(df: pl.DataFrame, spec: ModelSpec) -> tuple[np.ndarray, np.ndarray]:
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
        x, y = _xy(dev, spec)
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
    prior = (
        sorted(p.name for p in exp_dir.glob("*/predictions.parquet")) if exp_dir.exists() else []
    )
    cfg_hash = hashlib.sha256(config_text.encode()).hexdigest()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(started / 1e9))
    run_dir = exp_dir / f"{stamp}.{started // 1_000_000 % 1000:03d}Z-{cfg_hash[:8]}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.toml").write_text(config_text)

    prep = prepare(cfg)
    dev, test = usable_rows(prep, cfg)
    chosen, table, dev_oos = select_model(dev, cfg, prep.feature_names)
    x_dev, y_dev = _xy(dev, chosen)
    final: FittedModel = fit(chosen, x_dev, y_dev)
    x_test, _ = _xy(test, chosen)  # the only line that reads test-period features
    test_pred = test.with_columns(pred=pl.Series(final.predict(x_test)))

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
            "test_evaluations_before_this_run": len(prior),
            "provenance": provenance(),
        },
    )
    log.info("model selected", extra={"run": run_dir.name, "model": chosen.name})
    evaluate_run(run_dir)
    backtest_run(run_dir, prep)
    return run_dir


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    cfg = load_config(run_dir / "config.toml")
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
                    "mean_abs_ret": float(part["fwd_ret"].abs().mean()),  # type: ignore[arg-type]
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
    half_spread = float(dev["spread_bps"].median()) / 2 / 1e4  # type: ignore[arg-type]
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
        rules[f"diagnostic_top{round((1 - q) * 100)}pct"] = float(dev["pred"].abs().quantile(q))  # type: ignore[arg-type]
    return rules


def backtest_run(run_dir: Path, prep: Prepared | None = None) -> dict[str, Any]:
    cfg = load_config(run_dir / "config.toml")
    if prep is None:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        cfg.data.version = manifest["dataset"]["version"]  # replay the exact version
        prep = prepare(cfg)
    preds = pl.read_parquet(run_dir / "predictions.parquet")
    rules = strategy_thresholds(cfg, preds.filter(pl.col("split") == "dev_oos"))
    out: dict[str, Any] = {"thresholds": rules, "results": {}}
    for split in ("dev_oos", "test"):
        signals = preds.filter(pl.col("split") == split).select("t_ns", "session", "pred")
        lo, hi = signals["t_ns"].min(), signals["t_ns"].max()
        window = pl.col("available_time_ns").is_between(lo - 3600 * NS, hi + 3600 * NS)  # type: ignore[operator]
        book = BookTimeline.from_quotes(prep.quotes.filter(window))
        out["results"][split] = {}
        for rule, threshold in rules.items():
            target = threshold_target(threshold, cfg.backtest.position_size)
            result = run_backtest(book, signals, cfg.backtest, target, cfg.seed)
            result.fills.write_parquet(run_dir / f"fills_{split}_{rule}.parquet")
            result.equity.write_parquet(run_dir / f"equity_{split}_{rule}.parquet")
            out["results"][split][rule] = {
                **result.summary,
                "max_abs_forecast": float(signals["pred"].abs().max()),  # type: ignore[arg-type]
            }
    _write_json(run_dir / "backtest.json", out)
    return out
