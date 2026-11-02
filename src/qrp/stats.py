"""Predictive metrics and resampling for serially dependent samples.

Samples on a 1 s grid with a 10 s forward return overlap by construction, so neighbouring
errors are correlated and i.i.d. standard errors are too small. Uncertainty here comes from a
circular block bootstrap (Politis & Romano 1992) with blocks at least as long as the label
horizon, and significance from circular-shift nulls, which keep each series' autocorrelation
but break the alignment between forecast and outcome.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel

FloatArray = npt.NDArray[np.float64]


def ranks(x: FloatArray) -> FloatArray:
    """Average ranks (ties share the mean rank), as in Spearman's correlation."""
    return pl.Series(x).rank("average").to_numpy().astype(np.float64)


def pearson(x: FloatArray, y: FloatArray) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    denom = math.sqrt(float(xc @ xc) * float(yc @ yc))
    return float(xc @ yc / denom) if denom > 0 else 0.0


def rank_ic(pred: FloatArray, y: FloatArray) -> float:
    """Spearman correlation between forecast and outcome. 0 for a constant forecast."""
    return pearson(ranks(pred), ranks(y))


def oos_r2(pred: FloatArray, y: FloatArray) -> float:
    """1 - SSE(model) / SSE(zero forecast). Negative = worse than predicting no change."""
    return 1.0 - float(((y - pred) ** 2).sum()) / float((y**2).sum())


def hit_rate(pred: FloatArray, y: FloatArray) -> float:
    """Share of correct signs among samples where both forecast and outcome are non-zero."""
    mask = (pred != 0) & (y != 0)
    return float((np.sign(pred[mask]) == np.sign(y[mask])).mean()) if mask.any() else math.nan


class Interval(BaseModel):
    estimate: float
    low: float
    high: float
    se: float

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def block_indices(n: int, block_len: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    n_blocks = -(-n // block_len)
    starts = rng.integers(0, n, n_blocks)
    return ((starts[:, None] + np.arange(block_len)) % n).ravel()[:n]


def block_bootstrap(
    stat: Callable[[npt.NDArray[np.int64]], float],
    n: int,
    block_len: int,
    n_boot: int,
    seed: int,
    level: float = 0.95,
) -> Interval:
    """Percentile interval of ``stat(indices)`` over circular block resamples of 0..n-1."""
    rng = np.random.default_rng(seed)
    draws = np.array([stat(block_indices(n, block_len, rng)) for _ in range(n_boot)])
    tail = (1 - level) / 2
    return Interval(
        estimate=stat(np.arange(n)),
        low=float(np.quantile(draws, tail)),
        high=float(np.quantile(draws, 1 - tail)),
        se=float(draws.std(ddof=1)),
    )


class Metrics(BaseModel):
    n: int
    rank_ic: Interval
    oos_r2: Interval
    hit_rate: float
    mse: float
    calibration_slope: float  # OLS slope of outcome on forecast; 1 = calibrated, 0 = no signal


def evaluate(pred: FloatArray, y: FloatArray, block_len: int, n_boot: int, seed: int) -> Metrics:
    """Point metrics plus block-bootstrap intervals for rank IC and OOS R^2.

    The IC bootstrap resamples the precomputed rank pairs instead of re-ranking every resample:
    asymptotically equivalent and ~50x cheaper.
    """
    rp, ry = ranks(pred), ranks(y)
    pc = pred - pred.mean()
    slope = float(pc @ (y - y.mean()) / (pc @ pc)) if (pc @ pc) > 0 else 0.0
    return Metrics(
        n=len(y),
        rank_ic=block_bootstrap(lambda i: pearson(rp[i], ry[i]), len(y), block_len, n_boot, seed),
        oos_r2=block_bootstrap(lambda i: oos_r2(pred[i], y[i]), len(y), block_len, n_boot, seed),
        hit_rate=hit_rate(pred, y),
        mse=float(((y - pred) ** 2).mean()),
        calibration_slope=slope,
    )


def calibration_table(pred: FloatArray, y: FloatArray, bins: int = 10) -> pl.DataFrame:
    """Mean outcome by forecast decile. A calibrated forecast has mean_y close to mean_pred."""
    df = pl.DataFrame({"pred": pred, "y": y}).with_columns(
        bin=(pl.col("pred").rank("ordinal") * bins / (pl.len() + 1)).floor().cast(pl.Int32)
    )
    return (
        df.group_by("bin")
        .agg(n=pl.len(), mean_pred=pl.col("pred").mean(), mean_y=pl.col("y").mean())
        .sort("bin")
    )


def circular_shift_pvalue(
    pred: FloatArray, y: FloatArray, n_null: int, min_shift: int, seed: int
) -> tuple[float, FloatArray]:
    """Two-sided p-value of the observed rank IC against ICs with y rotated by random lags.

    Rotating keeps both series' autocorrelation intact, so the null reflects how large an IC
    two unrelated but persistent series produce by chance; a permutation null would not.
    """
    n = len(y)
    if n <= 2 * min_shift:
        raise ValueError("series too short for the requested minimum shift")
    rng = np.random.default_rng(seed)
    rp, ry = ranks(pred), ranks(y)
    observed = pearson(rp, ry)
    shifts = rng.integers(min_shift, n - min_shift, n_null)
    null = np.array([pearson(rp, np.roll(ry, s)) for s in shifts])
    p = (1 + int((np.abs(null) >= abs(observed)).sum())) / (n_null + 1)
    return p, null
