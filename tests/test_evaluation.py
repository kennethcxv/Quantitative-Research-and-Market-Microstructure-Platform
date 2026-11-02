from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qrp.models import ModelSpec, fit
from qrp.splits import walk_forward
from qrp.stats import (
    block_bootstrap,
    calibration_table,
    circular_shift_pvalue,
    evaluate,
    hit_rate,
    oos_r2,
    rank_ic,
)

NS = 1_000_000_000


@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(60, 400),
    horizon=st.integers(1, 20),
    embargo=st.integers(0, 10),
    folds=st.integers(1, 5),
)
def test_walk_forward_never_trains_on_overlapping_labels(
    n: int, horizon: int, embargo: int, folds: int
) -> None:
    t = np.arange(n, dtype=np.int64) * NS
    label_end = t + horizon * NS
    for fold in walk_forward(t, label_end, folds, embargo * NS):
        assert (label_end[fold.train] + embargo * NS <= fold.test_start_ns).all()
        assert t[fold.train].max(initial=-1) < fold.test_start_ns
        assert len(np.intersect1d(fold.train, fold.test)) == 0
        # Purge drops exactly the rows whose label (plus embargo) ends after the test start; a
        # label ending exactly at the start is already known, so it stays.
        assert fold.purged == min(horizon + embargo - 1, int(fold.test[0]))


def test_walk_forward_rejects_unsorted_times() -> None:
    t = np.array([0, 2, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="sorted"):
        walk_forward(t, t + 1, 1, 0)


def test_ols_matches_least_squares_and_ridge_shrinks() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 3)) * [1.0, 5.0, 0.1] + [0.0, 2.0, -1.0]
    y = 0.5 + x @ np.array([1.0, -0.2, 3.0]) + rng.normal(scale=0.1, size=500)
    spec = ModelSpec(name="ols", kind="ridge", features=["a", "b", "c"])
    model = fit(spec, x, y)
    design = np.column_stack([np.ones(500), x])
    expected = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(model.predict(x), expected, rtol=1e-9, atol=1e-9)
    ridge = fit(spec.model_copy(update={"alpha": 10.0}), x, y)
    assert np.abs(ridge.coef).sum() < np.abs(model.coef).sum()


def test_zero_and_mean_baselines() -> None:
    x = np.empty((4, 0))
    y = np.array([1.0, 2.0, 3.0, 6.0])
    assert (fit(ModelSpec(name="z", kind="zero"), x, y).predict(x) == 0).all()
    assert (fit(ModelSpec(name="m", kind="mean"), x, y).predict(x) == 3.0).all()


def test_constant_feature_does_not_produce_nan() -> None:
    x = np.column_stack([np.ones(50), np.arange(50.0)])
    model = fit(ModelSpec(name="r", kind="ridge", features=["c", "t"]), x, np.arange(50.0))
    assert np.isfinite(model.predict(x)).all() and model.coef[0] == 0


def test_point_metrics() -> None:
    y = np.array([1.0, -1.0, 2.0, -2.0, 0.0])
    pred = np.array([0.5, -0.5, 1.0, 1.0, 1.0])
    assert rank_ic(pred, y) == pytest.approx(
        pl.DataFrame({"p": pred, "y": y}).select(pl.corr("p", "y", method="spearman")).item()
    )
    assert oos_r2(np.zeros(5), y) == 0.0
    assert oos_r2(y, y) == 1.0
    assert hit_rate(pred, y) == 0.75  # the y == 0 row is excluded
    assert rank_ic(np.ones(5), y) == 0.0


def _ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_block_bootstrap_widens_for_persistent_series() -> None:
    rng = np.random.default_rng(1)
    x = _ar1(4000, 0.9, rng)
    iid = block_bootstrap(lambda i: float(x[i].mean()), len(x), 1, 400, seed=2)
    blocked = block_bootstrap(lambda i: float(x[i].mean()), len(x), 100, 400, seed=2)
    # Long-run variance of an AR(1) mean is (1+phi)/(1-phi) = 19x the i.i.d. variance.
    assert blocked.se / iid.se == pytest.approx(np.sqrt(19), rel=0.35)
    assert blocked.low < 0 < blocked.high  # true mean is 0


def test_circular_shift_null_separates_signal_from_persistence() -> None:
    rng = np.random.default_rng(3)
    a, b = _ar1(3000, 0.95, rng), _ar1(3000, 0.95, rng)
    p_unrelated, _ = circular_shift_pvalue(a, b, n_null=200, min_shift=100, seed=4)
    p_related, _ = circular_shift_pvalue(a, a + rng.normal(size=3000), 200, 100, seed=4)
    assert p_unrelated > 0.05
    assert p_related < 0.01


def test_evaluate_and_calibration_on_known_signal() -> None:
    rng = np.random.default_rng(5)
    signal = rng.normal(size=5000)
    y = 0.3 * signal + rng.normal(size=5000)
    m = evaluate(0.3 * signal, y, block_len=10, n_boot=200, seed=6)
    assert m.rank_ic.excludes_zero() and m.oos_r2.estimate > 0
    assert m.calibration_slope == pytest.approx(1.0, abs=0.1)
    table = calibration_table(0.3 * signal, y, bins=5)
    assert table["n"].sum() == 5000 and table["mean_y"].is_sorted()
