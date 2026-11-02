"""Causal market-microstructure features and forward-return labels on a clock-time grid.

Timing contract (docs/architecture.md section 3):

* Every input row gets an ``available_time_ns`` (see ``available_time``).
* The grid is ``t = k * interval`` per session (UTC date). A value at ``t`` uses only rows with
  ``available_time_ns <= t``: point values via backward as-of joins, window values over
  ``(t - W, t]`` as the difference of a cumulative sum taken as-of ``t`` and as-of ``t - W``.
* Labels look forward by construction and carry ``label_end_ns = t + horizon``.

Definitions (b/a = best bid/ask, qb/qa = sizes, m = (a + b) / 2) are in docs/features.md and in
``FEATURE_DOCS`` below, which the report prints verbatim.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
from pydantic import BaseModel

from qrp.book import ladder_columns, reconstruct
from qrp.schemas import TRADE, Capabilities

NS = 1_000_000_000


class TimingConfig(BaseModel):
    extra_latency_ms: float = 0.0  # our own processing delay on top of measured receive times
    assumed_feed_latency_ms: float | None = None  # required when the data has no receive times


class FeatureConfig(BaseModel):
    interval_ms: int = 1_000
    windows_s: list[int] = [5, 30]
    depth_levels: int = 5
    event_k: int = 20  # top-of-book changes per event-time return
    warmup_s: int = 60
    max_book_age_s: float = 60.0  # older quote at sample time = stale = invalid sample
    horizon_s: int = 10
    names: list[str] | None = None  # None = every feature the dataset supports


class FeatureError(ValueError):
    pass


def available_time(caps: Capabilities, timing: TimingConfig) -> pl.Expr:
    """The research clock. Never stored: it depends on the experiment's latency assumptions."""
    if caps.recv_time == "measured":
        return (pl.col("recv_time_ns") + int(timing.extra_latency_ms * 1e6)).alias(
            "available_time_ns"
        )
    if timing.assumed_feed_latency_ms is None:
        raise FeatureError(
            "dataset has no receive times: set timing.assumed_feed_latency_ms explicitly"
        )
    latency = int((timing.assumed_feed_latency_ms + timing.extra_latency_ms) * 1e6)
    return (pl.col("event_time_ns") + latency).alias("available_time_ns")


def feature_docs(cfg: FeatureConfig) -> dict[str, str]:
    n = cfg.depth_levels
    docs = {
        "spread_bps": "1e4 * (a - b) / m",
        "imbalance_l1": "(qb - qa) / (qb + qa); in [-1, 1]",
        f"imbalance_l{n}": f"same as imbalance_l1 with sizes summed over the top {n} levels",
        "microprice_dev_bps": (
            "1e4 * (micro - m) / m, micro = (a*qb + b*qa) / (qb + qa). Note micro - m = "
            "spread * imbalance_l1 / 2, so this is imbalance scaled by the spread"
        ),
        f"log_depth_l{n}": f"log of total size in the top {n} levels, both sides",
        f"ret_event_{cfg.event_k}": (
            f"log m now - log m {cfg.event_k} top-of-book changes ago (event clock)"
        ),
    }
    for w in cfg.windows_s:
        docs |= {
            f"ofi_{w}s": (
                f"order-flow imbalance (Cont, Kukanov & Stoikov 2014) summed over messages in "
                f"(t-{w}s, t]: e = 1[b>=b'] qb - 1[b<=b'] qb' - 1[a<=a'] qa + 1[a>=a'] qa', "
                "primes = previous message; 0 across unsynced or crossed books"
            ),
            f"trade_count_{w}s": f"number of prints in (t-{w}s, t]",
            f"log_volume_{w}s": f"log(1 + traded quantity in (t-{w}s, t])",
            f"trade_imbalance_{w}s": (
                f"(buy volume - sell volume) / volume in (t-{w}s, t], aggressor side; 0 if none"
            ),
            f"ret_{w}s": f"log m(t) - log m(t-{w}s) (clock time)",
            f"rv_{w}s": (
                f"sqrt(sum of squared {cfg.interval_ms}ms log-mid returns over the last {w}s)"
            ),
        }
    return docs


def _requirements(cfg: FeatureConfig) -> dict[str, tuple[int, bool]]:
    """feature -> (book levels needed, needs trades)."""
    reqs = {}
    for name in feature_docs(cfg):
        levels = cfg.depth_levels if name.endswith(f"_l{cfg.depth_levels}") else 1
        needs_trades = name.startswith(("trade_", "log_volume_"))
        reqs[name] = (levels, needs_trades)
    return reqs


def select_features(cfg: FeatureConfig, caps: Capabilities) -> list[str]:
    """Requested features, refusing any the dataset cannot support."""
    reqs = _requirements(cfg)
    names = cfg.names if cfg.names is not None else list(reqs)
    unknown = [n for n in names if n not in reqs]
    if unknown:
        raise FeatureError(f"unknown features {unknown}; known: {sorted(reqs)}")
    depth = caps.book_depth if caps.book_depth is not None else 10**9
    refused = [n for n in names if reqs[n][0] > depth or (reqs[n][1] and not caps.has_trades)]
    if refused:
        if cfg.names is not None:
            raise FeatureError(f"dataset cannot support {refused} (depth={caps.book_depth})")
        names = [n for n in names if n not in refused]
    return names


def _session(col: str) -> pl.Expr:
    return pl.from_epoch(col, time_unit="ns").dt.date().alias("session")


def _asof(left: pl.DataFrame, right: pl.DataFrame, on: str, cols: Sequence[str]) -> pl.DataFrame:
    """Values of ``cols`` from the last ``right`` row at or before ``left[on]``, per session."""
    return left.join_asof(
        right.select("session", "available_time_ns", *cols),
        left_on=on,
        right_on="available_time_ns",
        by="session",
        strategy="backward",
        check_sortedness=False,  # both sides are sorted by time; polars cannot verify with `by`
    ).select(cols)


def build_features(
    events: pl.DataFrame,
    caps: Capabilities,
    cfg: FeatureConfig,
    timing: TimingConfig,
) -> pl.DataFrame:
    """Features and labels for one symbol. Rows: one per grid time; ``valid`` marks usable rows."""
    if cfg.horizon_s * 1000 % cfg.interval_ms or any(
        w * 1000 % cfg.interval_ms for w in cfg.windows_s
    ):
        raise FeatureError("horizon and windows must be multiples of the sampling interval")
    names = select_features(cfg, caps)
    n_levels = min(cfg.depth_levels, caps.book_depth or cfg.depth_levels)
    sequencing = {k: v.sequencing for k, v in caps.streams.items()}
    clock = available_time(caps, timing)
    step = cfg.interval_ms * 1_000_000

    quotes, _ = reconstruct(events, sequencing, levels=n_levels)
    b, a = pl.col("bid_px_0"), pl.col("ask_px_0")
    qb, qa = pl.col("bid_sz_0"), pl.col("ask_sz_0")
    quotes = (
        quotes.with_columns(clock)
        .filter(pl.col("available_time_ns").is_not_null())
        .sort("available_time_ns", maintain_order=True)
        .with_columns(_session("available_time_ns"))
        .with_columns(
            quote_ok=pl.col("synced") & ~pl.col("crossed") & b.is_not_null() & a.is_not_null(),
            mid=(a + b) / 2,
        )
    )
    prev = {
        c: pl.col(c).shift(1).over("session")
        for c in ("bid_px_0", "ask_px_0", "bid_sz_0", "ask_sz_0", "quote_ok")
    }
    ofi = (
        pl.when(b >= prev["bid_px_0"]).then(qb).otherwise(0)
        - pl.when(b <= prev["bid_px_0"]).then(prev["bid_sz_0"]).otherwise(0)
        - pl.when(a <= prev["ask_px_0"]).then(qa).otherwise(0)
        + pl.when(a >= prev["ask_px_0"]).then(prev["ask_sz_0"]).otherwise(0)
    )
    top_changed = pl.any_horizontal(
        pl.col(c) != prev[c] for c in ("bid_px_0", "ask_px_0", "bid_sz_0", "ask_sz_0")
    )
    quotes = quotes.with_columns(
        cum_ofi=pl.when(pl.col("quote_ok") & prev["quote_ok"].fill_null(False))
        .then(ofi)
        .otherwise(0.0)
        .cum_sum()
        .over("session"),
    )
    tob = quotes.filter(pl.col("quote_ok") & top_changed.fill_null(True)).with_columns(
        ret_event=pl.col("mid").log() - pl.col("mid").log().shift(cfg.event_k).over("session")
    )

    trades = (
        events.filter(pl.col("kind") == TRADE)
        .with_columns(clock)
        .filter(pl.col("available_time_ns").is_not_null())
        .sort("available_time_ns", maintain_order=True)
        .with_columns(_session("available_time_ns"))
        .with_columns(
            cum_n=pl.col("size").cum_count().over("session"),
            cum_vol=pl.col("size").cum_sum().over("session"),
            cum_signed=(pl.col("size") * pl.col("side")).cum_sum().over("session"),
        )
    )

    # Grid: from the first usable quote + warmup to the last quote, per session.
    bounds = (
        quotes.filter("quote_ok")
        .group_by("session")
        .agg(first=pl.col("available_time_ns").min(), last=pl.col("available_time_ns").max())
    )
    warm = cfg.warmup_s * NS
    grid = (
        bounds.with_columns(
            t_ns=pl.int_ranges(
                ((pl.col("first") + warm) // step + 1) * step, pl.col("last") + 1, step
            )
        )
        .explode("t_ns")
        .drop_nulls("t_ns")
        .select("session", "t_ns")
        .sort("t_ns")
    )

    quotes = quotes.with_columns(quote_time_ns=pl.col("available_time_ns"))
    state = _asof(
        grid, quotes, "t_ns", [*ladder_columns(n_levels), "quote_ok", "quote_time_ns", "cum_ofi"]
    )
    out = pl.concat([grid, state], how="horizontal").with_columns(
        valid=(
            pl.col("quote_ok")
            & ((pl.col("t_ns") - pl.col("quote_time_ns")) <= int(cfg.max_book_age_s * NS))
        ).fill_null(False)
    )
    # A mid from an unsynced, crossed or stale book is not a price: null it so that returns,
    # volatility and labels spanning it are null too.
    out = out.with_columns(mid=pl.when("valid").then((a + b) / 2))
    mid_log = pl.col("mid").log()
    depth_bid = pl.sum_horizontal(pl.col(f"bid_sz_{i}").fill_nan(0) for i in range(n_levels))
    depth_ask = pl.sum_horizontal(pl.col(f"ask_sz_{i}").fill_nan(0) for i in range(n_levels))
    micro = (a * qb + b * qa) / (qb + qa)
    n = cfg.depth_levels
    cols: dict[str, pl.Expr] = {
        "spread_bps": 1e4 * (a - b) / pl.col("mid"),
        "imbalance_l1": (qb - qa) / (qb + qa),
        f"imbalance_l{n}": (depth_bid - depth_ask) / (depth_bid + depth_ask),
        "microprice_dev_bps": 1e4 * (micro - pl.col("mid")) / pl.col("mid"),
        f"log_depth_l{n}": (depth_bid + depth_ask).log(),
    }
    out = out.with_columns(**cols)

    tob_state = _asof(grid, tob, "t_ns", ["ret_event"]).rename(
        {"ret_event": f"ret_event_{cfg.event_k}"}
    )
    out = pl.concat([out, tob_state], how="horizontal")

    one_step_ret = mid_log - mid_log.shift(1).over("session")
    for w in cfg.windows_s:
        lag = w * 1000 // cfg.interval_ms
        lower = grid.with_columns(t_lo=pl.col("t_ns") - w * NS)
        ofi_lo = _asof(lower, quotes, "t_lo", ["cum_ofi"]).fill_null(0.0)
        tr_hi = _asof(grid, trades, "t_ns", ["cum_n", "cum_vol", "cum_signed"]).fill_null(0)
        tr_lo = _asof(lower, trades, "t_lo", ["cum_n", "cum_vol", "cum_signed"]).fill_null(0)
        volume = tr_hi["cum_vol"] - tr_lo["cum_vol"]
        signed = tr_hi["cum_signed"] - tr_lo["cum_signed"]
        out = out.with_columns(
            pl.Series(f"trade_count_{w}s", tr_hi["cum_n"] - tr_lo["cum_n"]),
            pl.Series(f"log_volume_{w}s", (volume + 1).log()),
            pl.Series(f"trade_imbalance_{w}s", (signed / volume).fill_nan(0.0).fill_null(0.0)),
            (pl.col("cum_ofi") - ofi_lo["cum_ofi"]).alias(f"ofi_{w}s"),
            (mid_log - mid_log.shift(lag).over("session")).alias(f"ret_{w}s"),
            # Rolling sums are updated incrementally; cancellation can leave tiny negatives.
            one_step_ret.pow(2)
            .rolling_sum(lag)
            .over("session")
            .clip(lower_bound=0.0)
            .sqrt()
            .alias(f"rv_{w}s"),
        )

    h = cfg.horizon_s * 1000 // cfg.interval_ms
    future_ok = pl.col("valid").shift(-h).over("session").fill_null(False)
    out = out.with_columns(
        fwd_ret=pl.when(pl.col("valid") & future_ok).then(
            mid_log.shift(-h).over("session") - mid_log
        ),
        label_end_ns=pl.col("t_ns") + cfg.horizon_s * NS,
    )
    symbol = events["symbol"][0]
    return out.select(
        pl.lit(symbol).alias("symbol"),
        "session",
        "t_ns",
        "valid",
        "mid",
        "bid_px_0",
        "ask_px_0",
        *names,
        "fwd_ret",
        "label_end_ns",
    )
