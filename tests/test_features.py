from __future__ import annotations

import math

import polars as pl
import pytest
from conftest import MS, T0, events

from qrp.features import FeatureConfig, FeatureError, TimingConfig, build_features, select_features
from qrp.schemas import Capabilities, StreamInfo
from qrp.synth import SynthConfig, generate, synthetic_capabilities

CAPS = Capabilities(
    book_depth=None,
    has_trades=True,
    recv_time="measured",
    event_time="exchange",
    streams={"feed": StreamInfo(sequencing="contiguous")},
)
SMALL = FeatureConfig(
    interval_ms=1000, windows_s=[2], depth_levels=2, event_k=1, warmup_s=0, horizon_s=1
)
S = 1000  # ms


@pytest.fixture
def hand_feed() -> pl.DataFrame:
    # recv = event + 1 ms (conftest). Grid points: T0+1s, +2s, +3s.
    return events(
        [
            (1, 0, "snapshot", 1, 99, 3),
            (1, 0, "snapshot", 1, 98, 1),
            (1, 0, "snapshot", -1, 101, 1),
            (1, 0, "snapshot", -1, 102, 4),
            (2, 500, "trade", 1, 101, 1),  # buy sweeps the 101 ask
            (2, 500, "book", -1, 101, 0),
            (3, 1500, "book", 1, 99, 5),
            (4, 2500, "book", -1, 100, 2),  # new ask inside the spread
            (5, 3200, "trade", -1, 99, 2),
            (5, 3200, "book", 1, 99, 3),
        ]
    )


def test_hand_computed_values(hand_feed: pl.DataFrame) -> None:
    f = build_features(hand_feed, CAPS, SMALL, TimingConfig())
    assert f["t_ns"].to_list() == [T0 + k * S * MS for k in (1, 2, 3)]
    assert f["valid"].all()
    r1, r2, r3 = f.rows(named=True)

    # t=1s sees message 2: bid 99x3, 98x1 / ask 102x4.
    assert r1["mid"] == 100.5
    assert r1["spread_bps"] == pytest.approx(1e4 * 3 / 100.5)
    assert r1["imbalance_l1"] == pytest.approx(-1 / 7)
    assert r1["imbalance_l2"] == pytest.approx(0.0)  # (3+1) vs 4
    micro = (102 * 3 + 99 * 4) / 7
    assert r1["microprice_dev_bps"] == pytest.approx(1e4 * (micro - 100.5) / 100.5)
    assert r1["ofi_2s"] == pytest.approx(1.0)  # the ask retreated: +1 (previous ask size)
    assert r1["trade_count_2s"] == 1
    assert r1["log_volume_2s"] == pytest.approx(math.log(2))
    assert r1["trade_imbalance_2s"] == 1.0

    # t=2s sees message 3: bid size 3 -> 5. OFI: +5 - 3 (bid) - 4 + 4 (ask) = +2.
    assert r2["imbalance_l1"] == pytest.approx(1 / 9)
    assert r2["ofi_2s"] == pytest.approx(3.0)  # messages 2 and 3 in (0s, 2s]
    assert r2["trade_count_2s"] == 1
    assert r2["ret_2s"] is None  # t-2s is before the first grid point

    # t=3s sees message 4: ask 100x2 joins. OFI for message 4: -2.
    assert r3["mid"] == 99.5
    assert r3["imbalance_l1"] == pytest.approx(3 / 7)
    assert r3["imbalance_l2"] == pytest.approx(0.0)  # 6 vs 6
    assert r3["ofi_2s"] == pytest.approx(0.0)  # +2 - 2 in (1s, 3s]
    assert r3["trade_count_2s"] == 0
    assert r3["trade_imbalance_2s"] == 0.0
    assert r3["ret_2s"] == pytest.approx(math.log(99.5 / 100.5))
    assert r3["rv_2s"] == pytest.approx(abs(math.log(99.5 / 100.5)))
    assert r3["ret_event_1"] == pytest.approx(math.log(99.5 / 100.5))

    assert r1["fwd_ret"] == pytest.approx(0.0)
    assert r2["fwd_ret"] == pytest.approx(math.log(99.5 / 100.5))
    assert r3["fwd_ret"] is None  # t+1s is past the end of the data
    assert r1["label_end_ns"] == r1["t_ns"] + S * MS


def test_message_received_after_t_is_invisible_at_t(hand_feed: pl.DataFrame) -> None:
    # Message 3 happened at 1.5s but only arrives at 2.4s: the t=2s sample must not see it.
    late = hand_feed.with_columns(
        pl.when(pl.col("seq") >= 3)
        .then(pl.col("recv_time_ns") + 900 * MS)
        .otherwise(pl.col("recv_time_ns"))
        .alias("recv_time_ns")
    )
    f = build_features(late, CAPS, SMALL, TimingConfig())
    assert f.row(1, named=True)["imbalance_l1"] == pytest.approx(-1 / 7)


def test_extra_latency_delays_everything(hand_feed: pl.DataFrame) -> None:
    f = build_features(hand_feed, CAPS, SMALL, TimingConfig(extra_latency_ms=600))
    # Message 2 (recv 0.501s) is now available at 1.101s, so t=1s still shows the snapshot.
    assert f.row(0, named=True)["imbalance_l1"] == pytest.approx((3 - 1) / 4)


def test_missing_receive_times_require_an_explicit_assumption(hand_feed: pl.DataFrame) -> None:
    no_recv = hand_feed.with_columns(pl.lit(None, pl.Int64).alias("recv_time_ns"))
    caps = CAPS.model_copy(update={"recv_time": "absent"})
    with pytest.raises(FeatureError, match="assumed_feed_latency_ms"):
        build_features(no_recv, caps, SMALL, TimingConfig())
    f = build_features(no_recv, caps, SMALL, TimingConfig(assumed_feed_latency_ms=1))
    assert f.equals(build_features(hand_feed, CAPS, SMALL, TimingConfig()))


def test_unsupported_features_are_refused() -> None:
    top_of_book = CAPS.model_copy(update={"book_depth": 1, "has_trades": False})
    assert "imbalance_l5" not in select_features(FeatureConfig(), top_of_book)
    assert "trade_count_5s" not in select_features(FeatureConfig(), top_of_book)
    with pytest.raises(FeatureError, match="cannot support"):
        select_features(FeatureConfig(names=["imbalance_l5"]), top_of_book)
    with pytest.raises(FeatureError, match="unknown"):
        select_features(FeatureConfig(names=["alpha_magic"]), CAPS)


def test_features_are_invariant_to_truncating_the_future() -> None:
    """Causality: values at t <= T must not change when data after T is deleted."""
    feed = generate(SynthConfig(n_days=1, session_seconds=1200, seed=5))
    cfg = FeatureConfig(warmup_s=10)
    full = build_features(feed, synthetic_capabilities(), cfg, TimingConfig())
    cut = full["t_ns"][len(full) // 2]
    truncated = build_features(
        feed.filter(pl.col("recv_time_ns") <= cut), synthetic_capabilities(), cfg, TimingConfig()
    )
    # The grid ends at the session's last quote, so the truncated grid may stop a point or two
    # earlier; it must be a prefix of the full grid, with identical values on every point.
    past = truncated.drop("fwd_ret", "label_end_ns")
    assert past.height > 500
    assert past.equals(full.head(past.height).drop("fwd_ret", "label_end_ns"))
    assert full.filter(pl.col("t_ns") <= cut).height - past.height <= 1


def test_invalid_book_rows_have_no_mid_or_label() -> None:
    feed = generate(SynthConfig(n_days=1, session_seconds=3600, seed=11))
    f = build_features(feed, synthetic_capabilities(), FeatureConfig(warmup_s=10), TimingConfig())
    invalid = f.filter(~pl.col("valid"))
    assert invalid.height > 0  # the injected packet loss
    assert invalid["mid"].null_count() == invalid.height
    assert invalid["fwd_ret"].null_count() == invalid.height
