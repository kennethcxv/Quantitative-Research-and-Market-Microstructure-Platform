from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from qrp.catalog import Catalog, CatalogError, ValidationFailed
from qrp.synth import SynthConfig, generate, synthetic_capabilities, synthetic_source

SMALL = SynthConfig(n_days=2, session_seconds=600, seed=3)


@pytest.fixture(scope="module")
def feed() -> pl.DataFrame:
    return generate(SMALL)


def _import(cat: Catalog, df: pl.DataFrame, digest: str = "abc", **kw: object):  # type: ignore[no-untyped-def]
    return cat.import_events(
        "synth",
        df,
        source=synthetic_source(),
        capabilities=synthetic_capabilities(),
        adapter="synthetic",
        adapter_params=SMALL.model_dump(mode="json"),
        input_sha256=[digest],
        **kw,  # type: ignore[arg-type]
    )


def test_generator_is_deterministic(feed: pl.DataFrame) -> None:
    assert generate(SMALL).equals(feed)
    assert not generate(SMALL.model_copy(update={"seed": 4})).equals(feed)


def test_import_roundtrip_partitions_and_manifest(tmp_path: Path, feed: pl.DataFrame) -> None:
    cat = Catalog(tmp_path)
    m = _import(cat, feed)
    assert m.n_rows == feed.height and m.duplicates_removed == 0
    assert [f.path.split("/")[2] for f in m.files] == ["date=2024-01-02", "date=2024-01-03"]
    assert m.validation["violations"]["seq_gap"] == 2  # one injected loss episode per day
    back = cat.scan("synth", m.version).collect()
    assert back.drop("date").equals(feed)  # partitioning and sorting preserve feed order
    assert cat.verify("synth", m.version) == []


def test_reimport_is_idempotent_and_revisions_are_versioned(
    tmp_path: Path, feed: pl.DataFrame
) -> None:
    cat = Catalog(tmp_path)
    first = _import(cat, feed)
    assert _import(cat, feed).version == first.version
    assert len(cat.versions("synth")) == 1
    corrected = feed.filter(pl.col("seq") != 10)
    second = _import(cat, corrected, digest="def", revises=first.version, note="drop msg 10")
    assert second.parent_version == first.version
    assert cat.resolve("synth").version == second.version
    assert cat.resolve("synth", as_of_ns=first.created_at_ns).version == first.version
    assert cat.resolve("synth", first.version).n_rows == feed.height  # old version untouched


def test_partition_pruning_filters(tmp_path: Path, feed: pl.DataFrame) -> None:
    cat = Catalog(tmp_path)
    m = _import(cat, feed)
    day2 = cat.scan("synth", m.version, symbols=["SYNTH"], dates=("2024-01-03", "2024-01-03"))
    assert day2.select(pl.col("date").unique()).collect().to_series().to_list()[0].day == 3
    assert cat.scan("synth", m.version, symbols=["OTHER"]).collect().height == 0


def test_duckdb_view_matches_polars(tmp_path: Path, feed: pl.DataFrame) -> None:
    cat = Catalog(tmp_path)
    m = _import(cat, feed)
    con = cat.duckdb("synth", m.version)
    n_trades, volume = con.execute(
        "SELECT count(*), sum(size) FROM events WHERE kind = 'trade'"
    ).fetchone()  # type: ignore[misc]
    trades = feed.filter(pl.col("kind") == "trade")
    assert n_trades == trades.height
    assert volume == pytest.approx(trades["size"].sum())


def test_tampered_file_is_detected(tmp_path: Path, feed: pl.DataFrame) -> None:
    cat = Catalog(tmp_path)
    m = _import(cat, feed)
    target = tmp_path / "synth" / m.version / m.files[0].path
    target.write_bytes(target.read_bytes()[:-10])
    assert cat.verify("synth", m.version) == [m.files[0].path]


def test_validation_errors_block_import_and_leave_nothing(
    tmp_path: Path, feed: pl.DataFrame
) -> None:
    cat = Catalog(tmp_path)
    broken = feed.with_columns(
        pl.when(pl.col("seq") == 50).then(-1.0).otherwise(pl.col("price")).alias("price")
    )
    with pytest.raises(ValidationFailed, match="nonpositive_price_or_size"):
        _import(cat, broken)
    assert cat.versions("synth") == []
    assert not any((tmp_path / "synth").glob("*/manifest.json"))


def test_redelivered_rows_are_dropped_and_counted(tmp_path: Path, feed: pl.DataFrame) -> None:
    cat = Catalog(tmp_path)
    redelivered = pl.concat([feed.head(100), feed.slice(98, 2), feed.slice(100)])
    m = _import(cat, redelivered)
    assert m.duplicates_removed == 2 and m.n_rows == feed.height


def test_capabilities_must_match_data(tmp_path: Path, feed: pl.DataFrame) -> None:
    cat = Catalog(tmp_path)
    with pytest.raises(CatalogError, match="has_trades"):
        _import(cat, feed.filter(pl.col("kind") != "trade"))


def test_changed_adapter_output_for_same_inputs_is_refused(
    tmp_path: Path, feed: pl.DataFrame
) -> None:
    cat = Catalog(tmp_path)
    first = _import(cat, feed)
    assert first.content_sha256 is not None
    # Same input hashes and adapter parameters, different events: the adapter code changed.
    with pytest.raises(CatalogError, match="adapter changed"):
        _import(cat, feed.with_columns(pl.col("size") * 2))


def test_reimport_repairs_a_missing_version_log_entry(tmp_path: Path, feed: pl.DataFrame) -> None:
    cat = Catalog(tmp_path)
    first = _import(cat, feed)
    (tmp_path / "synth" / "versions.jsonl").unlink()  # as if we crashed after the rename
    assert cat.versions("synth") == []
    assert _import(cat, feed).version == first.version
    assert [v["version"] for v in cat.versions("synth")] == [first.version]


def test_content_fingerprint_is_value_based(feed: pl.DataFrame) -> None:
    from qrp.catalog import content_fingerprint

    assert content_fingerprint(feed) == content_fingerprint(feed.clone())
    nudged = feed.with_columns(
        pl.when(pl.int_range(pl.len()) == 7).then(pl.col("price") + 1e-9).otherwise("price")
    )
    assert content_fingerprint(nudged) != content_fingerprint(feed)
    no_recv = feed.with_columns(pl.lit(None, pl.Int64).alias("recv_time_ns"))
    assert content_fingerprint(no_recv) != content_fingerprint(feed)
