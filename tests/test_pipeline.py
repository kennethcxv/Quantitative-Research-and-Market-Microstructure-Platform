"""End-to-end: the CLI workflow on generated data, and the qrp-v1 interface fixture."""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl
import pytest

from qrp.adapters import read_qrp_v1, write_qrp_v1
from qrp.book import reconstruct
from qrp.catalog import Catalog
from qrp.cli import main
from qrp.schemas import SchemaError

ROOT = Path(__file__).parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "project1_feed.ndjson"


def test_project1_fixture_conforms_to_the_interface(tmp_path: Path) -> None:
    imported = read_qrp_v1(FIXTURE)
    manifest = Catalog(tmp_path).import_events(
        "project1",
        imported.events,
        source=imported.source,
        capabilities=imported.capabilities,
        adapter=imported.adapter,
        adapter_params=imported.params,
        input_sha256=imported.input_sha256,
    )
    assert manifest.validation["ok"] and manifest.source.synthetic
    assert manifest.validation["violations"]["seq_gap"] == 1  # lost packets in the fixture
    sequencing = {k: v.sequencing for k, v in manifest.capabilities.streams.items()}
    quotes, stats = reconstruct(imported.events, sequencing)
    assert stats.gaps == 1 and stats.snapshots == 2  # start-up + recovery snapshot
    assert quotes.filter(pl.col("synced") & pl.col("crossed")).is_empty()
    assert quotes["synced"][-1]


def test_qrp_v1_reader_rejects_bad_exports(tmp_path: Path) -> None:
    events = read_qrp_v1(FIXTURE).events
    sidecar = json.loads(FIXTURE.with_name(FIXTURE.name + ".qrp.json").read_text())

    def export(name: str, frame: pl.DataFrame, meta: dict[str, object]) -> Path:
        path = tmp_path / name
        frame.write_ndjson(path)
        (tmp_path / f"{name}.qrp.json").write_text(json.dumps(meta))
        return path

    with pytest.raises(SchemaError, match="sidecar"):
        events.write_ndjson(tmp_path / "lonely.ndjson")
        read_qrp_v1(tmp_path / "lonely.ndjson")
    with pytest.raises(SchemaError, match="schema_version"):
        read_qrp_v1(export("v2.ndjson", events, {**sidecar, "schema_version": "qrp.events.v2"}))
    with pytest.raises(SchemaError, match="missing columns"):
        read_qrp_v1(export("cols.ndjson", events.drop("stream"), sidecar))
    garbage = tmp_path / "garbage.ndjson"
    garbage.write_text('{"symbol": "X"\nnot json\n')
    (tmp_path / "garbage.ndjson.qrp.json").write_text(json.dumps(sidecar))
    with pytest.raises(SchemaError):
        read_qrp_v1(garbage)
    # minor versions are accepted; CSV round-trips with nullable timestamps intact
    assert read_qrp_v1(
        export("v11.ndjson", events, {**sidecar, "schema_version": "qrp.events.v1.1"})
    )
    no_recv = events.with_columns(pl.lit(None, pl.Int64).alias("recv_time_ns"))
    imported = read_qrp_v1(FIXTURE)
    write_qrp_v1(no_recv, tmp_path / "rt.csv", imported.source, imported.capabilities)
    assert read_qrp_v1(tmp_path / "rt.csv").events.equals(no_recv)


def _config(tmp: Path, dataset: str) -> Path:
    text = (ROOT / "configs" / "synthetic_demo.toml").read_text()
    text = text.replace('catalog = "data/catalog"', f'catalog = "{tmp / "catalog"}"')
    text = text.replace('dataset = "synthetic"', f'dataset = "{dataset}"')
    text = re.sub(r"n_boot = \d+", "n_boot = 100", text)
    path = tmp / f"{dataset}.toml"
    path.write_text(text)
    return path


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("pipeline")
    feed = tmp / "feed.parquet"
    main(["fixture", "--out", str(feed), "--days", "3", "--seconds", "1800", "--seed", "3"])
    main(
        ["import", "--catalog", str(tmp / "catalog"), "--name", "synthetic", "--qrp-v1", str(feed)]
    )
    main(["validate", "--catalog", str(tmp / "catalog"), "--name", "synthetic"])
    main(["run", "--config", str(_config(tmp, "synthetic")), "--runs", str(tmp / "runs")])
    return tmp


def _runs(tmp: Path, name: str = "synthetic_imbalance") -> list[Path]:
    return sorted((tmp / "runs" / name).iterdir())


def test_full_workflow_produces_a_complete_run(workdir: Path) -> None:
    (run,) = _runs(workdir)
    for artifact in (
        "config.toml",
        "manifest.json",
        "selection.csv",
        "predictions.parquet",
        "metrics.json",
        "backtest.json",
        "controls.json",
        "report.md",
        "fills_test_cost_rule.parquet",
        "equity_test_cost_rule.parquet",
    ):
        assert (run / artifact).exists(), artifact
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["test_evaluations_before_this_run"] == 0
    assert manifest["dataset"]["source"]["synthetic"]
    book = manifest["book"]
    assert book["gaps"] == 3 and book["trades_checked"] > 0
    assert book["trades_inside_stale_quote"] < book["trades_checked"] / 100
    assert all(c["passed"] for c in json.loads((run / "controls.json").read_text()))
    report = (run / "report.md").read_text()
    assert "SYNTHETIC DATA" in report and "## Negative controls" in report
    fills = pl.read_parquet(run / "fills_test_diagnostic_top20pct.parquet")
    assert fills.height > 0 and fills["simulated"].all()


def test_rerun_is_deterministic_and_counts_test_evaluations(workdir: Path) -> None:
    main(
        [
            "run",
            "--config",
            str(workdir / "synthetic.toml"),
            "--runs",
            str(workdir / "runs"),
            "--no-controls",
        ]
    )
    first, second = _runs(workdir)[:2]
    assert (
        json.loads((second / "manifest.json").read_text())["test_evaluations_before_this_run"] == 1
    )
    for name in ("predictions.parquet", "fills_test_diagnostic_top20pct.parquet"):
        assert pl.read_parquet(first / name).equals(pl.read_parquet(second / name)), name
    for name in ("metrics.json", "backtest.json", "selection.csv", "model.json"):
        assert (first / name).read_text() == (second / name).read_text(), name


def test_test_period_data_cannot_influence_model_selection(workdir: Path) -> None:
    # Same events, except the test day's bid sizes are tripled: test features change, dev don't.
    events = pl.read_parquet(workdir / "feed.parquet")
    test_day = pl.from_epoch("recv_time_ns", time_unit="ns").dt.date() == pl.date(2024, 1, 4)
    altered = events.with_columns(
        pl.when(test_day & (pl.col("side") == 1) & (pl.col("kind") != "trade"))
        .then(pl.col("size") * 3)
        .otherwise(pl.col("size"))
        .alias("size")
    )
    imported = read_qrp_v1(workdir / "feed.parquet")
    write_qrp_v1(altered, workdir / "altered.parquet", imported.source, imported.capabilities)
    main(
        [
            "import",
            "--catalog",
            str(workdir / "catalog"),
            "--name",
            "altered",
            "--qrp-v1",
            str(workdir / "altered.parquet"),
        ]
    )
    main(
        [
            "run",
            "--config",
            str(_config(workdir, "altered")),
            "--runs",
            str(workdir / "runs"),
            "--no-controls",
        ]
    )
    clean, changed = _runs(workdir)[0], _runs(workdir)[-1]
    assert (clean / "selection.csv").read_text() == (changed / "selection.csv").read_text()

    def split(run: Path, name: str) -> pl.DataFrame:
        return pl.read_parquet(run / "predictions.parquet").filter(pl.col("split") == name)

    assert split(clean, "dev_oos").equals(split(changed, "dev_oos"))
    assert not split(clean, "test")["pred"].equals(split(changed, "test")["pred"])


def test_signals_cover_rows_without_labels(workdir: Path) -> None:
    run = _runs(workdir)[0]
    preds = pl.read_parquet(run / "predictions.parquet").filter(pl.col("split") == "test")
    signals = pl.read_parquet(run / "signals.parquet").filter(pl.col("split") == "test")
    # Every labelled test row has the same forecast in both; signals also cover the unlabelled
    # rows (e.g. the last horizon of the session), so they never depend on the future.
    joined = preds.join(signals, on="t_ns", suffix="_signal")
    assert joined.height == preds.height
    assert (joined["pred"] == joined["pred_signal"]).all()
    assert signals.height > preds.height


def test_stage_reruns_refuse_an_edited_config(workdir: Path) -> None:
    from qrp.experiment import RunIntegrityError, evaluate_run

    run = _runs(workdir)[0]
    original = (run / "config.toml").read_text()
    try:
        (run / "config.toml").write_text(
            original.replace("taker_fee_bps = 2.0", "taker_fee_bps = 0.0")
        )
        with pytest.raises(RunIntegrityError):
            evaluate_run(run)
        with pytest.raises(RunIntegrityError):
            main(["backtest", str(run)])
    finally:
        (run / "config.toml").write_text(original)


def test_test_access_ledger_counts_across_runs_directories(workdir: Path) -> None:
    ledger = workdir / "catalog" / "synthetic" / "test_access.jsonl"
    before = len(ledger.read_text().splitlines())
    elsewhere = workdir / "other_runs"
    main(
        [
            "run",
            "--config",
            str(workdir / "synthetic.toml"),
            "--runs",
            str(elsewhere),
            "--no-controls",
        ]
    )
    (run,) = sorted((elsewhere / "synthetic_imbalance").iterdir())
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["test_evaluations_before_this_run"] == before
    assert len(ledger.read_text().splitlines()) == before + 1


def test_features_command_excludes_the_test_period_by_default(workdir: Path) -> None:
    out = workdir / "features.parquet"
    main(["features", "--config", str(workdir / "synthetic.toml"), "--out", str(out)])
    sessions = pl.read_parquet(out)["session"].unique().sort().to_list()
    assert [str(d) for d in sessions] == ["2024-01-02", "2024-01-03"]


def test_as_of_resolves_latest_by_knowledge_time(workdir: Path) -> None:
    from qrp.catalog import CatalogError
    from qrp.experiment import load_config, prepare

    cfg = load_config(workdir / "synthetic.toml")
    cfg.data.as_of = "2000-01-01T00:00:00"  # before any version was imported
    with pytest.raises(CatalogError, match="no versions"):
        prepare(cfg)
    cfg.data.as_of = "2100-01-01T00:00:00+00:00"
    assert prepare(cfg).manifest.name == "synthetic"
