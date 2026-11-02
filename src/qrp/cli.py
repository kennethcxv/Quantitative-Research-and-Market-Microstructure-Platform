"""Command-line workflow. Run ``qrp <command> --help`` for options."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from qrp import adapters, experiment, log
from qrp.catalog import Catalog, ValidationFailed
from qrp.synth import SynthConfig, generate, synthetic_capabilities, synthetic_source

logger = logging.getLogger("qrp.cli")


def cmd_fixture(a: argparse.Namespace) -> None:
    cfg = SynthConfig(n_days=a.days, session_seconds=a.seconds, seed=a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    adapters.write_qrp_v1(generate(cfg), out, synthetic_source(), synthetic_capabilities())
    logger.info("wrote synthetic feed", extra={"path": str(out), "config": cfg.model_dump()})


def cmd_fetch_tardis(a: argparse.Namespace) -> None:
    for date in a.date:
        for path in adapters.fetch_tardis(a.exchange, a.symbol, date, Path(a.out)):
            logger.info("downloaded", extra={"path": str(path)})


def cmd_import(a: argparse.Namespace) -> None:
    if a.qrp_v1:
        imported = adapters.read_qrp_v1(Path(a.qrp_v1))
    else:
        raw = Path(a.raw_dir)
        books = [
            raw / adapters.tardis_filename(a.exchange, "incremental_book_L2", d, a.symbol)
            for d in a.tardis
        ]
        trades = [
            raw / adapters.tardis_filename(a.exchange, "trades", d, a.symbol) for d in a.tardis
        ]
        imported = adapters.read_tardis(books, trades, a.exchange, a.tardis)
    try:
        manifest = Catalog(a.catalog).import_events(
            a.name,
            imported.events,
            source=imported.source,
            capabilities=imported.capabilities,
            adapter=imported.adapter,
            adapter_params=imported.params,
            input_sha256=imported.input_sha256,
            revises=a.revises,
            note=a.note,
        )
    except ValidationFailed as exc:
        print(json.dumps(exc.failures, indent=2, default=str), file=sys.stderr)
        raise SystemExit(1) from exc
    violations = {k: v for k, v in manifest.validation["violations"].items() if v}
    summary = {"dataset": a.name, "version": manifest.version, "rows": manifest.n_rows}
    print(json.dumps({**summary, "violations": violations}, indent=2))


def cmd_validate(a: argparse.Namespace) -> None:
    catalog = Catalog(a.catalog)
    manifest = catalog.resolve(a.name, a.version)
    damaged = catalog.verify(a.name, manifest.version)
    report = {
        "dataset": a.name,
        "version": manifest.version,
        "files_intact": not damaged,
        "damaged_files": damaged,
        "rows": manifest.n_rows,
        "duplicates_removed": manifest.duplicates_removed,
        "warnings": {w["name"]: w["violations"] for w in manifest.validation["warnings"]},
        "source": manifest.source.name,
        "synthetic": manifest.source.synthetic,
    }
    print(json.dumps(report, indent=2))
    if damaged:
        raise SystemExit(1)


def cmd_features(a: argparse.Namespace) -> None:
    cfg = experiment.load_config(Path(a.config))
    prep = experiment.prepare(cfg)
    prep.features.write_parquet(a.out)
    logger.info("wrote features", extra={"path": a.out, "rows": prep.features.height})


def cmd_run(a: argparse.Namespace) -> None:
    from qrp.controls import run_controls
    from qrp.report import write_report

    path = Path(a.config)
    run_dir = experiment.run(experiment.load_config(path), path.read_text(), Path(a.runs))
    if not a.no_controls:
        failed = [c["name"] for c in run_controls(run_dir) if not c["passed"]]
        if failed:
            logger.warning("negative controls failed", extra={"controls": failed})
    print(write_report(run_dir))


def cmd_evaluate(a: argparse.Namespace) -> None:
    print(json.dumps(experiment.evaluate_run(Path(a.run_dir)), indent=2, default=str)[:4000])


def cmd_backtest(a: argparse.Namespace) -> None:
    print(json.dumps(experiment.backtest_run(Path(a.run_dir)), indent=2, default=str))


def cmd_controls(a: argparse.Namespace) -> None:
    from qrp.controls import run_controls

    results = run_controls(Path(a.run_dir))
    print(json.dumps(results, indent=2, default=str))
    if not all(r["passed"] for r in results):
        raise SystemExit(1)


def cmd_report(a: argparse.Namespace) -> None:
    from qrp.report import write_report

    print(write_report(Path(a.run_dir)))


def cmd_bench(a: argparse.Namespace) -> None:
    from qrp.bench import run_benchmarks

    run_benchmarks(rows=a.rows, repeats=a.repeats, workdir=Path(a.workdir), out=Path(a.out))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="qrp", description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fixture", help="write the synthetic qrp-v1 feed (+ sidecar)")
    p.add_argument("--out", required=True, help=".parquet, .csv or .ndjson")
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--seconds", type=int, default=4 * 3600)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_fixture)

    p = sub.add_parser("fetch-tardis", help="download free Tardis.dev first-of-month samples")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--symbol", default="LINKUSDT")
    p.add_argument("--date", nargs="+", default=["2024-03-01", "2024-04-01"])
    p.add_argument("--out", default="data/raw/tardis")
    p.set_defaults(func=cmd_fetch_tardis)

    p = sub.add_parser("import", help="validate and store a dataset version")
    p.add_argument("--catalog", default="data/catalog")
    p.add_argument("--name", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--qrp-v1", help="path to a qrp.events.v1 file with sidecar")
    src.add_argument("--tardis", nargs="+", metavar="DATE", help="Tardis sample dates")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--symbol", default="LINKUSDT")
    p.add_argument("--raw-dir", default="data/raw/tardis")
    p.add_argument("--revises", help="parent version this import corrects")
    p.add_argument("--note", help="revision note")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("validate", help="verify stored files and show the integrity report")
    p.add_argument("--catalog", default="data/catalog")
    p.add_argument("--name", required=True)
    p.add_argument("--version", default="latest")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("features", help="compute the feature table for a config")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("run", help="select, test once, evaluate, backtest and report")
    p.add_argument("--config", required=True)
    p.add_argument("--runs", default="runs")
    p.add_argument("--no-controls", action="store_true", help="skip negative controls")
    p.set_defaults(func=cmd_run)

    for name, func, text in (
        ("evaluate", cmd_evaluate, "recompute metrics for a run"),
        ("backtest", cmd_backtest, "re-run the backtest for a run"),
        ("controls", cmd_controls, "run negative controls for a run"),
        ("report", cmd_report, "write report.md for a run"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("run_dir")
        p.set_defaults(func=func)

    p = sub.add_parser("bench", help="benchmark storage/query/feature paths against pandas")
    p.add_argument("--rows", type=int, default=5_000_000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--workdir", default="benchmarks/work")
    p.add_argument("--out", default="benchmarks/results.json")
    p.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    log.configure(args.log_level)
    args.func(args)


if __name__ == "__main__":
    main()
