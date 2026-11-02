"""Versioned, partitioned Parquet storage for canonical events.

Layout (see docs/architecture.md section 2.3)::

    <root>/<dataset>/versions.jsonl
    <root>/<dataset>/<version>/manifest.json
    <root>/<dataset>/<version>/events/symbol=<s>/date=<YYYY-MM-DD>/part-0.parquet

A version id is a hash of the raw inputs plus the adapter and its parameters, so importing the
same input twice is a no-op and any change to input or transformation yields a new version.
Versions are written to a temporary directory and renamed into place, so a crash never leaves a
half-written version behind. Nothing ever modifies an existing version.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from qrp.schemas import (
    EVENT_SCHEMA,
    EVENTS_SCHEMA_VERSION,
    Capabilities,
    DatasetManifest,
    FileEntry,
    SourceInfo,
    conform,
)
from qrp.validate import ValidationConfig, deduplicate, validate_events

log = logging.getLogger(__name__)

_HIVE_SCHEMA = {"symbol": pl.String, "date": pl.Date}


class CatalogError(RuntimeError):
    pass


class ValidationFailed(CatalogError):
    def __init__(self, failures: list[dict[str, Any]]) -> None:
        names = ", ".join(f"{f['name']}={f['violations']}" for f in failures)
        super().__init__(f"validation errors block import: {names}")
        self.failures = failures


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def version_id(input_sha256: Sequence[str], adapter: str, params: dict[str, Any]) -> str:
    payload = json.dumps(
        [EVENTS_SCHEMA_VERSION, adapter, params, list(input_sha256)], sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Catalog:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _dir(self, name: str, version: str) -> Path:
        return self.root / name / version

    def versions(self, name: str) -> list[dict[str, Any]]:
        log_path = self.root / name / "versions.jsonl"
        if not log_path.exists():
            return []
        return [json.loads(line) for line in log_path.read_text().splitlines() if line]

    def resolve(
        self, name: str, version: str = "latest", as_of_ns: int | None = None
    ) -> DatasetManifest:
        """Load a manifest. ``latest`` honours ``as_of_ns`` (knowledge time) when given."""
        if version == "latest":
            known = [
                v for v in self.versions(name) if as_of_ns is None or v["created_at_ns"] <= as_of_ns
            ]
            if not known:
                raise CatalogError(f"no versions of {name!r} known as of {as_of_ns}")
            version = known[-1]["version"]
        path = self._dir(name, version) / "manifest.json"
        if not path.exists():
            raise CatalogError(f"unknown dataset version {name}/{version}")
        return DatasetManifest.model_validate_json(path.read_text())

    def import_events(
        self,
        name: str,
        events: pl.DataFrame,
        *,
        source: SourceInfo,
        capabilities: Capabilities,
        adapter: str,
        adapter_params: dict[str, Any],
        input_sha256: Sequence[str],
        revises: str | None = None,
        note: str | None = None,
        validation: ValidationConfig | None = None,
    ) -> DatasetManifest:
        """Validate ``events`` (in capture order) and store them as a new immutable version."""
        version = version_id(input_sha256, adapter, adapter_params)
        final_dir = self._dir(name, version)
        if (final_dir / "manifest.json").exists():
            log.info("dataset version already in catalog", extra={"version": version})
            return self.resolve(name, version)
        if revises is not None:
            self.resolve(name, revises)  # the parent must exist

        events, n_dupes = deduplicate(conform(events))
        sequencing = {k: v.sequencing for k, v in capabilities.streams.items()}
        report = validate_events(events, sequencing, validation)
        if not report.ok:
            raise ValidationFailed([c.model_dump() for c in report.failed("error")])
        if capabilities.has_trades != bool((events["kind"] == "trade").any()):
            raise CatalogError("capabilities.has_trades does not match the data")
        has_recv = events["recv_time_ns"].is_not_null().any()
        if (capabilities.recv_time == "measured") != has_recv:
            raise CatalogError("capabilities.recv_time does not match the data")

        tmp_dir = self.root / name / f".tmp-{version}-{time.time_ns()}"
        try:
            files = self._write_partitions(events, tmp_dir / "events")
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        t = pl.coalesce("recv_time_ns", "event_time_ns")
        bounds = events.select(t.min().alias("lo"), t.max().alias("hi")).row(0)
        manifest = DatasetManifest(
            name=name,
            version=version,
            created_at_ns=time.time_ns(),
            parent_version=revises,
            revision_note=note,
            source=source,
            capabilities=capabilities,
            adapter=adapter,
            adapter_params=adapter_params,
            input_sha256=list(input_sha256),
            n_rows=events.height,
            duplicates_removed=n_dupes,
            symbols=sorted(events["symbol"].unique().to_list()),
            time_min_ns=bounds[0],
            time_max_ns=bounds[1],
            files=files,
            validation={
                **report.summary(),
                "warnings": [c.model_dump() for c in report.failed("warning")],
            },
        )
        (tmp_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
        tmp_dir.rename(final_dir)
        with (self.root / name / "versions.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "version": version,
                        "created_at_ns": manifest.created_at_ns,
                        "parent_version": revises,
                        "note": note,
                    }
                )
                + "\n"
            )
        log.info(
            "imported dataset version",
            extra={"dataset": name, "version": version, "rows": events.height, "files": len(files)},
        )
        return manifest

    @staticmethod
    def _write_partitions(events: pl.DataFrame, out: Path) -> list[FileEntry]:
        # Stable sort: within a stream, feed order is seq order (validated), and rows inside a
        # message keep their original order.
        dated = events.with_columns(
            date=pl.from_epoch(pl.coalesce("recv_time_ns", "event_time_ns"), time_unit="ns")
            .dt.date()
            .cast(pl.String)
        ).sort("symbol", "date", "stream", "seq", maintain_order=True)
        files = []
        for (symbol, date), part in dated.group_by(["symbol", "date"], maintain_order=True):
            rel = Path("events") / f"symbol={symbol}" / f"date={date}" / "part-0.parquet"
            path = out.parent / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            part.drop("symbol", "date").write_parquet(path, compression="zstd", statistics=True)
            files.append(FileEntry(path=str(rel), sha256=sha256_file(path), rows=part.height))
        return files

    def verify(self, name: str, version: str) -> list[str]:
        """Return the files whose bytes no longer match the manifest (empty = intact)."""
        manifest = self.resolve(name, version)
        base = self._dir(name, manifest.version)
        return [
            f.path
            for f in manifest.files
            if not (base / f.path).exists() or sha256_file(base / f.path) != f.sha256
        ]

    def scan(
        self,
        name: str,
        version: str,
        symbols: Sequence[str] | None = None,
        dates: tuple[str, str] | None = None,
    ) -> pl.LazyFrame:
        """Lazy canonical events; symbol/date filters prune partitions before any file is read."""
        manifest = self.resolve(name, version)
        pattern = self._dir(name, manifest.version) / "events" / "**" / "*.parquet"
        lf = pl.scan_parquet(pattern, hive_partitioning=True, hive_schema=_HIVE_SCHEMA)
        if symbols is not None:
            lf = lf.filter(pl.col("symbol").is_in(list(symbols)))
        if dates is not None:
            lo, hi = (pl.lit(d).str.to_date() for d in dates)
            lf = lf.filter(pl.col("date").is_between(lo, hi))
        return lf.select(*EVENT_SCHEMA, "date")

    def duckdb(self, name: str, version: str) -> duckdb.DuckDBPyConnection:
        """In-memory DuckDB connection with an ``events`` view over one dataset version."""
        manifest = self.resolve(name, version)
        pattern = str(self._dir(name, manifest.version) / "events" / "**" / "*.parquet")
        quoted = pattern.replace("'", "''")
        con = duckdb.connect()
        con.execute(
            f"CREATE VIEW events AS SELECT * FROM read_parquet('{quoted}', hive_partitioning=true)"
        )
        return con
