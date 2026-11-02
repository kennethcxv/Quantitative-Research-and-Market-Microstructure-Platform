# Event interface `qrp.events.v1`

The contract between producers of market events (Project 1: the C++ market-data engine;
Project 2: the distributed event log) and this research platform. A producer conforms if
`qrp import --format qrp-v1` accepts its output with no `error`-severity validation failures.

## Files

An export is a data file plus a JSON sidecar describing it:

```
events.parquet            (or .csv / .ndjson; columns below, any row-group layout)
events.parquet.qrp.json   (sidecar, schema below)
```

Rows must be in **capture order per stream**: the order in which the producer's gateway
received them. Different streams may be interleaved arbitrarily.

## Columns

| Column | Type | Null | Semantics |
|--------|------|------|-----------|
| `symbol` | string | no | Instrument identifier as the venue spells it. |
| `stream` | string | no | Feed channel the row arrived on, e.g. `itch`, `depth`, `trades`. Must be declared in the sidecar. |
| `seq` | int64 | no | Message number within `(symbol, stream)`. All rows of one message share it. |
| `event_time_ns` | int64 | yes | Venue timestamp, UTC epoch nanoseconds. Null if the venue sends none — never a copy of `recv_time_ns`. |
| `recv_time_ns` | int64 | yes | Time the producer's gateway received the message, UTC epoch ns, from a disciplined clock (PTP/NTP). Null if not captured. |
| `kind` | string | no | `book`, `snapshot` or `trade`. |
| `side` | int8 | no | `book`/`snapshot`: `+1` bid, `-1` ask. `trade`: aggressor `+1` buy, `-1` sell, `0` unknown. |
| `price` | float64 | no | Level price or trade price, in quote currency. |
| `size` | float64 | no | `book`: new total size at the level (`0` deletes it). `snapshot`: size at the level (> 0). `trade`: quantity (> 0). |

At least one of the two timestamps must be present on every row.

### Kinds

* **`book`** — an incremental level update. Sizes are absolute, not deltas, so replaying a
  message twice is harmless and a lost delta cannot silently corrupt every later size.
* **`snapshot`** — all rows of a snapshot message together are the complete book (both sides).
  Producers send one at start-up and after any recovery. A message may not mix `snapshot` and
  `book` rows.
* **`trade`** — a print. It may share a message with the `book` rows it caused (a sweep is one
  message), which lets the consumer see the trade and the depleted levels atomically.

## Sidecar

```json
{
  "schema_version": "qrp.events.v1",
  "source": {
    "name": "market-data-engine replay of <feed>",
    "license": "who may use and redistribute this data",
    "synthetic": false,
    "url": null,
    "notes": ""
  },
  "capabilities": {
    "book_depth": null,
    "has_trades": true,
    "recv_time": "measured",
    "event_time": "exchange",
    "streams": {"itch": {"sequencing": "contiguous", "description": "per-channel packet seq"}}
  }
}
```

* `book_depth`: price levels per side the feed carries; `null` for full depth.
* `recv_time`: `measured` or `absent`. `event_time`: `exchange`, or `partial` if some rows lack it.
* `sequencing` per stream:
  * `contiguous` — `seq` comes from the protocol and increases by exactly one per message. A
    jump means messages were lost; the consumer invalidates the book until the next snapshot.
  * `monotonic` — `seq` comes from the protocol and increases, but may skip legitimately.
  * `derived` — the producer numbered messages itself; loss cannot be detected from `seq`.

## Versioning

Additive, backwards-compatible changes (a new optional column, a new sidecar field) bump the
minor version: `qrp.events.v1.1`. Anything that changes the meaning of an existing column is
`qrp.events.v2`, and the importer refuses versions it does not know.

## Reference fixture

`tests/fixtures/project1_feed.ndjson` (+ sidecar) is a small, openly redistributable synthetic
feed in this format, with a sweep, a sequence gap and a recovery snapshot. It is what the
cross-project integration test imports; a real Project 1 export should pass the same test.
