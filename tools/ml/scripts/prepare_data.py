#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PIECES = ["I", "O", "T", "S", "Z", "J", "L"]
SCHEMA_ID = "wishuponablock.train.v1"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            if isinstance(parsed, dict):
                yield parsed


def iter_csv(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            raw_data = row.get("data")
            if raw_data:
                try:
                    parsed = json.loads(raw_data)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    yield parsed
                continue
            if isinstance(row, dict):
                yield dict(row)


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from iter_csv(path)
        return
    if suffix in {".jsonl", ".ndjson"}:
        yield from iter_jsonl(path)
        return
    if suffix == ".json":
        text = path.read_text(encoding="utf-8")
        parsed = json.loads(text)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    yield item
        elif isinstance(parsed, dict):
            yield parsed
        return
    raise ValueError(f"Unsupported input file type: {path}")


def encode_row_occupancy(row: str) -> str:
    out: list[str] = []
    for ch in row:
        if ch == ".":
            out.append(".")
        elif ch.isdigit():
            out.append("." if int(ch) <= 0 else "#")
        else:
            out.append("#")
    return "".join(out)


def decode_board(raw: Any) -> list[str]:
    if isinstance(raw, list) and all(isinstance(row, str) for row in raw):
        rows = [encode_row_occupancy(row) for row in raw]
    elif isinstance(raw, str):
        rows = [encode_row_occupancy(row) for row in raw.split("/")]
    else:
        raise ValueError("Unsupported board format")

    if len(rows) != 20:
        raise ValueError(f"Expected 20 rows, got {len(rows)}")
    for row in rows:
        if len(row) != 10:
            raise ValueError(f"Expected row width 10, got {len(row)}")
    return rows


def decode_hold(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return None
        return value if value in DEFAULT_PIECES else None
    if isinstance(raw, (int, float)):
        idx = int(raw) - 1
        if 0 <= idx < len(DEFAULT_PIECES):
            return DEFAULT_PIECES[idx]
    return None


def decode_labels(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, str):
            continue
        piece = entry.strip()
        if piece not in DEFAULT_PIECES:
            continue
        if piece in seen:
            continue
        labels.append(piece)
        seen.add(piece)
    return labels


def read_source_session(record: dict[str, Any]) -> str | None:
    direct = record.get("session_id")
    if isinstance(direct, str) and direct:
        return direct

    source = record.get("source")
    if isinstance(source, dict):
        src = source.get("sessionId")
        if isinstance(src, str) and src:
            return src

    snapshot_meta = record.get("snapshot_meta")
    if isinstance(snapshot_meta, dict):
        session = snapshot_meta.get("session")
        if isinstance(session, dict):
            src = session.get("id")
            if isinstance(src, str) and src:
                return src
    return None


def read_source_sample_index(record: dict[str, Any]) -> int | None:
    direct = record.get("sample_index")
    if isinstance(direct, int):
        return direct

    source = record.get("source")
    if isinstance(source, dict):
        src = source.get("sampleIndex")
        if isinstance(src, int):
            return src

    snapshot_meta = record.get("snapshot_meta")
    if isinstance(snapshot_meta, dict):
        sample = snapshot_meta.get("sample")
        if isinstance(sample, dict):
            src = sample.get("index")
            if isinstance(src, int):
                return src
    return None


def normalize_record(record: dict[str, Any]) -> dict[str, Any] | None:
    try:
        board = decode_board(record.get("board"))
    except Exception:
        return None

    labels = decode_labels(record.get("labels"))
    if not labels:
        return None

    hold = decode_hold(record.get("hold"))
    session_id = read_source_session(record)
    sample_index = read_source_sample_index(record)

    out: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "board": board,
        "hold": hold,
        "labels": labels,
    }
    if session_id is not None:
        out["session_id"] = session_id
        out["source"] = {"sessionId": session_id}
        if sample_index is not None:
            out["source"]["sampleIndex"] = sample_index
            out["sample_index"] = sample_index
    return out


def dedupe_key(record: dict[str, Any]) -> str:
    source = record.get("source")
    session_id = None
    sample_index = None
    if isinstance(source, dict):
        session_id = source.get("sessionId")
        sample_index = source.get("sampleIndex")
    payload = {
        "board": record.get("board"),
        "hold": record.get("hold"),
        "labels": record.get("labels"),
        "sessionId": session_id,
        "sampleIndex": sample_index,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def collect_input_paths(raw_paths: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in raw_paths:
        if any(ch in raw for ch in ("*", "?", "[")):
            matched = sorted(Path().glob(raw))
            paths.extend(p for p in matched if p.is_file())
            continue
        path = Path(raw)
        if path.is_file():
            paths.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge and normalize label exports for training."
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Input files/globs (.jsonl/.csv/.json).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output normalized JSONL path.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate records.",
    )
    args = parser.parse_args()

    inputs = collect_input_paths(args.input)
    if not inputs:
        raise SystemExit("No input files found.")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    deduped = 0
    seen_keys: set[str] = set()

    with output_path.open("w", encoding="utf-8") as out:
        for path in inputs:
            for raw in iter_records(path):
                record = normalize_record(raw)
                if record is None:
                    skipped += 1
                    continue

                if not args.no_dedupe:
                    key = dedupe_key(record)
                    if key in seen_keys:
                        deduped += 1
                        continue
                    seen_keys.add(key)

                out.write(json.dumps(record, separators=(",", ":")))
                out.write("\n")
                written += 1

    print(f"Inputs: {len(inputs)} files")
    print(f"Wrote: {written}")
    print(f"Skipped invalid/empty-label records: {skipped}")
    if not args.no_dedupe:
        print(f"Dropped duplicates: {deduped}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
