#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PIECES = ["I", "O", "T", "S", "Z", "J", "L"]
SCHEMA_ID = "wishuponablock.train.v1"


def iter_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


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


def decode_board(encoded: Any) -> list[str]:
    if isinstance(encoded, list) and all(isinstance(row, str) for row in encoded):
        return [encode_row_occupancy(row) for row in encoded]
    if not isinstance(encoded, str):
        raise ValueError("Unsupported board encoding.")
    rows = encoded.split("/")
    return [encode_row_occupancy(row) for row in rows]


def decode_hold(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value if value else None
    if isinstance(value, (int, float)):
        idx = int(value) - 1
        if idx < 0:
            return None
        if idx < len(DEFAULT_PIECES):
            return DEFAULT_PIECES[idx]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert D1 label CSV export into training JSONL format."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        default=str(Path(__file__).with_name("online_labels.csv")),
        help="Path to online_labels.csv export.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default=str(Path(__file__).with_name("labels_v1.jsonl")),
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in iter_rows(input_path):
            raw = row.get("data")
            if not raw:
                skipped += 1
                continue
            try:
                record = json.loads(raw)
            except Exception:
                skipped += 1
                continue

            try:
                board = decode_board(record.get("board"))
            except Exception:
                skipped += 1
                continue

            hold = decode_hold(record.get("hold"))
            labels_raw = record.get("labels") or []
            labels: list[str] = [
                label for label in labels_raw if label in DEFAULT_PIECES
            ]
            source = record.get("source") or {}
            session_id = None
            if isinstance(source, dict):
                session_id = source.get("sessionId")

            payload = {
                "schema": SCHEMA_ID,
                "board": board,
                "hold": hold,
                "labels": labels,
                "session_id": session_id,
            }
            handle.write(json.dumps(payload))
            handle.write("\n")
            count += 1

    print(f"Wrote {count} records to {output_path} (skipped {skipped}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
