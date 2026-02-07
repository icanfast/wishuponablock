#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Any

DEFAULT_PIECES = ["I", "O", "T", "S", "Z", "J", "L"]
SCHEMA_ID = "wishuponablock.train.v1"


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            yield from data
        else:
            yield data
        return

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_piece_order(value: Any) -> list[str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return DEFAULT_PIECES


def decode_hold(value: Any, order: list[str]) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value if value else None
    if isinstance(value, (int, float)):
        idx = int(value) - 1
        if idx < 0:
            return None
        if idx < len(order):
            return order[idx]
        if idx < len(DEFAULT_PIECES):
            return DEFAULT_PIECES[idx]
    return None


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
    decoded: list[str] = []
    for row in rows:
        decoded.append(encode_row_occupancy(row))
    return decoded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert labels.jsonl to the intermediate training format.",
    )
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    parser.add_argument(
        "--include-source",
        action="store_true",
        help="Include source metadata in the output.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in iter_records(input_path):
            order = resolve_piece_order(record.get("pieceOrder"))
            board = decode_board(record.get("board"))
            hold = decode_hold(record.get("hold"), order)
            labels = record.get("labels") or []
            source = record.get("source") or {}
            session_id = None
            if isinstance(source, dict):
                session_id = source.get("sessionId")
            payload = {
                "schema": SCHEMA_ID,
                "board": board,
                "hold": hold,
                "labels": list(labels),
                "session_id": session_id,
            }
            if args.include_source and "source" in record:
                payload["source"] = record["source"]
            handle.write(json.dumps(payload))
            handle.write("\n")
            count += 1

    print(f"Wrote {count} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
