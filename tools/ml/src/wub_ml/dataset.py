from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset

DEFAULT_PIECES = ["I", "O", "T", "S", "Z", "J", "L"]


@dataclass
class Sample:
    board: torch.Tensor
    hold: int
    labels: list[int]
    session_id: str
    sample_index: int | None


def iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_paths(path: Path | str | Sequence[Path | str]) -> list[Path]:
    if isinstance(path, (list, tuple)):
        return [Path(p) for p in path]
    return [Path(path)]


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


def decode_board(rows: Any) -> list[str]:
    if isinstance(rows, list) and all(isinstance(r, str) for r in rows):
        return [encode_row_occupancy(r) for r in rows]
    if isinstance(rows, str):
        split_rows = rows.split("/")
        return [encode_row_occupancy(r) for r in split_rows]
    raise ValueError("Board must be a list of strings or '/'-delimited string.")


def board_to_tensor(rows: list[str]) -> torch.Tensor:
    if len(rows) != 20:
        raise ValueError(f"Expected 20 board rows, got {len(rows)}.")
    occupancy: list[list[float]] = []
    row_fill: list[list[float]] = []
    for row in rows:
        if len(row) != 10:
            raise ValueError("Each board row must have 10 columns.")
        occ = [0.0 if ch == "." else 1.0 for ch in row]
        occupancy.append(occ)
        fill_ratio = sum(occ) / len(occ)
        row_fill.append([fill_ratio] * len(occ))

    height = len(occupancy)
    width = len(occupancy[0])
    holes: list[list[float]] = [[0.0 for _ in range(width)] for _ in range(height)]
    for x in range(width):
        filled_seen = False
        for y in range(height):
            if occupancy[y][x] > 0:
                filled_seen = True
            elif filled_seen:
                holes[y][x] = 1.0

    tensor = torch.tensor([occupancy, holes, row_fill], dtype=torch.float32)
    return tensor


class LabelsDataset(Dataset[Sample]):
    def __init__(
        self,
        path: Path | str | Sequence[Path | str],
        piece_order: list[str] | None = None,
        drop_empty_labels: bool = True,
        mirror_prob: float = 0.0,
        virtual_session_size: int = 100,
    ) -> None:
        self.paths = normalize_paths(path)
        self.pieces = piece_order or DEFAULT_PIECES
        self.piece_to_idx = {p: i for i, p in enumerate(self.pieces)}
        self.samples: list[Sample] = []
        self.skipped = 0
        self.mirror_prob = max(0.0, min(1.0, float(mirror_prob)))
        self.virtual_session_size = max(0, int(virtual_session_size))
        self.input_channels = 3
        self.session_ids: list[str] = []

        for source_path in self.paths:
            for record in iter_json_records(source_path):
                try:
                    rows = decode_board(record.get("board"))
                    board = board_to_tensor(rows)
                except Exception:
                    self.skipped += 1
                    continue

                hold = record.get("hold")
                if isinstance(hold, (int, float)):
                    hold_idx = int(hold) - 1
                elif isinstance(hold, str):
                    hold_idx = self.piece_to_idx.get(hold, -1)
                else:
                    hold_idx = -1

                labels_raw = record.get("labels") or []
                labels: list[int] = []
                for label in labels_raw:
                    idx = self.piece_to_idx.get(label)
                    if idx is not None:
                        labels.append(idx)

                if drop_empty_labels and not labels:
                    self.skipped += 1
                    continue

                session_id = record.get("session_id")
                if not isinstance(session_id, str) or not session_id:
                    source = record.get("source")
                    if isinstance(source, dict):
                        source_id = source.get("sessionId")
                        if isinstance(source_id, str) and source_id:
                            session_id = source_id
                if not isinstance(session_id, str) or not session_id:
                    session_id = f"unknown_{len(self.samples)}"
                sample_index = read_sample_index(record)
                split_session_id = build_virtual_session_id(
                    session_id,
                    sample_index,
                    self.virtual_session_size,
                )
                self.samples.append(
                    Sample(
                        board=board,
                        hold=hold_idx,
                        labels=labels,
                        session_id=split_session_id,
                        sample_index=sample_index,
                    )
                )
                self.session_ids.append(split_session_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        sample = self.samples[idx]
        if self.mirror_prob <= 0:
            return sample
        if torch.rand(()) >= self.mirror_prob:
            return sample
        return mirror_sample(sample)


def collate_samples(batch: list[Sample]) -> dict[str, Any]:
    boards = torch.stack([item.board for item in batch], dim=0)
    holds = torch.tensor([item.hold for item in batch], dtype=torch.long)
    labels = [item.labels for item in batch]
    return {"board": boards, "hold": holds, "labels": labels}


def mirror_sample(sample: Sample) -> Sample:
    mirrored_board = torch.flip(sample.board, dims=[2])
    hold = mirror_piece_index(sample.hold)
    labels = [mirror_piece_index(idx) for idx in sample.labels]
    return Sample(
        board=mirrored_board,
        hold=hold,
        labels=labels,
        session_id=sample.session_id,
        sample_index=sample.sample_index,
    )


def mirror_piece_index(idx: int) -> int:
    # -1 means "no hold"
    if idx < 0:
        return idx
    mapping = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 6, 6: 5}
    return mapping.get(idx, idx)


def read_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def read_sample_index(record: dict[str, Any]) -> int | None:
    direct = read_int(record.get("sample_index"))
    if direct is not None:
        return direct

    source = record.get("source")
    if isinstance(source, dict):
        from_source = read_int(source.get("sampleIndex"))
        if from_source is not None:
            return from_source

    snapshot_meta = record.get("snapshot_meta")
    if isinstance(snapshot_meta, dict):
        sample = snapshot_meta.get("sample")
        if isinstance(sample, dict):
            from_meta = read_int(sample.get("index"))
            if from_meta is not None:
                return from_meta
    return None


def build_virtual_session_id(
    session_id: str,
    sample_index: int | None,
    virtual_session_size: int,
) -> str:
    if virtual_session_size <= 0 or sample_index is None:
        return session_id
    if sample_index < 0:
        return session_id
    batch_id = sample_index // virtual_session_size
    return f"{session_id}::b{batch_id}"
