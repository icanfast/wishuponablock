from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

DEFAULT_PIECES = ["I", "O", "T", "S", "Z", "J", "L"]


@dataclass
class Sample:
    board: torch.Tensor
    hold: int
    labels: list[int]
    session_id: str


def iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def decode_board(rows: Any) -> list[str]:
    if not isinstance(rows, list) or not all(isinstance(r, str) for r in rows):
        raise ValueError("Board must be a list of strings.")
    return list(rows)


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
        path: Path | str,
        piece_order: list[str] | None = None,
        drop_empty_labels: bool = True,
        mirror_prob: float = 0.0,
    ) -> None:
        self.path = Path(path)
        self.pieces = piece_order or DEFAULT_PIECES
        self.piece_to_idx = {p: i for i, p in enumerate(self.pieces)}
        self.samples: list[Sample] = []
        self.skipped = 0
        self.mirror_prob = max(0.0, min(1.0, float(mirror_prob)))
        self.input_channels = 3
        self.session_ids: list[str] = []

        for record in iter_json_records(self.path):
            try:
                rows = decode_board(record.get("board"))
                board = board_to_tensor(rows)
            except Exception:
                self.skipped += 1
                continue

            hold = record.get("hold")
            hold_idx = self.piece_to_idx.get(hold, -1) if hold else -1

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
                session_id = f"unknown_{len(self.samples)}"
            self.samples.append(
                Sample(
                    board=board,
                    hold=hold_idx,
                    labels=labels,
                    session_id=session_id,
                )
            )
            self.session_ids.append(session_id)

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
    )


def mirror_piece_index(idx: int) -> int:
    # -1 means "no hold"
    if idx < 0:
        return idx
    mapping = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 6, 6: 5}
    return mapping.get(idx, idx)
