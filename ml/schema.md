# Intermediate Training Format (v1)

**File type:** JSON Lines (`.jsonl`) - one JSON object per line.

Each record contains only the information needed for training.

```json
{
  "schema": "wishuponablock.train.v1",
  "board": [
    "..........",
    "....##....",
    "...###....",
    ".........."
  ],
  "hold": "I",
  "labels": ["T", "S", "I"],
  "session_id": "2026-02-05T10:12:48.123Z"
}
```

## Notes

- `board` rows are **top to bottom**, 10 columns each.
- `.` means empty, `#` means filled.
- `hold` is optional and may be `null`.
- `labels` preserves the order the user selected pieces.
- `session_id` groups samples from the same recording session.

The conversion script accepts raw `labels.jsonl` from the labeling tool and outputs this format.
