# Training Input Format (v1)

**File type:** JSON Lines (`.jsonl`) - one JSON object per line.

The trainer now accepts the raw labeling output directly. It only uses the
fields described below and ignores the rest.

```json
{
  "schema": "wishuponablock.train.v1",
  "board": "0000000000/0000000000/0000000000/0000000000/0000000000/0000000000/0000000000/0000000000/0000000000/0000000000/3000000000/3300044000/3666445500/7776666550/7777226220/7744226220/7446666660/7766664430/1111644330/2222111130",
  "hold": "I",
  "labels": ["T", "S", "I"],
  "source": {
    "sessionId": "2026-02-05T10:12:48.123Z"
  }
}
```

## Notes

- `board` rows are **top to bottom**, 10 columns each.
- Digits represent piece IDs; the trainer only uses occupancy (empty vs filled).
- `hold` is optional and may be `null`.
- `labels` preserves the order the user selected pieces.
- `session_id` is inferred from `source.sessionId` when available.
- `sample_index` is inferred from `source.sampleIndex` when available.
