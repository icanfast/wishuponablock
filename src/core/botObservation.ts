import type { Board, GameState, PieceKind } from './types';
import { PIECES } from './types';
import { buildModelHeadInput, type LoadedModel } from './wubModel';

export type BotObservationSpace = 'model_head_v1' | 'raw_v1';

const PIECE_INDEX = new Map(PIECES.map((piece, idx) => [piece, idx]));

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

export type BotObservationParts = {
  board: Board;
  hold: PieceKind | null;
  active: PieceKind;
  next: PieceKind | null;
  nextQueue?: Array<PieceKind | null> | null;
  canHold: boolean;
  totalLinesCleared: number;
  timeMs: number;
  level: number;
  score: number;
  lineGoal: number | null;
};

const RAW_VISIBLE_QUEUE_SLOTS = 5;

const encodeModelHeadObservationFromParts = (
  model: LoadedModel,
  parts: BotObservationParts,
): Float32Array => {
  const headInput = buildModelHeadInput(model, parts.board, parts.hold);
  const contextDim = PIECES.length + PIECES.length + 5;
  const out = new Float32Array(headInput.length + contextDim);
  out.set(headInput, 0);

  let offset = headInput.length;
  const activeIdx = PIECE_INDEX.get(parts.active);
  if (activeIdx != null) out[offset + activeIdx] = 1;
  offset += PIECES.length;

  const nextIdx = parts.next == null ? null : PIECE_INDEX.get(parts.next);
  if (nextIdx != null) out[offset + nextIdx] = 1;
  offset += PIECES.length;

  const lineGoal =
    parts.lineGoal != null && parts.lineGoal > 0 ? parts.lineGoal : null;
  const progress =
    lineGoal != null
      ? clamp(parts.totalLinesCleared / lineGoal, 0, 2)
      : clamp(parts.totalLinesCleared / 80, 0, 2);
  out[offset++] = progress;
  out[offset++] = clamp(parts.timeMs / 180_000, 0, 2);
  out[offset++] = clamp(parts.level / 20, 0, 2);
  out[offset++] = clamp(parts.score / 200_000, 0, 2);
  out[offset++] = parts.canHold ? 1 : 0;
  return out;
};

const encodeRawObservationFromParts = (
  parts: BotObservationParts,
): Float32Array => {
  const rows = parts.board.length;
  const cols = parts.board[0]?.length ?? 0;
  const boardSize = rows * cols;
  // board + active one-hot + hold one-hot(+none) + next one-hot + scalar context
  // + visible queue (up to 5 next pieces as 5 * 7 one-hots)
  const out = new Float32Array(
    boardSize +
      PIECES.length +
      (PIECES.length + 1) +
      PIECES.length +
      5 +
      RAW_VISIBLE_QUEUE_SLOTS * PIECES.length,
  );

  let offset = 0;
  for (let y = 0; y < rows; y += 1) {
    const row = parts.board[y];
    for (let x = 0; x < cols; x += 1) {
      out[offset++] = row[x] == null ? 0 : 1;
    }
  }

  const activeIdx = PIECE_INDEX.get(parts.active);
  if (activeIdx != null) out[offset + activeIdx] = 1;
  offset += PIECES.length;

  if (parts.hold == null) {
    out[offset] = 1;
  } else {
    const holdIdx = PIECE_INDEX.get(parts.hold);
    if (holdIdx != null) out[offset + 1 + holdIdx] = 1;
  }
  offset += PIECES.length + 1;

  const nextIdx = parts.next == null ? null : PIECE_INDEX.get(parts.next);
  if (nextIdx != null) out[offset + nextIdx] = 1;
  offset += PIECES.length;

  const lineGoal =
    parts.lineGoal != null && parts.lineGoal > 0 ? parts.lineGoal : null;
  const progress =
    lineGoal != null
      ? clamp(parts.totalLinesCleared / lineGoal, 0, 2)
      : clamp(parts.totalLinesCleared / 80, 0, 2);
  out[offset++] = progress;
  out[offset++] = clamp(parts.timeMs / 180_000, 0, 2);
  out[offset++] = clamp(parts.level / 20, 0, 2);
  out[offset++] = clamp(parts.score / 200_000, 0, 2);
  out[offset++] = parts.canHold ? 1 : 0;

  const queueSource = Array.isArray(parts.nextQueue) ? parts.nextQueue : [];
  for (let slot = 0; slot < RAW_VISIBLE_QUEUE_SLOTS; slot += 1) {
    const piece = queueSource[slot] ?? null;
    const pieceIdx = piece == null ? null : PIECE_INDEX.get(piece);
    if (pieceIdx != null) out[offset + pieceIdx] = 1;
    offset += PIECES.length;
  }
  return out;
};

export const normalizeBotObservationSpace = (
  value: string | null | undefined,
): BotObservationSpace => (value === 'raw_v1' ? 'raw_v1' : 'model_head_v1');

export const encodeBotObservationFromParts = (options: {
  observationSpace: BotObservationSpace;
  model?: LoadedModel | null;
  parts: BotObservationParts;
}): Float32Array => {
  if (options.observationSpace === 'raw_v1') {
    return encodeRawObservationFromParts(options.parts);
  }
  if (!options.model) {
    throw new Error(
      'encodeBotObservationFromParts: model is required for model_head_v1.',
    );
  }
  return encodeModelHeadObservationFromParts(options.model, options.parts);
};

export const encodeBotObservation = (options: {
  observationSpace: BotObservationSpace;
  model: LoadedModel;
  state: GameState;
}): Float32Array =>
  encodeBotObservationFromParts({
    observationSpace: options.observationSpace,
    model: options.model,
    parts: {
      board: options.state.board,
      hold: options.state.hold,
      active: options.state.active.k,
      next: options.state.next[0] ?? null,
      nextQueue: options.state.next.slice(0, RAW_VISIBLE_QUEUE_SLOTS),
      canHold: options.state.canHold,
      totalLinesCleared: Math.max(
        0,
        Math.trunc(options.state.totalLinesCleared),
      ),
      timeMs: Math.max(0, Math.trunc(options.state.timeMs)),
      level: Math.max(1, Math.trunc(options.state.level)),
      score: Math.max(0, Math.trunc(options.state.score)),
      lineGoal:
        options.state.lineGoal != null && options.state.lineGoal > 0
          ? Math.trunc(options.state.lineGoal)
          : null,
    },
  });
