import { COLS, ROWS } from './constants';
import { TETROMINOES, rotAdd } from './tetromino';
import { getSrsKickTests } from './srs';
import type { ActivePiece, Board, Rotation, Vec2 } from './types';

export function cellsOf(
  piece: ActivePiece,
  r: Rotation = piece.r,
  dx = 0,
  dy = 0,
): Vec2[] {
  const shape = TETROMINOES[piece.k][r];
  return shape.map(([x, y]) => [piece.x + x + dx, piece.y + y + dy]);
}

export function collides(
  board: Board,
  piece: ActivePiece,
  r: Rotation = piece.r,
  dx = 0,
  dy = 0,
): boolean {
  for (const [cx, cy] of cellsOf(piece, r, dx, dy)) {
    if (cx < 0 || cx >= COLS || cy >= ROWS) return true;
    if (cy >= 0 && board[cy][cx] != null) return true;
  }
  return false;
}

export function merge(board: Board, piece: ActivePiece): void {
  for (const [cx, cy] of cellsOf(piece)) {
    if (cy >= 0) board[cy][cx] = piece.k;
  }
}

export function dropDistance(board: Board, piece: ActivePiece): number {
  let d = 0;
  while (!collides(board, piece, piece.r, 0, d + 1)) d++;
  return d;
}

export function tryRotateSRS(
  board: Board,
  piece: ActivePiece,
  dir: -1 | 1,
): boolean {
  const from = piece.r;
  const to = rotAdd(from, dir);

  // SRS defines a list of translation tests depending on (from -> to) and piece type. :contentReference[oaicite:5]{index=5}
  const tests = getSrsKickTests(piece.k, from, to);

  for (const [dx, dy] of tests) {
    if (!collides(board, piece, to, dx, dy)) {
      piece.r = to;
      piece.x += dx;
      piece.y += dy;
      return true;
    }
  }
  return false;
}

export function tryRotate180PreferDirect(
  board: Board,
  piece: ActivePiece,
): boolean {
  const to = ((piece.r + 2) % 4) as Rotation;

  // 1) Direct 180, no kicks, no intermediate state
  if (!collides(board, piece, to, 0, 0)) {
    piece.r = to;
    return true;
  }

  // 2) Fallback: two 90-degree SRS rotations (with kicks)
  return tryRotate180ViaSRS(board, piece);
}

/**
 * This implementation tries CW+CW, then CCW+CCW, using SRS kicks for each 90° step.
 */
export function tryRotate180ViaSRS(board: Board, piece: ActivePiece): boolean {
  const attempt = (dir: -1 | 1): boolean => {
    const tmp: ActivePiece = { ...piece };
    if (!tryRotateSRS(board, tmp, dir)) return false;
    if (!tryRotateSRS(board, tmp, dir)) return false;
    piece.x = tmp.x;
    piece.y = tmp.y;
    piece.r = tmp.r;
    return true;
  };

  return attempt(1) || attempt(-1);
}
