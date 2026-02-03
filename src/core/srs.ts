import type { PieceKind, Rotation, Vec2 } from './types';

// SRS basic-rotation kick data (y is "up" here).
// We convert to our game coords (y down) by flipping dy.

type KickTable = Record<
  Rotation,
  Partial<Record<Rotation, readonly Vec2[]>>
>;

const JLSTZ_KICKS: KickTable = {
  0: {
    1: [
      [0, 0],
      [-1, 0],
      [-1, 1],
      [0, -2],
      [-1, -2],
    ],
    3: [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, -2],
      [1, -2],
    ],
  },
  1: {
    0: [
      [0, 0],
      [1, 0],
      [1, -1],
      [0, 2],
      [1, 2],
    ],
    2: [
      [0, 0],
      [1, 0],
      [1, -1],
      [0, 2],
      [1, 2],
    ],
  },
  2: {
    1: [
      [0, 0],
      [-1, 0],
      [-1, 1],
      [0, -2],
      [-1, -2],
    ],
    3: [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, -2],
      [1, -2],
    ],
  },
  3: {
    2: [
      [0, 0],
      [-1, 0],
      [-1, -1],
      [0, 2],
      [-1, 2],
    ],
    0: [
      [0, 0],
      [-1, 0],
      [-1, -1],
      [0, 2],
      [-1, 2],
    ],
  },
};

const I_KICKS: KickTable = {
  0: {
    1: [
      [0, 0],
      [-2, 0],
      [1, 0],
      [-2, -1],
      [1, 2],
    ],
    3: [
      [0, 0],
      [-1, 0],
      [2, 0],
      [-1, 2],
      [2, -1],
    ],
  },
  1: {
    0: [
      [0, 0],
      [2, 0],
      [-1, 0],
      [2, 1],
      [-1, -2],
    ],
    2: [
      [0, 0],
      [-1, 0],
      [2, 0],
      [-1, 2],
      [2, -1],
    ],
  },
  2: {
    1: [
      [0, 0],
      [1, 0],
      [-2, 0],
      [1, -2],
      [-2, 1],
    ],
    3: [
      [0, 0],
      [2, 0],
      [-1, 0],
      [2, 1],
      [-1, -2],
    ],
  },
  3: {
    2: [
      [0, 0],
      [-2, 0],
      [1, 0],
      [-2, -1],
      [1, 2],
    ],
    0: [
      [0, 0],
      [1, 0],
      [-2, 0],
      [1, -2],
      [-2, 1],
    ],
  },
};

function srsToGame(dx: number, dyUp: number): Vec2 {
  // Our y grows down, SRS y grows up → flip sign.
  return [dx, -dyUp];
}

/**
 * Returns the list of translation tests (dx,dy) to apply *after* basic rotation,
 * in our game coordinate system (y down).
 *
 * For JLSTZ + I: 5 tests. For O: no kicks.
 */
export function getSrsKickTests(
  kind: PieceKind,
  from: Rotation,
  to: Rotation,
): Vec2[] {
  if (kind === 'O') return [[0, 0]];

  const table = kind === 'I' ? I_KICKS : JLSTZ_KICKS;
  const tests = table[from]?.[to];
  if (!tests) return [];

  return tests.map(([x, y]) => srsToGame(x, y));
}
