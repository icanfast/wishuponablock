import { COLS, ROWS } from './constants';
import type { Board } from './types';

export function makeBoard(): Board {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(null));
}

export function clearLines(board: Board): number {
  let cleared = 0;
  for (let y = ROWS - 1; y >= 0; y--) {
    if (board[y].every((c) => c != null)) {
      board.splice(y, 1);
      board.unshift(Array(COLS).fill(null));
      cleared++;
      y++;
    }
  }
  return cleared;
}
