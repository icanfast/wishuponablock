import { Graphics } from 'pixi.js';
import { COLS, ROWS } from '../core/constants';
import { cellsOf } from '../core/piece';
import type { GameState, PieceKind } from '../core/types';

const COLORS: Record<PieceKind, number> = {
  I: 0x4dd3ff,
  O: 0xffd84d,
  T: 0xc77dff,
  S: 0x6eea6e,
  Z: 0xff6b6b,
  J: 0x4d7cff,
  L: 0xffa94d,
};

export class PixiRenderer {
  constructor(
    private gfx: Graphics,
    private cell = 28,
    private boardX = 40,
    private boardY = 40,
  ) {}

  render(state: GameState): void {
    const { gfx, cell, boardX, boardY } = this;

    gfx.clear();

    // background + frame
    gfx
      .rect(boardX - 2, boardY - 2, COLS * cell + 4, ROWS * cell + 4)
      .fill(0x121a24);
    gfx.rect(boardX, boardY, COLS * cell, ROWS * cell).fill(0x0b0f14);

    // settled blocks
    for (let y = 0; y < ROWS; y++) {
      for (let x = 0; x < COLS; x++) {
        const k = state.board[y][x];
        if (!k) continue;
        drawCell(gfx, boardX, boardY, cell, x, y, COLORS[k]);
      }
    }

    if (state.gameOver) return;

    // ghost
    const ghostColor = dim(COLORS[state.active.k], 0.35);
    const ghostPiece = { ...state.active, y: state.ghostY };
    for (const [x, y] of cellsOf(ghostPiece)) {
      if (y < 0) continue;
      drawCell(gfx, boardX, boardY, cell, x, y, ghostColor);
    }

    // active
    const color = COLORS[state.active.k];
    for (const [x, y] of cellsOf(state.active)) {
      if (y < 0) continue;
      drawCell(gfx, boardX, boardY, cell, x, y, color);
    }
  }
}

function drawCell(
  gfx: Graphics,
  boardX: number,
  boardY: number,
  cell: number,
  x: number,
  y: number,
  color: number,
): void {
  const px = boardX + x * cell;
  const py = boardY + y * cell;
  const pad = 2;
  gfx.rect(px + pad, py + pad, cell - pad * 2, cell - pad * 2).fill(color);
}

function dim(color: number, factor: number): number {
  const r = ((color >> 16) & 0xff) * factor;
  const g = ((color >> 8) & 0xff) * factor;
  const b = (color & 0xff) * factor;
  return ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
}
