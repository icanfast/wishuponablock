import { Graphics } from 'pixi.js';
import {
  BOARD_CELL_PX,
  BOARD_X,
  BOARD_Y,
  COLS,
  HOLD_COLS,
  HOLD_INNER_Y,
  HOLD_LABEL_HEIGHT,
  HOLD_PANEL_HEIGHT,
  HOLD_WIDTH,
  HOLD_X,
  HOLD_Y,
  NEXT_COUNT,
  QUEUE_COLS,
  QUEUE_GAP_PX,
  QUEUE_PREVIEW_HEIGHT,
  QUEUE_X,
  QUEUE_Y,
  ROWS,
} from '../core/constants';
import { cellsOf } from '../core/piece';
import { TETROMINOES } from '../core/tetromino';
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

const BORDER_COLOR = 0x1f2a37;
const BOARD_BORDER = 3;
const PANEL_BORDER = 3;

export class PixiRenderer {
  constructor(
    private gfx: Graphics,
    private cell = BOARD_CELL_PX,
    private boardX = BOARD_X,
    private boardY = BOARD_Y,
  ) {}

  render(state: GameState): void {
    const { gfx, cell, boardX, boardY } = this;

    gfx.clear();

    // board background + frame
    gfx
      .rect(
        boardX - BOARD_BORDER,
        boardY - BOARD_BORDER,
        COLS * cell + BOARD_BORDER * 2,
        ROWS * cell + BOARD_BORDER * 2,
      )
      .fill(BORDER_COLOR);
    gfx.rect(boardX, boardY, COLS * cell, ROWS * cell).fill(0x0b0f14);

    this.renderHold(state);
    this.renderQueue(state);

    // settled blocks
    for (let y = 0; y < ROWS; y++) {
      for (let x = 0; x < COLS; x++) {
        const k = state.board[y][x];
        if (!k) continue;
        drawCell(gfx, boardX, boardY, cell, x, y, COLORS[k]);
      }
    }

    if (state.gameOver || state.gameWon) return;

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

  renderBoardOnly(board: GameState['board'], hold?: PieceKind | null): void {
    const { gfx, cell, boardX, boardY } = this;
    gfx.clear();

    gfx
      .rect(
        boardX - BOARD_BORDER,
        boardY - BOARD_BORDER,
        COLS * cell + BOARD_BORDER * 2,
        ROWS * cell + BOARD_BORDER * 2,
      )
      .fill(BORDER_COLOR);
    gfx.rect(boardX, boardY, COLS * cell, ROWS * cell).fill(0x0b0f14);

    if (hold !== undefined) {
      this.renderHoldPiece(hold);
    }

    for (let y = 0; y < ROWS; y++) {
      for (let x = 0; x < COLS; x++) {
        const k = board[y][x];
        if (!k) continue;
        drawCell(gfx, boardX, boardY, cell, x, y, COLORS[k]);
      }
    }
  }

  private renderHoldPiece(hold: PieceKind | null): void {
    const { gfx, cell } = this;

    gfx
      .rect(
        HOLD_X - PANEL_BORDER,
        HOLD_Y - PANEL_BORDER,
        HOLD_WIDTH + PANEL_BORDER * 2,
        HOLD_PANEL_HEIGHT + PANEL_BORDER * 2,
      )
      .fill(BORDER_COLOR);
    gfx.rect(HOLD_X, HOLD_Y, HOLD_WIDTH, HOLD_PANEL_HEIGHT).fill(0x0b0f14);
    gfx.rect(HOLD_X, HOLD_Y, HOLD_WIDTH, HOLD_LABEL_HEIGHT).fill(0x121a24);

    if (!hold) return;

    const offsetX = Math.floor((HOLD_COLS - 4) / 2) * cell;
    this.drawPreviewPiece(hold, HOLD_X + offsetX, HOLD_INNER_Y, cell);
  }

  private renderQueue(state: GameState): void {
    const { gfx, cell } = this;
    const count = Math.min(state.next.length, NEXT_COUNT);
    if (count === 0) return;

    const panelHeight =
      count * QUEUE_PREVIEW_HEIGHT + (count - 1) * QUEUE_GAP_PX;
    const panelWidth = QUEUE_COLS * cell;

    gfx
      .rect(
        QUEUE_X - PANEL_BORDER,
        QUEUE_Y - PANEL_BORDER,
        panelWidth + PANEL_BORDER * 2,
        panelHeight + PANEL_BORDER * 2,
      )
      .fill(BORDER_COLOR);
    gfx.rect(QUEUE_X, QUEUE_Y, panelWidth, panelHeight).fill(0x0b0f14);

    const offsetX = Math.floor((QUEUE_COLS - 4) / 2) * cell;

    for (let i = 0; i < count; i++) {
      const kind = state.next[i];
      const boxY = QUEUE_Y + i * (QUEUE_PREVIEW_HEIGHT + QUEUE_GAP_PX);
      this.drawPreviewPiece(kind, QUEUE_X + offsetX, boxY, cell);
    }
  }

  private renderHold(state: GameState): void {
    const { gfx, cell } = this;

    gfx
      .rect(
        HOLD_X - PANEL_BORDER,
        HOLD_Y - PANEL_BORDER,
        HOLD_WIDTH + PANEL_BORDER * 2,
        HOLD_PANEL_HEIGHT + PANEL_BORDER * 2,
      )
      .fill(BORDER_COLOR);
    gfx.rect(HOLD_X, HOLD_Y, HOLD_WIDTH, HOLD_PANEL_HEIGHT).fill(0x0b0f14);
    gfx.rect(HOLD_X, HOLD_Y, HOLD_WIDTH, HOLD_LABEL_HEIGHT).fill(0x121a24);

    if (!state.hold) return;

    const offsetX = Math.floor((HOLD_COLS - 4) / 2) * cell;
    this.drawPreviewPiece(state.hold, HOLD_X + offsetX, HOLD_INNER_Y, cell);
  }

  private drawPreviewPiece(
    kind: PieceKind,
    originX: number,
    originY: number,
    cell: number,
  ): void {
    const color = COLORS[kind];
    const shape = TETROMINOES[kind][0];
    for (const [x, y] of shape) {
      drawCell(this.gfx, originX, originY, cell, x, y, color);
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
