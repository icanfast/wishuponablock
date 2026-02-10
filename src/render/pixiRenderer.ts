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
  HOLD_ROWS,
  HOLD_WIDTH,
  HOLD_X,
  HOLD_Y,
  NEXT_COUNT,
  QUEUE_COLS,
  QUEUE_GAP_PX,
  QUEUE_LABEL_HEIGHT,
  QUEUE_PREVIEW_ROWS,
  QUEUE_PREVIEW_HEIGHT,
  QUEUE_X,
  QUEUE_Y,
  ROWS,
} from '../core/constants';
import { cellsOf } from '../core/piece';
import { TETROMINOES } from '../core/tetromino';
import type { ActivePiece, GameState, PieceKind } from '../core/types';
import {
  PIECE_COLORS,
  PIECE_COLORS_COLORBLIND,
  PIECE_COLORS_HIGH_CONTRAST,
} from '../core/palette';

const BOARD_FILL = 0x0b0f14;
const BOARD_FILL_HIGH = 0x0a0f14;
const PANEL_FILL = 0x0b0f14;
const PANEL_FILL_HIGH = 0x0a0f14;
const LABEL_FILL = 0x121a24;
const LABEL_FILL_HIGH = 0x1a2533;
const OUTLINE_COLOR_HIGH = 0x0b111a;

const BORDER_COLOR = 0x1f2a37;
const BORDER_COLOR_HIGH = 0x334155;
const BOARD_BORDER = 3;
const PANEL_BORDER = 3;

export class PixiRenderer {
  constructor(
    private gfx: Graphics,
    private cell = BOARD_CELL_PX,
    private boardX = BOARD_X,
    private boardY = BOARD_Y,
  ) {}

  private gridlineOpacity = 0;
  private highContrast = false;
  private colorblindMode = false;

  setGridlineOpacity(opacity: number): void {
    this.gridlineOpacity = Math.max(0, Math.min(1, opacity));
  }

  setHighContrast(enabled: boolean): void {
    this.highContrast = enabled;
  }

  setColorblindMode(enabled: boolean): void {
    this.colorblindMode = enabled;
  }

  private getBorderColor(): number {
    return this.highContrast ? BORDER_COLOR_HIGH : BORDER_COLOR;
  }

  private getBoardFill(): number {
    return this.highContrast ? BOARD_FILL_HIGH : BOARD_FILL;
  }

  private getPanelFill(): number {
    return this.highContrast ? PANEL_FILL_HIGH : PANEL_FILL;
  }

  private getLabelFill(): number {
    return this.highContrast ? LABEL_FILL_HIGH : LABEL_FILL;
  }

  private getOutlineColor(): number | undefined {
    return this.highContrast ? OUTLINE_COLOR_HIGH : undefined;
  }

  render(state: GameState): void {
    const { gfx, cell, boardX, boardY } = this;

    gfx.clear();

    // board background + frame
    const borderColor = this.getBorderColor();
    const boardFill = this.getBoardFill();

    gfx
      .rect(
        boardX - BOARD_BORDER,
        boardY - BOARD_BORDER,
        COLS * cell + BOARD_BORDER * 2,
        ROWS * cell + BOARD_BORDER * 2,
      )
      .fill(borderColor);
    gfx.rect(boardX, boardY, COLS * cell, ROWS * cell).fill(boardFill);

    this.renderGridlines();
    this.renderHold(state);
    this.renderQueue(state);

    // settled blocks
    const outlineColor = this.getOutlineColor();
    for (let y = 0; y < ROWS; y++) {
      for (let x = 0; x < COLS; x++) {
        const k = state.board[y][x];
        if (!k) continue;
        drawCell(
          gfx,
          boardX,
          boardY,
          cell,
          x,
          y,
          this.getPieceColor(k),
          outlineColor,
        );
      }
    }

    if (state.gameOver || state.gameWon) return;

    // ghost
    const ghostColor = dim(this.getPieceColor(state.active.k), 0.35);
    const ghostPiece = { ...state.active, y: state.ghostY };
    for (const [x, y] of cellsOf(ghostPiece)) {
      if (y < 0) continue;
      drawCell(gfx, boardX, boardY, cell, x, y, ghostColor, outlineColor);
    }

    // active
    const color = this.getPieceColor(state.active.k);
    for (const [x, y] of cellsOf(state.active)) {
      if (y < 0) continue;
      drawCell(gfx, boardX, boardY, cell, x, y, color, outlineColor);
    }
  }

  renderBoardOnly(board: GameState['board'], hold?: PieceKind | null): void {
    this.renderBoardPreview(board, hold, null);
  }

  renderBoardPreview(
    board: GameState['board'],
    hold: PieceKind | null | undefined,
    ghost: ActivePiece | null,
  ): void {
    const { gfx, cell, boardX, boardY } = this;
    gfx.clear();

    const borderColor = this.getBorderColor();
    const boardFill = this.getBoardFill();

    gfx
      .rect(
        boardX - BOARD_BORDER,
        boardY - BOARD_BORDER,
        COLS * cell + BOARD_BORDER * 2,
        ROWS * cell + BOARD_BORDER * 2,
      )
      .fill(borderColor);
    gfx.rect(boardX, boardY, COLS * cell, ROWS * cell).fill(boardFill);
    this.renderGridlines();

    if (hold !== undefined) {
      this.renderHoldPiece(hold);
    }

    const outlineColor = this.getOutlineColor();
    for (let y = 0; y < ROWS; y++) {
      for (let x = 0; x < COLS; x++) {
        const k = board[y][x];
        if (!k) continue;
        drawCell(
          gfx,
          boardX,
          boardY,
          cell,
          x,
          y,
          this.getPieceColor(k),
          outlineColor,
        );
      }
    }

    if (ghost) {
      const ghostColor = dim(this.getPieceColor(ghost.k), 0.35);
      for (const [x, y] of cellsOf(ghost)) {
        if (y < 0) continue;
        drawCell(gfx, boardX, boardY, cell, x, y, ghostColor, outlineColor);
      }
    }
  }

  private renderHoldPiece(hold: PieceKind | null): void {
    const { gfx, cell } = this;

    const borderColor = this.getBorderColor();
    const panelFill = this.getPanelFill();
    const labelFill = this.getLabelFill();

    gfx
      .rect(
        HOLD_X - PANEL_BORDER,
        HOLD_Y - PANEL_BORDER,
        HOLD_WIDTH + PANEL_BORDER * 2,
        HOLD_PANEL_HEIGHT + PANEL_BORDER * 2,
      )
      .fill(borderColor);
    gfx.rect(HOLD_X, HOLD_Y, HOLD_WIDTH, HOLD_PANEL_HEIGHT).fill(panelFill);
    gfx.rect(HOLD_X, HOLD_Y, HOLD_WIDTH, HOLD_LABEL_HEIGHT).fill(labelFill);

    if (!hold) return;

    const offsetX = Math.floor((HOLD_COLS - 4) / 2) * cell;
    this.drawPreviewPiece(
      hold,
      HOLD_X + offsetX,
      HOLD_INNER_Y,
      cell,
      4,
      HOLD_ROWS,
    );
  }

  private renderQueue(state: GameState): void {
    const { gfx, cell } = this;
    const count = Math.min(state.next.length, NEXT_COUNT);
    if (count === 0) return;

    const panelHeight =
      QUEUE_LABEL_HEIGHT +
      count * QUEUE_PREVIEW_HEIGHT +
      (count - 1) * QUEUE_GAP_PX;
    const panelWidth = QUEUE_COLS * cell;
    const borderColor = this.getBorderColor();
    const panelFill = this.getPanelFill();
    const labelFill = this.getLabelFill();

    gfx
      .rect(
        QUEUE_X - PANEL_BORDER,
        QUEUE_Y - PANEL_BORDER,
        panelWidth + PANEL_BORDER * 2,
        panelHeight + PANEL_BORDER * 2,
      )
      .fill(borderColor);
    gfx.rect(QUEUE_X, QUEUE_Y, panelWidth, panelHeight).fill(panelFill);
    gfx.rect(QUEUE_X, QUEUE_Y, panelWidth, QUEUE_LABEL_HEIGHT).fill(labelFill);

    const offsetX = Math.floor((QUEUE_COLS - 4) / 2) * cell;

    for (let i = 0; i < count; i++) {
      const kind = state.next[i];
      const boxY =
        QUEUE_Y +
        QUEUE_LABEL_HEIGHT +
        i * (QUEUE_PREVIEW_HEIGHT + QUEUE_GAP_PX);
      this.drawPreviewPiece(
        kind,
        QUEUE_X + offsetX,
        boxY,
        cell,
        4,
        QUEUE_PREVIEW_ROWS,
      );
    }
  }

  private renderGridlines(): void {
    if (this.gridlineOpacity <= 0) return;
    const { gfx, cell, boardX, boardY } = this;
    const alpha = this.highContrast
      ? Math.max(this.gridlineOpacity, 0.25)
      : this.gridlineOpacity;
    const color = this.getBorderColor();
    for (let x = 1; x < COLS; x++) {
      gfx
        .rect(boardX + x * cell - 0.5, boardY, 1, ROWS * cell)
        .fill({ color, alpha });
    }
    for (let y = 1; y < ROWS; y++) {
      gfx
        .rect(boardX, boardY + y * cell - 0.5, COLS * cell, 1)
        .fill({ color, alpha });
    }
  }

  private getPieceColor(kind: PieceKind): number {
    if (this.colorblindMode) return PIECE_COLORS_COLORBLIND[kind];
    return this.highContrast
      ? PIECE_COLORS_HIGH_CONTRAST[kind]
      : PIECE_COLORS[kind];
  }

  private renderHold(state: GameState): void {
    const { gfx, cell } = this;

    const borderColor = this.getBorderColor();
    const panelFill = this.getPanelFill();
    const labelFill = this.getLabelFill();

    gfx
      .rect(
        HOLD_X - PANEL_BORDER,
        HOLD_Y - PANEL_BORDER,
        HOLD_WIDTH + PANEL_BORDER * 2,
        HOLD_PANEL_HEIGHT + PANEL_BORDER * 2,
      )
      .fill(borderColor);
    gfx.rect(HOLD_X, HOLD_Y, HOLD_WIDTH, HOLD_PANEL_HEIGHT).fill(panelFill);
    gfx.rect(HOLD_X, HOLD_Y, HOLD_WIDTH, HOLD_LABEL_HEIGHT).fill(labelFill);

    if (!state.hold) return;

    const offsetX = Math.floor((HOLD_COLS - 4) / 2) * cell;
    this.drawPreviewPiece(
      state.hold,
      HOLD_X + offsetX,
      HOLD_INNER_Y,
      cell,
      4,
      HOLD_ROWS,
    );
  }

  private drawPreviewPiece(
    kind: PieceKind,
    originX: number,
    originY: number,
    cell: number,
    boxCols = 4,
    boxRows = 4,
  ): void {
    const color = this.getPieceColor(kind);
    const shape = TETROMINOES[kind][0];
    const bounds = getShapeBounds(shape);
    const dx = (boxCols - bounds.width) / 2 - bounds.minX;
    const dy = (boxRows - bounds.height) / 2 - bounds.minY;
    const outlineColor = this.getOutlineColor();
    for (const [x, y] of shape) {
      drawCell(
        this.gfx,
        originX,
        originY,
        cell,
        x + dx,
        y + dy,
        color,
        outlineColor,
      );
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
  outlineColor?: number,
): void {
  const px = boardX + x * cell;
  const py = boardY + y * cell;
  const pad = 2;
  if (outlineColor !== undefined) {
    gfx
      .rect(px + pad - 1, py + pad - 1, cell - pad * 2 + 2, cell - pad * 2 + 2)
      .fill(outlineColor);
  }
  gfx.rect(px + pad, py + pad, cell - pad * 2, cell - pad * 2).fill(color);
}

function dim(color: number, factor: number): number {
  const r = ((color >> 16) & 0xff) * factor;
  const g = ((color >> 8) & 0xff) * factor;
  const b = (color & 0xff) * factor;
  return ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
}

function getShapeBounds(shape: ReadonlyArray<readonly [number, number]>): {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  width: number;
  height: number;
} {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [x, y] of shape) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const width = maxX - minX + 1;
  const height = maxY - minY + 1;
  return { minX, maxX, minY, maxY, width, height };
}
