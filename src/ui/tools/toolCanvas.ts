import { makeBoard } from '../../core/board';
import type { ActivePiece, Board, PieceKind } from '../../core/types';
import type { PixiRenderer } from '../../render/pixiRenderer';

export type ToolCanvas = {
  render: (
    board: Board,
    hold: PieceKind | null,
    ghost?: ActivePiece | null,
    active?: ActivePiece | null,
  ) => void;
  clear: () => void;
};

export function createToolCanvas(renderer: PixiRenderer): ToolCanvas {
  return {
    render: (board, hold, ghost, active) => {
      if (ghost || active) {
        renderer.renderBoardPreview(board, hold, ghost ?? null, active ?? null);
      } else {
        renderer.renderBoardOnly(board, hold);
      }
    },
    clear: () => {
      renderer.renderBoardOnly(makeBoard(), null);
    },
  };
}
