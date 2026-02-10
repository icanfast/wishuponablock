import { makeBoard } from '../../core/board';
import type { ActivePiece, Board, PieceKind } from '../../core/types';
import type { PixiRenderer } from '../../render/pixiRenderer';

export type ToolCanvas = {
  render: (
    board: Board,
    hold: PieceKind | null,
    ghost?: ActivePiece | null,
  ) => void;
  clear: () => void;
};

export function createToolCanvas(renderer: PixiRenderer): ToolCanvas {
  return {
    render: (board, hold, ghost) => {
      if (ghost) {
        renderer.renderBoardPreview(board, hold, ghost);
      } else {
        renderer.renderBoardOnly(board, hold);
      }
    },
    clear: () => {
      renderer.renderBoardOnly(makeBoard(), null);
    },
  };
}
