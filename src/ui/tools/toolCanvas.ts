import { makeBoard } from '../../core/board';
import type { Board, PieceKind } from '../../core/types';
import type { PixiRenderer } from '../../render/pixiRenderer';

export type ToolCanvas = {
  render: (board: Board, hold: PieceKind | null) => void;
  clear: () => void;
};

export function createToolCanvas(renderer: PixiRenderer): ToolCanvas {
  return {
    render: (board, hold) => {
      renderer.renderBoardOnly(board, hold);
    },
    clear: () => {
      renderer.renderBoardOnly(makeBoard(), null);
    },
  };
}
