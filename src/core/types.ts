export const PIECES = ['I', 'O', 'T', 'S', 'Z', 'J', 'L'] as const;
export type PieceKind = (typeof PIECES)[number];

export type Rotation = 0 | 1 | 2 | 3;
export type Vec2 = readonly [number, number];

export type Cell = PieceKind | null;
export type Board = Cell[][];

export interface ActivePiece {
  k: PieceKind;
  r: Rotation;
  x: number;
  y: number; // can be negative while spawning
}

export interface GameState {
  board: Board;
  active: ActivePiece;
  ghostY: number;
  hold: PieceKind | null;
  canHold: boolean;
  next: PieceKind[]; // preview window (derived from generator)
  gameOver: boolean;
  gameWon: boolean;
  combo: number;
  timeMs: number;
  totalLinesCleared: number;
  lineGoal: number | null;
  level: number;
  score: number;
  scoringEnabled: boolean;
}

export interface InputFrame {
  /**
   * Signed number of horizontal steps to attempt this frame.
   * Can be +/-Infinity to indicate "instant ARR" to the wall.
   */
  moveX: number;
  /**
   * True when moveX was produced by DAS/ARR repeat (post-initial press).
   * Used to avoid applying mis-input effects after DAS.
   */
  moveXFromRepeat?: boolean;
  rotate: -1 | 0 | 1; // -1 = CCW, +1 = CW
  rotate180: boolean;
  softDrop: boolean;
  hardDrop: boolean;
  hold: boolean;
  restart: boolean;
}
