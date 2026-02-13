import type { Board, PieceKind, PieceProbability } from './types';

export interface PieceGenerator {
  next(): PieceKind;
  peek(n: number): PieceKind[];
  reset(seed: number): void;
  onLock?(board: Board, hold: PieceKind | null): void;
  getLastSampleDistribution?(): PieceProbability[] | null;
}
