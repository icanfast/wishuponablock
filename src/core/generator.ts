import type { PieceKind } from './types';

export interface PieceGenerator {
  next(): PieceKind;
  peek(n: number): PieceKind[];
  reset(seed: number): void;
}
