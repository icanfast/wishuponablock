import { PIECES, type PieceKind } from './types';
import { XorShift32 } from './rng';
import type { PieceGenerator } from './generator';

export class RandomGenerator implements PieceGenerator {
  private rng: XorShift32;

  constructor(seed: number) {
    this.rng = new XorShift32(seed);
  }

  reset(seed: number): void {
    this.rng = new XorShift32(seed);
  }

  next(): PieceKind {
    const i = this.rng.nextInt(PIECES.length);
    return PIECES[i];
  }

  peek(n: number): PieceKind[] {
    const out: PieceKind[] = [];
    const rng = this.rng.clone();
    for (let i = 0; i < n; i++) {
      out.push(PIECES[rng.nextInt(PIECES.length)]);
    }
    return out;
  }
}
