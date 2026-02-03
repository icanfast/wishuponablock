import { PIECES, type PieceKind } from './types';
import { XorShift32 } from './rng';
import type { PieceGenerator } from './generator';

export class NesGenerator implements PieceGenerator {
  private rng: XorShift32;
  private last: PieceKind | null = null;

  constructor(seed: number) {
    this.rng = new XorShift32(seed);
  }

  reset(seed: number): void {
    this.rng = new XorShift32(seed);
    this.last = null;
  }

  next(): PieceKind {
    const first = this.randomPiece();
    if (this.last == null) {
      this.last = first;
      return first;
    }

    if (first === this.last) {
      const second = this.randomPiece();
      this.last = second;
      return second;
    }

    this.last = first;
    return first;
  }

  peek(n: number): PieceKind[] {
    const out: PieceKind[] = [];
    const rng = this.rng.clone();
    let last = this.last;

    const nextLocal = (): PieceKind => {
      const first = PIECES[rng.nextInt(PIECES.length)];
      if (last == null) {
        last = first;
        return first;
      }
      if (first === last) {
        const second = PIECES[rng.nextInt(PIECES.length)];
        last = second;
        return second;
      }
      last = first;
      return first;
    };

    for (let i = 0; i < n; i++) out.push(nextLocal());
    return out;
  }

  private randomPiece(): PieceKind {
    return PIECES[this.rng.nextInt(PIECES.length)];
  }
}
