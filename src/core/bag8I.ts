import { PIECES, type PieceKind } from './types';
import { XorShift32, shuffleInPlace } from './rng';
import type { PieceGenerator } from './generator';

/**
 * 8-piece bag with an extra I piece.
 */
export class Bag8I implements PieceGenerator {
  private rng: XorShift32;
  private q: PieceKind[] = [];

  constructor(seed: number) {
    this.rng = new XorShift32(seed);
    this.refill();
  }

  reset(seed: number): void {
    this.rng = new XorShift32(seed);
    this.q = [];
    this.refill();
  }

  next(): PieceKind {
    if (this.q.length === 0) this.refill();
    return this.q.shift()!;
  }

  ensure(n: number): void {
    while (this.q.length < n) this.refill();
  }

  peek(n: number): PieceKind[] {
    this.ensure(n);
    return this.q.slice(0, n);
  }

  private refill(): void {
    const bag = [...PIECES, 'I'] as PieceKind[];
    shuffleInPlace(bag, this.rng);
    this.q.push(...bag);
  }
}
