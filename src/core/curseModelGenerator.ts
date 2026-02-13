import type { Board, PieceKind, PieceProbability } from './types';
import { PIECES } from './types';
import { XorShift32 } from './rng';
import type { PieceGenerator } from './generator';
import type { LoadedModel } from './wubModel';
import { predictLogits } from './wubModel';
import { inferCurseDistribution } from './curseInference';

export class CurseModelGenerator implements PieceGenerator {
  private rng: XorShift32;
  private model: LoadedModel | null;
  private pending: PieceKind | null = null;
  private pendingDistribution: PieceProbability[] | null = null;
  private lastSampleDistribution: PieceProbability[] | null = null;

  constructor(
    seed: number,
    model: LoadedModel | null,
    modelPromise?: Promise<LoadedModel | null>,
  ) {
    this.rng = new XorShift32(seed);
    this.model = model ?? null;
    modelPromise?.then((loaded) => {
      if (loaded) this.model = loaded;
    });
  }

  reset(seed: number): void {
    this.rng = new XorShift32(seed);
    this.pending = null;
    this.pendingDistribution = null;
    this.lastSampleDistribution = null;
  }

  next(): PieceKind {
    if (this.pending) {
      const next = this.pending;
      this.pending = null;
      this.lastSampleDistribution = this.pendingDistribution
        ? this.pendingDistribution.map((entry) => ({ ...entry }))
        : null;
      this.pendingDistribution = null;
      return next;
    }
    this.lastSampleDistribution = null;
    return this.sampleFallback();
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  peek(_n: number): PieceKind[] {
    return [];
  }

  getLastSampleDistribution(): PieceProbability[] | null {
    if (!this.lastSampleDistribution) return null;
    return this.lastSampleDistribution.map((entry) => ({ ...entry }));
  }

  onLock(board: Board, hold: PieceKind | null): void {
    if (!this.model) {
      this.pending = null;
      this.pendingDistribution = null;
      return;
    }
    const logits = predictLogits(this.model, board, hold);
    const probs = inferCurseDistribution(logits);
    const pieces = this.model.pieces ?? PIECES;
    this.pendingDistribution = pieces.map((piece, index) => ({
      piece,
      probability: Number.isFinite(probs[index]) ? probs[index] : 0,
    }));
    this.pending = pieces[this.sampleIndex(probs)] ?? PIECES[0];
  }

  private sampleFallback(): PieceKind {
    const pieces = this.model?.pieces ?? PIECES;
    return pieces[this.rng.nextInt(pieces.length)];
  }

  private sampleIndex(probs: Float32Array): number {
    let total = 0;
    for (const p of probs) total += p;
    if (total <= 0) return this.rng.nextInt(probs.length);
    const r = (this.rng.nextU32() / 0x100000000) * total;
    let acc = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (r <= acc) return i;
    }
    return probs.length - 1;
  }
}
