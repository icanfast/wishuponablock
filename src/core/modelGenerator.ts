import type { Board, PieceKind, PieceProbability } from './types';
import { PIECES } from './types';
import { XorShift32 } from './rng';
import type { PieceGenerator } from './generator';
import type { LoadedModel } from './wubModel';
import { predictLogits, softmax } from './wubModel';

type InferenceStrategy = 'clean_uniform' | 'threshold';

type InferenceOptions = {
  strategy?: InferenceStrategy;
  temperature?: number;
  threshold?: number;
};

export class ModelGenerator implements PieceGenerator {
  private rng: XorShift32;
  private model: LoadedModel | null;
  private pending: PieceKind | null = null;
  private pendingDistribution: PieceProbability[] | null = null;
  private lastSampleDistribution: PieceProbability[] | null = null;
  private strategy: InferenceStrategy;
  private temperature: number;
  private threshold: number;

  constructor(
    seed: number,
    model: LoadedModel | null,
    modelPromise?: Promise<LoadedModel | null>,
    options: InferenceOptions = {},
  ) {
    this.rng = new XorShift32(seed);
    this.model = model ?? null;
    this.strategy = options.strategy ?? 'clean_uniform';
    this.temperature = options.temperature ?? 1;
    this.threshold = options.threshold ?? 0;
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
    const probs =
      this.strategy === 'threshold'
        ? thresholdedSoftmax(logits, this.temperature, this.threshold)
        : softmax(logits);
    if (this.strategy === 'clean_uniform') {
      const blend = getCleanBlend(board);
      if (blend > 0) {
        const uniform = 1 / probs.length;
        for (let i = 0; i < probs.length; i++) {
          probs[i] = probs[i] * (1 - blend) + uniform * blend;
        }
      }
    }
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

const thresholdedSoftmax = (
  logits: Float32Array,
  temperature: number,
  threshold: number,
): Float32Array => {
  const temp = temperature > 0 ? temperature : 1;
  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    scaled[i] = logits[i] / temp;
  }
  const base = softmax(scaled);
  if (threshold <= 0) return base;
  const filtered = new Float32Array(base.length);
  let sum = 0;
  for (let i = 0; i < base.length; i++) {
    if (base[i] >= threshold) {
      filtered[i] = base[i];
      sum += base[i];
    }
  }
  if (sum <= 0) return base;
  for (let i = 0; i < filtered.length; i++) {
    if (filtered[i] > 0) {
      filtered[i] /= sum;
    }
  }
  return filtered;
};

const CLEAN_SCORE_THRESHOLD = 0.98;
const CLEAN_HEIGHT_THRESHOLD = 4;
const CLEAN_UNIFORM_BLEND = 0.2;

const getCleanBlend = (board: Board): number => {
  const { holes, height, filled } = getBoardStats(board);
  const score = filled === 0 ? 1 : 1 - holes / Math.max(1, filled);
  if (score < CLEAN_SCORE_THRESHOLD) return 0;
  if (height > CLEAN_HEIGHT_THRESHOLD) return 0;
  return CLEAN_UNIFORM_BLEND;
};

const getBoardStats = (
  board: Board,
): { holes: number; height: number; filled: number } => {
  const rows = board.length;
  const cols = board[0]?.length ?? 0;
  let filled = 0;
  let firstFilledRow = rows;

  for (let y = 0; y < rows; y++) {
    let rowHasBlock = false;
    for (let x = 0; x < cols; x++) {
      if (board[y][x] != null) {
        filled++;
        rowHasBlock = true;
      }
    }
    if (rowHasBlock && firstFilledRow === rows) {
      firstFilledRow = y;
    }
  }

  let holes = 0;
  for (let x = 0; x < cols; x++) {
    let seenFilled = false;
    for (let y = 0; y < rows; y++) {
      const filledCell = board[y][x] != null;
      if (filledCell) {
        seenFilled = true;
      } else if (seenFilled) {
        holes++;
      }
    }
  }

  const height = firstFilledRow < rows ? rows - firstFilledRow : 0;
  return { holes, height, filled };
};
