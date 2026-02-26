import type { Board, PieceKind, PieceProbability } from './types';
import { PIECES } from './types';
import { XorShift32 } from './rng';
import type { PieceGenerator } from './generator';
import { createModelRunner, type ModelRunner } from './modelRunner';
import type { LoadedModel } from './wubModel';
import { softmax } from './wubModel';
import { hasPerfMetricsSink, recordPerfDuration } from './perfMetrics';

type InferenceStrategy = 'clean_uniform' | 'threshold';
type QueuePolicyId = 'next_piece_v1' | 'bag_shuffle_v1';

const DEFAULT_QUEUE_POLICY_ID: QueuePolicyId = 'next_piece_v1';
const BAG_ZERO_LOGIT = -30;

export type ModelGeneratorDecisionEvent = {
  board: Board;
  hold: PieceKind | null;
  action: PieceKind;
  pieces: PieceKind[];
  logits: Float32Array;
  probabilities: Float32Array;
  inferenceMs: number;
  samplingMs: number;
  totalMs: number;
  wallTimeMs: number;
};

type InferenceOptions = {
  strategy?: InferenceStrategy;
  temperature?: number;
  threshold?: number;
  postSharpness?: number;
  queuePolicyId?: string;
  onDecision?: (event: ModelGeneratorDecisionEvent) => void;
};

type QueuedDecision = {
  action: PieceKind;
  pieces: PieceKind[];
  logits: Float32Array;
  probabilities: Float32Array;
  distribution: PieceProbability[];
};

export class ModelGenerator implements PieceGenerator {
  private rng: XorShift32;
  private model: LoadedModel | null;
  private runner: ModelRunner;
  private queue: QueuedDecision[] = [];
  private lastSampleDistribution: PieceProbability[] | null = null;
  private strategy: InferenceStrategy;
  private temperature: number;
  private threshold: number;
  private postSharpness: number;
  private queuePolicyId: QueuePolicyId;
  private onDecision: ((event: ModelGeneratorDecisionEvent) => void) | null;

  constructor(
    seed: number,
    model: LoadedModel | null,
    modelPromise?: Promise<LoadedModel | null>,
    runner?: ModelRunner,
    options: InferenceOptions = {},
  ) {
    this.rng = new XorShift32(seed);
    this.model = model ?? null;
    this.runner = runner ?? createModelRunner().runner;
    this.strategy = options.strategy ?? 'clean_uniform';
    this.temperature = options.temperature ?? 1;
    this.threshold = options.threshold ?? 0;
    this.postSharpness = options.postSharpness ?? 1;
    this.queuePolicyId = normalizeQueuePolicyId(options.queuePolicyId);
    this.onDecision = options.onDecision ?? null;
    modelPromise?.then((loaded) => {
      if (loaded) this.model = loaded;
    });
  }

  reset(seed: number): void {
    this.rng = new XorShift32(seed);
    this.queue = [];
    this.lastSampleDistribution = null;
  }

  next(): PieceKind {
    if (this.queue.length > 0) {
      const nextDecision = this.queue.shift()!;
      this.lastSampleDistribution = nextDecision.distribution.map((entry) => ({
        ...entry,
      }));
      return nextDecision.action;
    }
    this.lastSampleDistribution = null;
    return this.sampleFallback();
  }

  peek(n: number): PieceKind[] {
    const limit = Math.max(0, Math.trunc(n));
    if (limit === 0 || this.queue.length === 0) return [];
    return this.queue.slice(0, limit).map((entry) => entry.action);
  }

  getLastSampleDistribution(): PieceProbability[] | null {
    if (!this.lastSampleDistribution) return null;
    return this.lastSampleDistribution.map((entry) => ({ ...entry }));
  }

  onLock(board: Board, hold: PieceKind | null): void {
    if (!this.model) {
      this.queue = [];
      this.lastSampleDistribution = null;
      return;
    }

    const perfEnabled = hasPerfMetricsSink();
    const lockStartMs = performance.now();
    let inferenceMs = 0;
    let samplingMs = 0;
    const pieces = resolvePieces(this.model.pieces);

    if (this.queuePolicyId === 'bag_shuffle_v1') {
      if (this.queue.length === 0) {
        const logitsStartMs = performance.now();
        const logits = this.runner.predictLogits(this.model, board, hold);
        const logitsEndMs = performance.now();
        inferenceMs = logitsEndMs - logitsStartMs;

        const sampleStartMs = performance.now();
        const probabilities = this.resolveProbabilities(logits, board);
        this.queue = this.buildBagQueue(pieces, probabilities);
        samplingMs = performance.now() - sampleStartMs;
      }
    } else {
      const logitsStartMs = performance.now();
      const logits = this.runner.predictLogits(this.model, board, hold);
      const logitsEndMs = performance.now();
      inferenceMs = logitsEndMs - logitsStartMs;

      const sampleStartMs = performance.now();
      const probabilities = this.resolveProbabilities(logits, board);
      this.queue = [this.buildSingleDecision(pieces, logits, probabilities)];
      samplingMs = performance.now() - sampleStartMs;
    }

    const nextDecision = this.queue[0] ?? null;
    const nowMs = performance.now();
    const totalMs = nowMs - lockStartMs;

    if (this.onDecision && nextDecision) {
      this.onDecision({
        board,
        hold,
        action: nextDecision.action,
        pieces: [...nextDecision.pieces],
        logits: new Float32Array(nextDecision.logits),
        probabilities: new Float32Array(nextDecision.probabilities),
        inferenceMs,
        samplingMs,
        totalMs,
        wallTimeMs: nowMs,
      });
    }

    if (perfEnabled) {
      if (inferenceMs > 0) {
        recordPerfDuration('ml.predict_logits_ms', inferenceMs, nowMs);
      }
      if (samplingMs > 0) {
        recordPerfDuration('ml.sample_distribution_ms', samplingMs, nowMs);
      }
      recordPerfDuration('ml.on_lock_ms', nowMs - lockStartMs, nowMs);
    }
  }

  private resolveProbabilities(
    logits: Float32Array,
    board: Board,
  ): Float32Array {
    const probabilities =
      this.strategy === 'threshold'
        ? thresholdedSoftmax(
            logits,
            this.temperature,
            this.threshold,
            this.postSharpness,
          )
        : softmax(logits);
    if (this.strategy === 'clean_uniform') {
      const blend = getCleanBlend(board);
      if (blend > 0) {
        const uniform = 1 / probabilities.length;
        for (let i = 0; i < probabilities.length; i++) {
          probabilities[i] = probabilities[i] * (1 - blend) + uniform * blend;
        }
      }
    }
    return probabilities;
  }

  private buildSingleDecision(
    pieces: PieceKind[],
    logits: Float32Array,
    probabilities: Float32Array,
  ): QueuedDecision {
    const sampledIndex = this.sampleIndex(probabilities);
    const action = pieces[sampledIndex] ?? PIECES[0];
    return {
      action,
      pieces: [...pieces],
      logits: new Float32Array(logits),
      probabilities: new Float32Array(probabilities),
      distribution: this.buildDistributionFromProbabilities(
        pieces,
        probabilities,
      ),
    };
  }

  private buildBagQueue(
    pieces: PieceKind[],
    probabilities: Float32Array,
  ): QueuedDecision[] {
    const count = pieces.length;
    if (count === 0) return [];

    const weights = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const weight = probabilities[i];
      weights[i] = Number.isFinite(weight) && weight > 0 ? weight : 0;
    }

    const available = new Array<boolean>(count).fill(true);
    const queue: QueuedDecision[] = [];
    for (let draw = 0; draw < count; draw++) {
      const stepProbabilities = new Float32Array(count);
      let totalWeight = 0;
      let remaining = 0;
      for (let i = 0; i < count; i++) {
        if (!available[i]) continue;
        remaining += 1;
        totalWeight += weights[i];
      }
      if (remaining <= 0) break;

      if (totalWeight <= 0) {
        const uniform = 1 / remaining;
        for (let i = 0; i < count; i++) {
          if (!available[i]) continue;
          stepProbabilities[i] = uniform;
        }
      } else {
        for (let i = 0; i < count; i++) {
          if (!available[i]) continue;
          stepProbabilities[i] = weights[i] / totalWeight;
        }
      }

      const sampledIndex = this.sampleFromAvailable(
        stepProbabilities,
        available,
      );
      const stepLogits = new Float32Array(count);
      for (let i = 0; i < count; i++) {
        const p = stepProbabilities[i];
        stepLogits[i] = p > 0 ? Math.log(p) : BAG_ZERO_LOGIT;
      }

      queue.push({
        action: pieces[sampledIndex] ?? PIECES[0],
        pieces: [...pieces],
        logits: stepLogits,
        probabilities: stepProbabilities,
        distribution: this.buildDistributionFromProbabilities(
          pieces,
          stepProbabilities,
        ),
      });

      available[sampledIndex] = false;
      weights[sampledIndex] = 0;
    }
    return queue;
  }

  private buildDistributionFromProbabilities(
    pieces: PieceKind[],
    probabilities: Float32Array,
  ): PieceProbability[] {
    const distribution: PieceProbability[] = [];
    for (let i = 0; i < pieces.length; i++) {
      distribution.push({
        piece: pieces[i],
        probability:
          Number.isFinite(probabilities[i]) && probabilities[i] > 0
            ? probabilities[i]
            : 0,
      });
    }
    return distribution;
  }

  private sampleFromAvailable(
    probabilities: Float32Array,
    available: boolean[],
  ): number {
    let total = 0;
    let remaining = 0;
    for (let i = 0; i < probabilities.length; i++) {
      if (!available[i]) continue;
      remaining += 1;
      const p = probabilities[i];
      if (p > 0 && Number.isFinite(p)) total += p;
    }

    if (remaining <= 0) return 0;
    if (total <= 0) {
      const choices: number[] = [];
      for (let i = 0; i < available.length; i++) {
        if (available[i]) choices.push(i);
      }
      return choices[this.rng.nextInt(choices.length)] ?? 0;
    }

    const r = (this.rng.nextU32() / 0x100000000) * total;
    let acc = 0;
    let lastPositive = -1;
    for (let i = 0; i < probabilities.length; i++) {
      if (!available[i]) continue;
      const p = probabilities[i];
      if (!Number.isFinite(p) || p <= 0) continue;
      acc += p;
      lastPositive = i;
      if (r <= acc) return i;
    }

    if (lastPositive >= 0) return lastPositive;
    for (let i = 0; i < available.length; i++) {
      if (available[i]) return i;
    }
    return 0;
  }

  private sampleFallback(): PieceKind {
    const pieces = this.model?.pieces ?? PIECES;
    return pieces[this.rng.nextInt(pieces.length)];
  }

  private sampleIndex(probs: Float32Array): number {
    let total = 0;
    for (const p of probs) {
      if (Number.isFinite(p) && p > 0) total += p;
    }
    if (total <= 0) return this.rng.nextInt(probs.length);
    const r = (this.rng.nextU32() / 0x100000000) * total;
    let acc = 0;
    let lastPositive = -1;
    for (let i = 0; i < probs.length; i++) {
      const p = probs[i];
      if (!Number.isFinite(p) || p <= 0) continue;
      acc += p;
      lastPositive = i;
      if (r <= acc) return i;
    }
    if (lastPositive >= 0) return lastPositive;
    return this.rng.nextInt(probs.length);
  }
}

const normalizeQueuePolicyId = (value: unknown): QueuePolicyId =>
  value === 'bag_shuffle_v1' ? 'bag_shuffle_v1' : DEFAULT_QUEUE_POLICY_ID;

const resolvePieces = (source: PieceKind[] | null | undefined): PieceKind[] => {
  const resolved: PieceKind[] = [];
  const seen = new Set<PieceKind>();
  for (const piece of source ?? PIECES) {
    if (seen.has(piece)) continue;
    seen.add(piece);
    resolved.push(piece);
  }
  for (const piece of PIECES) {
    if (seen.has(piece)) continue;
    seen.add(piece);
    resolved.push(piece);
  }
  return resolved;
};

const thresholdedSoftmax = (
  logits: Float32Array,
  temperature: number,
  threshold: number,
  postSharpness: number,
): Float32Array => {
  const temp = temperature > 0 ? temperature : 1;
  const sharpness = postSharpness > 0 ? postSharpness : 1;
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
  if (Math.abs(sharpness - 1) < 1e-6) return filtered;
  let sharpenedSum = 0;
  for (let i = 0; i < filtered.length; i++) {
    if (filtered[i] > 0) {
      filtered[i] = filtered[i] ** sharpness;
      sharpenedSum += filtered[i];
    }
  }
  if (sharpenedSum <= 0) return filtered;
  for (let i = 0; i < filtered.length; i++) {
    if (filtered[i] > 0) {
      filtered[i] /= sharpenedSum;
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
