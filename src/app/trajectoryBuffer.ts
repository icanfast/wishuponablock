import type { Board, PieceKind } from '../core/types';
import type { ModelGeneratorDecisionEvent } from '../core/modelGenerator';

export const TRAJECTORY_SCHEMA_V1 = 'wishuponablock.trajectory.v1';

export type TrajectoryDecisionSample = {
  schema: typeof TRAJECTORY_SCHEMA_V1;
  id: string;
  modeId: string;
  createdAtMs: number;
  deliberationMs: number | null;
  boardOccupancy: number[][];
  hold: PieceKind | null;
  action: PieceKind;
  actionIndex: number;
  pieces: PieceKind[];
  logits: number[];
  probabilities: number[];
  inferenceMs: number;
  samplingMs: number;
  totalDecisionMs: number;
  reward: number | null;
};

export type TrajectoryBufferStats = {
  totalSamples: number;
  byMode: Record<string, number>;
  lastSampleAtMs: number | null;
};

export type TrajectoryBuffer = {
  recordDecision: (options: {
    modeId: string;
    decision: ModelGeneratorDecisionEvent;
  }) => TrajectoryDecisionSample;
  listSamples: (options?: {
    modeId?: string;
    limit?: number;
  }) => TrajectoryDecisionSample[];
  clear: () => number;
  size: () => number;
  getStats: () => TrajectoryBufferStats;
};

type TrajectoryBufferOptions = {
  maxSamples?: number;
};

const DEFAULT_MAX_SAMPLES = 2500;

const clampLimit = (value: number | undefined, fallback: number): number => {
  if (value == null || !Number.isFinite(value)) return fallback;
  return Math.max(1, Math.trunc(value));
};

const cloneBoardOccupancy = (board: Board): number[][] =>
  board.map((row) => row.map((cell) => (cell != null ? 1 : 0)));

const cloneDecision = (
  sample: TrajectoryDecisionSample,
): TrajectoryDecisionSample => ({
  ...sample,
  boardOccupancy: sample.boardOccupancy.map((row) => row.slice()),
  pieces: [...sample.pieces],
  logits: [...sample.logits],
  probabilities: [...sample.probabilities],
});

export function createTrajectoryBuffer(
  options: TrajectoryBufferOptions = {},
): TrajectoryBuffer {
  const maxSamples = clampLimit(options.maxSamples, DEFAULT_MAX_SAMPLES);
  const samples: TrajectoryDecisionSample[] = [];
  let nextId = 1;
  let lastDecisionAtMs: number | null = null;

  const trimToMax = () => {
    if (samples.length <= maxSamples) return;
    samples.splice(0, samples.length - maxSamples);
  };

  return {
    recordDecision: ({ modeId, decision }) => {
      const actionIndex = decision.pieces.indexOf(decision.action);
      const deliberationMs =
        lastDecisionAtMs == null
          ? null
          : Math.max(0, decision.wallTimeMs - lastDecisionAtMs);
      lastDecisionAtMs = decision.wallTimeMs;

      const sample: TrajectoryDecisionSample = {
        schema: TRAJECTORY_SCHEMA_V1,
        id: `traj_${nextId++}`,
        modeId,
        createdAtMs: Date.now(),
        deliberationMs,
        boardOccupancy: cloneBoardOccupancy(decision.board),
        hold: decision.hold,
        action: decision.action,
        actionIndex,
        pieces: [...decision.pieces],
        logits: Array.from(decision.logits),
        probabilities: Array.from(decision.probabilities),
        inferenceMs: decision.inferenceMs,
        samplingMs: decision.samplingMs,
        totalDecisionMs: decision.totalMs,
        reward: null,
      };
      samples.push(sample);
      trimToMax();
      return cloneDecision(sample);
    },
    listSamples: (query) => {
      const modeId = query?.modeId?.trim();
      const limit = query?.limit;
      const filtered =
        modeId && modeId.length > 0
          ? samples.filter((sample) => sample.modeId === modeId)
          : samples;
      const sliced =
        limit != null && Number.isFinite(limit) && limit > 0
          ? filtered.slice(-Math.trunc(limit))
          : filtered.slice();
      return sliced.map(cloneDecision);
    },
    clear: () => {
      const removed = samples.length;
      samples.length = 0;
      lastDecisionAtMs = null;
      return removed;
    },
    size: () => samples.length,
    getStats: () => {
      const byMode: Record<string, number> = {};
      for (const sample of samples) {
        byMode[sample.modeId] = (byMode[sample.modeId] ?? 0) + 1;
      }
      return {
        totalSamples: samples.length,
        byMode,
        lastSampleAtMs:
          samples.length > 0 ? samples[samples.length - 1].createdAtMs : null,
      };
    },
  };
}
