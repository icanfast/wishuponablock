import type { Board, PieceKind } from '../core/types';
import type { ModelGeneratorDecisionEvent } from '../core/modelGenerator';
import type { TrajectoryReplayStepV1 } from '../core/trajectoryProtocol';

export const TRAJECTORY_SCHEMA_V1 = 'wishuponablock.trajectory.v1';

export type TrajectorySampleAxes = {
  arch: string;
  rewardProfileId: string;
  queuePolicyId: string;
};

export type TrajectoryDecisionSample = {
  schema: typeof TRAJECTORY_SCHEMA_V1;
  id: string;
  modeId: string;
  arch: string;
  rewardProfileId: string;
  queuePolicyId: string;
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
  replay?: TrajectoryReplayStepV1;
};

export type TrajectoryBufferStats = {
  totalSamples: number;
  byMode: Record<string, number>;
  byModeAndAxes: Record<string, number>;
  lastSampleAtMs: number | null;
};

export type TrajectoryBuffer = {
  recordDecision: (options: {
    modeId: string;
    modelAxes: TrajectorySampleAxes;
    decision: ModelGeneratorDecisionEvent;
    replay?: TrajectoryReplayStepV1 | null;
  }) => TrajectoryDecisionSample;
  listSamples: (options?: {
    modeId?: string;
    modelAxes?: Partial<TrajectorySampleAxes>;
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

const normalizeAxis = (value: unknown, fallback: string): string => {
  if (typeof value !== 'string') return fallback;
  const normalized = value.trim().toLowerCase();
  if (!/^[a-z0-9_-]{1,64}$/.test(normalized)) return fallback;
  return normalized;
};

const buildModeAxesKey = (sample: {
  modeId: string;
  arch: string;
  rewardProfileId: string;
  queuePolicyId: string;
}): string =>
  `${sample.modeId}:${sample.arch}:${sample.rewardProfileId}:${sample.queuePolicyId}`;

const cloneDecision = (
  sample: TrajectoryDecisionSample,
): TrajectoryDecisionSample => ({
  ...sample,
  boardOccupancy: sample.boardOccupancy.map((row) => row.slice()),
  pieces: [...sample.pieces],
  logits: [...sample.logits],
  probabilities: [...sample.probabilities],
  ...(sample.replay
    ? {
        replay: {
          lockPiece: sample.replay.lockPiece,
          lockRotation: sample.replay.lockRotation,
          lockX: sample.replay.lockX,
          lockY: sample.replay.lockY,
          holdUsed: sample.replay.holdUsed,
          gameTimeMs: sample.replay.gameTimeMs,
          totalLinesCleared: sample.replay.totalLinesCleared,
          score: sample.replay.score,
        },
      }
    : {}),
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
    recordDecision: ({ modeId, modelAxes, decision, replay }) => {
      const actionIndex = decision.pieces.indexOf(decision.action);
      const deliberationMs =
        lastDecisionAtMs == null
          ? null
          : Math.max(0, decision.wallTimeMs - lastDecisionAtMs);
      lastDecisionAtMs = decision.wallTimeMs;
      const arch = normalizeAxis(modelAxes.arch, 'full');
      const rewardProfileId = normalizeAxis(
        modelAxes.rewardProfileId,
        'default',
      );
      const queuePolicyId = normalizeAxis(
        modelAxes.queuePolicyId,
        'next_piece_v1',
      );

      const sample: TrajectoryDecisionSample = {
        schema: TRAJECTORY_SCHEMA_V1,
        id: `traj_${nextId++}`,
        modeId,
        arch,
        rewardProfileId,
        queuePolicyId,
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
        ...(replay
          ? {
              replay: {
                lockPiece: replay.lockPiece,
                lockRotation: Math.max(
                  0,
                  Math.min(3, Math.trunc(replay.lockRotation)),
                ),
                lockX: Math.trunc(replay.lockX),
                lockY: Math.trunc(replay.lockY),
                holdUsed: Boolean(replay.holdUsed),
                gameTimeMs: Math.max(0, Math.trunc(replay.gameTimeMs)),
                totalLinesCleared: Math.max(
                  0,
                  Math.trunc(replay.totalLinesCleared),
                ),
                score: Math.max(0, Math.trunc(replay.score)),
              },
            }
          : {}),
      };
      samples.push(sample);
      trimToMax();
      return cloneDecision(sample);
    },
    listSamples: (query) => {
      const modeId = query?.modeId?.trim();
      const archFilter = normalizeAxis(query?.modelAxes?.arch, '');
      const rewardProfileIdFilter = normalizeAxis(
        query?.modelAxes?.rewardProfileId,
        '',
      );
      const queuePolicyIdFilter = normalizeAxis(
        query?.modelAxes?.queuePolicyId,
        '',
      );
      const limit = query?.limit;
      const filtered = samples.filter((sample) => {
        if (modeId && modeId.length > 0 && sample.modeId !== modeId) {
          return false;
        }
        if (archFilter && sample.arch !== archFilter) {
          return false;
        }
        if (
          rewardProfileIdFilter &&
          sample.rewardProfileId !== rewardProfileIdFilter
        ) {
          return false;
        }
        if (
          queuePolicyIdFilter &&
          sample.queuePolicyId !== queuePolicyIdFilter
        ) {
          return false;
        }
        return true;
      });
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
      const byModeAndAxes: Record<string, number> = {};
      for (const sample of samples) {
        byMode[sample.modeId] = (byMode[sample.modeId] ?? 0) + 1;
        const axesKey = buildModeAxesKey(sample);
        byModeAndAxes[axesKey] = (byModeAndAxes[axesKey] ?? 0) + 1;
      }
      return {
        totalSamples: samples.length,
        byMode,
        byModeAndAxes,
        lastSampleAtMs:
          samples.length > 0 ? samples[samples.length - 1].createdAtMs : null,
      };
    },
  };
}
