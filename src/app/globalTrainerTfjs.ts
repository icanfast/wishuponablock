import type { TrajectorySessionV1 } from '../core/trajectoryProtocol';
import type { LoadedModel } from '../core/wubModel';
import type {
  PersonalTrainer,
  PersonalTrainerResult,
} from './personalTrainerTfjs';
import type { ResolvedPersonalTrainingPipeline } from './trainingPipelines';
import type { TrajectoryDecisionSample } from './trajectoryBuffer';

export type GlobalTrainingRunConfig = {
  model: LoadedModel;
  recordings: TrajectorySessionV1[];
  pipeline: ResolvedPersonalTrainingPipeline;
  trainer: PersonalTrainer;
  train?: {
    epochs?: number;
    learningRate?: number;
    l2?: number;
    sampleLimit?: number;
    backendPreference?: 'auto' | 'webgl' | 'cpu';
  };
};

export type GlobalTrainingRunResult = {
  ok: boolean;
  message: string;
  pipelineId: string;
  samplesUsed: number;
  finalLoss: number | null;
  holdoutDelta: number | null;
  updatedModelBytes: ArrayBuffer | null;
};

const toDecisionSample = (
  session: TrajectorySessionV1,
  sample: TrajectorySessionV1['samples'][number],
  pipeline: ResolvedPersonalTrainingPipeline,
): TrajectoryDecisionSample => {
  const arch = session.meta?.modelArchId?.trim().toLowerCase() || pipeline.arch;
  const rewardProfileId =
    session.meta?.rewardProfileId?.trim().toLowerCase() ||
    pipeline.rewardProfileId;
  const queuePolicyId =
    session.meta?.queuePolicyId?.trim().toLowerCase() || pipeline.queuePolicyId;
  const actionIndex =
    sample.actionIndex >= 0 && sample.actionIndex < sample.pieces.length
      ? sample.actionIndex
      : sample.pieces.indexOf(sample.action);

  return {
    schema: 'wishuponablock.trajectory.v1',
    id: `${session.sessionId}:${sample.id}`,
    modeId: session.modeId,
    arch,
    rewardProfileId,
    queuePolicyId,
    createdAtMs: sample.createdAtMs,
    deliberationMs: sample.deliberationMs,
    boardOccupancy: sample.boardOccupancy.map((row) => row.slice()),
    hold: sample.hold,
    action: sample.action,
    actionIndex: Math.max(0, actionIndex),
    pieces: [...sample.pieces],
    logits: [...sample.logits],
    probabilities: [...sample.probabilities],
    inferenceMs: sample.inferenceMs,
    samplingMs: sample.samplingMs,
    totalDecisionMs: sample.totalDecisionMs,
    reward: sample.reward,
  };
};

const toTrainingSamples = (
  sessions: TrajectorySessionV1[],
  pipeline: ResolvedPersonalTrainingPipeline,
): TrajectoryDecisionSample[] => {
  const out: TrajectoryDecisionSample[] = [];
  for (const session of sessions) {
    if (session.modeId !== pipeline.modeId) continue;
    for (const sample of session.samples) {
      if (sample.pieces.length === 0) continue;
      if (sample.logits.length !== sample.pieces.length) continue;
      if (sample.probabilities.length !== sample.pieces.length) continue;
      out.push(toDecisionSample(session, sample, pipeline));
    }
  }
  return out;
};

const normalizeResult = (
  result: PersonalTrainerResult,
): GlobalTrainingRunResult => ({
  ok: result.ok,
  message: result.message,
  pipelineId: result.pipelineId,
  samplesUsed: result.samplesUsed,
  finalLoss: result.finalLoss,
  holdoutDelta: result.holdoutDelta,
  updatedModelBytes: result.updatedModelBytes,
});

export const runGlobalTrainingOneShot = async (
  config: GlobalTrainingRunConfig,
): Promise<GlobalTrainingRunResult> => {
  const samples = toTrainingSamples(config.recordings, config.pipeline);
  const result = await config.trainer.trainHeadOnly({
    model: config.model,
    samples,
    pipeline: config.pipeline,
    train: config.train,
  });
  return normalizeResult(result);
};
