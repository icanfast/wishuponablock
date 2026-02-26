import { MIN_TRAJECTORY_SAMPLES_PER_SESSION } from '../core/trajectoryProtocol';
import type { TrajectoryRewardPolicyId } from './trajectoryRewardPolicy';

export type PersonalTrainingStrategyId = 'mlp_head_policy_v1';

export type PersonalTrainingPipelineId = 'personal_rl_v1';

export type PersonalTrainingBackendPreference = 'auto' | 'webgl' | 'cpu';

export type PersonalTrainingTrainDefaults = {
  epochs: number;
  learningRate: number;
  l2: number;
  sampleLimit: number;
  backendPreference: PersonalTrainingBackendPreference;
};

export type PersonalTrainingEvalGate = {
  holdoutRatio: number;
  minHoldoutSamples: number;
  minTrainSamples: number;
  minObjectiveGain: number;
};

export type PersonalTrainingModeOverrides = {
  minSamples?: number;
  rewardPolicyId?: TrajectoryRewardPolicyId;
  trainDefaults?: Partial<PersonalTrainingTrainDefaults>;
  evalGate?: Partial<PersonalTrainingEvalGate>;
};

export type PersonalTrainingPipeline = {
  id: PersonalTrainingPipelineId;
  label: string;
  rewardPolicyId: TrajectoryRewardPolicyId;
  strategyId: PersonalTrainingStrategyId;
  trainableLayers: readonly string[];
  minSamples: number;
  trainDefaults: PersonalTrainingTrainDefaults;
  evalGate: PersonalTrainingEvalGate;
  modeOverrides?: Record<string, PersonalTrainingModeOverrides>;
};

const PERSONAL_TRAINING_PIPELINE_REGISTRY: Record<
  PersonalTrainingPipelineId,
  PersonalTrainingPipeline
> = {
  personal_rl_v1: {
    id: 'personal_rl_v1',
    label: 'Personal RL v1',
    rewardPolicyId: 'comfort_v1',
    strategyId: 'mlp_head_policy_v1',
    trainableLayers: ['mlp.0', 'mlp.2'],
    minSamples: MIN_TRAJECTORY_SAMPLES_PER_SESSION,
    trainDefaults: {
      epochs: 8,
      learningRate: 0.003,
      l2: 0.00005,
      sampleLimit: 512,
      backendPreference: 'auto',
    },
    evalGate: {
      holdoutRatio: 0.2,
      minHoldoutSamples: 2,
      minTrainSamples: 6,
      minObjectiveGain: 0,
    },
    modeOverrides: {
      practice: {
        trainDefaults: {
          epochs: 8,
          learningRate: 0.003,
          sampleLimit: 512,
        },
      },
      sprint: {
        trainDefaults: {
          epochs: 10,
          learningRate: 0.0025,
          sampleLimit: 640,
        },
        evalGate: {
          holdoutRatio: 0.25,
          minHoldoutSamples: 4,
          minTrainSamples: 8,
          minObjectiveGain: 0.0005,
        },
      },
      classic: {
        trainDefaults: {
          epochs: 8,
          learningRate: 0.002,
          sampleLimit: 640,
        },
        evalGate: {
          holdoutRatio: 0.2,
          minHoldoutSamples: 3,
          minTrainSamples: 8,
          minObjectiveGain: 0,
        },
      },
      charcuterie: {
        trainDefaults: {
          epochs: 6,
          learningRate: 0.0015,
          sampleLimit: 768,
        },
        evalGate: {
          holdoutRatio: 0.3,
          minHoldoutSamples: 6,
          minTrainSamples: 12,
          minObjectiveGain: 0.001,
        },
      },
    },
  },
};

export const DEFAULT_PERSONAL_TRAINING_PIPELINE_ID: PersonalTrainingPipelineId =
  'personal_rl_v1';

export const resolvePersonalTrainingPipelineId = (
  value: string | null | undefined,
): PersonalTrainingPipelineId => {
  if (!value) return DEFAULT_PERSONAL_TRAINING_PIPELINE_ID;
  const normalized = value.trim();
  if (normalized in PERSONAL_TRAINING_PIPELINE_REGISTRY) {
    return normalized as PersonalTrainingPipelineId;
  }
  return DEFAULT_PERSONAL_TRAINING_PIPELINE_ID;
};

export const getPersonalTrainingPipeline = (
  value: string | null | undefined,
): PersonalTrainingPipeline =>
  PERSONAL_TRAINING_PIPELINE_REGISTRY[resolvePersonalTrainingPipelineId(value)];

export type ResolvedPersonalTrainingPipeline = Omit<
  PersonalTrainingPipeline,
  'modeOverrides'
> & {
  modeId: string;
};

const normalizeModeId = (value: string | null | undefined): string =>
  typeof value === 'string' ? value.trim().toLowerCase() : '';

export const resolvePersonalTrainingPipelineForMode = (
  pipelineValue: string | null | undefined,
  modeId: string | null | undefined,
): ResolvedPersonalTrainingPipeline => {
  const base = getPersonalTrainingPipeline(pipelineValue);
  const normalizedMode = normalizeModeId(modeId);
  const modeOverride = base.modeOverrides?.[normalizedMode];
  return {
    id: base.id,
    label: base.label,
    rewardPolicyId: modeOverride?.rewardPolicyId ?? base.rewardPolicyId,
    strategyId: base.strategyId,
    trainableLayers: base.trainableLayers,
    minSamples: Math.max(
      1,
      Math.trunc(modeOverride?.minSamples ?? base.minSamples),
    ),
    trainDefaults: {
      ...base.trainDefaults,
      ...(modeOverride?.trainDefaults ?? {}),
    },
    evalGate: {
      ...base.evalGate,
      ...(modeOverride?.evalGate ?? {}),
    },
    modeId: normalizedMode || 'unknown',
  };
};
