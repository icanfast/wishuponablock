import { MIN_TRAJECTORY_SAMPLES_PER_SESSION } from '../core/trajectoryProtocol';
import type { TrajectoryRewardPolicyId } from './trajectoryRewardPolicy';
import {
  DEFAULT_MODEL_ARCH,
  DEFAULT_QUEUE_POLICY_ID,
  DEFAULT_REWARD_PROFILE_ID,
} from '../core/modelAxes';

export type PersonalTrainingStrategyId = 'mlp_head_policy_v1';

export type PersonalTrainingPipelineId =
  | 'personal_rl_next_piece_v1'
  | 'personal_rl_bag_shuffle_v1';

export type PersonalTrainingBackendPreference = 'auto' | 'webgl' | 'cpu';

export type PersonalTrainingAxes = {
  arch: string;
  rewardProfileId: string;
  queuePolicyId: string;
};

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

export type PersonalTrainingAxesOverrides = {
  arch?: string;
  rewardProfileId?: string;
  queuePolicyId?: string;
  pipelineId?: PersonalTrainingPipelineId;
  minSamples?: number;
  rewardPolicyId?: TrajectoryRewardPolicyId;
  trainDefaults?: Partial<PersonalTrainingTrainDefaults>;
  evalGate?: Partial<PersonalTrainingEvalGate>;
};

const PERSONAL_TRAINING_PIPELINE_REGISTRY: Record<
  PersonalTrainingPipelineId,
  PersonalTrainingPipeline
> = {
  personal_rl_next_piece_v1: {
    id: 'personal_rl_next_piece_v1',
    label: 'Personal RL Next Piece v1',
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
        minSamples: 2,
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
  personal_rl_bag_shuffle_v1: {
    id: 'personal_rl_bag_shuffle_v1',
    label: 'Personal RL Bag Shuffle v1',
    rewardPolicyId: 'comfort_v1',
    strategyId: 'mlp_head_policy_v1',
    trainableLayers: ['mlp.0', 'mlp.2'],
    minSamples: Math.max(MIN_TRAJECTORY_SAMPLES_PER_SESSION, 12),
    trainDefaults: {
      epochs: 10,
      learningRate: 0.002,
      l2: 0.00008,
      sampleLimit: 768,
      backendPreference: 'auto',
    },
    evalGate: {
      holdoutRatio: 0.25,
      minHoldoutSamples: 4,
      minTrainSamples: 10,
      minObjectiveGain: 0.0003,
    },
    modeOverrides: {
      sprint: {
        trainDefaults: {
          epochs: 12,
          learningRate: 0.0018,
          sampleLimit: 896,
        },
      },
      charcuterie: {
        minSamples: 2,
        trainDefaults: {
          epochs: 8,
          learningRate: 0.0015,
          sampleLimit: 1024,
        },
        evalGate: {
          holdoutRatio: 0.3,
          minHoldoutSamples: 6,
          minTrainSamples: 12,
          minObjectiveGain: 0.0008,
        },
      },
    },
  },
};

export const DEFAULT_PERSONAL_TRAINING_PIPELINE_ID: PersonalTrainingPipelineId =
  'personal_rl_next_piece_v1';

const LEGACY_PIPELINE_ALIAS = 'personal_rl_v1';

const PERSONAL_TRAINING_AXES_OVERRIDES: PersonalTrainingAxesOverrides[] = [
  {
    queuePolicyId: 'bag_shuffle_v1',
    pipelineId: 'personal_rl_bag_shuffle_v1',
  },
  {
    arch: 'lean',
    trainDefaults: {
      sampleLimit: 384,
    },
  },
];

export const resolvePersonalTrainingPipelineId = (
  value: string | null | undefined,
): PersonalTrainingPipelineId => {
  if (!value) return DEFAULT_PERSONAL_TRAINING_PIPELINE_ID;
  const normalized = value.trim().toLowerCase();
  if (normalized === LEGACY_PIPELINE_ALIAS) {
    return DEFAULT_PERSONAL_TRAINING_PIPELINE_ID;
  }
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
  arch: string;
  rewardProfileId: string;
  queuePolicyId: string;
};

const normalizeModeId = (value: string | null | undefined): string =>
  typeof value === 'string' ? value.trim().toLowerCase() : '';

const normalizeAxis = (value: unknown, fallback: string): string => {
  if (typeof value !== 'string') return fallback;
  const normalized = value.trim().toLowerCase();
  if (!/^[a-z0-9_-]{1,64}$/.test(normalized)) return fallback;
  return normalized;
};

const normalizeAxes = (
  value?: Partial<PersonalTrainingAxes> | null,
): PersonalTrainingAxes => ({
  arch: normalizeAxis(value?.arch, DEFAULT_MODEL_ARCH),
  rewardProfileId: normalizeAxis(
    value?.rewardProfileId,
    DEFAULT_REWARD_PROFILE_ID,
  ),
  queuePolicyId: normalizeAxis(value?.queuePolicyId, DEFAULT_QUEUE_POLICY_ID),
});

const matchesAxesOverride = (
  axes: PersonalTrainingAxes,
  override: PersonalTrainingAxesOverrides,
): boolean => {
  if (override.arch && override.arch !== axes.arch) return false;
  if (
    override.rewardProfileId &&
    override.rewardProfileId !== axes.rewardProfileId
  ) {
    return false;
  }
  if (override.queuePolicyId && override.queuePolicyId !== axes.queuePolicyId) {
    return false;
  }
  return true;
};

const findAxesOverride = (
  axes: PersonalTrainingAxes,
): PersonalTrainingAxesOverrides | null => {
  for (const override of PERSONAL_TRAINING_AXES_OVERRIDES) {
    if (matchesAxesOverride(axes, override)) return override;
  }
  return null;
};

export const resolvePersonalTrainingPipelineForMode = (
  pipelineValue: string | null | undefined,
  modeId: string | null | undefined,
): ResolvedPersonalTrainingPipeline => {
  return resolvePersonalTrainingPipelineForContext(pipelineValue, {
    modeId,
  });
};

export const resolvePersonalTrainingPipelineForContext = (
  pipelineValue: string | null | undefined,
  context: {
    modeId?: string | null;
    arch?: string | null;
    rewardProfileId?: string | null;
    queuePolicyId?: string | null;
  },
): ResolvedPersonalTrainingPipeline => {
  const axes = normalizeAxes({
    arch: context.arch ?? undefined,
    rewardProfileId: context.rewardProfileId ?? undefined,
    queuePolicyId: context.queuePolicyId ?? undefined,
  });
  const axesOverride = findAxesOverride(axes);
  const pipelineId = pipelineValue
    ? resolvePersonalTrainingPipelineId(pipelineValue)
    : (axesOverride?.pipelineId ?? DEFAULT_PERSONAL_TRAINING_PIPELINE_ID);
  const base = getPersonalTrainingPipeline(pipelineId);
  const normalizedMode = normalizeModeId(context.modeId);
  const modeOverride = base.modeOverrides?.[normalizedMode];
  const mergedTrainDefaults = {
    ...base.trainDefaults,
    ...(modeOverride?.trainDefaults ?? {}),
    ...(axesOverride?.trainDefaults ?? {}),
  };
  const mergedEvalGate = {
    ...base.evalGate,
    ...(modeOverride?.evalGate ?? {}),
    ...(axesOverride?.evalGate ?? {}),
  };
  return {
    id: base.id,
    label: base.label,
    rewardPolicyId:
      axesOverride?.rewardPolicyId ??
      modeOverride?.rewardPolicyId ??
      base.rewardPolicyId,
    strategyId: base.strategyId,
    trainableLayers: base.trainableLayers,
    minSamples: Math.max(
      1,
      Math.trunc(
        axesOverride?.minSamples ?? modeOverride?.minSamples ?? base.minSamples,
      ),
    ),
    trainDefaults: mergedTrainDefaults,
    evalGate: mergedEvalGate,
    modeId: normalizedMode || 'unknown',
    arch: axes.arch,
    rewardProfileId: axes.rewardProfileId,
    queuePolicyId: axes.queuePolicyId,
  };
};
