import { MIN_TRAJECTORY_SAMPLES_PER_SESSION } from '../core/trajectoryProtocol';
import type { TrajectoryRewardPolicyId } from './trajectoryRewardPolicy';

export type PersonalTrainingStrategyId = 'mlp_head_policy_v1';

export type PersonalTrainingPipelineId = 'personal_rl_v1';

export type PersonalTrainingEvalGate = {
  holdoutRatio: number;
  minHoldoutSamples: number;
  minTrainSamples: number;
  minObjectiveGain: number;
};

export type PersonalTrainingPipeline = {
  id: PersonalTrainingPipelineId;
  label: string;
  rewardPolicyId: TrajectoryRewardPolicyId;
  strategyId: PersonalTrainingStrategyId;
  trainableLayers: readonly string[];
  minSamples: number;
  evalGate: PersonalTrainingEvalGate;
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
    evalGate: {
      holdoutRatio: 0.2,
      minHoldoutSamples: 2,
      minTrainSamples: 6,
      minObjectiveGain: 0,
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
