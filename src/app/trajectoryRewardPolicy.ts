import type { TrajectoryDecisionSample } from './trajectoryBuffer';

export type TrajectoryRewardPolicyId = 'comfort_v1';
export type TrajectoryRewardKind = 'discounted_return';

export type TrajectoryRewardTerminalStats = {
  totalLinesCleared: number;
  score: number;
  timeMs: number;
};

export type TrajectoryRewardContext = {
  modeId: string;
  outcome: string;
  terminal: TrajectoryRewardTerminalStats | null;
};

export type TrajectoryRewardComputation = {
  policyId: TrajectoryRewardPolicyId;
  kind: TrajectoryRewardKind;
  gamma: number;
  rewards: number[];
  diagnostics: {
    sum: number;
    mean: number;
    min: number;
    max: number;
  };
};

type RewardPolicy = (
  samples: TrajectoryDecisionSample[],
  context: TrajectoryRewardContext,
) => TrajectoryRewardComputation;

type ComfortModeConfig = {
  targetDeliberationMs: number;
  speedWeight: number;
  stepReward: number;
  successBonus: number;
  failurePenalty: number;
  linesWeight: number;
  scoreWeight: number;
  gamma: number;
};

const DEFAULT_POLICY_ID: TrajectoryRewardPolicyId = 'comfort_v1';

const DEFAULT_COMFORT_CONFIG: ComfortModeConfig = {
  targetDeliberationMs: 900,
  speedWeight: 0.3,
  stepReward: 0.02,
  successBonus: 0.4,
  failurePenalty: 0.25,
  linesWeight: 0.04,
  scoreWeight: 0.02,
  gamma: 0.995,
};

const COMFORT_MODE_OVERRIDES: Partial<
  Record<string, Partial<ComfortModeConfig>>
> = {
  sprint: {
    targetDeliberationMs: 700,
    speedWeight: 0.35,
    stepReward: 0.025,
    successBonus: 0.8,
    failurePenalty: 0.3,
    linesWeight: 0.08,
    scoreWeight: 0,
    gamma: 0.996,
  },
  classic: {
    targetDeliberationMs: 850,
    speedWeight: 0.25,
    stepReward: 0.02,
    successBonus: 0.2,
    failurePenalty: 0.2,
    linesWeight: 0.03,
    scoreWeight: 0.06,
    gamma: 0.997,
  },
  cheese: {
    targetDeliberationMs: 750,
    speedWeight: 0.3,
    stepReward: 0.02,
    successBonus: 0.6,
    failurePenalty: 0.25,
    linesWeight: 0.06,
    scoreWeight: 0,
    gamma: 0.996,
  },
  charcuterie: {
    targetDeliberationMs: 1000,
    speedWeight: 0.2,
    stepReward: 0.015,
    successBonus: 0.4,
    failurePenalty: 0.2,
    linesWeight: 0,
    scoreWeight: 0,
    gamma: 0.994,
  },
};

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

const toNonNegative = (value: number): number =>
  Number.isFinite(value) ? Math.max(0, value) : 0;

const getComfortConfig = (modeId: string): ComfortModeConfig => {
  const override = COMFORT_MODE_OVERRIDES[modeId] ?? {};
  return {
    ...DEFAULT_COMFORT_CONFIG,
    ...override,
  };
};

const computeTerminalBonus = (
  modeId: string,
  outcome: string,
  terminal: TrajectoryRewardTerminalStats | null,
  config: ComfortModeConfig,
): number => {
  let bonus = 0;
  if (outcome === 'game_won') bonus += config.successBonus;
  if (outcome === 'game_over') bonus -= config.failurePenalty;
  if (!terminal) return bonus;

  const lines = toNonNegative(terminal.totalLinesCleared);
  const score = toNonNegative(terminal.score);
  const timeMs = toNonNegative(terminal.timeMs);
  bonus += config.linesWeight * clamp(lines / 40, 0, 4);
  bonus += config.scoreWeight * clamp(score / 100_000, 0, 8);

  if (modeId === 'sprint' && outcome === 'game_won' && timeMs > 0) {
    const speedScore = clamp((180_000 - timeMs) / 180_000, -1, 1);
    bonus += 0.5 * speedScore;
  }

  return bonus;
};

const computeDiscountedReturns = (
  immediateRewards: number[],
  gamma: number,
): number[] => {
  const rewards = new Array<number>(immediateRewards.length);
  let running = 0;
  for (let i = immediateRewards.length - 1; i >= 0; i -= 1) {
    running = immediateRewards[i] + gamma * running;
    rewards[i] = running;
  }
  return rewards;
};

const withDiagnostics = (
  policyId: TrajectoryRewardPolicyId,
  kind: TrajectoryRewardKind,
  gamma: number,
  rewards: number[],
): TrajectoryRewardComputation => {
  if (rewards.length === 0) {
    return {
      policyId,
      kind,
      gamma,
      rewards,
      diagnostics: { sum: 0, mean: 0, min: 0, max: 0 },
    };
  }
  let sum = 0;
  let min = rewards[0];
  let max = rewards[0];
  for (const reward of rewards) {
    sum += reward;
    if (reward < min) min = reward;
    if (reward > max) max = reward;
  }
  return {
    policyId,
    kind,
    gamma,
    rewards,
    diagnostics: {
      sum,
      mean: sum / rewards.length,
      min,
      max,
    },
  };
};

const computeComfortV1: RewardPolicy = (samples, context) => {
  const config = getComfortConfig(context.modeId);
  if (samples.length === 0) {
    return withDiagnostics('comfort_v1', 'discounted_return', config.gamma, []);
  }

  const immediateRewards = new Array<number>(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    const sample = samples[i];
    const deliberationMs =
      sample.deliberationMs == null
        ? config.targetDeliberationMs
        : clamp(sample.deliberationMs, 0, 5000);
    const speedScore = clamp(
      (config.targetDeliberationMs - deliberationMs) /
        config.targetDeliberationMs,
      -1,
      1,
    );
    immediateRewards[i] = config.stepReward + config.speedWeight * speedScore;
  }
  immediateRewards[immediateRewards.length - 1] += computeTerminalBonus(
    context.modeId,
    context.outcome,
    context.terminal,
    config,
  );

  const rewards = computeDiscountedReturns(immediateRewards, config.gamma);
  return withDiagnostics(
    'comfort_v1',
    'discounted_return',
    config.gamma,
    rewards,
  );
};

const REWARD_POLICY_REGISTRY: Record<TrajectoryRewardPolicyId, RewardPolicy> = {
  comfort_v1: computeComfortV1,
};

export const resolveTrajectoryRewardPolicyId = (
  value: string | null | undefined,
): TrajectoryRewardPolicyId => {
  if (!value) return DEFAULT_POLICY_ID;
  const normalized = value.trim().toLowerCase();
  if (normalized in REWARD_POLICY_REGISTRY) {
    return normalized as TrajectoryRewardPolicyId;
  }
  return DEFAULT_POLICY_ID;
};

export const computeTrajectoryRewards = (
  samples: TrajectoryDecisionSample[],
  context: TrajectoryRewardContext,
  policyId: TrajectoryRewardPolicyId,
): TrajectoryRewardComputation =>
  REWARD_POLICY_REGISTRY[policyId](samples, context);
