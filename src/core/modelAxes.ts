export const DEFAULT_MODEL_ARCH = 'full' as const;
export const DEFAULT_REWARD_PROFILE_ID = 'default' as const;
export const DEFAULT_QUEUE_POLICY_ID = 'next_piece_v1' as const;

export const MODEL_ARCH_OPTIONS = ['full', 'lean'] as const;
export const REWARD_PROFILE_OPTIONS = ['default'] as const;
export const QUEUE_POLICY_OPTIONS = [
  'next_piece_v1',
  'bag_shuffle_v1',
] as const;

export type ModelArchOption = (typeof MODEL_ARCH_OPTIONS)[number];
export type RewardProfileOption = (typeof REWARD_PROFILE_OPTIONS)[number];
export type QueuePolicyOption = (typeof QUEUE_POLICY_OPTIONS)[number];

export type ModelAxes = {
  arch: string;
  rewardProfileId: string;
  queuePolicyId: string;
};

const isValidAxis = (value: string): boolean =>
  /^[a-z0-9_-]{1,64}$/.test(value.trim().toLowerCase());

const normalizeAxis = (value: unknown, fallback: string): string => {
  if (typeof value !== 'string') return fallback;
  const normalized = value.trim().toLowerCase();
  if (!normalized || !isValidAxis(normalized)) return fallback;
  return normalized;
};

export const normalizeModelAxes = (value: unknown): ModelAxes => {
  const obj =
    value && typeof value === 'object' && !Array.isArray(value)
      ? (value as {
          arch?: unknown;
          rewardProfileId?: unknown;
          queuePolicyId?: unknown;
        })
      : null;
  return {
    arch: normalizeAxis(obj?.arch, DEFAULT_MODEL_ARCH),
    rewardProfileId: normalizeAxis(
      obj?.rewardProfileId,
      DEFAULT_REWARD_PROFILE_ID,
    ),
    queuePolicyId: normalizeAxis(obj?.queuePolicyId, DEFAULT_QUEUE_POLICY_ID),
  };
};

export const modelAxesKey = (value: ModelAxes): string =>
  `${value.arch}:${value.rewardProfileId}:${value.queuePolicyId}`;
