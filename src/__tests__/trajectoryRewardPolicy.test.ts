import { describe, expect, it } from 'vitest';
import type { PieceKind } from '../core/types';
import {
  TRAJECTORY_SCHEMA_V1,
  type TrajectoryDecisionSample,
} from '../app/trajectoryBuffer';
import {
  computeTrajectoryRewards,
  resolveTrajectoryRewardPolicyId,
} from '../app/trajectoryRewardPolicy';

const PIECES: PieceKind[] = ['I', 'O', 'T', 'S', 'Z', 'J', 'L'];

const makeBoard = (): number[][] =>
  Array.from({ length: 20 }, () => Array.from({ length: 10 }, () => 0));

const makeSample = (
  id: string,
  deliberationMs: number,
): TrajectoryDecisionSample => ({
  schema: TRAJECTORY_SCHEMA_V1,
  id,
  modeId: 'practice',
  arch: 'full',
  rewardProfileId: 'default',
  queuePolicyId: 'next_piece_v1',
  createdAtMs: Date.now(),
  deliberationMs,
  boardOccupancy: makeBoard(),
  hold: null,
  action: 'I',
  actionIndex: 0,
  pieces: [...PIECES],
  logits: [0, 0, 0, 0, 0, 0, 0],
  probabilities: [1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7],
  inferenceMs: 1,
  samplingMs: 0.3,
  totalDecisionMs: 1.3,
  reward: null,
});

describe('trajectory reward policy', () => {
  it('rewards quicker deliberation more than slower deliberation', () => {
    const fast = [
      makeSample('a', 150),
      makeSample('b', 200),
      makeSample('c', 250),
    ];
    const slow = [
      makeSample('a', 1400),
      makeSample('b', 1500),
      makeSample('c', 1600),
    ];
    const policyId = resolveTrajectoryRewardPolicyId('comfort_v1');
    const fastRewards = computeTrajectoryRewards(
      fast,
      {
        modeId: 'practice',
        outcome: 'manual',
        terminal: null,
      },
      policyId,
    );
    const slowRewards = computeTrajectoryRewards(
      slow,
      {
        modeId: 'practice',
        outcome: 'manual',
        terminal: null,
      },
      policyId,
    );
    expect(fastRewards.diagnostics.mean).toBeGreaterThan(
      slowRewards.diagnostics.mean,
    );
  });

  it('applies a better terminal return for a winning outcome', () => {
    const samples = [
      makeSample('a', 400),
      makeSample('b', 500),
      makeSample('c', 600),
    ];
    const policyId = resolveTrajectoryRewardPolicyId('comfort_v1');
    const won = computeTrajectoryRewards(
      samples,
      {
        modeId: 'sprint',
        outcome: 'game_won',
        terminal: { totalLinesCleared: 40, score: 12000, timeMs: 78_000 },
      },
      policyId,
    );
    const lost = computeTrajectoryRewards(
      samples,
      {
        modeId: 'sprint',
        outcome: 'game_over',
        terminal: { totalLinesCleared: 18, score: 9000, timeMs: 120_000 },
      },
      policyId,
    );
    expect(won.rewards[won.rewards.length - 1]).toBeGreaterThan(
      lost.rewards[lost.rewards.length - 1],
    );
  });

  it('falls back to comfort_v1 for unknown policy ids', () => {
    expect(resolveTrajectoryRewardPolicyId('future_policy')).toBe('comfort_v1');
  });
});
