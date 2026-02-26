import { describe, expect, it } from 'vitest';
import { createTrajectoryBuffer } from '../app/trajectoryBuffer';
import type { ModelGeneratorDecisionEvent } from '../core/modelGenerator';

const EMPTY_BOARD = Array.from({ length: 20 }, () =>
  Array.from({ length: 10 }, () => null),
);

const makeDecision = (
  action: 'I' | 'O' | 'T',
  wallTimeMs: number,
): ModelGeneratorDecisionEvent => ({
  board: EMPTY_BOARD,
  hold: null,
  action,
  pieces: ['I', 'O', 'T', 'S', 'Z', 'J', 'L'],
  logits: new Float32Array([1, 0, 0, 0, 0, 0, 0]),
  probabilities: new Float32Array([1, 0, 0, 0, 0, 0, 0]),
  inferenceMs: 1,
  samplingMs: 0.2,
  totalMs: 1.2,
  wallTimeMs,
});

describe('trajectoryBuffer axes filters', () => {
  it('filters samples by mode and model axes', () => {
    const buffer = createTrajectoryBuffer({ maxSamples: 20 });
    buffer.recordDecision({
      modeId: 'practice',
      modelAxes: {
        arch: 'full',
        rewardProfileId: 'default',
        queuePolicyId: 'next_piece_v1',
      },
      decision: makeDecision('I', 1000),
    });
    buffer.recordDecision({
      modeId: 'practice',
      modelAxes: {
        arch: 'lean',
        rewardProfileId: 'default',
        queuePolicyId: 'next_piece_v1',
      },
      decision: makeDecision('O', 1100),
    });
    buffer.recordDecision({
      modeId: 'sprint',
      modelAxes: {
        arch: 'full',
        rewardProfileId: 'default',
        queuePolicyId: 'bag_shuffle_v1',
      },
      decision: makeDecision('T', 1200),
    });

    const filtered = buffer.listSamples({
      modeId: 'practice',
      modelAxes: {
        arch: 'full',
        rewardProfileId: 'default',
        queuePolicyId: 'next_piece_v1',
      },
    });
    expect(filtered).toHaveLength(1);
    expect(filtered[0].action).toBe('I');
  });

  it('tracks stats buckets by mode and axes', () => {
    const buffer = createTrajectoryBuffer({ maxSamples: 20 });
    buffer.recordDecision({
      modeId: 'practice',
      modelAxes: {
        arch: 'full',
        rewardProfileId: 'default',
        queuePolicyId: 'next_piece_v1',
      },
      decision: makeDecision('I', 2000),
    });
    buffer.recordDecision({
      modeId: 'practice',
      modelAxes: {
        arch: 'full',
        rewardProfileId: 'default',
        queuePolicyId: 'next_piece_v1',
      },
      decision: makeDecision('O', 2100),
    });

    const stats = buffer.getStats();
    expect(stats.totalSamples).toBe(2);
    expect(stats.byMode.practice).toBe(2);
    expect(stats.byModeAndAxes['practice:full:default:next_piece_v1']).toBe(2);
  });
});
