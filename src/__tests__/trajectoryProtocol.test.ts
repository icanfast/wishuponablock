import { describe, expect, it } from 'vitest';
import {
  parseTrajectorySessionV1,
  TRAJECTORY_SESSION_SCHEMA_V1,
} from '../core/trajectoryProtocol';

const PIECES = ['I', 'O', 'T', 'S', 'Z', 'J', 'L'] as const;

const makeSample = (id: string, createdAtMs: number) => ({
  id,
  createdAtMs,
  deliberationMs: 120,
  boardOccupancy: Array.from({ length: 20 }, () =>
    Array.from({ length: 10 }, () => 0),
  ),
  hold: null,
  action: 'I',
  actionIndex: 0,
  pieces: [...PIECES],
  logits: [0, 0, 0, 0, 0, 0, 0],
  probabilities: [1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7],
  inferenceMs: 1.2,
  samplingMs: 0.2,
  totalDecisionMs: 1.4,
  reward: 0.1,
});

const makePayload = (sampleCount: number, meta: Record<string, unknown> = {}) =>
  ({
    schema: TRAJECTORY_SESSION_SCHEMA_V1,
    sessionId: 'session_12345678',
    modeId: 'practice',
    buildVersion: '0.2.4-dev',
    startedAtMs: 1_000,
    endedAtMs: 2_000,
    durationMs: 1_000,
    samples: Array.from({ length: sampleCount }, (_, index) =>
      makeSample(`sample_${index}`, 1_000 + index),
    ),
    meta: Object.keys(meta).length > 0 ? meta : null,
  }) as const;

describe('trajectory protocol parser', () => {
  it('rejects short trajectories when minSamples is required', () => {
    const parsed = parseTrajectorySessionV1(makePayload(2), {
      minSamples: 8,
    });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.error).toContain('too few samples');
  });

  it('accepts short trajectories when no minSamples option is provided', () => {
    const parsed = parseTrajectorySessionV1(makePayload(1));
    if (!parsed.ok) {
      throw new Error(parsed.error);
    }
    expect(parsed.ok).toBe(true);
  });

  it('parses pipeline metadata descriptor fields', () => {
    const parsed = parseTrajectorySessionV1(
      makePayload(8, {
        pipelineId: 'personal_rl_v1',
        rewardPolicyId: 'comfort_v1',
        rewardProfileId: 'default',
        queuePolicyId: 'bag_shuffle_v1',
        pipelineMode: 'practice',
        modelArchId: 'full',
        modelArch: 'ic11_conv12x12_pool1x1_mlp256_extra7_out7',
      }),
      { minSamples: 8 },
    );
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.meta?.pipelineId).toBe('personal_rl_v1');
    expect(parsed.value.meta?.rewardPolicyId).toBe('comfort_v1');
    expect(parsed.value.meta?.rewardProfileId).toBe('default');
    expect(parsed.value.meta?.queuePolicyId).toBe('bag_shuffle_v1');
    expect(parsed.value.meta?.pipelineMode).toBe('practice');
    expect(parsed.value.meta?.modelArchId).toBe('full');
    expect(parsed.value.meta?.modelArch).toContain('conv12x12');
  });

  it('parses optional replay telemetry on samples', () => {
    const payload = makePayload(8);
    (payload.samples[0] as Record<string, unknown>).replay = {
      lockPiece: 'T',
      lockRotation: 1,
      lockX: 4,
      lockY: 18,
      holdUsed: true,
      gameTimeMs: 5000,
      totalLinesCleared: 7,
      score: 1200,
    };
    const parsed = parseTrajectorySessionV1(payload, { minSamples: 8 });
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.value.samples[0].replay?.lockPiece).toBe('T');
    expect(parsed.value.samples[0].replay?.holdUsed).toBe(true);
  });

  it('rejects invalid replay telemetry payloads', () => {
    const payload = makePayload(8);
    (payload.samples[0] as Record<string, unknown>).replay = {
      lockPiece: 'I',
      lockRotation: 9,
      lockX: 4,
      lockY: 18,
      holdUsed: false,
      gameTimeMs: 1000,
      totalLinesCleared: 0,
      score: 0,
    };
    const parsed = parseTrajectorySessionV1(payload, { minSamples: 8 });
    expect(parsed.ok).toBe(false);
    if (parsed.ok) return;
    expect(parsed.error).toContain('Invalid trajectory sample');
  });
});
