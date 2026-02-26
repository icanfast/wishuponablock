import { describe, expect, it } from 'vitest';
import { ModelGenerator } from '../core/modelGenerator';
import type { ModelRunner } from '../core/modelRunner';
import { PIECES, type Board, type PieceKind } from '../core/types';
import type { LoadedModel } from '../core/wubModel';

const EMPTY_BOARD: Board = Array.from({ length: 20 }, () =>
  Array.from({ length: 10 }, () => null),
);

const TEST_MODEL: LoadedModel = {
  config: {
    input_channels: 1,
    conv_channels: [],
    mlp_hidden: 1,
    extra_features: 0,
    num_outputs: PIECES.length,
  },
  params: {},
  pieces: [...PIECES],
  boardChannels: ['occupancy'],
};

const createFixedRunner = (logits: number[]): ModelRunner => ({
  prepare: async () => {},
  getInfo: () => ({
    requestedBackend: 'native',
    activeBackend: 'native',
    runtimeBackend: null,
    fallbackReason: null,
  }),
  predictLogits: () => new Float32Array(logits),
});

describe('ModelGenerator queue policy', () => {
  it('bag_shuffle_v1 creates deterministic weighted bag order', () => {
    const runner = createFixedRunner([7, 6, 5, 4, 3, 2, 1]);
    const a = new ModelGenerator(12345, TEST_MODEL, undefined, runner, {
      queuePolicyId: 'bag_shuffle_v1',
    });
    const b = new ModelGenerator(12345, TEST_MODEL, undefined, runner, {
      queuePolicyId: 'bag_shuffle_v1',
    });

    a.onLock(EMPTY_BOARD, null);
    b.onLock(EMPTY_BOARD, null);

    const previewA = a.peek(PIECES.length);
    const previewB = b.peek(PIECES.length);
    expect(previewA).toEqual(previewB);
    expect(new Set(previewA)).toEqual(new Set(PIECES));

    const drawA = Array.from({ length: PIECES.length }, () => a.next());
    const drawB = Array.from({ length: PIECES.length }, () => b.next());
    expect(drawA).toEqual(previewA);
    expect(drawA).toEqual(drawB);
    expect(new Set(drawA).size).toBe(PIECES.length);
  });

  it('bag_shuffle_v1 emits a decision on every lock using queued entries', () => {
    const runner = createFixedRunner([4, 3, 2, 1, 0.5, 0.25, 0.125]);
    const actions: PieceKind[] = [];
    const generator = new ModelGenerator(777, TEST_MODEL, undefined, runner, {
      queuePolicyId: 'bag_shuffle_v1',
      onDecision: (event) => {
        actions.push(event.action);
      },
    });

    for (let i = 0; i < 5; i++) {
      generator.onLock(EMPTY_BOARD, null);
      expect(actions.length).toBe(i + 1);
      expect(generator.next()).toBe(actions[i]);
    }
  });
});
