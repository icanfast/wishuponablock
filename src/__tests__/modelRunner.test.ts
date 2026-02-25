import { describe, expect, it } from 'vitest';
import { createModelRunner } from '../core/modelRunner';
import {
  parseWubModel,
  predictLogits,
  type ExportedModel,
} from '../core/wubModel';
import type { Board } from '../core/types';

function createBoard(): Board {
  const rows = 20;
  const cols = 10;
  const board: Board = Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => null),
  );
  board[19][0] = 'I';
  board[19][1] = 'O';
  board[18][4] = 'T';
  board[17][5] = 'L';
  board[16][2] = 'S';
  return board;
}

function createTinyModel() {
  const payload: ExportedModel = {
    schema: 'wub/v1',
    model: {
      input_channels: 3,
      conv_channels: [1],
      mlp_hidden: 4,
      extra_features: 0,
      num_outputs: 7,
      pool_shape: [1, 1],
      feature_norm: null,
      feature_norm_eps: 1e-5,
      dropout_p: 0,
    },
    params: {
      'conv.0.weight': {
        shape: [1, 3, 3, 3],
        data: Array.from({ length: 27 }, (_, i) => (i - 13) * 0.01),
      },
      'conv.0.bias': {
        shape: [1],
        data: [0.02],
      },
      'mlp.0.weight': {
        shape: [4, 1],
        data: [0.2, -0.3, 0.15, 0.4],
      },
      'mlp.0.bias': {
        shape: [4],
        data: [0.01, -0.02, 0.03, 0.04],
      },
      'mlp.2.weight': {
        shape: [7, 4],
        data: Array.from({ length: 28 }, (_, i) =>
          i % 5 === 0 ? 0.11 : -0.07,
        ),
      },
      'mlp.2.bias': {
        shape: [7],
        data: [0.01, 0.02, -0.03, 0.04, 0.01, -0.01, 0.03],
      },
    },
    pieces: ['I', 'O', 'T', 'S', 'Z', 'J', 'L'],
    board_channels: ['occupancy', 'holes', 'row_fill'],
  };
  return parseWubModel(payload);
}

describe('Model runner', () => {
  it('matches native predictLogits when native backend is selected', async () => {
    const model = createTinyModel();
    const board = createBoard();
    const { runner } = createModelRunner({ preferredBackend: 'native' });
    await runner.prepare(model);
    const info = runner.getInfo();
    expect(info.requestedBackend).toBe('native');
    expect(info.activeBackend).toBe('native');
    expect(info.fallbackReason).toBeNull();

    const expected = predictLogits(model, board, 'I');
    const actual = runner.predictLogits(model, board, 'I');
    expect(Array.from(actual)).toEqual(Array.from(expected));
  });

  it('falls back to native when tfjs backend is requested in node runtime', async () => {
    const model = createTinyModel();
    const board = createBoard();
    const { runner } = createModelRunner({ preferredBackend: 'tfjs' });
    await runner.prepare(model);
    const info = runner.getInfo();
    expect(info.requestedBackend).toBe('tfjs');
    expect(info.activeBackend).toBe('native');
    expect(info.fallbackReason).toContain('browser runtime');

    const expected = predictLogits(model, board, null);
    const actual = runner.predictLogits(model, board, null);
    expect(Array.from(actual)).toEqual(Array.from(expected));
  });
});
