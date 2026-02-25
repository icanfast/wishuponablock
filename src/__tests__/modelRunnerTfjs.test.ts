import { describe, expect, it } from 'vitest';
import { reorderInputChwToNhwc } from '../core/modelRunnerTfjs';

describe('TFJS runner helpers', () => {
  it('reorders model input from CHW to NHWC layout', () => {
    const channels = 2;
    const rows = 2;
    const cols = 3;
    const chw = new Float32Array([
      // c0
      1, 2, 3, 4, 5, 6,
      // c1
      101, 102, 103, 104, 105, 106,
    ]);
    const actual = reorderInputChwToNhwc(chw, channels, rows, cols);
    const expected = new Float32Array([
      // y=0, x=0..2
      1, 101, 2, 102, 3, 103,
      // y=1, x=0..2
      4, 104, 5, 105, 6, 106,
    ]);
    expect(Array.from(actual)).toEqual(Array.from(expected));
  });
});
