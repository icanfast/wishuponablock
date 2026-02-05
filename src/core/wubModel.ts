import type { Board, PieceKind } from './types';
import { PIECES } from './types';

export type ModelTensor = {
  shape: number[];
  data: Float32Array;
};

export type ExportedModel = {
  schema: string;
  model: {
    input_channels: number;
    conv_channels: number[];
    mlp_hidden: number;
    extra_features: number;
    num_outputs: number;
  };
  params: Record<string, { shape: number[]; data: number[] }>;
  pieces?: string[];
  board_channels?: string[];
};

export type LoadedModel = {
  config: ExportedModel['model'];
  params: Record<string, ModelTensor>;
  pieces: PieceKind[];
  boardChannels: string[];
};

const DEFAULT_BOARD_CHANNELS = ['occupancy', 'holes', 'row_fill'];

export async function loadWubModel(url: string): Promise<LoadedModel> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load model (${res.status})`);
  }
  const json = (await res.json()) as ExportedModel;
  return parseWubModel(json);
}

export function parseWubModel(payload: ExportedModel): LoadedModel {
  const params: Record<string, ModelTensor> = {};
  for (const [key, value] of Object.entries(payload.params)) {
    params[key] = {
      shape: value.shape,
      data: new Float32Array(value.data),
    };
  }

  const pieces =
    payload.pieces && payload.pieces.length === PIECES.length
      ? ([...payload.pieces] as PieceKind[])
      : [...PIECES];

  const boardChannels =
    payload.board_channels && payload.board_channels.length > 0
      ? payload.board_channels
      : DEFAULT_BOARD_CHANNELS;

  return {
    config: payload.model,
    params,
    pieces,
    boardChannels,
  };
}

export function predictLogits(
  model: LoadedModel,
  board: Board,
  hold: PieceKind | null,
): Float32Array {
  const { config } = model;
  const rows = board.length;
  const cols = board[0]?.length ?? 0;
  let input = buildInputChannels(board, model.boardChannels);
  const expectedLength = config.input_channels * rows * cols;
  if (input.length !== expectedLength) {
    const padded = new Float32Array(expectedLength);
    padded.set(input.subarray(0, expectedLength), 0);
    input = padded;
  }

  let current = input;
  let inChannels = config.input_channels;
  const height = rows;
  const width = cols;

  for (let i = 0; i < config.conv_channels.length; i++) {
    const layerIndex = i * 2;
    const weight = getParam(model, `conv.${layerIndex}.weight`);
    const bias = getParam(model, `conv.${layerIndex}.bias`);
    current = conv2d(
      current,
      inChannels,
      config.conv_channels[i],
      height,
      width,
      weight.data,
      bias.data,
    );
    inChannels = config.conv_channels[i];
  }

  const pooled = globalAveragePool(current, inChannels, height, width);

  const extra = buildExtraFeatures(config.extra_features, hold, model.pieces);

  const mlpInput = concatFeatures(pooled, extra);
  const hidden = linear(
    mlpInput,
    getParam(model, 'mlp.0.weight').data,
    getParam(model, 'mlp.0.bias').data,
  );
  reluInPlace(hidden);
  const logits = linear(
    hidden,
    getParam(model, 'mlp.2.weight').data,
    getParam(model, 'mlp.2.bias').data,
  );
  return logits;
}

export function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (const v of logits) {
    if (v > max) max = v;
  }
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const val = Math.exp(logits[i] - max);
    out[i] = val;
    sum += val;
  }
  if (sum === 0) {
    const uniform = 1 / out.length;
    out.fill(uniform);
    return out;
  }
  for (let i = 0; i < out.length; i++) {
    out[i] /= sum;
  }
  return out;
}

function buildInputChannels(board: Board, channels: string[]): Float32Array {
  const rows = board.length;
  const cols = board[0]?.length ?? 0;
  const occupancy = new Float32Array(rows * cols);
  const rowFill = new Float32Array(rows * cols);

  for (let y = 0; y < rows; y++) {
    let rowCount = 0;
    for (let x = 0; x < cols; x++) {
      const filled = board[y][x] != null ? 1 : 0;
      occupancy[y * cols + x] = filled;
      rowCount += filled;
    }
    const ratio = cols > 0 ? rowCount / cols : 0;
    for (let x = 0; x < cols; x++) {
      rowFill[y * cols + x] = ratio;
    }
  }

  const holes = new Float32Array(rows * cols);
  for (let x = 0; x < cols; x++) {
    let filledSeen = false;
    for (let y = 0; y < rows; y++) {
      const filled = occupancy[y * cols + x] > 0;
      if (filled) {
        filledSeen = true;
      } else if (filledSeen) {
        holes[y * cols + x] = 1;
      }
    }
  }

  const input = new Float32Array(channels.length * rows * cols);
  channels.forEach((name, channelIndex) => {
    const offset = channelIndex * rows * cols;
    let source: Float32Array | null = null;
    switch (name) {
      case 'occupancy':
        source = occupancy;
        break;
      case 'holes':
        source = holes;
        break;
      case 'row_fill':
        source = rowFill;
        break;
      default:
        source = null;
    }
    if (source) {
      input.set(source, offset);
    }
  });

  return input;
}

function buildExtraFeatures(
  extraFeatures: number,
  hold: PieceKind | null,
  pieces: PieceKind[],
): Float32Array | null {
  if (extraFeatures <= 0) return null;
  const out = new Float32Array(extraFeatures);
  const holdIndex = hold ? pieces.indexOf(hold) : -1;
  const idx = holdIndex >= 0 ? holdIndex + 1 : 0;
  if (idx < out.length) {
    out[idx] = 1;
  }
  return out;
}

function concatFeatures(
  base: Float32Array,
  extra: Float32Array | null,
): Float32Array {
  if (!extra || extra.length === 0) return base;
  const out = new Float32Array(base.length + extra.length);
  out.set(base, 0);
  out.set(extra, base.length);
  return out;
}

function getParam(model: LoadedModel, name: string): ModelTensor {
  const tensor = model.params[name];
  if (!tensor) {
    throw new Error(`Missing model param: ${name}`);
  }
  return tensor;
}

function conv2d(
  input: Float32Array,
  inChannels: number,
  outChannels: number,
  height: number,
  width: number,
  weight: Float32Array,
  bias: Float32Array,
): Float32Array {
  const output = new Float32Array(outChannels * height * width);
  const kernelSize = 3;
  const pad = 1;
  for (let oc = 0; oc < outChannels; oc++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = bias[oc] ?? 0;
        for (let ic = 0; ic < inChannels; ic++) {
          for (let ky = 0; ky < kernelSize; ky++) {
            for (let kx = 0; kx < kernelSize; kx++) {
              const iy = y + ky - pad;
              const ix = x + kx - pad;
              if (iy < 0 || iy >= height || ix < 0 || ix >= width) {
                continue;
              }
              const inputIdx = (ic * height + iy) * width + ix;
              const weightIdx =
                ((oc * inChannels + ic) * kernelSize + ky) * kernelSize + kx;
              sum += input[inputIdx] * weight[weightIdx];
            }
          }
        }
        const outIdx = (oc * height + y) * width + x;
        output[outIdx] = sum > 0 ? sum : 0;
      }
    }
  }
  return output;
}

function globalAveragePool(
  input: Float32Array,
  channels: number,
  height: number,
  width: number,
): Float32Array {
  const spatial = height * width;
  const out = new Float32Array(channels);
  for (let c = 0; c < channels; c++) {
    let sum = 0;
    const offset = c * spatial;
    for (let i = 0; i < spatial; i++) {
      sum += input[offset + i];
    }
    out[c] = sum / spatial;
  }
  return out;
}

function linear(
  input: Float32Array,
  weight: Float32Array,
  bias: Float32Array,
): Float32Array {
  const outFeatures = bias.length;
  const inFeatures = input.length;
  const out = new Float32Array(outFeatures);
  for (let o = 0; o < outFeatures; o++) {
    let sum = bias[o] ?? 0;
    const wOffset = o * inFeatures;
    for (let i = 0; i < inFeatures; i++) {
      sum += weight[wOffset + i] * input[i];
    }
    out[o] = sum;
  }
  return out;
}

function reluInPlace(values: Float32Array): void {
  for (let i = 0; i < values.length; i++) {
    if (values[i] < 0) values[i] = 0;
  }
}
