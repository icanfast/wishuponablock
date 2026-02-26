import type { Board, PieceKind } from './types';
import { PIECES } from './types';
import { hasPerfMetricsSink, recordPerfDuration } from './perfMetrics';

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
    pool_shape?: number[];
    feature_norm?: string | null;
    feature_norm_eps?: number;
    dropout_p?: number;
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
const MODEL_SCHEMA_V1 = 'wishuponablock.model.v1';

export async function loadWubModel(url: string): Promise<LoadedModel> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load model (${res.status})`);
  }
  const bytes = await res.arrayBuffer();
  return parseWubModelFromBytes(bytes);
}

export function parseWubModelFromBytes(
  payload: ArrayBuffer | ArrayBufferView,
): LoadedModel {
  const view =
    payload instanceof ArrayBuffer
      ? new Uint8Array(payload)
      : new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength);
  const text = new TextDecoder().decode(view);
  return parseWubModelFromJsonText(text);
}

export function parseWubModelFromJsonText(payload: string): LoadedModel {
  let json: ExportedModel;
  try {
    json = JSON.parse(payload) as ExportedModel;
  } catch {
    throw new Error('Invalid model payload: expected JSON.');
  }
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

export function serializeWubModel(model: LoadedModel): ExportedModel {
  const params: Record<string, { shape: number[]; data: number[] }> = {};
  for (const [name, tensor] of Object.entries(model.params)) {
    params[name] = {
      shape: [...tensor.shape],
      data: Array.from(tensor.data),
    };
  }
  return {
    schema: MODEL_SCHEMA_V1,
    model: {
      ...model.config,
      conv_channels: [...model.config.conv_channels],
      pool_shape: model.config.pool_shape
        ? [...model.config.pool_shape]
        : undefined,
    },
    params,
    pieces: [...model.pieces],
    board_channels: [...model.boardChannels],
  };
}

export function serializeWubModelToJsonText(model: LoadedModel): string {
  return JSON.stringify(serializeWubModel(model));
}

export function serializeWubModelToBytes(model: LoadedModel): ArrayBuffer {
  const text = serializeWubModelToJsonText(model);
  return new TextEncoder().encode(text).buffer;
}

export function buildModelHeadInput(
  model: LoadedModel,
  board: Board,
  hold: PieceKind | null,
): Float32Array {
  const { config } = model;
  const rows = board.length;
  const cols = board[0]?.length ?? 0;

  let input = buildModelInputChannels(board, model.boardChannels);
  const expectedLength = config.input_channels * rows * cols;
  if (input.length !== expectedLength) {
    const padded = new Float32Array(expectedLength);
    padded.set(input.subarray(0, expectedLength), 0);
    input = padded;
  }

  let current = input;
  let inChannels = config.input_channels;
  const poolShape = getModelPoolShape(config.pool_shape);

  for (let i = 0; i < config.conv_channels.length; i++) {
    const layerIndex = i * 2;
    const weight = getParam(model, `conv.${layerIndex}.weight`);
    const bias = getParam(model, `conv.${layerIndex}.bias`);
    current = conv2d(
      current,
      inChannels,
      config.conv_channels[i],
      rows,
      cols,
      weight.data,
      bias.data,
    );
    inChannels = config.conv_channels[i];
  }

  const pooled = adaptiveAveragePool(
    current,
    inChannels,
    rows,
    cols,
    poolShape[0],
    poolShape[1],
  );
  const extra = buildModelExtraFeatures(
    config.extra_features,
    hold,
    model.pieces,
  );
  let mlpInput = concatFeatures(pooled, extra);
  if (config.feature_norm === 'layernorm') {
    mlpInput = layerNorm1d(mlpInput, config.feature_norm_eps ?? 1e-5);
  }
  return mlpInput;
}

export function predictLogits(
  model: LoadedModel,
  board: Board,
  hold: PieceKind | null,
): Float32Array {
  const perfEnabled = hasPerfMetricsSink();
  const predictStartMs = perfEnabled ? performance.now() : 0;
  const { config } = model;
  const rows = board.length;
  const cols = board[0]?.length ?? 0;
  const encodeStartMs = perfEnabled ? performance.now() : 0;
  let input = buildModelInputChannels(board, model.boardChannels);
  const expectedLength = config.input_channels * rows * cols;
  if (input.length !== expectedLength) {
    const padded = new Float32Array(expectedLength);
    padded.set(input.subarray(0, expectedLength), 0);
    input = padded;
  }
  if (perfEnabled) {
    const nowMs = performance.now();
    recordPerfDuration(
      'ml.model.input_encode_ms',
      nowMs - encodeStartMs,
      nowMs,
    );
  }

  let current = input;
  let inChannels = config.input_channels;
  const height = rows;
  const width = cols;
  const poolShape = getModelPoolShape(config.pool_shape);
  const computeStartMs = perfEnabled ? performance.now() : 0;
  const convStartMs = perfEnabled ? performance.now() : 0;

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
  if (perfEnabled) {
    const nowMs = performance.now();
    recordPerfDuration('ml.model.conv_stack_ms', nowMs - convStartMs, nowMs);
  }

  const poolStartMs = perfEnabled ? performance.now() : 0;
  const pooled = adaptiveAveragePool(
    current,
    inChannels,
    height,
    width,
    poolShape[0],
    poolShape[1],
  );
  if (perfEnabled) {
    const nowMs = performance.now();
    recordPerfDuration('ml.model.pool_ms', nowMs - poolStartMs, nowMs);
  }

  const headStartMs = perfEnabled ? performance.now() : 0;
  const extra = buildModelExtraFeatures(
    config.extra_features,
    hold,
    model.pieces,
  );

  let mlpInput = concatFeatures(pooled, extra);
  if (config.feature_norm === 'layernorm') {
    mlpInput = layerNorm1d(mlpInput, config.feature_norm_eps ?? 1e-5);
  }
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
  if (perfEnabled) {
    const nowMs = performance.now();
    recordPerfDuration('ml.model.head_ms', nowMs - headStartMs, nowMs);
    recordPerfDuration('ml.model.compute_ms', nowMs - computeStartMs, nowMs);
    recordPerfDuration('ml.model.total_ms', nowMs - predictStartMs, nowMs);
  }
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

export function buildModelInputChannels(
  board: Board,
  channels: string[],
): Float32Array {
  const rows = board.length;
  const cols = board[0]?.length ?? 0;
  const size = rows * cols;
  const occupancy = new Float32Array(rows * cols);
  const rowFill = new Float32Array(size);
  const coordX = new Float32Array(size);
  const coordY = new Float32Array(size);
  const colHeight = new Float32Array(size);
  const wellDepth = new Float32Array(size);
  const reachableEmpty = new Float32Array(size);
  const coarseOccV2 = new Float32Array(size);

  const denomX = Math.max(1, cols - 1);
  const denomY = Math.max(1, rows - 1);

  for (let y = 0; y < rows; y++) {
    let rowCount = 0;
    for (let x = 0; x < cols; x++) {
      const idx = y * cols + x;
      const filled = board[y][x] != null ? 1 : 0;
      occupancy[idx] = filled;
      coordX[idx] = x / denomX;
      coordY[idx] = y / denomY;
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

  for (let x = 0; x < cols; x++) {
    let firstFilled = -1;
    for (let y = 0; y < rows; y++) {
      if (occupancy[y * cols + x] > 0) {
        firstFilled = y;
        break;
      }
    }
    const value =
      firstFilled < 0 ? 0 : (rows - firstFilled) / Math.max(1, rows);
    for (let y = 0; y < rows; y++) {
      colHeight[y * cols + x] = value;
    }
  }

  for (let x = 0; x < cols; x++) {
    let depth = 0;
    for (let y = 0; y < rows; y++) {
      const idx = y * cols + x;
      if (occupancy[idx] > 0) {
        depth = 0;
        continue;
      }
      const leftBlocked = x === 0 || occupancy[idx - 1] > 0;
      const rightBlocked = x === cols - 1 || occupancy[idx + 1] > 0;
      if (leftBlocked && rightBlocked) {
        depth += 1;
        wellDepth[idx] = depth / Math.max(1, rows);
      } else {
        depth = 0;
      }
    }
  }

  const queueY = new Int16Array(size);
  const queueX = new Int16Array(size);
  let head = 0;
  let tail = 0;
  const visited = new Uint8Array(size);
  for (let x = 0; x < cols; x++) {
    const idx = x;
    if (occupancy[idx] > 0) continue;
    visited[idx] = 1;
    queueY[tail] = 0;
    queueX[tail] = x;
    tail += 1;
  }
  while (head < tail) {
    const y = queueY[head];
    const x = queueX[head];
    head += 1;
    const idx = y * cols + x;
    reachableEmpty[idx] = 1;
    const neighbors: [number, number][] = [
      [y - 1, x],
      [y + 1, x],
      [y, x - 1],
      [y, x + 1],
    ];
    for (const [ny, nx] of neighbors) {
      if (ny < 0 || ny >= rows || nx < 0 || nx >= cols) continue;
      const nIdx = ny * cols + nx;
      if (visited[nIdx] > 0) continue;
      if (occupancy[nIdx] > 0) continue;
      visited[nIdx] = 1;
      queueY[tail] = ny;
      queueX[tail] = nx;
      tail += 1;
    }
  }

  for (let y0 = 0; y0 < rows; y0 += 2) {
    const y1 = Math.min(rows, y0 + 2);
    for (let x = 0; x < cols; x++) {
      let value = 0;
      for (let y = y0; y < y1; y++) {
        value = Math.max(value, occupancy[y * cols + x]);
      }
      for (let y = y0; y < y1; y++) {
        coarseOccV2[y * cols + x] = value;
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
      case 'coord_x':
        source = coordX;
        break;
      case 'coord_y':
        source = coordY;
        break;
      case 'col_height':
        source = colHeight;
        break;
      case 'well_depth':
        source = wellDepth;
        break;
      case 'reachable_empty':
        source = reachableEmpty;
        break;
      case 'coarse_occ_v2':
        source = coarseOccV2;
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

export function buildModelExtraFeatures(
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

export function getModelPoolShape(
  value: number[] | undefined,
): [number, number] {
  if (!Array.isArray(value) || value.length !== 2) {
    return [1, 1];
  }
  const h = Math.max(1, Math.trunc(Number(value[0]) || 1));
  const w = Math.max(1, Math.trunc(Number(value[1]) || 1));
  return [h, w];
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

function adaptiveAveragePool(
  input: Float32Array,
  channels: number,
  height: number,
  width: number,
  outHeight: number,
  outWidth: number,
): Float32Array {
  const out = new Float32Array(channels * outHeight * outWidth);
  for (let c = 0; c < channels; c++) {
    const channelOffset = c * height * width;
    for (let oy = 0; oy < outHeight; oy++) {
      const yStart = Math.floor((oy * height) / outHeight);
      const yEnd = Math.ceil(((oy + 1) * height) / outHeight);
      for (let ox = 0; ox < outWidth; ox++) {
        const xStart = Math.floor((ox * width) / outWidth);
        const xEnd = Math.ceil(((ox + 1) * width) / outWidth);
        let sum = 0;
        let count = 0;
        for (let iy = yStart; iy < yEnd; iy++) {
          for (let ix = xStart; ix < xEnd; ix++) {
            sum += input[channelOffset + iy * width + ix];
            count += 1;
          }
        }
        const outIdx = (c * outHeight + oy) * outWidth + ox;
        out[outIdx] = count > 0 ? sum / count : 0;
      }
    }
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

function layerNorm1d(values: Float32Array, eps = 1e-5): Float32Array {
  const out = new Float32Array(values.length);
  if (values.length === 0) return out;
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
  }
  const mean = sum / values.length;
  let varSum = 0;
  for (let i = 0; i < values.length; i++) {
    const diff = values[i] - mean;
    varSum += diff * diff;
  }
  const variance = varSum / values.length;
  const denom = Math.sqrt(variance + Math.max(1e-12, eps));
  for (let i = 0; i < values.length; i++) {
    out[i] = (values[i] - mean) / denom;
  }
  return out;
}
