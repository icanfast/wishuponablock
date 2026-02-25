import type { Board, PieceKind } from './types';
import {
  buildModelExtraFeatures,
  buildModelInputChannels,
  getModelPoolShape,
  predictLogits as predictLogitsNative,
  type LoadedModel,
  type ModelTensor,
} from './wubModel';
import type {
  MlBackend,
  ModelRunner,
  ModelRunnerInfo,
  TfjsBackendPreference,
} from './modelRunner';
import { hasPerfMetricsSink, recordPerfDuration } from './perfMetrics';

type TensorData = Float32Array | Int32Array | Uint8Array;

type TfTensor = {
  dataSync: () => TensorData;
  dispose: () => void;
  reshape: (shape: number[]) => TfTensor;
};

type TfjsModule = {
  ready: () => Promise<void>;
  getBackend: () => string;
  setBackend: (backend: string) => Promise<boolean>;
  tensor1d: (values: Float32Array | Int32Array, dtype?: string) => TfTensor;
  tensor2d: (values: Float32Array, shape: [number, number]) => TfTensor;
  tensor4d: (
    values: Float32Array,
    shape: [number, number, number, number],
  ) => TfTensor;
  scalar: (value: number) => TfTensor;
  conv2d: (
    x: TfTensor,
    filter: TfTensor,
    strides: number | [number, number],
    pad: 'same' | 'valid',
  ) => TfTensor;
  add: (a: TfTensor, b: TfTensor) => TfTensor;
  relu: (x: TfTensor) => TfTensor;
  avgPool: (
    x: TfTensor,
    filterSize: number | [number, number],
    strides: number | [number, number],
    pad: 'same' | 'valid',
  ) => TfTensor;
  transpose: (x: TfTensor, perm?: number[]) => TfTensor;
  concat: (tensors: TfTensor[], axis: number) => TfTensor;
  matMul: (a: TfTensor, b: TfTensor) => TfTensor;
  mean: (x: TfTensor, axis: number | number[], keepDims?: boolean) => TfTensor;
  square: (x: TfTensor) => TfTensor;
  sub: (a: TfTensor, b: TfTensor) => TfTensor;
  div: (a: TfTensor, b: TfTensor) => TfTensor;
  sqrt: (x: TfTensor) => TfTensor;
  tidy: <T>(fn: () => T) => T;
  dispose: (tensor: TfTensor | TfTensor[] | null | undefined) => void;
};

type PreparedWeights = {
  model: LoadedModel;
  convKernels: TfTensor[];
  convBiases: TfTensor[];
  mlp0KernelT: TfTensor;
  mlp0Bias: TfTensor;
  mlp2KernelT: TfTensor;
  mlp2Bias: TfTensor;
  poolShape: [number, number];
  pooledFeatureCount: number;
};

const TFJS_CDN_URL = 'https://esm.sh/@tensorflow/tfjs@4.22.0';
let tfjsModulePromise: Promise<TfjsModule> | null = null;

const describeError = (error: unknown): string =>
  error instanceof Error ? error.message : String(error);

const loadTfjsModule = async (): Promise<TfjsModule> => {
  if (!tfjsModulePromise) {
    tfjsModulePromise = import(/* @vite-ignore */ TFJS_CDN_URL)
      .then((mod) => {
        const candidate = (mod.default ?? mod) as unknown;
        if (!candidate || typeof candidate !== 'object') {
          throw new Error('TFJS module did not export an object.');
        }
        const tf = candidate as Partial<TfjsModule>;
        if (
          typeof tf.tidy !== 'function' ||
          typeof tf.tensor4d !== 'function'
        ) {
          throw new Error('TFJS module is missing required APIs.');
        }
        return tf as TfjsModule;
      })
      .catch((error) => {
        tfjsModulePromise = null;
        throw error;
      });
  }
  return tfjsModulePromise;
};

const selectTfjsBackend = async (
  tf: TfjsModule,
  preferredBackend: TfjsBackendPreference,
): Promise<string> => {
  const candidates =
    preferredBackend === 'webgl'
      ? ['webgl']
      : preferredBackend === 'cpu'
        ? ['cpu']
        : ['webgl', 'cpu'];
  for (const backend of candidates) {
    try {
      const changed = await tf.setBackend(backend);
      if (changed) {
        await tf.ready();
        return tf.getBackend();
      }
    } catch {
      // Ignore backend registration failures and continue probing.
    }
  }
  await tf.ready();
  return tf.getBackend();
};

const requireParam = (model: LoadedModel, name: string): ModelTensor => {
  const tensor = model.params[name];
  if (!tensor) {
    throw new Error(`Missing model param: ${name}`);
  }
  return tensor;
};

const expectShape = (
  tensor: ModelTensor,
  expectedLength: number,
  paramName: string,
): void => {
  if (tensor.shape.length !== expectedLength) {
    throw new Error(
      `Invalid shape rank for ${paramName}: expected rank ${expectedLength}, got ${tensor.shape.length}.`,
    );
  }
};

const convertConvKernelOichwToHwio = (
  tensor: ModelTensor,
  outChannels: number,
  inChannels: number,
): Float32Array => {
  expectShape(tensor, 4, 'conv weight');
  const [shapeOut, shapeIn, kH, kW] = tensor.shape;
  if (
    shapeOut !== outChannels ||
    shapeIn !== inChannels ||
    kH !== 3 ||
    kW !== 3
  ) {
    throw new Error(
      `Unexpected conv weight shape ${tensor.shape.join(
        'x',
      )}; expected ${outChannels}x${inChannels}x3x3.`,
    );
  }
  const out = new Float32Array(kH * kW * inChannels * outChannels);
  for (let oc = 0; oc < outChannels; oc++) {
    for (let ic = 0; ic < inChannels; ic++) {
      for (let ky = 0; ky < kH; ky++) {
        for (let kx = 0; kx < kW; kx++) {
          const srcIdx = ((oc * inChannels + ic) * kH + ky) * kW + kx;
          const dstIdx = ((ky * kW + kx) * inChannels + ic) * outChannels + oc;
          out[dstIdx] = tensor.data[srcIdx];
        }
      }
    }
  }
  return out;
};

const transposeLinearWeightOutInToInOut = (
  tensor: ModelTensor,
  outFeatures: number,
  inFeatures: number,
  paramName: string,
): Float32Array => {
  expectShape(tensor, 2, paramName);
  const [shapeOut, shapeIn] = tensor.shape;
  if (shapeOut !== outFeatures || shapeIn !== inFeatures) {
    throw new Error(
      `Unexpected ${paramName} shape ${tensor.shape.join(
        'x',
      )}; expected ${outFeatures}x${inFeatures}.`,
    );
  }
  const out = new Float32Array(inFeatures * outFeatures);
  for (let o = 0; o < outFeatures; o++) {
    for (let i = 0; i < inFeatures; i++) {
      out[i * outFeatures + o] = tensor.data[o * inFeatures + i];
    }
  }
  return out;
};

export const reorderInputChwToNhwc = (
  input: Float32Array,
  channels: number,
  rows: number,
  cols: number,
): Float32Array => {
  const out = new Float32Array(channels * rows * cols);
  let dst = 0;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      for (let c = 0; c < channels; c++) {
        const src = (c * rows + y) * cols + x;
        out[dst++] = input[src];
      }
    }
  }
  return out;
};

const disposePreparedWeights = (
  tf: TfjsModule | null,
  state: PreparedWeights | null,
): void => {
  if (!tf || !state) return;
  tf.dispose(state.convKernels);
  tf.dispose(state.convBiases);
  tf.dispose(state.mlp0KernelT);
  tf.dispose(state.mlp0Bias);
  tf.dispose(state.mlp2KernelT);
  tf.dispose(state.mlp2Bias);
};

const createInfo = (requestedBackend: MlBackend): ModelRunnerInfo => ({
  requestedBackend,
  activeBackend: requestedBackend,
  runtimeBackend: null,
  fallbackReason: null,
});

export function createTfjsModelRunner(
  requestedBackend: MlBackend,
  tfjsBackendPreference: TfjsBackendPreference,
): ModelRunner {
  const info = createInfo(requestedBackend);
  let tf: TfjsModule | null = null;
  let prepared: PreparedWeights | null = null;

  const setFallback = (reason: string) => {
    if (info.activeBackend !== 'native' || info.fallbackReason !== reason) {
      console.warn(
        `[ML] TFJS backend disabled; using native backend: ${reason}`,
      );
    }
    info.activeBackend = 'native';
    info.runtimeBackend = null;
    info.fallbackReason = reason;
    disposePreparedWeights(tf, prepared);
    prepared = null;
  };

  const prepare = async (model: LoadedModel): Promise<void> => {
    if (prepared?.model === model && info.activeBackend === 'tfjs') {
      return;
    }
    if (typeof window === 'undefined') {
      setFallback('tfjs backend is only available in browser runtime.');
      return;
    }
    try {
      if (!tf) {
        tf = await loadTfjsModule();
      }
      const backend = await selectTfjsBackend(tf, tfjsBackendPreference);

      const convKernels: TfTensor[] = [];
      const convBiases: TfTensor[] = [];
      let inChannels = model.config.input_channels;
      for (let i = 0; i < model.config.conv_channels.length; i++) {
        const outChannels = model.config.conv_channels[i];
        const layerIndex = i * 2;
        const weight = requireParam(model, `conv.${layerIndex}.weight`);
        const bias = requireParam(model, `conv.${layerIndex}.bias`);
        if (bias.shape.length !== 1 || bias.shape[0] !== outChannels) {
          throw new Error(
            `Unexpected conv.${layerIndex}.bias shape ${bias.shape.join(
              'x',
            )}; expected ${outChannels}.`,
          );
        }
        const kernel = convertConvKernelOichwToHwio(
          weight,
          outChannels,
          inChannels,
        );
        convKernels.push(tf.tensor4d(kernel, [3, 3, inChannels, outChannels]));
        convBiases.push(tf.tensor1d(bias.data));
        inChannels = outChannels;
      }

      const poolShape = getModelPoolShape(model.config.pool_shape);
      const pooledFeatureCount = inChannels * poolShape[0] * poolShape[1];
      const mlpInputFeatures =
        pooledFeatureCount + Math.max(0, model.config.extra_features);
      const mlpHiddenFeatures = model.config.mlp_hidden;
      const outputFeatures = model.config.num_outputs;

      const mlp0Weight = requireParam(model, 'mlp.0.weight');
      const mlp0Bias = requireParam(model, 'mlp.0.bias');
      const mlp2Weight = requireParam(model, 'mlp.2.weight');
      const mlp2Bias = requireParam(model, 'mlp.2.bias');

      if (
        mlp0Bias.shape.length !== 1 ||
        mlp0Bias.shape[0] !== mlpHiddenFeatures
      ) {
        throw new Error(
          `Unexpected mlp.0.bias shape ${mlp0Bias.shape.join(
            'x',
          )}; expected ${mlpHiddenFeatures}.`,
        );
      }
      if (mlp2Bias.shape.length !== 1 || mlp2Bias.shape[0] !== outputFeatures) {
        throw new Error(
          `Unexpected mlp.2.bias shape ${mlp2Bias.shape.join(
            'x',
          )}; expected ${outputFeatures}.`,
        );
      }

      const mlp0KernelT = transposeLinearWeightOutInToInOut(
        mlp0Weight,
        mlpHiddenFeatures,
        mlpInputFeatures,
        'mlp.0.weight',
      );
      const mlp2KernelT = transposeLinearWeightOutInToInOut(
        mlp2Weight,
        outputFeatures,
        mlpHiddenFeatures,
        'mlp.2.weight',
      );

      const nextPrepared: PreparedWeights = {
        model,
        convKernels,
        convBiases,
        mlp0KernelT: tf.tensor2d(mlp0KernelT, [
          mlpInputFeatures,
          mlpHiddenFeatures,
        ]),
        mlp0Bias: tf.tensor1d(mlp0Bias.data),
        mlp2KernelT: tf.tensor2d(mlp2KernelT, [
          mlpHiddenFeatures,
          outputFeatures,
        ]),
        mlp2Bias: tf.tensor1d(mlp2Bias.data),
        poolShape,
        pooledFeatureCount,
      };

      disposePreparedWeights(tf, prepared);
      prepared = nextPrepared;
      info.activeBackend = 'tfjs';
      info.runtimeBackend = backend;
      info.fallbackReason = null;
      console.info(
        `[ML] tfjs backend ready (backend=${backend}, preference=${tfjsBackendPreference})`,
      );
    } catch (error) {
      setFallback(describeError(error));
    }
  };

  const predictWithTfjs = (
    model: LoadedModel,
    board: Board,
    hold: PieceKind | null,
  ): Float32Array => {
    const tfModule = tf;
    const preparedState = prepared;
    if (!tfModule || !preparedState || preparedState.model !== model) {
      throw new Error('TFJS runner was not prepared for this model.');
    }
    const rows = board.length;
    const cols = board[0]?.length ?? 0;
    const expectedLength = model.config.input_channels * rows * cols;
    const poolShape = preparedState.poolShape;
    if (rows % poolShape[0] !== 0 || cols % poolShape[1] !== 0) {
      throw new Error(
        `Board shape ${rows}x${cols} is incompatible with pool_shape ${poolShape[0]}x${poolShape[1]} for tfjs avgPool path.`,
      );
    }

    const perfEnabled = hasPerfMetricsSink();
    const predictStartMs = perfEnabled ? performance.now() : 0;

    const encodeStartMs = perfEnabled ? performance.now() : 0;
    let input = buildModelInputChannels(board, model.boardChannels);
    if (input.length !== expectedLength) {
      const padded = new Float32Array(expectedLength);
      padded.set(input.subarray(0, expectedLength), 0);
      input = padded;
    }
    const nhwcInput = reorderInputChwToNhwc(
      input,
      model.config.input_channels,
      rows,
      cols,
    );
    const extra = buildModelExtraFeatures(
      model.config.extra_features,
      hold,
      model.pieces,
    );
    if (perfEnabled) {
      const nowMs = performance.now();
      recordPerfDuration(
        'ml.model.input_encode_ms',
        nowMs - encodeStartMs,
        nowMs,
      );
    }

    const convStartMs = perfEnabled ? performance.now() : 0;
    const convOutput = tfModule.tidy(() => {
      let x = tfModule.tensor4d(nhwcInput, [
        1,
        rows,
        cols,
        model.config.input_channels,
      ]);
      for (let i = 0; i < preparedState.convKernels.length; i++) {
        const conv = tfModule.conv2d(
          x,
          preparedState.convKernels[i],
          1,
          'same',
        );
        const biased = tfModule.add(conv, preparedState.convBiases[i]);
        x = tfModule.relu(biased);
      }
      return x;
    });
    if (perfEnabled) {
      const nowMs = performance.now();
      recordPerfDuration('ml.model.conv_stack_ms', nowMs - convStartMs, nowMs);
    }

    const poolStartMs = perfEnabled ? performance.now() : 0;
    const poolKernel: [number, number] = [
      rows / poolShape[0],
      cols / poolShape[1],
    ];
    const pooledFlat = tfModule.tidy(() => {
      const pooled = tfModule.avgPool(
        convOutput,
        poolKernel,
        poolKernel,
        'valid',
      );
      const pooledNchw = tfModule.transpose(pooled, [0, 3, 1, 2]);
      return pooledNchw.reshape([1, preparedState.pooledFeatureCount]);
    });
    tfModule.dispose(convOutput);
    if (perfEnabled) {
      const nowMs = performance.now();
      recordPerfDuration('ml.model.pool_ms', nowMs - poolStartMs, nowMs);
    }

    const headStartMs = perfEnabled ? performance.now() : 0;
    const logits = tfModule.tidy(() => {
      let mlpInput = pooledFlat;
      if (extra && extra.length > 0) {
        const extraTensor = tfModule.tensor2d(extra, [1, extra.length]);
        mlpInput = tfModule.concat([pooledFlat, extraTensor], 1);
      }
      if (model.config.feature_norm === 'layernorm') {
        const mean = tfModule.mean(mlpInput, 1, true);
        const centered = tfModule.sub(mlpInput, mean);
        const variance = tfModule.mean(tfModule.square(centered), 1, true);
        const eps = tfModule.scalar(
          Math.max(1e-12, model.config.feature_norm_eps ?? 1e-5),
        );
        const denom = tfModule.sqrt(tfModule.add(variance, eps));
        mlpInput = tfModule.div(centered, denom);
      }
      const hidden = tfModule.relu(
        tfModule.add(
          tfModule.matMul(mlpInput, preparedState.mlp0KernelT),
          preparedState.mlp0Bias,
        ),
      );
      const logitsTensor = tfModule.add(
        tfModule.matMul(hidden, preparedState.mlp2KernelT),
        preparedState.mlp2Bias,
      );
      return Float32Array.from(logitsTensor.dataSync());
    });
    tfModule.dispose(pooledFlat);

    if (perfEnabled) {
      const nowMs = performance.now();
      recordPerfDuration('ml.model.head_ms', nowMs - headStartMs, nowMs);
      recordPerfDuration('ml.model.total_ms', nowMs - predictStartMs, nowMs);
    }
    return logits;
  };

  return {
    prepare,
    getInfo: () => ({ ...info }),
    predictLogits: (model, board, hold) => {
      if (info.activeBackend === 'tfjs') {
        try {
          return predictWithTfjs(model, board, hold);
        } catch (error) {
          setFallback(describeError(error));
        }
      }
      return predictLogitsNative(model, board, hold);
    },
  };
}
