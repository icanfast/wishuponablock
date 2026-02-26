import type { LoadedModel } from '../core/wubModel';
import { serializeWubModelToBytes } from '../core/wubModel';
import type { TrajectoryDecisionSample } from './trajectoryBuffer';

const TFJS_CDN_URL = 'https://esm.sh/@tensorflow/tfjs@4.22.0';

type TfjsBackendPreference = 'auto' | 'webgl' | 'cpu';

type TrainOptions = {
  epochs?: number;
  learningRate?: number;
  l2?: number;
  sampleLimit?: number;
  backendPreference?: TfjsBackendPreference;
};

export type PersonalTrainerResult = {
  ok: boolean;
  message: string;
  backend: string | null;
  samplesUsed: number;
  epochs: number;
  finalLoss: number | null;
  updatedModelBytes: ArrayBuffer | null;
};

export type PersonalTrainer = {
  trainBiasOnly: (options: {
    model: LoadedModel;
    samples: TrajectoryDecisionSample[];
    train?: TrainOptions;
  }) => Promise<PersonalTrainerResult>;
};

type TfTensor = {
  dataSync: () => Float32Array | Int32Array | Uint8Array;
  dispose: () => void;
};

type TfTrainOptimizer = {
  minimize: (
    fn: () => TfTensor,
    returnCost?: boolean,
    varList?: TfTensor[],
  ) => TfTensor | null;
};

type TfjsModule = {
  ready: () => Promise<void>;
  getBackend: () => string;
  setBackend: (backend: string) => Promise<boolean>;
  tensor1d: (values: Float32Array | Int32Array, dtype?: string) => TfTensor;
  tensor2d: (
    values: Float32Array,
    shape: [number, number],
    dtype?: string,
  ) => TfTensor;
  scalar: (value: number, dtype?: string) => TfTensor;
  variable: (initialValue: TfTensor) => TfTensor;
  oneHot: (indices: TfTensor, depth: number) => TfTensor;
  sum: (x: TfTensor, axis?: number | number[], keepDims?: boolean) => TfTensor;
  mean: (x: TfTensor, axis?: number | number[], keepDims?: boolean) => TfTensor;
  add: (a: TfTensor, b: TfTensor) => TfTensor;
  mul: (a: TfTensor, b: TfTensor) => TfTensor;
  neg: (x: TfTensor) => TfTensor;
  square: (x: TfTensor) => TfTensor;
  logSoftmax: (x: TfTensor, axis?: number) => TfTensor;
  tidy: <T>(fn: () => T) => T;
  dispose: (value: unknown) => void;
  train: {
    adam: (learningRate: number) => TfTrainOptimizer;
  };
};

let tfPromise: Promise<TfjsModule> | null = null;

const loadTf = async (): Promise<TfjsModule> => {
  if (!tfPromise) {
    tfPromise = import(/* @vite-ignore */ TFJS_CDN_URL)
      .then((mod) => {
        const tf = (mod.default ?? mod) as unknown;
        if (!tf || typeof tf !== 'object') {
          throw new Error('TFJS module did not export an object.');
        }
        return tf as TfjsModule;
      })
      .catch((error) => {
        tfPromise = null;
        throw error;
      });
  }
  return tfPromise;
};

const selectBackend = async (
  tf: TfjsModule,
  preference: TfjsBackendPreference,
): Promise<string> => {
  const candidates =
    preference === 'webgl'
      ? ['webgl']
      : preference === 'cpu'
        ? ['cpu']
        : ['webgl', 'cpu'];
  for (const candidate of candidates) {
    try {
      const changed = await tf.setBackend(candidate);
      if (changed) {
        await tf.ready();
        return tf.getBackend();
      }
    } catch {
      // keep probing
    }
  }
  await tf.ready();
  return tf.getBackend();
};

const clampInt = (value: number | undefined, fallback: number): number => {
  if (value == null || !Number.isFinite(value)) return fallback;
  return Math.max(1, Math.trunc(value));
};

const clampFloat = (
  value: number | undefined,
  fallback: number,
  min: number,
): number => {
  if (value == null || !Number.isFinite(value)) return fallback;
  return Math.max(min, value);
};

const buildBiasAdjustedModelBytes = (
  baseModel: LoadedModel,
  biasDelta: Float32Array,
): ArrayBuffer => {
  const updated: LoadedModel = {
    config: {
      ...baseModel.config,
      conv_channels: [...baseModel.config.conv_channels],
      pool_shape: baseModel.config.pool_shape
        ? [...baseModel.config.pool_shape]
        : undefined,
    },
    pieces: [...baseModel.pieces],
    boardChannels: [...baseModel.boardChannels],
    params: {},
  };

  for (const [name, tensor] of Object.entries(baseModel.params)) {
    updated.params[name] = {
      shape: [...tensor.shape],
      data: new Float32Array(tensor.data),
    };
  }

  const outputBias = updated.params['mlp.2.bias'];
  if (!outputBias) {
    throw new Error('Model does not have mlp.2.bias.');
  }
  if (outputBias.data.length !== biasDelta.length) {
    throw new Error('Bias delta length does not match model outputs.');
  }
  for (let i = 0; i < biasDelta.length; i++) {
    outputBias.data[i] += biasDelta[i];
  }
  return serializeWubModelToBytes(updated);
};

export function createPersonalTrainerTfjs(): PersonalTrainer {
  return {
    trainBiasOnly: async ({ model, samples, train }) => {
      const numOutputs = model.config.num_outputs;
      const sampleLimit = clampInt(train?.sampleLimit, 512);
      const epochs = clampInt(train?.epochs, 8);
      const learningRate = clampFloat(train?.learningRate, 0.01, 1e-6);
      const l2 = clampFloat(train?.l2, 1e-4, 0);
      const backendPreference = train?.backendPreference ?? 'auto';

      const eligible = samples
        .filter(
          (sample) =>
            sample.logits.length === numOutputs &&
            sample.actionIndex >= 0 &&
            sample.actionIndex < numOutputs,
        )
        .slice(-sampleLimit);
      if (eligible.length < 8) {
        return {
          ok: false,
          message: 'Not enough trajectory samples for training (need >= 8).',
          backend: null,
          samplesUsed: eligible.length,
          epochs: 0,
          finalLoss: null,
          updatedModelBytes: null,
        };
      }

      let tf: TfjsModule | null = null;
      let backend: string | null = null;
      let logitsTensor: TfTensor | null = null;
      let actionTensor: TfTensor | null = null;
      let oneHotTensor: TfTensor | null = null;
      let advantageTensor: TfTensor | null = null;
      let biasDelta: TfTensor | null = null;
      let finalLoss: number | null = null;
      try {
        tf = await loadTf();
        backend = await selectBackend(tf, backendPreference);

        const logitsFlat = new Float32Array(eligible.length * numOutputs);
        const actions = new Int32Array(eligible.length);
        const advantages = new Float32Array(eligible.length);

        for (let i = 0; i < eligible.length; i++) {
          const sample = eligible[i];
          actions[i] = sample.actionIndex;
          const deliberationPenalty =
            sample.deliberationMs != null &&
            Number.isFinite(sample.deliberationMs)
              ? Math.max(0.1, Math.min(5, sample.deliberationMs / 1000))
              : 1;
          advantages[i] = 1 / deliberationPenalty;
          for (let j = 0; j < numOutputs; j++) {
            logitsFlat[i * numOutputs + j] = sample.logits[j];
          }
        }

        logitsTensor = tf.tensor2d(logitsFlat, [eligible.length, numOutputs]);
        actionTensor = tf.tensor1d(actions, 'int32');
        oneHotTensor = tf.oneHot(actionTensor, numOutputs);
        advantageTensor = tf.tensor1d(advantages);
        biasDelta = tf.variable(tf.tensor1d(new Float32Array(numOutputs)));
        const optimizer = tf.train.adam(learningRate);

        for (let epoch = 0; epoch < epochs; epoch++) {
          const lossTensor = optimizer.minimize(
            () =>
              tf!.tidy(() => {
                const shifted = tf!.add(logitsTensor!, biasDelta!);
                const logProbs = tf!.logSoftmax(shifted, 1);
                const selectedLogProb = tf!.sum(
                  tf!.mul(logProbs, oneHotTensor!),
                  1,
                );
                const weighted = tf!.mul(selectedLogProb, advantageTensor!);
                const policyLoss = tf!.neg(tf!.mean(weighted));
                const reg = tf!.mul(
                  tf!.mean(tf!.square(biasDelta!)),
                  tf!.scalar(l2),
                );
                return tf!.add(policyLoss, reg);
              }),
            true,
            [biasDelta],
          );
          if (!lossTensor) {
            throw new Error('Optimizer returned null loss tensor.');
          }
          const lossData = lossTensor.dataSync();
          finalLoss = Number(lossData[0]);
          lossTensor.dispose();
        }

        const biasDeltaValues = biasDelta.dataSync();
        const biasDeltaArray = new Float32Array(biasDeltaValues.length);
        biasDeltaArray.set(biasDeltaValues);
        const updatedModelBytes = buildBiasAdjustedModelBytes(
          model,
          biasDeltaArray,
        );
        return {
          ok: true,
          message: `Bias-only training complete (backend=${backend}).`,
          backend,
          samplesUsed: eligible.length,
          epochs,
          finalLoss,
          updatedModelBytes,
        };
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        return {
          ok: false,
          message: `Training failed: ${detail}`,
          backend,
          samplesUsed: eligible.length,
          epochs: 0,
          finalLoss: null,
          updatedModelBytes: null,
        };
      } finally {
        if (tf) {
          tf.dispose([
            logitsTensor,
            actionTensor,
            oneHotTensor,
            advantageTensor,
            biasDelta,
          ]);
        }
      }
    },
  };
}
