import type { Board, PieceKind } from '../core/types';
import {
  buildModelHeadInput,
  serializeWubModelToBytes,
  type LoadedModel,
} from '../core/wubModel';
import type { ResolvedPersonalTrainingPipeline } from './trainingPipelines';
import type { TrajectoryDecisionSample } from './trajectoryBuffer';

const TFJS_CDN_URL = 'https://esm.sh/@tensorflow/tfjs@4.22.0';
const OCCUPIED_CELL_PIECE: PieceKind = 'I';

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
  pipelineId: string;
  backend: string | null;
  samplesUsed: number;
  epochs: number;
  finalLoss: number | null;
  updatedModelBytes: ArrayBuffer | null;
};

export type PersonalTrainer = {
  trainHeadOnly: (options: {
    model: LoadedModel;
    samples: TrajectoryDecisionSample[];
    pipeline: ResolvedPersonalTrainingPipeline;
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
  sub: (a: TfTensor, b: TfTensor) => TfTensor;
  mul: (a: TfTensor, b: TfTensor) => TfTensor;
  neg: (x: TfTensor) => TfTensor;
  square: (x: TfTensor) => TfTensor;
  logSoftmax: (x: TfTensor, axis?: number) => TfTensor;
  matMul: (
    a: TfTensor,
    b: TfTensor,
    transposeA?: boolean,
    transposeB?: boolean,
  ) => TfTensor;
  relu: (x: TfTensor) => TfTensor;
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

const fallbackDeliberationAdvantage = (
  sample: TrajectoryDecisionSample,
): number => {
  const deliberationPenalty =
    sample.deliberationMs != null && Number.isFinite(sample.deliberationMs)
      ? Math.max(0.1, Math.min(5, sample.deliberationMs / 1000))
      : 1;
  return 1 / deliberationPenalty;
};

const resolveSampleAdvantage = (sample: TrajectoryDecisionSample): number => {
  if (sample.reward != null && Number.isFinite(sample.reward)) {
    return sample.reward;
  }
  return fallbackDeliberationAdvantage(sample);
};

const normalizeInPlace = (values: Float32Array): void => {
  if (values.length <= 1) return;
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
  }
  const mean = sum / values.length;
  let variance = 0;
  for (let i = 0; i < values.length; i++) {
    const diff = values[i] - mean;
    variance += diff * diff;
  }
  variance /= values.length;
  const std = Math.sqrt(Math.max(1e-8, variance));
  for (let i = 0; i < values.length; i++) {
    values[i] = (values[i] - mean) / std;
  }
};

const toBoardFromOccupancy = (boardOccupancy: number[][]): Board =>
  boardOccupancy.map((row) =>
    row.map((cell) => (cell > 0 ? OCCUPIED_CELL_PIECE : null)),
  );

const resolveModelActionIndex = (
  model: LoadedModel,
  sample: TrajectoryDecisionSample,
): number | null => {
  const index = model.pieces.indexOf(sample.action);
  if (index < 0) return null;
  if (
    sample.actionIndex >= 0 &&
    sample.actionIndex < sample.pieces.length &&
    sample.pieces[sample.actionIndex] !== sample.action
  ) {
    return null;
  }
  return index;
};

const cloneModel = (baseModel: LoadedModel): LoadedModel => {
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
  return updated;
};

const getLinearParamShape = (
  model: LoadedModel,
  weightName: string,
  biasName: string,
): { inFeatures: number; outFeatures: number } => {
  const weight = model.params[weightName];
  const bias = model.params[biasName];
  if (!weight || !bias) {
    throw new Error(`Missing model params: ${weightName}/${biasName}.`);
  }
  const outFeatures = bias.data.length;
  if (outFeatures <= 0) {
    throw new Error(`Invalid bias size for ${biasName}.`);
  }
  const inFeatures = weight.data.length / outFeatures;
  if (!Number.isFinite(inFeatures) || Math.trunc(inFeatures) !== inFeatures) {
    throw new Error(`Invalid weight size for ${weightName}.`);
  }
  return { inFeatures: Math.trunc(inFeatures), outFeatures };
};

const overwriteLinearHeadParams = (options: {
  baseModel: LoadedModel;
  mlp0Weight: Float32Array;
  mlp0Bias: Float32Array;
  mlp2Weight: Float32Array;
  mlp2Bias: Float32Array;
}): ArrayBuffer => {
  const updated = cloneModel(options.baseModel);
  const assign = (name: string, source: Float32Array): void => {
    const param = updated.params[name];
    if (!param) throw new Error(`Missing model param: ${name}.`);
    if (param.data.length !== source.length) {
      throw new Error(`Unexpected trained tensor shape for ${name}.`);
    }
    param.data = new Float32Array(source);
  };
  assign('mlp.0.weight', options.mlp0Weight);
  assign('mlp.0.bias', options.mlp0Bias);
  assign('mlp.2.weight', options.mlp2Weight);
  assign('mlp.2.bias', options.mlp2Bias);
  return serializeWubModelToBytes(updated);
};

export function createPersonalTrainerTfjs(): PersonalTrainer {
  return {
    trainHeadOnly: async ({ model, samples, pipeline, train }) => {
      const sampleLimit = clampInt(
        train?.sampleLimit,
        pipeline.trainDefaults.sampleLimit,
      );
      const epochs = clampInt(train?.epochs, pipeline.trainDefaults.epochs);
      const learningRate = clampFloat(
        train?.learningRate,
        pipeline.trainDefaults.learningRate,
        1e-6,
      );
      const l2 = clampFloat(train?.l2, pipeline.trainDefaults.l2, 0);
      const backendPreference =
        train?.backendPreference ?? pipeline.trainDefaults.backendPreference;
      const minSamples = Math.max(1, Math.trunc(pipeline.minSamples));
      const holdoutRatio = Math.min(
        0.5,
        Math.max(0.05, pipeline.evalGate.holdoutRatio),
      );
      const minHoldoutSamples = Math.max(
        1,
        Math.trunc(pipeline.evalGate.minHoldoutSamples),
      );
      const minTrainSamples = Math.max(
        1,
        Math.trunc(pipeline.evalGate.minTrainSamples),
      );
      const minObjectiveGain = Number.isFinite(
        pipeline.evalGate.minObjectiveGain,
      )
        ? pipeline.evalGate.minObjectiveGain
        : 0;

      if (pipeline.strategyId !== 'mlp_head_policy_v1') {
        return {
          ok: false,
          message: `Unsupported training strategy: ${pipeline.strategyId}.`,
          pipelineId: pipeline.id,
          backend: null,
          samplesUsed: 0,
          epochs: 0,
          finalLoss: null,
          updatedModelBytes: null,
        };
      }

      let tf: TfjsModule | null = null;
      let backend: string | null = null;
      let headInputTensor: TfTensor | null = null;
      let actionTensor: TfTensor | null = null;
      let oneHotTensor: TfTensor | null = null;
      let advantageTensor: TfTensor | null = null;
      let holdoutInputTensor: TfTensor | null = null;
      let holdoutActionTensor: TfTensor | null = null;
      let holdoutOneHotTensor: TfTensor | null = null;
      let holdoutAdvantageTensor: TfTensor | null = null;
      let mlp0WeightBase: TfTensor | null = null;
      let mlp0BiasBase: TfTensor | null = null;
      let mlp2WeightBase: TfTensor | null = null;
      let mlp2BiasBase: TfTensor | null = null;
      let mlp0WeightVar: TfTensor | null = null;
      let mlp0BiasVar: TfTensor | null = null;
      let mlp2WeightVar: TfTensor | null = null;
      let mlp2BiasVar: TfTensor | null = null;
      let finalLoss: number | null = null;
      let eligibleCount = 0;
      try {
        const mlp0Shape = getLinearParamShape(
          model,
          'mlp.0.weight',
          'mlp.0.bias',
        );
        const mlp2Shape = getLinearParamShape(
          model,
          'mlp.2.weight',
          'mlp.2.bias',
        );
        if (mlp2Shape.inFeatures !== mlp0Shape.outFeatures) {
          throw new Error('MLP head shape mismatch (mlp.0 -> mlp.2).');
        }
        if (mlp2Shape.outFeatures !== model.config.num_outputs) {
          throw new Error('MLP output shape does not match model outputs.');
        }

        const eligible = samples.slice(-sampleLimit);
        const encodedExamples: Array<{
          actionIndex: number;
          advantage: number;
          headInput: Float32Array;
        }> = [];
        let rewardBackedCount = 0;

        for (const sample of eligible) {
          const actionIndex = resolveModelActionIndex(model, sample);
          if (actionIndex == null) continue;
          const board = toBoardFromOccupancy(sample.boardOccupancy);
          const headInput = buildModelHeadInput(model, board, sample.hold);
          if (headInput.length !== mlp0Shape.inFeatures) continue;
          const advantage = resolveSampleAdvantage(sample);
          if (sample.reward != null && Number.isFinite(sample.reward)) {
            rewardBackedCount += 1;
          }
          encodedExamples.push({ actionIndex, advantage, headInput });
        }

        eligibleCount = encodedExamples.length;
        if (eligibleCount < minSamples) {
          return {
            ok: false,
            message: `Not enough trajectory samples for training (need >= ${minSamples}).`,
            pipelineId: pipeline.id,
            backend: null,
            samplesUsed: eligibleCount,
            epochs: 0,
            finalLoss: null,
            updatedModelBytes: null,
          };
        }

        if (eligibleCount < minHoldoutSamples + minTrainSamples) {
          return {
            ok: false,
            message:
              `Need at least ${minHoldoutSamples + minTrainSamples} samples ` +
              `for eval gate (train >= ${minTrainSamples}, holdout >= ${minHoldoutSamples}).`,
            pipelineId: pipeline.id,
            backend: null,
            samplesUsed: eligibleCount,
            epochs: 0,
            finalLoss: null,
            updatedModelBytes: null,
          };
        }

        let holdoutCount = Math.max(
          minHoldoutSamples,
          Math.round(eligibleCount * holdoutRatio),
        );
        holdoutCount = Math.min(holdoutCount, eligibleCount - minTrainSamples);
        const trainCount = eligibleCount - holdoutCount;
        if (holdoutCount < minHoldoutSamples || trainCount < minTrainSamples) {
          return {
            ok: false,
            message: 'Could not derive a valid train/holdout split.',
            pipelineId: pipeline.id,
            backend: null,
            samplesUsed: eligibleCount,
            epochs: 0,
            finalLoss: null,
            updatedModelBytes: null,
          };
        }

        const trainExamples = encodedExamples.slice(0, trainCount);
        const holdoutExamples = encodedExamples.slice(trainCount);

        const headInputData = new Float32Array(
          trainCount * mlp0Shape.inFeatures,
        );
        const actionData = new Int32Array(trainCount);
        const advantageData = new Float32Array(trainCount);
        for (let i = 0; i < trainExamples.length; i += 1) {
          const sample = trainExamples[i];
          actionData[i] = sample.actionIndex;
          advantageData[i] = sample.advantage;
          headInputData.set(sample.headInput, i * mlp0Shape.inFeatures);
        }
        normalizeInPlace(advantageData);

        const holdoutInputData = new Float32Array(
          holdoutCount * mlp0Shape.inFeatures,
        );
        const holdoutActionData = new Int32Array(holdoutCount);
        const holdoutAdvantageData = new Float32Array(holdoutCount);
        for (let i = 0; i < holdoutExamples.length; i += 1) {
          const sample = holdoutExamples[i];
          holdoutActionData[i] = sample.actionIndex;
          holdoutAdvantageData[i] = sample.advantage;
          holdoutInputData.set(sample.headInput, i * mlp0Shape.inFeatures);
        }
        normalizeInPlace(holdoutAdvantageData);

        tf = await loadTf();
        backend = await selectBackend(tf, backendPreference);

        headInputTensor = tf.tensor2d(headInputData, [
          trainCount,
          mlp0Shape.inFeatures,
        ]);
        actionTensor = tf.tensor1d(actionData, 'int32');
        oneHotTensor = tf.oneHot(actionTensor, mlp2Shape.outFeatures);
        advantageTensor = tf.tensor1d(advantageData);
        holdoutInputTensor = tf.tensor2d(holdoutInputData, [
          holdoutCount,
          mlp0Shape.inFeatures,
        ]);
        holdoutActionTensor = tf.tensor1d(holdoutActionData, 'int32');
        holdoutOneHotTensor = tf.oneHot(
          holdoutActionTensor,
          mlp2Shape.outFeatures,
        );
        holdoutAdvantageTensor = tf.tensor1d(holdoutAdvantageData);

        const mlp0WeightData = new Float32Array(
          model.params['mlp.0.weight'].data,
        );
        const mlp0BiasData = new Float32Array(model.params['mlp.0.bias'].data);
        const mlp2WeightData = new Float32Array(
          model.params['mlp.2.weight'].data,
        );
        const mlp2BiasData = new Float32Array(model.params['mlp.2.bias'].data);

        mlp0WeightBase = tf.tensor2d(mlp0WeightData, [
          mlp0Shape.outFeatures,
          mlp0Shape.inFeatures,
        ]);
        mlp0BiasBase = tf.tensor1d(mlp0BiasData);
        mlp2WeightBase = tf.tensor2d(mlp2WeightData, [
          mlp2Shape.outFeatures,
          mlp2Shape.inFeatures,
        ]);
        mlp2BiasBase = tf.tensor1d(mlp2BiasData);

        mlp0WeightVar = tf.variable(
          tf.tensor2d(mlp0WeightData, [
            mlp0Shape.outFeatures,
            mlp0Shape.inFeatures,
          ]),
        );
        mlp0BiasVar = tf.variable(tf.tensor1d(mlp0BiasData));
        mlp2WeightVar = tf.variable(
          tf.tensor2d(mlp2WeightData, [
            mlp2Shape.outFeatures,
            mlp2Shape.inFeatures,
          ]),
        );
        mlp2BiasVar = tf.variable(tf.tensor1d(mlp2BiasData));

        const optimizer = tf.train.adam(learningRate);

        for (let epoch = 0; epoch < epochs; epoch += 1) {
          const lossTensor = optimizer.minimize(
            () =>
              tf!.tidy(() => {
                const hiddenLinear = tf!.add(
                  tf!.matMul(headInputTensor!, mlp0WeightVar!, false, true),
                  mlp0BiasVar!,
                );
                const hidden = tf!.relu(hiddenLinear);
                const logits = tf!.add(
                  tf!.matMul(hidden, mlp2WeightVar!, false, true),
                  mlp2BiasVar!,
                );
                const logProbs = tf!.logSoftmax(logits, 1);
                const selectedLogProb = tf!.sum(
                  tf!.mul(logProbs, oneHotTensor!),
                  1,
                );
                const weighted = tf!.mul(selectedLogProb, advantageTensor!);
                const policyLoss = tf!.neg(tf!.mean(weighted));

                const regComponents = tf!.add(
                  tf!.add(
                    tf!.mean(
                      tf!.square(tf!.sub(mlp0WeightVar!, mlp0WeightBase!)),
                    ),
                    tf!.mean(tf!.square(tf!.sub(mlp0BiasVar!, mlp0BiasBase!))),
                  ),
                  tf!.add(
                    tf!.mean(
                      tf!.square(tf!.sub(mlp2WeightVar!, mlp2WeightBase!)),
                    ),
                    tf!.mean(tf!.square(tf!.sub(mlp2BiasVar!, mlp2BiasBase!))),
                  ),
                );
                const reg = tf!.mul(regComponents, tf!.scalar(l2));
                return tf!.add(policyLoss, reg);
              }),
            true,
            [mlp0WeightVar, mlp0BiasVar, mlp2WeightVar, mlp2BiasVar],
          );
          if (!lossTensor) {
            throw new Error('Optimizer returned null loss tensor.');
          }
          const lossData = lossTensor.dataSync();
          finalLoss = Number(lossData[0]);
          lossTensor.dispose();
        }

        const evaluatePolicyObjective = (
          weight0: TfTensor,
          bias0: TfTensor,
          weight2: TfTensor,
          bias2: TfTensor,
        ): number => {
          const objectiveTensor = tf!.tidy(() => {
            const hiddenLinear = tf!.add(
              tf!.matMul(holdoutInputTensor!, weight0, false, true),
              bias0,
            );
            const hidden = tf!.relu(hiddenLinear);
            const logits = tf!.add(
              tf!.matMul(hidden, weight2, false, true),
              bias2,
            );
            const logProbs = tf!.logSoftmax(logits, 1);
            const selectedLogProb = tf!.sum(
              tf!.mul(logProbs, holdoutOneHotTensor!),
              1,
            );
            return tf!.mean(tf!.mul(selectedLogProb, holdoutAdvantageTensor!));
          });
          const value = Number(objectiveTensor.dataSync()[0]);
          objectiveTensor.dispose();
          return value;
        };

        const baselineObjective = evaluatePolicyObjective(
          mlp0WeightBase!,
          mlp0BiasBase!,
          mlp2WeightBase!,
          mlp2BiasBase!,
        );
        const candidateObjective = evaluatePolicyObjective(
          mlp0WeightVar!,
          mlp0BiasVar!,
          mlp2WeightVar!,
          mlp2BiasVar!,
        );
        if (
          !Number.isFinite(baselineObjective) ||
          !Number.isFinite(candidateObjective)
        ) {
          throw new Error('Eval gate produced non-finite objective.');
        }
        const objectiveDelta = candidateObjective - baselineObjective;
        if (objectiveDelta + 1e-9 < minObjectiveGain) {
          return {
            ok: false,
            message:
              `Eval gate rejected update (${pipeline.id}). ` +
              `Holdout objective Δ=${objectiveDelta.toFixed(6)} ` +
              `(candidate=${candidateObjective.toFixed(6)}, baseline=${baselineObjective.toFixed(6)}, ` +
              `required>=${minObjectiveGain.toFixed(6)}).`,
            pipelineId: pipeline.id,
            backend,
            samplesUsed: eligibleCount,
            epochs,
            finalLoss,
            updatedModelBytes: null,
          };
        }

        const updatedModelBytes = overwriteLinearHeadParams({
          baseModel: model,
          mlp0Weight: new Float32Array(mlp0WeightVar.dataSync()),
          mlp0Bias: new Float32Array(mlp0BiasVar.dataSync()),
          mlp2Weight: new Float32Array(mlp2WeightVar.dataSync()),
          mlp2Bias: new Float32Array(mlp2BiasVar.dataSync()),
        });

        return {
          ok: true,
          message:
            `Head-only training complete (${pipeline.id}, backend=${backend}, ` +
            `rewards=${rewardBackedCount}/${eligibleCount}, ` +
            `train=${trainCount}, holdout=${holdoutCount}, ` +
            `holdout Δ=${objectiveDelta.toFixed(6)}).`,
          pipelineId: pipeline.id,
          backend,
          samplesUsed: eligibleCount,
          epochs,
          finalLoss,
          updatedModelBytes,
        };
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        return {
          ok: false,
          message: `Training failed: ${detail}`,
          pipelineId: pipeline.id,
          backend,
          samplesUsed: eligibleCount,
          epochs: 0,
          finalLoss: null,
          updatedModelBytes: null,
        };
      } finally {
        if (tf) {
          tf.dispose([
            headInputTensor,
            actionTensor,
            oneHotTensor,
            advantageTensor,
            holdoutInputTensor,
            holdoutActionTensor,
            holdoutOneHotTensor,
            holdoutAdvantageTensor,
            mlp0WeightBase,
            mlp0BiasBase,
            mlp2WeightBase,
            mlp2BiasBase,
            mlp0WeightVar,
            mlp0BiasVar,
            mlp2WeightVar,
            mlp2BiasVar,
          ]);
        }
      }
    },
  };
}
