import type { Board, PieceKind } from './types';
import { predictLogits, type LoadedModel } from './wubModel';
import { createTfjsModelRunner } from './modelRunnerTfjs';

export type MlBackend = 'native' | 'tfjs';
export type TfjsBackendPreference = 'auto' | 'webgl' | 'cpu';

export type ModelRunner = {
  prepare: (model: LoadedModel) => Promise<void>;
  getInfo: () => ModelRunnerInfo;
  predictLogits: (
    model: LoadedModel,
    board: Board,
    hold: PieceKind | null,
  ) => Float32Array;
};

export type ModelRunnerInfo = {
  requestedBackend: MlBackend;
  activeBackend: MlBackend;
  runtimeBackend: string | null;
  fallbackReason: string | null;
};

type CreateModelRunnerOptions = {
  preferredBackend?: MlBackend | null;
  tfjsBackendPreference?: TfjsBackendPreference | null;
};

const normalizeBackend = (value: unknown): MlBackend =>
  value === 'tfjs' ? 'tfjs' : 'native';

const createNativeModelRunner = (): ModelRunner => {
  const info: ModelRunnerInfo = {
    requestedBackend: 'native',
    activeBackend: 'native',
    runtimeBackend: null,
    fallbackReason: null,
  };
  return {
    prepare: async () => {},
    getInfo: () => ({ ...info }),
    predictLogits: (model, board, hold) => predictLogits(model, board, hold),
  };
};

export function createModelRunner(options: CreateModelRunnerOptions = {}): {
  runner: ModelRunner;
} {
  const requestedBackend = normalizeBackend(options.preferredBackend);
  const tfjsBackendPreference =
    options.tfjsBackendPreference === 'webgl' ||
    options.tfjsBackendPreference === 'cpu'
      ? options.tfjsBackendPreference
      : 'auto';

  if (requestedBackend === 'tfjs') {
    return {
      runner: createTfjsModelRunner(requestedBackend, tfjsBackendPreference),
    };
  }

  return {
    runner: createNativeModelRunner(),
  };
}
