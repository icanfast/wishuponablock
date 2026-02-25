import { parseWubModelFromBytes, type LoadedModel } from '../core/wubModel';
import {
  createModelRunner,
  type MlBackend,
  type ModelRunner,
  type ModelRunnerInfo,
  type TfjsBackendPreference,
} from '../core/modelRunner';

export type ModelStatus = 'idle' | 'loading' | 'ready' | 'failed';

export type ModelService = {
  getModel: () => LoadedModel | null;
  getModelBytes: () => ArrayBuffer | null;
  getModelPromise: () => Promise<LoadedModel | null> | null;
  getRunner: () => ModelRunner;
  getRunnerInfo: () => ModelRunnerInfo;
  getStatus: () => ModelStatus;
  ensureLoaded: () => Promise<LoadedModel | null>;
  ensureModelBytes: () => Promise<ArrayBuffer | null>;
  replaceModelFromBytes: (
    payload: ArrayBuffer | ArrayBufferView,
    sourceTag?: string,
  ) => Promise<LoadedModel>;
  reloadDefaultModel: () => Promise<LoadedModel | null>;
  setStatusListener: (listener: ((status: ModelStatus) => void) | null) => void;
};

type ModelServiceOptions = {
  modelUrl: string;
  preferredBackend?: MlBackend;
  tfjsBackendPreference?: TfjsBackendPreference;
};

export function createModelService(options: ModelServiceOptions): ModelService {
  const {
    modelUrl,
    preferredBackend = 'native',
    tfjsBackendPreference = 'auto',
  } = options;
  let model: LoadedModel | null = null;
  let modelBytes: ArrayBuffer | null = null;
  let modelPromise: Promise<LoadedModel | null> | null = null;
  let status: ModelStatus = 'idle';
  let listener: ((status: ModelStatus) => void) | null = null;
  const modelRunner = createModelRunner({
    preferredBackend,
    tfjsBackendPreference,
  }).runner;
  const logRunnerState = (prefix = '[ML] backend') => {
    const runnerInfo = modelRunner.getInfo();
    if (runnerInfo.fallbackReason) {
      console.info(
        `${prefix} fallback requested=${runnerInfo.requestedBackend} active=${runnerInfo.activeBackend}: ${runnerInfo.fallbackReason}`,
      );
    } else {
      console.info(
        `${prefix} selected requested=${runnerInfo.requestedBackend} active=${runnerInfo.activeBackend}`,
      );
    }
  };
  logRunnerState();

  const notify = (next: ModelStatus) => {
    status = next;
    listener?.(status);
  };

  const toOwnedArrayBuffer = (
    payload: ArrayBuffer | ArrayBufferView,
  ): ArrayBuffer => {
    const view =
      payload instanceof ArrayBuffer
        ? new Uint8Array(payload)
        : new Uint8Array(
            payload.buffer,
            payload.byteOffset,
            payload.byteLength,
          );
    const copy = new Uint8Array(view.byteLength);
    copy.set(view);
    return copy.buffer;
  };

  const fetchModelBytes = async (url: string): Promise<ArrayBuffer> => {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Failed to load model (${response.status})`);
    }
    return await response.arrayBuffer();
  };

  const applyLoadedModel = async (
    payload: ArrayBuffer | ArrayBufferView,
    sourceTag: string,
  ): Promise<LoadedModel> => {
    const bytes = toOwnedArrayBuffer(payload);
    const loaded = parseWubModelFromBytes(bytes);
    await modelRunner.prepare(loaded);
    model = loaded;
    modelBytes = bytes;
    modelPromise = Promise.resolve(loaded);
    notify('ready');
    logRunnerState(`[ML] backend after ${sourceTag}`);
    return loaded;
  };

  const restorePreviousModel = (
    previous: LoadedModel | null,
    previousBytes: ArrayBuffer | null,
    previousPromise: Promise<LoadedModel | null> | null,
  ): void => {
    model = previous;
    modelBytes = previousBytes;
    modelPromise = previousPromise;
    notify(previous ? 'ready' : 'failed');
  };

  const ensureLoaded = (): Promise<LoadedModel | null> => {
    if (model) {
      notify('ready');
      return Promise.resolve(model);
    }
    if (modelPromise) return modelPromise;
    notify('loading');
    modelPromise = fetchModelBytes(modelUrl)
      .then(async (bytes) => {
        return await applyLoadedModel(bytes, 'prepare');
      })
      .catch((err) => {
        console.warn('Failed to load ML model:', err);
        model = null;
        modelBytes = null;
        modelPromise = null;
        notify('failed');
        return null;
      });
    return modelPromise;
  };

  const ensureModelBytes = async (): Promise<ArrayBuffer | null> => {
    if (modelBytes) return toOwnedArrayBuffer(modelBytes);
    const loaded = await ensureLoaded();
    if (!loaded || !modelBytes) return null;
    return toOwnedArrayBuffer(modelBytes);
  };

  const replaceModelFromBytes = async (
    payload: ArrayBuffer | ArrayBufferView,
    sourceTag = 'bytes payload',
  ): Promise<LoadedModel> => {
    const previousModel = model;
    const previousBytes = modelBytes;
    const previousPromise = modelPromise;
    notify('loading');
    try {
      return await applyLoadedModel(payload, sourceTag);
    } catch (error) {
      console.warn(`Failed to apply model from ${sourceTag}:`, error);
      restorePreviousModel(previousModel, previousBytes, previousPromise);
      throw error;
    }
  };

  const reloadDefaultModel = async (): Promise<LoadedModel | null> => {
    const previousModel = model;
    const previousBytes = modelBytes;
    const previousPromise = modelPromise;
    notify('loading');
    try {
      const bytes = await fetchModelBytes(modelUrl);
      return await applyLoadedModel(bytes, 'default model reload');
    } catch (error) {
      console.warn('Failed to reload default ML model:', error);
      restorePreviousModel(previousModel, previousBytes, previousPromise);
      return null;
    }
  };

  return {
    getModel: () => model,
    getModelBytes: () => (modelBytes ? toOwnedArrayBuffer(modelBytes) : null),
    getModelPromise: () => modelPromise,
    getRunner: () => modelRunner,
    getRunnerInfo: () => modelRunner.getInfo(),
    getStatus: () => status,
    ensureLoaded,
    ensureModelBytes,
    replaceModelFromBytes,
    reloadDefaultModel,
    setStatusListener: (next) => {
      listener = next;
    },
  };
}
