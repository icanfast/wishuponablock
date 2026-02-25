import { loadWubModel, type LoadedModel } from '../core/wubModel';
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
  getModelPromise: () => Promise<LoadedModel | null> | null;
  getRunner: () => ModelRunner;
  getRunnerInfo: () => ModelRunnerInfo;
  getStatus: () => ModelStatus;
  ensureLoaded: () => Promise<LoadedModel | null>;
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

  const ensureLoaded = (): Promise<LoadedModel | null> => {
    if (model) {
      notify('ready');
      return Promise.resolve(model);
    }
    if (modelPromise) return modelPromise;
    notify('loading');
    modelPromise = loadWubModel(modelUrl)
      .then(async (loaded) => {
        model = loaded;
        if (loaded) {
          await modelRunner.prepare(loaded);
          logRunnerState('[ML] backend after prepare');
        }
        notify(loaded ? 'ready' : 'failed');
        return loaded;
      })
      .catch((err) => {
        console.warn('Failed to load ML model:', err);
        model = null;
        modelPromise = null;
        notify('failed');
        return null;
      });
    return modelPromise;
  };

  return {
    getModel: () => model,
    getModelPromise: () => modelPromise,
    getRunner: () => modelRunner,
    getRunnerInfo: () => modelRunner.getInfo(),
    getStatus: () => status,
    ensureLoaded,
    setStatusListener: (next) => {
      listener = next;
    },
  };
}
