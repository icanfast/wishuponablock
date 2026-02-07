import { loadWubModel, type LoadedModel } from '../core/wubModel';

export type ModelStatus = 'idle' | 'loading' | 'ready' | 'failed';

export type ModelService = {
  getModel: () => LoadedModel | null;
  getModelPromise: () => Promise<LoadedModel | null> | null;
  getStatus: () => ModelStatus;
  ensureLoaded: () => Promise<LoadedModel | null>;
  setStatusListener: (listener: ((status: ModelStatus) => void) | null) => void;
};

type ModelServiceOptions = {
  modelUrl: string;
};

export function createModelService(options: ModelServiceOptions): ModelService {
  const { modelUrl } = options;
  let model: LoadedModel | null = null;
  let modelPromise: Promise<LoadedModel | null> | null = null;
  let status: ModelStatus = 'idle';
  let listener: ((status: ModelStatus) => void) | null = null;

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
      .then((loaded) => {
        model = loaded;
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
    getStatus: () => status,
    ensureLoaded,
    setStatusListener: (next) => {
      listener = next;
    },
  };
}
