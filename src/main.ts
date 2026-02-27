import { Application, Graphics } from 'pixi.js';
import {
  COLS,
  ML_BACKEND_PREFERENCE_STORAGE_KEY,
  ML_MODEL_URL,
  PLAY_HEIGHT,
  PLAY_WIDTH,
  ROWS,
} from './core/constants';
import {
  GENERATOR_TYPES,
  usesModelGenerator,
  type GeneratorType,
} from './core/generators';
import {
  createModelRunner,
  type MlBackend,
  type ModelRunnerInfo,
  type TfjsBackendPreference,
} from './core/modelRunner';
import type { GameSession } from './core/gameSession';
import { createSettingsStore } from './core/settingsStore';
import { createGameRuntime, type GameRuntime } from './app/runtime';
import { createUploadService } from './app/uploadService';
import { createModeController } from './app/modeController';
import { createInputService } from './app/inputService';
import { createIdentityService } from './app/identityService';
import { createSoundService } from './app/soundService';
import { createSessionController } from './app/sessionController';
import { createScreenManager } from './app/screenManager';
import { createUiController } from './app/uiController';
import { createScreenFlowController } from './app/screenFlowController';
import { createSettingsController } from './app/settingsController';
import { createAuthService } from './app/authService';
import { createPersonalModelService } from './app/personalModelService';
import {
  createTrajectoryBuffer,
  type TrajectoryDecisionSample,
} from './app/trajectoryBuffer';
import { createTrajectoryRecordingService } from './app/trajectoryRecordingService';
import {
  createAdminRecordingsService,
  type AdminRecordingsManifestPage,
  type AdminRecordingsPage,
} from './app/adminRecordingsService';
import {
  createAdminBotPolicyService,
  type BotPolicyRecord as AdminBotPolicyRecord,
} from './app/adminBotPolicyService';
import {
  computeTrajectoryRewards,
  resolveTrajectoryRewardPolicyId,
  type TrajectoryRewardComputation,
  type TrajectoryRewardPolicyId,
  type TrajectoryRewardTerminalStats,
} from './app/trajectoryRewardPolicy';
import { createPersonalTrainerTfjs } from './app/personalTrainerTfjs';
import { runGlobalTrainingOneShot } from './app/globalTrainerTfjs';
import { resolvePersonalTrainingPipelineForContext } from './app/trainingPipelines';
import {
  createGuiInspectBotInputSource,
  generateBotTrajectoryBatch,
  runHeadlessBotValidation,
  runCapabilityBenchmark,
  trainBotPolicyOneShot,
  type BotPieceSourceProfile,
  type BotTrainingAlgorithm,
  type BotPolicyArtifact,
} from './app/headlessBotService';
import {
  MIN_TRAJECTORY_SAMPLES_PER_SESSION,
  TRAJECTORY_SESSION_SCHEMA_V1,
  type TrajectoryReplayStepV1,
  type TrajectorySessionMetaV1,
  type TrajectorySessionV1,
} from './core/trajectoryProtocol';
import type {
  CharcuterieHoleWeights,
  CharcuterieScoreWeights,
} from './app/charcuterieService';
import { createModelService, type ModelStatus } from './app/modelService';
import {
  createSnapshotService,
  type SnapshotService,
} from './app/snapshotService';
import { createGameSessionFactory } from './app/gameFactory';
import { PixiRenderer } from './render/pixiRenderer';
import { type Board, type PieceKind, type GameState } from './core/types';
import {
  createMenuScreen,
  type MenuAuthState,
  type MenuMlBackendPreference,
  type MenuMlParityResult,
  type MenuAuthStatusTone,
  type MenuModelAxes,
  type MenuAdminRecordingsPage,
  type MenuAdminRecordingPreview,
  type MenuAdminRecordingsQuery,
  type MenuAdminManifestQuery,
  type MenuAdminGlobalBaselineSummary,
  type MenuBotPoliciesPage,
  type MenuScreen,
} from './ui/screens/menuScreen';
import {
  MODEL_ARCH_OPTIONS,
  QUEUE_POLICY_OPTIONS,
  REWARD_PROFILE_OPTIONS,
  modelAxesKey,
  normalizeModelAxes,
} from './core/modelAxes';
import { parseWubModelFromBytes, type LoadedModel } from './core/wubModel';
import { createGameScreen, type GameScreen } from './ui/screens/gameScreen';
import {
  createToolHost,
  type ToolController,
  type ToolHost,
} from './ui/tools/toolHost';
import type { InputSource } from './core/runner';
import { createLabelingTool } from './ui/tools/labelingTool';
import { createConstructorTool } from './ui/tools/constructorTool';
import { createToolCanvas } from './ui/tools/toolCanvas';
import { getPiecePalette, type PiecePalette } from './core/palette';
import { createPerfOverlay } from './app/perfOverlay';
import { setPerfMetricsSink } from './core/perfMetrics';
import pkg from '../package.json';

function hasWebGL(): boolean {
  const c = document.createElement('canvas');
  return !!(c.getContext('webgl2') || c.getContext('webgl'));
}

const resolveMlBackendPreference = (
  value: string | null | undefined,
  fallback: MenuMlBackendPreference,
): MenuMlBackendPreference => {
  if (
    value === 'native' ||
    value === 'tfjs_auto' ||
    value === 'tfjs_webgl' ||
    value === 'tfjs_cpu'
  ) {
    return value;
  }
  return fallback;
};

const getMlBackendPreferenceParts = (
  preference: MenuMlBackendPreference,
): {
  preferredBackend: MlBackend;
  tfjsBackendPreference: TfjsBackendPreference;
} => {
  if (preference === 'native') {
    return { preferredBackend: 'native', tfjsBackendPreference: 'auto' };
  }
  if (preference === 'tfjs_webgl') {
    return { preferredBackend: 'tfjs', tfjsBackendPreference: 'webgl' };
  }
  if (preference === 'tfjs_cpu') {
    return { preferredBackend: 'tfjs', tfjsBackendPreference: 'cpu' };
  }
  return { preferredBackend: 'tfjs', tfjsBackendPreference: 'auto' };
};

const formatMlRuntimeSummary = (info: ModelRunnerInfo): string => {
  const runtime = info.runtimeBackend ? ` (${info.runtimeBackend})` : '';
  const fallback = info.fallbackReason
    ? `\nFallback: ${info.fallbackReason}`
    : '';
  return `Requested: ${info.requestedBackend}\nActive: ${info.activeBackend}${runtime}${fallback}`;
};

type ActiveModelSource =
  | { kind: 'global' }
  | {
      kind: 'personal';
      mode: string;
      version: number | null;
    };

async function boot() {
  const APP_VERSION = pkg.version;
  const GAME_SCREEN_Y_OFFSET = 20;
  if (!hasWebGL()) {
    document.body.innerHTML = `<div style="padding:16px;color:#fff;background:#000;height:100vh">
      WebGL is disabled/unavailable. Enable hardware acceleration.
    </div>`;
    return;
  }

  const app = new Application();
  await app.init({
    width: PLAY_WIDTH,
    height: PLAY_HEIGHT,
    backgroundColor: 0x0b0f14,
    preference: 'webgl',
    powerPreference: 'high-performance',
    antialias: false,
  });

  const root = document.getElementById('app') ?? document.body;
  root.innerHTML = '';
  Object.assign(document.body.style, { margin: '0', overflow: 'hidden' });
  Object.assign(root.style, {
    position: 'fixed',
    inset: '0',
    overflow: 'hidden',
  });

  const playWindow = document.createElement('div');
  Object.assign(playWindow.style, {
    position: 'relative',
    width: `${PLAY_WIDTH}px`,
    height: `${PLAY_HEIGHT}px`,
  });
  root.appendChild(playWindow);
  playWindow.appendChild(app.canvas);

  const gameGfx = new Graphics();
  const toolGfx = new Graphics();
  gameGfx.y = GAME_SCREEN_Y_OFFSET;
  gameGfx.visible = false;
  toolGfx.visible = false;
  app.stage.addChild(gameGfx);
  app.stage.addChild(toolGfx);

  const uiLayer = document.createElement('div');
  Object.assign(uiLayer.style, {
    position: 'absolute',
    inset: '0',
    pointerEvents: 'none',
  });
  playWindow.appendChild(uiLayer);

  const makeScreenLayer = () => {
    const layer = document.createElement('div');
    Object.assign(layer.style, {
      position: 'absolute',
      inset: '0',
      pointerEvents: 'none',
      display: 'none',
    });
    return layer;
  };

  const gameScreen = makeScreenLayer();
  const toolScreen = makeScreenLayer();
  const menuScreen = makeScreenLayer();

  uiLayer.appendChild(gameScreen);
  uiLayer.appendChild(toolScreen);
  uiLayer.appendChild(menuScreen);

  const screenManager = createScreenManager();

  const settingsStore = createSettingsStore();
  const settings = settingsStore.get();
  const SHOW_EXPERIMENTAL_GAMEPLAY_CONTROLS =
    import.meta.env.VITE_SHOW_EXPERIMENTAL_GAMEPLAY_CONTROLS === 'true';
  const ENABLE_LEGACY_DATA_TOOLS =
    import.meta.env.VITE_ENABLE_LEGACY_DATA_TOOLS === 'true';
  const SHOW_PERF_OVERLAY =
    import.meta.env.VITE_SHOW_PERF_OVERLAY === 'true' || import.meta.env.DEV;
  const perfOverlay = SHOW_PERF_OVERLAY
    ? createPerfOverlay({
        root: playWindow,
        windowMs: 10_000,
        targetFps: 120,
        initialVisible: false,
      })
    : null;
  setPerfMetricsSink(perfOverlay);
  const uploadService = createUploadService({
    envMode: import.meta.env.VITE_UPLOAD_MODE as string | undefined,
    envBaseUrl: import.meta.env.VITE_UPLOAD_URL as string | undefined,
  });
  const uploadClient = uploadService.uploadClient;
  const uploadBaseUrl = uploadService.baseUrl;
  const authService = createAuthService({ baseUrl: uploadBaseUrl });
  const personalModelService = createPersonalModelService({
    baseUrl: uploadBaseUrl,
  });
  const trajectoryRecordingService = createTrajectoryRecordingService({
    baseUrl: uploadBaseUrl,
  });
  const adminRecordingsService = createAdminRecordingsService({
    baseUrl: uploadBaseUrl,
  });
  const adminBotPolicyService = createAdminBotPolicyService({
    baseUrl: uploadBaseUrl,
  });
  const authErrorMessages: Record<string, string> = {
    oauth_not_configured: 'OAuth provider is not configured.',
    oauth_denied: 'OAuth sign-in was cancelled.',
    oauth_state_mismatch: 'OAuth session expired. Please try again.',
    oauth_failed: 'OAuth sign-in failed. Please try again.',
  };
  const toErrorMessage = (error: unknown, fallback: string): string => {
    if (error instanceof Error && error.message.trim()) return error.message;
    return fallback;
  };
  const parseApiErrorMessage = async (
    response: Response,
    fallback: string,
  ): Promise<string> => {
    const payload = (await response.json().catch(() => null)) as {
      error?: unknown;
    } | null;
    if (typeof payload?.error === 'string' && payload.error.trim()) {
      return payload.error;
    }
    return fallback;
  };
  const readUrlToken = (value: string | null): string | null => {
    if (!value) return null;
    const token = value.trim();
    return token ? token : null;
  };
  const startupUrl = new URL(window.location.href);
  let authInitialResetToken: string | null = null;
  let authInitialStatus: {
    message: string;
    tone: MenuAuthStatusTone;
  } | null = null;
  let openPanelOnBoot: 'account' | 'admin' | null = null;

  const authError = startupUrl.searchParams.get('auth_error');
  if (authError) {
    const message = authErrorMessages[authError] ?? 'Authentication failed.';
    authInitialStatus = { message, tone: 'error' };
    openPanelOnBoot = 'account';
    startupUrl.searchParams.delete('auth_error');
  }

  if (startupUrl.pathname === '/auth/password/reset') {
    authInitialResetToken = readUrlToken(startupUrl.searchParams.get('token'));
    authInitialStatus = {
      message: authInitialResetToken
        ? 'Paste your new password and submit reset.'
        : 'Missing password reset token.',
      tone: authInitialResetToken ? 'neutral' : 'error',
    };
    openPanelOnBoot = 'account';
    startupUrl.pathname = '/';
    startupUrl.searchParams.delete('token');
  }

  if (startupUrl.pathname === '/auth/email/verify') {
    const verifyToken = readUrlToken(startupUrl.searchParams.get('token'));
    if (!verifyToken) {
      authInitialStatus = {
        message: 'Missing email verification token.',
        tone: 'error',
      };
    } else {
      try {
        await authService.consumeEmailVerification(verifyToken);
        authInitialStatus = {
          message: 'Email verified. You are signed in.',
          tone: 'success',
        };
      } catch (error) {
        authInitialStatus = {
          message: toErrorMessage(error, 'Email verification failed.'),
          tone: 'error',
        };
      }
    }
    openPanelOnBoot = 'account';
    startupUrl.pathname = '/';
    startupUrl.searchParams.delete('token');
  }

  if (startupUrl.pathname === '/admin') {
    openPanelOnBoot = 'admin';
    startupUrl.pathname = '/';
  }

  const cleanedHref = `${startupUrl.pathname}${startupUrl.search}${startupUrl.hash}`;
  if (
    cleanedHref !==
    `${window.location.pathname}${window.location.search}${window.location.hash}`
  ) {
    window.history.replaceState(null, '', cleanedHref);
  }
  const useRemoteUpload = ENABLE_LEGACY_DATA_TOOLS && uploadService.useRemote;
  const toolUsesRemote =
    ENABLE_LEGACY_DATA_TOOLS && uploadService.toolUsesRemote;
  const LABELING_PROGRESS_TARGET = 1000;
  const defaultMlBackendPreference: MenuMlBackendPreference =
    import.meta.env.VITE_ML_BACKEND === 'native' ? 'native' : 'tfjs_auto';
  const storedMlBackendPreference = resolveMlBackendPreference(
    localStorage.getItem(ML_BACKEND_PREFERENCE_STORAGE_KEY),
    defaultMlBackendPreference,
  );
  const { preferredBackend: preferredMlBackend, tfjsBackendPreference } =
    getMlBackendPreferenceParts(storedMlBackendPreference);
  let modelStatusLabel: HTMLDivElement | null = null;
  let pausedByModel = false;
  let activeModelSource: ActiveModelSource = { kind: 'global' };
  let activeModelContextKey: string | null = null;
  let modelSyncRequestId = 0;
  const syncedModelShaByMode = new Map<string, string>();
  let setScreen: (screen: 'menu' | 'game' | 'tool') => void = () => {};
  let requestStartGame: () => void = () => {};
  let runtime: GameRuntime | null = null;
  const modelService = createModelService({
    modelUrl: ML_MODEL_URL,
    preferredBackend: preferredMlBackend,
    tfjsBackendPreference,
  });
  const botModelRunner = createModelRunner({
    preferredBackend: preferredMlBackend,
    tfjsBackendPreference,
  }).runner;
  modelService.setStatusListener((status) => {
    updateModelStatusUI(status);
  });
  const getModelGeneratorLabel = (): string => {
    const type = settingsStore.get().generator.type;
    return type === 'curse' ? 'Curse Upon a Block' : 'Wish Upon a Block';
  };
  const getModelSourceLabel = (): string => {
    if (activeModelSource.kind === 'personal') {
      const version =
        activeModelSource.version != null
          ? ` v${activeModelSource.version}`
          : '';
      return `personal:${activeModelSource.mode}${version}`;
    }
    return 'global';
  };
  const setActiveModelSource = (next: ActiveModelSource): void => {
    activeModelSource = next;
    updateModelStatusUI(modelService.getStatus());
  };
  const getHttpStatus = (error: unknown): number | null => {
    if (
      typeof (error as { status?: unknown })?.status === 'number' &&
      Number.isFinite((error as { status: number }).status)
    ) {
      return Math.trunc((error as { status: number }).status);
    }
    return null;
  };
  const sha256HexFromBuffer = async (buffer: ArrayBuffer): Promise<string> => {
    const digest = await crypto.subtle.digest('SHA-256', buffer);
    const bytes = new Uint8Array(digest);
    return Array.from(bytes, (value) =>
      value.toString(16).padStart(2, '0'),
    ).join('');
  };
  const updateModelStatusUI = (status: ModelStatus): void => {
    if (!modelStatusLabel) return;
    const generatorType = settingsStore.get().generator.type;
    if (!usesModelGenerator(generatorType)) {
      modelStatusLabel.textContent = '';
      modelStatusLabel.style.display = 'none';
      if (pausedByModel) {
        pausedByModel = false;
        runtime?.setPausedByModel(false);
      }
      return;
    }
    const generatorLabel = getModelGeneratorLabel();
    const runnerInfo = modelService.getRunnerInfo();
    const runtimeLabel =
      runnerInfo.activeBackend === 'tfjs' && runnerInfo.runtimeBackend
        ? `:${runnerInfo.runtimeBackend}`
        : '';
    const backendSuffix =
      runnerInfo.requestedBackend === runnerInfo.activeBackend
        ? ` (${runnerInfo.activeBackend}${runtimeLabel})`
        : ` (${runnerInfo.activeBackend}${runtimeLabel} fallback)`;
    const sourceSuffix = ` [${getModelSourceLabel()}]`;
    modelStatusLabel.style.display = 'block';
    let text = `${generatorLabel}: idle (RNG fallback)${backendSuffix}${sourceSuffix}`;
    let color = '#f4b266';
    let shouldPause = false;
    if (status === 'ready') {
      text = `${generatorLabel}: loaded${backendSuffix}${sourceSuffix}`;
      color = '#8fd19e';
      shouldPause = false;
    } else if (status === 'loading') {
      text = `${generatorLabel}: loading (RNG fallback)${backendSuffix}${sourceSuffix}`;
      color = '#f4b266';
      shouldPause = false;
    } else if (status === 'failed') {
      text = `${generatorLabel}: failed to load model${backendSuffix}${sourceSuffix}`;
      color = '#f28b82';
      shouldPause = true;
    }
    modelStatusLabel.textContent = text;
    modelStatusLabel.style.color = color;
    if (pausedByModel !== shouldPause) {
      pausedByModel = shouldPause;
      runtime?.setPausedByModel(shouldPause);
    }
  };
  const getMlRuntimeSummary = (): string =>
    `${formatMlRuntimeSummary(modelService.getRunnerInfo())}\nModel source: ${getModelSourceLabel()}`;
  const applyMlBackendPreference = (next: MenuMlBackendPreference): void => {
    localStorage.setItem(ML_BACKEND_PREFERENCE_STORAGE_KEY, next);
    window.location.reload();
  };
  void modelService.ensureLoaded();
  const trajectoryBuffer = createTrajectoryBuffer({ maxSamples: 2500 });
  const modeController = createModeController({
    initialModeId: 'practice',
  });
  const trainingPipelineValue = import.meta.env.VITE_TRAINING_PIPELINE as
    | string
    | undefined;
  const getTrainingPipelineForContext = (
    modeId: string,
    axes: {
      arch: string;
      rewardProfileId: string;
      queuePolicyId: string;
    },
  ) =>
    resolvePersonalTrainingPipelineForContext(trainingPipelineValue, {
      modeId,
      arch: axes.arch,
      rewardProfileId: axes.rewardProfileId,
      queuePolicyId: axes.queuePolicyId,
    });
  const initialModelAxes = normalizeModelAxes(settingsStore.get().modelAxes);
  const defaultTrainingPipeline = getTrainingPipelineForContext(
    modeController.getState().mode.id,
    initialModelAxes,
  );
  const DEFAULT_MIN_TRAJECTORY_SAMPLES_FOR_UPLOAD = Math.max(
    MIN_TRAJECTORY_SAMPLES_PER_SESSION,
    defaultTrainingPipeline.minSamples,
  );
  const configuredTrajectoryRewardPolicy = import.meta.env
    .VITE_TRAJECTORY_REWARD_POLICY as string | undefined;
  const getTrajectoryRewardPolicyId = (
    modeId: string,
    axes: {
      arch: string;
      rewardProfileId: string;
      queuePolicyId: string;
    },
  ): TrajectoryRewardPolicyId =>
    resolveTrajectoryRewardPolicyId(
      configuredTrajectoryRewardPolicy ??
        getTrainingPipelineForContext(modeId, axes).rewardPolicyId,
    );
  const personalTrainer = createPersonalTrainerTfjs();
  let menuUi: MenuScreen | null = null;
  const charcuterieDefaultSimCount = 10000;
  const charcuterieScoreWeights: CharcuterieScoreWeights = {
    height: 10,
    holes: 20,
    blocks: 0.01,
    clears: 100,
  };
  const charcuterieHoleWeights: CharcuterieHoleWeights = {
    bottom: 5,
    mid: 2,
  };

  const inputService = createInputService({ settings });
  const identityService = createIdentityService();
  let manualInputSource = inputService.getInputSource();
  let botGuiInputSource: InputSource | null = null;
  let botGuiInspectEnabled = false;
  const activeInputSource: InputSource = {
    sample: (state, dtMs) => {
      if (botGuiInspectEnabled && botGuiInputSource) {
        return botGuiInputSource.sample(state, dtMs);
      }
      return manualInputSource.sample(state, dtMs);
    },
    reset: (seed) => {
      manualInputSource.reset?.(seed);
      botGuiInputSource?.reset?.(seed);
    },
  };
  let authState: MenuAuthState = {
    loading: true,
    authenticated: false,
    user: null,
  };

  const applyAuthState = (next: MenuAuthState) => {
    authState = next;
    menuUi?.setAuthState(next);
    identityService.setUserId(
      next.authenticated ? (next.user?.id ?? null) : null,
    );
  };

  const getActiveModelAxes = () =>
    normalizeModelAxes(settingsStore.get().modelAxes);
  const buildModelSyncKey = (mode: string): string =>
    `${mode}:${modelAxesKey(getActiveModelAxes())}`;
  const buildModelContextKey = (userId: string | null, mode: string): string =>
    `${userId ?? 'anon'}:${buildModelSyncKey(mode)}`;
  let botGlobalModelCache: {
    mode: string;
    key: string;
    modelId: string;
    model: LoadedModel;
  } | null = null;
  const loadBotReferenceModel = async (
    mode: string,
  ): Promise<{ model: LoadedModel; modelId: string }> => {
    if (!authState.authenticated || !authState.user) {
      throw new Error('Sign in to use bot training.');
    }
    const selector = getActiveModelAxes();
    const key = `${mode}:${modelAxesKey(selector)}`;
    const globalModel = await personalModelService.downloadGlobalCurrent(
      mode,
      selector,
    );
    if (
      botGlobalModelCache &&
      botGlobalModelCache.mode === mode &&
      botGlobalModelCache.key === key &&
      botGlobalModelCache.modelId === globalModel.id
    ) {
      await botModelRunner.prepare(botGlobalModelCache.model);
      return {
        model: botGlobalModelCache.model,
        modelId: botGlobalModelCache.modelId,
      };
    }
    const parsed = parseWubModelFromBytes(globalModel.bytes);
    await botModelRunner.prepare(parsed);
    botGlobalModelCache = {
      mode,
      key,
      modelId: globalModel.id,
      model: parsed,
    };
    return { model: parsed, modelId: globalModel.id };
  };

  const syncActiveModelForContext = async (options: {
    mode: string;
    reason: string;
    force?: boolean;
    interactive?: boolean;
  }): Promise<{ source: 'global' | 'personal'; message: string }> => {
    const { mode, reason, force = false, interactive = false } = options;
    const userId =
      authState.authenticated && authState.user ? authState.user.id : null;
    const contextKey = buildModelContextKey(userId, mode);
    const selector = getActiveModelAxes();
    if (!force && activeModelContextKey === contextKey) {
      return {
        source: activeModelSource.kind,
        message: `Using ${getModelSourceLabel()} model.`,
      };
    }

    const requestId = ++modelSyncRequestId;
    const isCurrentRequest = (): boolean => requestId === modelSyncRequestId;

    if (!userId) {
      if (activeModelSource.kind !== 'global' || force) {
        await modelService.reloadDefaultModel();
        if (!isCurrentRequest()) {
          return {
            source: activeModelSource.kind,
            message: 'Model context was superseded.',
          };
        }
        setActiveModelSource({ kind: 'global' });
        sessionController.rebuildSession();
      }
      activeModelContextKey = contextKey;
      return {
        source: 'global',
        message: 'Using global model.',
      };
    }

    try {
      const result = await personalModelService.downloadCurrent(mode, selector);
      if (!isCurrentRequest()) {
        return {
          source: activeModelSource.kind,
          message: 'Model context was superseded.',
        };
      }

      await modelService.replaceModelFromBytes(
        result.bytes,
        `personal model (${mode})`,
      );
      if (!isCurrentRequest()) {
        return {
          source: activeModelSource.kind,
          message: 'Model context was superseded.',
        };
      }

      const resolvedSha =
        result.sha256 ?? (await sha256HexFromBuffer(result.bytes));
      syncedModelShaByMode.set(buildModelSyncKey(mode), resolvedSha);
      setActiveModelSource({
        kind: 'personal',
        mode: result.mode,
        version: result.version,
      });
      activeModelContextKey = contextKey;
      sessionController.rebuildSession();
      const versionLabel =
        result.version != null ? `v${result.version}` : 'latest';
      return {
        source: 'personal',
        message: `Loaded ${versionLabel} personal model for mode "${result.mode}".`,
      };
    } catch (error) {
      if (!isCurrentRequest()) {
        return {
          source: activeModelSource.kind,
          message: 'Model context was superseded.',
        };
      }
      const status = getHttpStatus(error);
      if (status !== 404) {
        console.warn(
          `[models] personal model load failed (mode=${mode}, reason=${reason})`,
          error,
        );
      }

      await modelService.reloadDefaultModel();
      if (!isCurrentRequest()) {
        return {
          source: activeModelSource.kind,
          message: 'Model context was superseded.',
        };
      }

      setActiveModelSource({ kind: 'global' });
      activeModelContextKey = contextKey;
      sessionController.rebuildSession();

      if (status === 404) {
        if (interactive) {
          throw new Error(`No cloud model saved for mode "${mode}" yet.`);
        }
        return {
          source: 'global',
          message: `No personal model for mode "${mode}", using global model.`,
        };
      }
      if (interactive) throw error;
      return {
        source: 'global',
        message: `Using global model (personal load failed for mode "${mode}").`,
      };
    }
  };

  const autoSavePersonalModelIfDirty = async (
    mode: string,
    reason: string,
  ): Promise<void> => {
    if (!authState.authenticated || !authState.user) return;
    if (
      activeModelSource.kind !== 'personal' ||
      activeModelSource.mode !== mode
    )
      return;

    const selector = getActiveModelAxes();
    const bytes = await modelService.ensureModelBytes();
    if (!bytes) return;
    const currentSha = await sha256HexFromBuffer(bytes);
    const syncKey = buildModelSyncKey(mode);
    const syncedSha = syncedModelShaByMode.get(syncKey) ?? null;
    if (syncedSha && syncedSha === currentSha) return;

    try {
      const result = await personalModelService.uploadCurrent(
        mode,
        bytes,
        selector,
      );
      syncedModelShaByMode.set(syncKey, result.sha256 ?? currentSha);
      setActiveModelSource({
        kind: 'personal',
        mode,
        version: result.version ?? activeModelSource.version,
      });
      console.info(
        `[models] auto-saved personal model (mode=${mode}, reason=${reason})`,
      );
    } catch (error) {
      console.warn(
        `[models] auto-save failed (mode=${mode}, reason=${reason})`,
        error,
      );
    }
  };

  const refreshAuthState = async () => {
    applyAuthState({ ...authState, loading: true });
    try {
      const session = await authService.getSession();
      applyAuthState({
        loading: false,
        authenticated: session.authenticated,
        user: session.user,
      });
      const currentMode = modeController.getState().mode.id;
      await syncActiveModelForContext({
        mode: currentMode,
        reason: 'auth_refresh',
      });
    } catch (error) {
      console.error('[auth] /me failed', error);
      applyAuthState({
        loading: false,
        authenticated: false,
        user: null,
      });
      throw error;
    }
  };

  const logoutAuthState = async () => {
    const currentMode = modeController.getState().mode.id;
    await autoSavePersonalModelIfDirty(currentMode, 'logout');
    await authService.logout();
    activeModelContextKey = null;
    await refreshAuthState();
  };

  const sendAuthVerificationEmail = async () => {
    const result = await authService.sendVerificationEmail();
    await refreshAuthState().catch(() => {});
    return result;
  };
  const signupWithEmail = async (payload: {
    email: string;
    password: string;
    username?: string;
  }) => {
    await authService.emailSignup(payload);
    await refreshAuthState();
  };
  const loginWithEmail = async (payload: {
    email: string;
    password: string;
  }) => {
    await authService.emailLogin(payload);
    await refreshAuthState();
  };
  const sendPasswordReset = async (email: string) => {
    await authService.passwordForgot(email);
  };
  const resetPasswordWithToken = async (payload: {
    token: string;
    password: string;
  }) => {
    await authService.passwordReset(payload);
    await refreshAuthState();
  };
  const loadCurrentModePersonalModel = async (): Promise<string> => {
    if (!authState.authenticated) {
      throw new Error('Sign in to load a personal model.');
    }
    const mode = modeController.getState().mode.id;
    const result = await syncActiveModelForContext({
      mode,
      reason: 'manual_load',
      force: true,
      interactive: true,
    });
    return result.message;
  };
  const saveCurrentModePersonalModel = async (): Promise<string> => {
    if (!authState.authenticated) {
      throw new Error('Sign in to save a personal model.');
    }
    const mode = modeController.getState().mode.id;
    const selector = getActiveModelAxes();
    const bytes = await modelService.ensureModelBytes();
    if (!bytes) {
      throw new Error('No model is loaded to upload.');
    }
    const currentSha = await sha256HexFromBuffer(bytes);
    const result = await personalModelService.uploadCurrent(
      mode,
      bytes,
      selector,
    );
    syncedModelShaByMode.set(
      buildModelSyncKey(mode),
      result.sha256 ?? currentSha,
    );
    if (
      activeModelSource.kind === 'personal' &&
      activeModelSource.mode === mode
    ) {
      setActiveModelSource({
        kind: 'personal',
        mode,
        version: result.version ?? activeModelSource.version,
      });
    }
    const versionLabel =
      result.version != null ? `v${result.version}` : 'saved';
    return `Saved ${versionLabel} model for mode "${result.mode}".`;
  };
  const listCurrentModeGlobalModels = async () => {
    if (!authState.authenticated) {
      throw new Error('Sign in to browse global baselines.');
    }
    const mode = modeController.getState().mode.id;
    return await personalModelService.listGlobal(mode, getActiveModelAxes());
  };
  const resetCurrentModeToGlobalModel = async (
    globalModelId: string | null,
  ): Promise<string> => {
    if (!authState.authenticated) {
      throw new Error('Sign in to reset your model.');
    }
    const mode = modeController.getState().mode.id;
    const selector = getActiveModelAxes();
    const result = await personalModelService.resetCurrentFromGlobal(
      mode,
      globalModelId,
      selector,
    );
    await syncActiveModelForContext({
      mode,
      reason: 'global_reset',
      force: true,
      interactive: true,
    });
    const versionLabel =
      result.model.version != null ? `v${result.model.version}` : 'latest';
    const globalLabel = result.globalModelLabel ?? result.globalModelId;
    const globalSuffix = globalLabel ? ` from "${globalLabel}"` : '';
    return `Reset and loaded ${versionLabel} personal model${globalSuffix}.`;
  };

  const toMenuAdminRecordingsPage = (
    page: AdminRecordingsPage,
  ): MenuAdminRecordingsPage => ({
    recordings: page.recordings.map((recording) => ({
      id: recording.id,
      userId: recording.userId,
      mode: recording.mode,
      buildVersion: recording.buildVersion,
      startedAtMs: recording.startedAtMs,
      durationMs: recording.durationMs,
      samples: recording.samples,
    })),
    page: {
      limit: page.page.limit,
      nextCursor: page.page.nextCursor,
      returned: page.page.returned,
    },
  });

  const listAdminRecordings = async (
    query: MenuAdminRecordingsQuery,
  ): Promise<MenuAdminRecordingsPage> =>
    toMenuAdminRecordingsPage(
      await adminRecordingsService.listRecordings(query),
    );

  const loadAdminRecordingPreview = async (
    id: string,
  ): Promise<MenuAdminRecordingPreview> => {
    const session = await adminRecordingsService.loadRecordingObject(id);
    const rewards = session.samples
      .map((sample) => sample.reward)
      .filter(
        (value): value is number => value != null && Number.isFinite(value),
      );
    const avgReward =
      rewards.length > 0
        ? rewards.reduce((sum, value) => sum + value, 0) / rewards.length
        : null;
    const deliberations = session.samples
      .map((sample) => sample.deliberationMs)
      .filter(
        (value): value is number => value != null && Number.isFinite(value),
      );
    const meanDeliberationMs =
      deliberations.length > 0
        ? deliberations.reduce((sum, value) => sum + value, 0) /
          deliberations.length
        : null;
    return {
      id,
      sessionId: session.sessionId,
      modeId: session.modeId,
      buildVersion: session.buildVersion,
      samples: session.samples.length,
      durationMs: session.durationMs,
      startedAtMs: session.startedAtMs,
      endedAtMs: session.endedAtMs,
      avgReward,
      meanDeliberationMs,
      rewardPolicy: session.meta?.rewardPolicy ?? null,
      rewardPolicyId: session.meta?.rewardPolicyId ?? null,
      rewardProfileId: session.meta?.rewardProfileId ?? null,
      queuePolicyId: session.meta?.queuePolicyId ?? null,
      rewardKind: session.meta?.rewardKind ?? null,
      rewardGamma: session.meta?.rewardGamma ?? null,
      pipelineId: session.meta?.pipelineId ?? null,
      pipelineMode: session.meta?.pipelineMode ?? null,
      modelArchId: session.meta?.modelArchId ?? null,
      modelArch: session.meta?.modelArch ?? null,
      outcome: session.meta?.outcome ?? null,
    };
  };
  const listAdminGlobalBaselines = async (): Promise<
    MenuAdminGlobalBaselineSummary[]
  > => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const mode = modeController.getState().mode.id;
    const selector = getActiveModelAxes();
    const url = new URL(
      `${uploadBaseUrl}/admin/models/global/index`,
      window.location.origin,
    );
    url.searchParams.set('mode', mode);
    url.searchParams.set('arch', selector.arch);
    url.searchParams.set('reward_profile', selector.rewardProfileId);
    url.searchParams.set('queue_policy', selector.queuePolicyId);
    const response = await fetch(url.toString(), {
      method: 'GET',
      credentials: 'include',
      cache: 'no-store',
    });
    if (!response.ok) {
      throw new Error(
        await parseApiErrorMessage(
          response,
          `Baseline list failed (${response.status}).`,
        ),
      );
    }
    const payload = (await response.json().catch(() => null)) as {
      models?: Array<Record<string, unknown>> | null;
    } | null;
    const rows = Array.isArray(payload?.models) ? payload.models : [];
    const baselines: MenuAdminGlobalBaselineSummary[] = [];
    for (const row of rows) {
      const id = typeof row.id === 'string' ? row.id : null;
      const modeValue = typeof row.mode === 'string' ? row.mode : null;
      if (!id || !modeValue) continue;
      baselines.push({
        id,
        mode: modeValue,
        arch: typeof row.arch === 'string' ? row.arch : null,
        rewardProfileId:
          typeof row.rewardProfileId === 'string' ? row.rewardProfileId : null,
        queuePolicyId:
          typeof row.queuePolicyId === 'string' ? row.queuePolicyId : null,
        pipelineId: typeof row.pipelineId === 'string' ? row.pipelineId : null,
        label: typeof row.label === 'string' ? row.label : null,
        isDefault: row.isDefault === true || row.isDefault === 1,
        sizeBytes:
          typeof row.sizeBytes === 'number' && Number.isFinite(row.sizeBytes)
            ? Math.trunc(row.sizeBytes)
            : null,
        updatedAtMs:
          typeof row.updatedAtMs === 'number' &&
          Number.isFinite(row.updatedAtMs)
            ? Math.trunc(row.updatedAtMs)
            : null,
        retiredAtMs:
          typeof row.retiredAtMs === 'number' &&
          Number.isFinite(row.retiredAtMs)
            ? Math.trunc(row.retiredAtMs)
            : null,
      });
    }
    return baselines;
  };
  const setAdminGlobalBaselineDefault = async (id: string): Promise<string> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const response = await fetch(
      `${uploadBaseUrl}/admin/models/global/set-default`,
      {
        method: 'POST',
        credentials: 'include',
        cache: 'no-store',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({ id }),
      },
    );
    if (!response.ok) {
      throw new Error(
        await parseApiErrorMessage(
          response,
          `Set default failed (${response.status}).`,
        ),
      );
    }
    return 'Default baseline updated.';
  };
  const retireAdminGlobalBaseline = async (id: string): Promise<string> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const response = await fetch(
      `${uploadBaseUrl}/admin/models/global/retire`,
      {
        method: 'POST',
        credentials: 'include',
        cache: 'no-store',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({ id }),
      },
    );
    if (!response.ok) {
      throw new Error(
        await parseApiErrorMessage(
          response,
          `Retire failed (${response.status}).`,
        ),
      );
    }
    const payload = (await response.json().catch(() => null)) as {
      replacementDefaultId?: unknown;
    } | null;
    const replacementId =
      typeof payload?.replacementDefaultId === 'string'
        ? payload.replacementDefaultId
        : null;
    return replacementId
      ? `Baseline retired. New default: ${replacementId}.`
      : 'Baseline retired.';
  };
  const publishModelBytesAsGlobalBaseline = async (options: {
    bytes: ArrayBuffer;
    pipelineId: string;
    label?: string;
    setDefault?: boolean;
  }): Promise<string> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const mode = modeController.getState().mode.id;
    const selector = getActiveModelAxes();
    const label =
      typeof options.label === 'string' && options.label.trim()
        ? options.label.trim()
        : '';
    const setDefault = options.setDefault === true;
    const url = new URL(
      `${uploadBaseUrl}/admin/models/global/publish`,
      window.location.origin,
    );
    url.searchParams.set('mode', mode);
    url.searchParams.set('arch', selector.arch);
    url.searchParams.set('reward_profile', selector.rewardProfileId);
    url.searchParams.set('queue_policy', selector.queuePolicyId);
    url.searchParams.set('pipeline_id', options.pipelineId);
    if (label) {
      url.searchParams.set('label', label);
    }
    if (setDefault) {
      url.searchParams.set('set_default', '1');
    }
    const response = await fetch(url.toString(), {
      method: 'POST',
      credentials: 'include',
      cache: 'no-store',
      headers: {
        'content-type': 'application/octet-stream',
      },
      body: options.bytes,
    });
    if (!response.ok) {
      throw new Error(
        await parseApiErrorMessage(
          response,
          `Publish failed (${response.status}).`,
        ),
      );
    }
    const responsePayload = (await response.json().catch(() => null)) as {
      model?: { id?: unknown; label?: unknown } | null;
    } | null;
    const id =
      typeof responsePayload?.model?.id === 'string'
        ? responsePayload.model.id
        : 'unknown';
    const createdLabel =
      typeof responsePayload?.model?.label === 'string'
        ? responsePayload.model.label
        : '';
    return createdLabel
      ? `Published baseline: ${createdLabel} (${id}).`
      : `Published baseline id: ${id}.`;
  };
  const publishCurrentModelAsGlobalBaseline = async (options?: {
    label?: string;
    setDefault?: boolean;
  }): Promise<string> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const bytes = await modelService.ensureModelBytes();
    if (!bytes) {
      throw new Error('No model is loaded to publish.');
    }
    return await publishModelBytesAsGlobalBaseline({
      bytes,
      pipelineId: getLocalTrainingPreset().pipelineId,
      label: options?.label,
      setDefault: options?.setDefault,
    });
  };

  let adminManifestPage: AdminRecordingsManifestPage | null = null;
  let adminManifestSessions: TrajectorySessionV1[] = [];
  let adminGlobalTrainingCandidate: {
    bytes: ArrayBuffer;
    pipelineId: string;
    samplesUsed: number;
    holdoutDelta: number | null;
    finalLoss: number | null;
  } | null = null;
  let adminBotPolicy: BotPolicyArtifact | null = null;
  let adminBotPolicyRecord: AdminBotPolicyRecord | null = null;
  let adminBenchmarkSuggestedArch: 'full' | 'lean' | null = null;
  let botGuiInspectGeneratorBackup: GeneratorType | null = null;

  const prepareAdminManifest = async (
    query: MenuAdminManifestQuery,
  ): Promise<string> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const mode = query.mode?.trim() || modeController.getState().mode.id;
    const selector = getActiveModelAxes();
    const pipeline = getTrainingPipelineForContext(mode, selector);
    const page = await adminRecordingsService.listTrainingManifest({
      mode,
      build: query.build,
      limit: query.limit,
      cursor: query.cursor ?? null,
      minSamples: query.minSamples,
      actorType:
        query.actorType === 'human' || query.actorType === 'bot'
          ? query.actorType
          : undefined,
      arch: selector.arch,
      rewardProfileId: selector.rewardProfileId,
      queuePolicyId: selector.queuePolicyId,
      pipelineId: pipeline.id,
    });
    adminManifestPage = page;
    adminManifestSessions = [];
    adminGlobalTrainingCandidate = null;
    const sampleCount = page.recordings.reduce(
      (sum, recording) => sum + Math.max(0, recording.samples),
      0,
    );
    const next = page.page.nextCursor ? 'yes' : 'no';
    return (
      `Manifest ready: ${page.page.returned} recordings, ${sampleCount} samples, ` +
      `pipeline=${pipeline.id}, next_page=${next}.`
    );
  };

  const loadAdminManifestSessions = async (): Promise<
    TrajectorySessionV1[]
  > => {
    if (!adminManifestPage || adminManifestPage.recordings.length === 0) {
      throw new Error('Manifest is empty. Run PREPARE MANIFEST first.');
    }
    const sessions: TrajectorySessionV1[] = [];
    for (const recording of adminManifestPage.recordings) {
      const session = await adminRecordingsService.loadRecordingObject(
        recording.id,
      );
      sessions.push(session);
    }
    adminManifestSessions = sessions;
    return sessions;
  };

  const runAdminGlobalTrainingOneShot = async (): Promise<string> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const modeId = modeController.getState().mode.id;
    const model = await modelService.ensureLoaded();
    if (!model) {
      throw new Error('Model is not loaded.');
    }
    const selector = getActiveModelAxes();
    const pipeline = getTrainingPipelineForContext(modeId, selector);
    const sessions =
      adminManifestSessions.length > 0
        ? adminManifestSessions
        : await loadAdminManifestSessions();
    const result = await runGlobalTrainingOneShot({
      model,
      recordings: sessions,
      pipeline,
      trainer: personalTrainer,
      train: {
        backendPreference: pipeline.trainDefaults.backendPreference,
        epochs: pipeline.trainDefaults.epochs,
        learningRate: pipeline.trainDefaults.learningRate,
        l2: pipeline.trainDefaults.l2,
        sampleLimit: pipeline.trainDefaults.sampleLimit,
      },
    });
    if (!result.ok || !result.updatedModelBytes) {
      throw new Error(result.message);
    }
    adminGlobalTrainingCandidate = {
      bytes: result.updatedModelBytes,
      pipelineId: result.pipelineId,
      samplesUsed: result.samplesUsed,
      holdoutDelta: result.holdoutDelta,
      finalLoss: result.finalLoss,
    };
    const holdoutSuffix =
      result.holdoutDelta != null
        ? ` holdoutΔ=${result.holdoutDelta.toFixed(6)}`
        : '';
    const lossSuffix =
      result.finalLoss != null
        ? ` finalLoss=${result.finalLoss.toExponential(3)}`
        : '';
    return (
      `Global training candidate ready (${result.pipelineId}). ` +
      `samples=${result.samplesUsed}.${holdoutSuffix}${lossSuffix}`
    );
  };

  const publishAdminGlobalTrainingCandidate = async (): Promise<string> => {
    if (!adminGlobalTrainingCandidate) {
      throw new Error('No global candidate ready. Run TRAIN GLOBAL first.');
    }
    return await publishModelBytesAsGlobalBaseline({
      bytes: adminGlobalTrainingCandidate.bytes,
      pipelineId: adminGlobalTrainingCandidate.pipelineId,
      label: `candidate_${modeController.getState().mode.id}_${Date.now()}`,
      setDefault: false,
    });
  };

  const getBotPolicySelector = () => {
    const modeId = modeController.getState().mode.id;
    const axes = getActiveModelAxes();
    return {
      modeId,
      archId: axes.arch,
      queuePolicyId: axes.queuePolicyId,
    };
  };

  const ensureBotPolicyLoaded = (): {
    policy: BotPolicyArtifact;
    policyId: string;
  } => {
    if (!adminBotPolicy) {
      throw new Error('No bot policy loaded. Train or load one first.');
    }
    return {
      policy: adminBotPolicy,
      policyId: adminBotPolicyRecord?.id ?? adminBotPolicy.id,
    };
  };

  const fetchCurrentAdminBotPolicy = async (): Promise<string> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const selector = getBotPolicySelector();
    const current = await adminBotPolicyService.getCurrent(selector);
    if (!current.current) {
      adminBotPolicyRecord = null;
      return `No published current bot policy for ${selector.modeId}/${selector.archId}/${selector.queuePolicyId}.`;
    }
    const loaded = await adminBotPolicyService.loadObject(current.current.id);
    adminBotPolicyRecord = current.current;
    adminBotPolicy = loaded.artifact;
    adminBotPolicy.id = current.current.id;
    adminBotPolicy.modeId = current.current.modeId;
    return `Loaded current bot policy: ${current.current.id} (v${current.current.version}).`;
  };

  const listAdminBotPolicies = async (options?: {
    limit?: number;
    cursor?: string | null;
  }): Promise<MenuBotPoliciesPage> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const page = await adminBotPolicyService.list(getBotPolicySelector(), {
      limit: options?.limit,
      cursor: options?.cursor ?? null,
    });
    return {
      policies: page.policies.map((policy) => ({
        id: policy.id,
        modeId: policy.modeId,
        archId: policy.archId,
        queuePolicyId: policy.queuePolicyId,
        pipelineId: policy.pipelineId,
        pieceSourceProfile: policy.pieceSourceProfile,
        version: policy.version,
        isPinned: policy.isPinned,
        createdAtMs: policy.createdAtMs,
      })),
      page: {
        returned: page.page.returned,
        nextCursor: page.page.nextCursor,
      },
    };
  };

  const loadAdminBotPolicyById = async (id: string): Promise<string> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const loaded = await adminBotPolicyService.loadObject(id);
    adminBotPolicyRecord = loaded.policy;
    adminBotPolicy = loaded.artifact;
    adminBotPolicy.id = loaded.policy.id;
    adminBotPolicy.modeId = loaded.policy.modeId;
    return `Loaded bot policy ${loaded.policy.id} (v${loaded.policy.version}).`;
  };

  const publishAdminBotPolicy = async (options?: {
    pin?: boolean;
    setCurrent?: boolean;
    pipelineId?: string;
    pieceSourceProfile?: BotPieceSourceProfile;
    metrics?: Record<string, unknown> | null;
  }): Promise<string> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    if (!adminBotPolicy) {
      throw new Error('No bot policy loaded to publish.');
    }
    const pipelineId =
      options?.pipelineId ?? adminBotPolicy.pipelineId ?? 'bot_reinforce_v2';
    const selector = getBotPolicySelector();
    const published = await adminBotPolicyService.publish({
      selector,
      policyArtifact: {
        ...adminBotPolicy,
        id: adminBotPolicyRecord?.id ?? adminBotPolicy.id,
        modeId: selector.modeId,
        archId: selector.archId,
        queuePolicyId: selector.queuePolicyId,
        pipelineId,
        pieceSourceProfile: options?.pieceSourceProfile ?? 'bag7',
      },
      pipelineId,
      pieceSourceProfile: options?.pieceSourceProfile ?? 'bag7',
      metrics: options?.metrics ?? null,
      pin: options?.pin === true,
      setCurrent: options?.setCurrent !== false,
    });
    adminBotPolicyRecord = published;
    adminBotPolicy.id = published.id;
    adminBotPolicy.modeId = published.modeId;
    adminBotPolicy.archId = published.archId;
    adminBotPolicy.queuePolicyId = published.queuePolicyId;
    adminBotPolicy.pipelineId = published.pipelineId;
    adminBotPolicy.pieceSourceProfile =
      (published.pieceSourceProfile as BotPieceSourceProfile) ?? 'bag7';
    return `Published bot policy ${published.id} (v${published.version}).`;
  };

  const selectAdminBotPolicyAsCurrent = async (id: string): Promise<string> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const selected = await adminBotPolicyService.selectCurrent(id);
    adminBotPolicyRecord = selected;
    if (!adminBotPolicy || adminBotPolicy.id !== selected.id) {
      const loaded = await adminBotPolicyService.loadObject(selected.id);
      adminBotPolicy = loaded.artifact;
      adminBotPolicy.id = selected.id;
      adminBotPolicy.modeId = selected.modeId;
    }
    return `Selected current policy ${selected.id}.`;
  };

  const setAdminBotPolicyPinState = async (
    id: string,
    pin: boolean,
  ): Promise<string> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const record = pin
      ? await adminBotPolicyService.pin(id)
      : await adminBotPolicyService.unpin(id);
    if (adminBotPolicyRecord?.id === id) {
      adminBotPolicyRecord = record;
    }
    return pin ? `Pinned ${id}.` : `Unpinned ${id}.`;
  };

  const runAdminTrainBotPolicyOneShot = async (options: {
    episodes?: number;
    maxPiecesPerEpisode?: number;
    seed?: number;
    pieceSourceProfile?: BotPieceSourceProfile;
    warmStartFromLoaded?: boolean;
    algorithm?: BotTrainingAlgorithm;
  }): Promise<string> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const modeId = modeController.getState().mode.id;
    const reference = await loadBotReferenceModel(modeId);
    const initialPolicy =
      options.warmStartFromLoaded === false ? null : adminBotPolicy;
    const result = await trainBotPolicyOneShot({
      modeId,
      settings: settingsStore.get(),
      model: reference.model,
      modelRunner: botModelRunner,
      modelAxes: getActiveModelAxes(),
      initialPolicy,
      algorithm: options.algorithm ?? 'reinforce',
      episodes: options.episodes,
      maxPiecesPerEpisode: options.maxPiecesPerEpisode,
      seed: options.seed,
      pieceSourceProfile: options.pieceSourceProfile ?? 'bag7',
    });
    if (!result.ok || !result.policyArtifact) {
      throw new Error(result.message);
    }
    adminBotPolicy = result.policyArtifact;
    adminBotPolicyRecord = null;
    const lossSuffix =
      result.finalLoss != null
        ? ` finalLoss=${result.finalLoss.toExponential(3)}`
        : '';
    return (
      `${result.message} meanReturn=${result.meanReturn.toFixed(4)}${lossSuffix}\n` +
      `policyId=${result.policyArtifact.id} baseGlobal=${reference.modelId}`
    );
  };

  const runAdminGenerateBotRecordings = async (options: {
    sessions?: number;
    maxPiecesPerEpisode?: number;
    pieceSourceProfile?: BotPieceSourceProfile;
  }): Promise<string> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const { policy, policyId } = ensureBotPolicyLoaded();
    const modeId = modeController.getState().mode.id;
    const reference = await loadBotReferenceModel(modeId);
    const generated = await generateBotTrajectoryBatch({
      modeId,
      settings: settingsStore.get(),
      model: reference.model,
      modelRunner: botModelRunner,
      modelAxes: getActiveModelAxes(),
      policy,
      sessions: Math.max(1, options.sessions ?? 1),
      maxPiecesPerEpisode: options.maxPiecesPerEpisode,
      trainingIntent: 'bot_generation_v1',
      pieceSourceProfile: options.pieceSourceProfile ?? 'active_generator',
    });

    let uploadedSessions = 0;
    let uploadedSamples = 0;
    const axes = getActiveModelAxes();
    const pipeline = getTrainingPipelineForContext(modeId, axes);
    const modelArch = getTrajectoryModelArch();
    for (const draft of generated.drafts) {
      if (draft.samples.length < MIN_TRAJECTORY_SAMPLES_PER_SESSION) {
        continue;
      }
      const rewardPolicyId = resolveTrajectoryRewardPolicyId(draft.modeId);
      const rewardInput: TrajectoryDecisionSample[] = draft.samples.map(
        (sample) => ({
          schema: 'wishuponablock.trajectory.v1',
          id: sample.id,
          modeId: draft.modeId,
          arch: axes.arch,
          rewardProfileId: axes.rewardProfileId,
          queuePolicyId: axes.queuePolicyId,
          createdAtMs: sample.createdAtMs,
          deliberationMs: sample.deliberationMs,
          boardOccupancy: sample.boardOccupancy.map((row) => row.slice()),
          hold: sample.hold,
          action: sample.action,
          actionIndex: sample.actionIndex,
          pieces: [...sample.pieces],
          logits: [...sample.logits],
          probabilities: [...sample.probabilities],
          inferenceMs: sample.inferenceMs,
          samplingMs: sample.samplingMs,
          totalDecisionMs: sample.totalDecisionMs,
          reward: sample.reward,
          replay: sample.replay
            ? {
                lockPiece: sample.replay.lockPiece,
                lockRotation: sample.replay.lockRotation,
                lockX: sample.replay.lockX,
                lockY: sample.replay.lockY,
                holdUsed: sample.replay.holdUsed,
                gameTimeMs: sample.replay.gameTimeMs,
                totalLinesCleared: sample.replay.totalLinesCleared,
                score: sample.replay.score,
              }
            : undefined,
        }),
      );
      const rewards = computeTrajectoryRewards(
        rewardInput,
        {
          modeId: draft.modeId,
          outcome: draft.outcome,
          terminal: draft.terminal,
        },
        rewardPolicyId,
      );
      const sessionId = crypto.randomUUID();
      const pieceSourceProfile =
        options.pieceSourceProfile ??
        policy.pieceSourceProfile ??
        'active_generator';
      const session: TrajectorySessionV1 = {
        schema: TRAJECTORY_SESSION_SCHEMA_V1,
        sessionId,
        modeId: draft.modeId,
        buildVersion: APP_VERSION,
        startedAtMs: draft.startedAtMs,
        endedAtMs: draft.endedAtMs,
        durationMs: draft.durationMs,
        samples: draft.samples.map((sample, index) => ({
          ...sample,
          boardOccupancy: sample.boardOccupancy.map((row) => row.slice()),
          pieces: [...sample.pieces],
          logits: [...sample.logits],
          probabilities: [...sample.probabilities],
          reward: rewards.rewards[index] ?? null,
          replay: sample.replay
            ? {
                lockPiece: sample.replay.lockPiece,
                lockRotation: sample.replay.lockRotation,
                lockX: sample.replay.lockX,
                lockY: sample.replay.lockY,
                holdUsed: sample.replay.holdUsed,
                gameTimeMs: sample.replay.gameTimeMs,
                totalLinesCleared: sample.replay.totalLinesCleared,
                score: sample.replay.score,
              }
            : undefined,
        })),
        meta: {
          outcome: draft.outcome,
          channel: import.meta.env.MODE,
          modelSource: activeModelSource.kind,
          modelMode:
            activeModelSource.kind === 'personal'
              ? activeModelSource.mode
              : undefined,
          modelVersion:
            activeModelSource.kind === 'personal'
              ? activeModelSource.version
              : null,
          modelArchId: axes.arch,
          rewardProfileId: axes.rewardProfileId,
          queuePolicyId: axes.queuePolicyId,
          rewardPolicy: rewards.policyId,
          rewardPolicyId: rewards.policyId,
          rewardKind: rewards.kind,
          rewardGamma: rewards.gamma,
          pipelineId: pipeline.id,
          pipelineMode: draft.modeId,
          modelArch,
          actorType: 'bot',
          actorPolicyId: policyId,
          trainingIntent: generated.trainingIntent ?? undefined,
          pieceSourceProfile,
        },
      };
      await trajectoryRecordingService.uploadSession(session);
      uploadedSessions += 1;
      uploadedSamples += session.samples.length;
    }
    return (
      `Bot generation complete. sessions=${uploadedSessions}, samples=${uploadedSamples}, ` +
      `policy=${policyId}.`
    );
  };

  const runAdminCapabilityBenchmark = async (options?: {
    episodes?: number;
    maxPiecesPerEpisode?: number;
    pieceSourceProfile?: BotPieceSourceProfile;
  }): Promise<string> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const { policy } = ensureBotPolicyLoaded();
    const modeId = modeController.getState().mode.id;
    const reference = await loadBotReferenceModel(modeId);
    const result = await runCapabilityBenchmark({
      modeId,
      settings: settingsStore.get(),
      model: reference.model,
      modelRunner: botModelRunner,
      modelAxes: getActiveModelAxes(),
      policy,
      episodes: options?.episodes,
      maxPiecesPerEpisode: options?.maxPiecesPerEpisode,
      pieceSourceProfile: options?.pieceSourceProfile ?? 'bag7',
    });
    adminBenchmarkSuggestedArch = result.recommendedArch;
    return (
      `Benchmark: verdict=${result.verdict}, recommend=${result.recommendedArch}, ` +
      `decision p95=${result.metrics.p95DecisionMs.toFixed(2)}ms, ` +
      `tick p95=${result.metrics.p95TickMs.toFixed(2)}ms, ` +
      `sim=${(result.metrics.simThroughput * 100).toFixed(1)}% of ${result.metrics.targetSimFps}fps.`
    );
  };

  const runAdminHeadlessBotValidation = async (options?: {
    maxPieces?: number;
    seed?: number;
    pieceSourceProfile?: BotPieceSourceProfile;
  }): Promise<string> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const { policy } = ensureBotPolicyLoaded();
    const modeId = modeController.getState().mode.id;
    const reference = await loadBotReferenceModel(modeId);
    const validation = await runHeadlessBotValidation({
      modeId,
      settings: settingsStore.get(),
      model: reference.model,
      modelRunner: botModelRunner,
      modelAxes: getActiveModelAxes(),
      policy,
      maxPieces: options?.maxPieces ?? 10_000,
      seed: options?.seed,
      pieceSourceProfile: options?.pieceSourceProfile ?? 'bag7',
    });
    return (
      `Headless validate: pieces=${validation.piecesSurvived}, outcome=${validation.outcome}, ` +
      `decision p95=${validation.p95DecisionMs.toFixed(2)}ms, ` +
      `tick p95=${validation.p95TickMs.toFixed(2)}ms, ` +
      `meanΔscore=${validation.meanBoardScoreDelta.toFixed(4)}, ` +
      `totalReward=${validation.totalReward.toFixed(3)}.`
    );
  };

  const applyBotGuiPieceSourceProfile = (
    profile: BotPieceSourceProfile,
  ): void => {
    const current = settingsStore.get();
    if (profile === 'bag7') {
      if (botGuiInspectGeneratorBackup == null) {
        botGuiInspectGeneratorBackup = current.generator.type;
      }
      if (current.generator.type !== 'bag7') {
        settingsStore.apply({
          generator: {
            ...current.generator,
            type: 'bag7',
          },
        });
      }
      return;
    }
    if (botGuiInspectGeneratorBackup) {
      settingsStore.apply({
        generator: {
          ...current.generator,
          type: botGuiInspectGeneratorBackup,
        },
      });
      botGuiInspectGeneratorBackup = null;
    }
  };

  const startAdminBotGuiInspect = async (options?: {
    apmInput?: number;
    seed?: number;
    pieceSourceProfile?: BotPieceSourceProfile;
    pieces?: number;
    greedy?: boolean;
  }): Promise<string> => {
    if (
      !authState.authenticated ||
      !authState.user ||
      !authState.user.isAdmin
    ) {
      throw new Error('Admin account required.');
    }
    const { policy, policyId } = ensureBotPolicyLoaded();
    const modeId = modeController.getState().mode.id;
    const reference = await loadBotReferenceModel(modeId);
    const apmInput = Math.max(
      20,
      Math.min(1200, Math.trunc(options?.apmInput ?? 240)),
    );
    const seed = options?.seed ?? 42_030;
    const pieceSourceProfile =
      options?.pieceSourceProfile ?? policy.pieceSourceProfile ?? 'bag7';
    botGuiInputSource = createGuiInspectBotInputSource({
      model: reference.model,
      policy,
      apmInput,
      seed,
      greedy: options?.greedy !== false,
    });
    botGuiInspectEnabled = true;
    applyBotGuiPieceSourceProfile(pieceSourceProfile);
    modeController.startCharcuterie(Math.max(1, options?.pieces ?? 20), {
      simCount: charcuterieDefaultSimCount,
      ...(Number.isFinite(seed) ? { seed: Math.trunc(seed) } : {}),
    });
    requestStartGame();
    return (
      `GUI inspect started for ${policyId}. ` +
      `APM=${apmInput}, piece_source=${pieceSourceProfile}, ` +
      `policy_mode=${options?.greedy === false ? 'sampled' : 'greedy'}.`
    );
  };

  const stopAdminBotGuiInspect = (): string => {
    botGuiInspectEnabled = false;
    botGuiInputSource = null;
    applyBotGuiPieceSourceProfile('active_generator');
    return 'GUI inspect stopped.';
  };

  const applyAdminBenchmarkSuggestedArch = async (): Promise<string> => {
    if (!adminBenchmarkSuggestedArch) {
      throw new Error('No benchmark recommendation yet.');
    }
    const current = getActiveModelAxes();
    if (current.arch === adminBenchmarkSuggestedArch) {
      return `Model arch already "${current.arch}".`;
    }
    const next = {
      ...current,
      arch: adminBenchmarkSuggestedArch,
    };
    settingsStore.apply({ modelAxes: next });
    const mode = modeController.getState().mode.id;
    const sync = await syncActiveModelForContext({
      mode,
      reason: 'benchmark_arch_apply',
      force: true,
      interactive: true,
    });
    return (
      `Applied benchmark recommendation: arch=${adminBenchmarkSuggestedArch}. ` +
      sync.message
    );
  };
  const getMenuModelAxes = (): MenuModelAxes => {
    const axes = getActiveModelAxes();
    return {
      arch: axes.arch,
      rewardProfileId: axes.rewardProfileId,
      queuePolicyId: axes.queuePolicyId,
    };
  };
  const setMenuModelAxes = async (next: MenuModelAxes): Promise<string> => {
    const current = getActiveModelAxes();
    const normalized = normalizeModelAxes(next);
    if (modelAxesKey(current) === modelAxesKey(normalized)) {
      return 'Model axes unchanged.';
    }
    settingsStore.apply({ modelAxes: normalized });
    const mode = modeController.getState().mode.id;
    const result = await syncActiveModelForContext({
      mode,
      reason: 'model_axes_change',
      force: true,
    });
    return `Model axes updated (${normalized.arch}/${normalized.rewardProfileId}/${normalized.queuePolicyId}). ${result.message}`;
  };

  type TrajectoryRunState = {
    sessionId: string;
    modeId: string;
    startedAtMs: number;
    axes: {
      arch: string;
      rewardProfileId: string;
      queuePolicyId: string;
    };
    pipelineId: string;
    minSamplesForUpload: number;
    rewardPolicyId: TrajectoryRewardPolicyId;
  };
  let activeTrajectoryRun: TrajectoryRunState | null = null;
  let pendingTrajectoryRunModeId: string | null = null;
  let pendingTrajectorySession: TrajectorySessionV1 | null = null;
  let pendingTrajectoryReplayStep: TrajectoryReplayStepV1 | null = null;
  let holdUsedSinceLastLock = false;
  let lastTrajectoryMinSamplesRequired =
    DEFAULT_MIN_TRAJECTORY_SAMPLES_FOR_UPLOAD;
  let lastTrajectoryUploadMessage = 'No uploads yet.';
  let lastTrajectoryUploadAtMs: number | null = null;
  let lastTrajectoryUploadSamples = 0;
  let lastTrajectoryUploadError: string | null = null;

  const getTrajectoryModelArch = (): string => {
    const model = modelService.getModel();
    if (!model) return 'unknown';
    const cfg = model.config;
    const conv = cfg.conv_channels.map((value) => Math.trunc(value)).join('x');
    const poolShape =
      cfg.pool_shape && cfg.pool_shape.length >= 2
        ? `${Math.trunc(cfg.pool_shape[0])}x${Math.trunc(cfg.pool_shape[1])}`
        : '1x1';
    return `ic${Math.trunc(cfg.input_channels)}_conv${conv}_pool${poolShape}_mlp${Math.trunc(cfg.mlp_hidden)}_extra${Math.trunc(cfg.extra_features)}_out${Math.trunc(cfg.num_outputs)}`;
  };

  const toTrajectoryMeta = (
    run: TrajectoryRunState,
    outcome: string,
    rewards: TrajectoryRewardComputation | null,
  ): TrajectorySessionMetaV1 => {
    const meta: TrajectorySessionMetaV1 = {
      outcome,
      channel: import.meta.env.MODE,
      modelSource: activeModelSource.kind,
      pipelineId: run.pipelineId,
      pipelineMode: run.modeId,
      modelArchId: run.axes.arch,
      rewardProfileId: run.axes.rewardProfileId,
      queuePolicyId: run.axes.queuePolicyId,
      modelArch: getTrajectoryModelArch(),
      pieceSourceProfile:
        settingsStore.get().generator.type === 'bag7'
          ? 'bag7'
          : 'active_generator',
    };
    if (activeModelSource.kind === 'personal') {
      meta.modelMode = activeModelSource.mode;
      if (activeModelSource.version != null) {
        meta.modelVersion = activeModelSource.version;
      }
    }
    if (rewards) {
      meta.rewardPolicy = rewards.policyId;
      meta.rewardPolicyId = rewards.policyId;
      meta.rewardKind = rewards.kind;
      meta.rewardGamma = rewards.gamma;
    }
    return meta;
  };

  const toTrajectorySamples = (
    samples: TrajectoryDecisionSample[],
    rewards: number[],
  ): TrajectorySessionV1['samples'] =>
    samples.map((sample, index) => {
      const reward = rewards[index];
      return {
        id: sample.id,
        createdAtMs: sample.createdAtMs,
        deliberationMs: sample.deliberationMs,
        boardOccupancy: sample.boardOccupancy.map((row) => row.slice()),
        hold: sample.hold,
        action: sample.action,
        actionIndex: sample.actionIndex,
        pieces: [...sample.pieces],
        logits: [...sample.logits],
        probabilities: [...sample.probabilities],
        inferenceMs: sample.inferenceMs,
        samplingMs: sample.samplingMs,
        totalDecisionMs: sample.totalDecisionMs,
        reward:
          typeof reward === 'number' && Number.isFinite(reward)
            ? reward
            : (sample.reward ?? null),
        ...(sample.replay
          ? {
              replay: {
                lockPiece: sample.replay.lockPiece,
                lockRotation: sample.replay.lockRotation,
                lockX: sample.replay.lockX,
                lockY: sample.replay.lockY,
                holdUsed: sample.replay.holdUsed,
                gameTimeMs: sample.replay.gameTimeMs,
                totalLinesCleared: sample.replay.totalLinesCleared,
                score: sample.replay.score,
              },
            }
          : {}),
      };
    });

  const beginTrajectoryRun = (modeId: string): void => {
    const axes = getActiveModelAxes();
    const pipeline = getTrainingPipelineForContext(modeId, axes);
    activeTrajectoryRun = {
      sessionId: crypto.randomUUID(),
      modeId,
      startedAtMs: Date.now(),
      axes,
      pipelineId: pipeline.id,
      minSamplesForUpload: Math.max(
        MIN_TRAJECTORY_SAMPLES_PER_SESSION,
        pipeline.minSamples,
      ),
      rewardPolicyId: getTrajectoryRewardPolicyId(modeId, axes),
    };
  };

  const scheduleTrajectoryRunStart = (modeId: string): void => {
    pendingTrajectoryRunModeId = modeId;
  };

  const listRunSamples = (
    run: TrajectoryRunState,
  ): TrajectoryDecisionSample[] =>
    trajectoryBuffer
      .listSamples({
        modeId: run.modeId,
        modelAxes: run.axes,
      })
      .filter((sample) => sample.createdAtMs >= run.startedAtMs);

  const buildTrajectorySession = (
    run: TrajectoryRunState,
    outcome: string,
    terminal: TrajectoryRewardTerminalStats | null = null,
  ): TrajectorySessionV1 | null => {
    const runSamples = listRunSamples(run);
    if (runSamples.length < run.minSamplesForUpload) return null;
    const rewards = computeTrajectoryRewards(
      runSamples,
      {
        modeId: run.modeId,
        outcome,
        terminal,
      },
      run.rewardPolicyId,
    );
    const endedAtMs = Math.max(
      Date.now(),
      runSamples[runSamples.length - 1].createdAtMs,
    );
    return {
      schema: TRAJECTORY_SESSION_SCHEMA_V1,
      sessionId: run.sessionId,
      modeId: run.modeId,
      buildVersion: APP_VERSION,
      startedAtMs: run.startedAtMs,
      endedAtMs,
      durationMs: Math.max(0, endedAtMs - run.startedAtMs),
      samples: toTrajectorySamples(runSamples, rewards.rewards),
      meta: toTrajectoryMeta(run, outcome, rewards),
    };
  };

  const finalizeTrajectoryRun = (
    outcome: string,
    terminal: TrajectoryRewardTerminalStats | null = null,
  ): TrajectorySessionV1 | null => {
    const run = activeTrajectoryRun;
    activeTrajectoryRun = null;
    pendingTrajectoryRunModeId = null;
    pendingTrajectoryReplayStep = null;
    holdUsedSinceLastLock = false;
    if (!run) return null;
    lastTrajectoryMinSamplesRequired = run.minSamplesForUpload;
    const session = buildTrajectorySession(run, outcome, terminal);
    if (session) {
      pendingTrajectorySession = session;
    }
    return session;
  };

  const uploadTrajectorySession = async (
    session: TrajectorySessionV1,
  ): Promise<string> => {
    const uploaded = await trajectoryRecordingService.uploadSession(session);
    if (pendingTrajectorySession?.sessionId === session.sessionId) {
      pendingTrajectorySession = null;
    }
    lastTrajectoryUploadAtMs = uploaded.createdAtMs;
    lastTrajectoryUploadSamples = uploaded.samples;
    lastTrajectoryUploadError = null;
    lastTrajectoryUploadMessage = `Uploaded ${uploaded.samples} samples for mode "${uploaded.mode}".`;
    return `${lastTrajectoryUploadMessage} (id: ${uploaded.id})`;
  };

  const finalizeAndUploadTrajectoryRun = async (
    outcome: string,
    terminal: TrajectoryRewardTerminalStats | null = null,
  ): Promise<string> => {
    const session = finalizeTrajectoryRun(outcome, terminal);
    if (!session) {
      lastTrajectoryUploadMessage =
        'No trajectory samples captured for this run.';
      return lastTrajectoryUploadMessage;
    }
    try {
      return await uploadTrajectorySession(session);
    } catch (error) {
      const message = toErrorMessage(error, 'Trajectory upload failed.');
      lastTrajectoryUploadError = message;
      lastTrajectoryUploadMessage = `Upload failed: ${message}`;
      throw error;
    }
  };

  const uploadLatestTrajectory = async (): Promise<string> => {
    let candidate = pendingTrajectorySession;
    if (!candidate && activeTrajectoryRun) {
      candidate = buildTrajectorySession(activeTrajectoryRun, 'manual');
    }
    if (!candidate) {
      if (activeTrajectoryRun) {
        const count = listRunSamples(activeTrajectoryRun).length;
        throw new Error(
          `Need at least ${activeTrajectoryRun.minSamplesForUpload} samples before upload (currently ${count}).`,
        );
      }
      throw new Error('No trajectory recording available to upload.');
    }
    try {
      return await uploadTrajectorySession(candidate);
    } catch (error) {
      const message = toErrorMessage(error, 'Trajectory upload failed.');
      lastTrajectoryUploadError = message;
      lastTrajectoryUploadMessage = `Upload failed: ${message}`;
      throw error;
    }
  };

  const toTrajectoryTerminalStats = (
    state: GameState,
  ): TrajectoryRewardTerminalStats => ({
    totalLinesCleared: Math.max(0, Math.trunc(state.totalLinesCleared)),
    score: Math.max(0, Math.trunc(state.score)),
    timeMs: Math.max(0, Math.trunc(state.timeMs)),
  });

  const getTrajectoryUploadSummary = (): string => {
    const lines: string[] = [];
    const run = activeTrajectoryRun;
    if (run) {
      const activeSamples = listRunSamples(run).length;
      lines.push(`Active run: ${run.modeId} (${activeSamples} samples)`);
    } else {
      lines.push('Active run: none');
    }
    if (pendingTrajectorySession) {
      lines.push(
        `Pending upload: ${pendingTrajectorySession.modeId} (${pendingTrajectorySession.samples.length} samples)`,
      );
    } else {
      lines.push('Pending upload: none');
    }
    lines.push(`Last upload: ${lastTrajectoryUploadMessage}`);
    if (lastTrajectoryUploadAtMs != null) {
      lines.push(
        `Uploaded at: ${new Date(lastTrajectoryUploadAtMs).toLocaleTimeString()} (${lastTrajectoryUploadSamples} samples)`,
      );
    }
    if (lastTrajectoryUploadError) {
      lines.push(`Last error: ${lastTrajectoryUploadError}`);
    }
    return lines.join('\n');
  };

  const runLocalHeadTraining = async (options?: {
    modeId?: string;
    sampleLimit?: number;
    epochs?: number;
    learningRate?: number;
    l2?: number;
    backendPreference?: 'auto' | 'webgl' | 'cpu';
  }): Promise<{
    ok: boolean;
    message: string;
    samplesUsed: number;
    finalLoss: number | null;
  }> => {
    const model = await modelService.ensureLoaded();
    if (!model) {
      return {
        ok: false,
        message: 'Model is not loaded.',
        samplesUsed: 0,
        finalLoss: null,
      };
    }
    const modeId = options?.modeId ?? modeController.getState().mode.id;
    const axes = getActiveModelAxes();
    const trainingPipeline = getTrainingPipelineForContext(modeId, axes);
    const samples = trajectoryBuffer.listSamples({
      modeId,
      modelAxes: axes,
      limit: options?.sampleLimit,
    });
    const result = await personalTrainer.trainHeadOnly({
      model,
      samples,
      pipeline: trainingPipeline,
      train: {
        sampleLimit: options?.sampleLimit,
        epochs: options?.epochs,
        learningRate: options?.learningRate,
        l2: options?.l2,
        backendPreference: options?.backendPreference,
      },
    });
    if (!result.ok || !result.updatedModelBytes) {
      return {
        ok: false,
        message: result.message,
        samplesUsed: result.samplesUsed,
        finalLoss: result.finalLoss,
      };
    }
    await modelService.replaceModelFromBytes(
      result.updatedModelBytes,
      `local ${result.pipelineId} training`,
    );
    sessionController.rebuildSession();
    updateModelStatusUI(modelService.getStatus());
    return {
      ok: true,
      message: result.message,
      samplesUsed: result.samplesUsed,
      finalLoss: result.finalLoss,
    };
  };
  const getLocalTrainingPreset = () => {
    const modeId = modeController.getState().mode.id;
    const pipeline = getTrainingPipelineForContext(
      modeId,
      getActiveModelAxes(),
    );
    return {
      pipelineId: pipeline.id,
      modeId,
      minSamples: pipeline.minSamples,
      trainDefaults: { ...pipeline.trainDefaults },
      evalGate: { ...pipeline.evalGate },
    };
  };
  const getLocalTrainingStats = () => {
    const stats = trajectoryBuffer.getStats();
    const currentModeId = modeController.getState().mode.id;
    const axes = getActiveModelAxes();
    const modeAxesKey = `${currentModeId}:${modelAxesKey(axes)}`;
    return {
      currentModeId,
      currentModeAxesSamples: stats.byModeAndAxes[modeAxesKey] ?? 0,
      currentModeSamples: stats.byMode[currentModeId] ?? 0,
      totalSamples: stats.totalSamples,
      lastSampleAtMs: stats.lastSampleAtMs,
    };
  };
  applyAuthState(authState);

  const soundService = createSoundService({ settings });
  let suppressLockEffects = false;
  let snapshotService: SnapshotService | null = null;
  const startRecordingSession = () => {
    snapshotService?.start();
  };
  const stopRecordingSession = (options?: {
    promptForFolder?: boolean;
  }): Promise<void> => snapshotService?.stop(options) ?? Promise.resolve();
  const restartRecordingSession = () => {
    snapshotService?.restart();
  };

  let pendingLineClearSound = false;
  const handlePieceLock = (board: Board, hold: PieceKind | null) => {
    if (suppressLockEffects) return;
    if (!pendingLineClearSound) {
      soundService.playLock();
    }
    pendingLineClearSound = false;
    const state = session.getGame().state;
    pendingTrajectoryReplayStep = {
      lockPiece: state.active.k,
      lockRotation: Math.max(0, Math.min(3, Math.trunc(state.active.r))),
      lockX: Math.trunc(state.active.x),
      lockY: Math.trunc(state.active.y),
      holdUsed: holdUsedSinceLastLock,
      gameTimeMs: Math.max(0, Math.trunc(state.timeMs)),
      totalLinesCleared: Math.max(0, Math.trunc(state.totalLinesCleared)),
      score: Math.max(0, Math.trunc(state.score)),
    };
    holdUsedSinceLastLock = false;
    const linesLeft =
      state.lineGoal != null
        ? Math.max(0, state.lineGoal - state.totalLinesCleared)
        : undefined;
    snapshotService?.handleLock(board, hold, {
      active: state.active,
      next: state.next,
      odds: state.mlQueueProbabilities,
      linesLeft,
      level: state.level,
      score: state.score,
    });
  };
  const handleLineClear = (combo: number) => {
    if (suppressLockEffects) return;
    pendingLineClearSound = true;
    soundService.playCombo(combo);
  };
  const handleHoldSnapshot = (board: Board, hold: PieceKind | null) => {
    if (suppressLockEffects) return;
    holdUsedSinceLastLock = true;
    const state = session.getGame().state;
    const linesLeft =
      state.lineGoal != null
        ? Math.max(0, state.lineGoal - state.totalLinesCleared)
        : undefined;
    snapshotService?.handleHold(board, hold, {
      active: state.active,
      next: state.next,
      odds: state.mlQueueProbabilities,
      linesLeft,
      level: state.level,
      score: state.score,
    });
  };

  const initialModeState = modeController.getState();
  let currentModeForModelSync = initialModeState.mode.id;
  const session: GameSession = createGameSessionFactory({
    settings,
    initialMode: initialModeState.mode,
    initialModeOptions: initialModeState.options,
    modelService,
    onPieceLock: handlePieceLock,
    onHold: handleHoldSnapshot,
    onLineClear: handleLineClear,
    onBeforeRestart: () => {
      restartRecordingSession();
      previousRunEnded = false;
      activeTrajectoryRun = null;
      pendingTrajectoryReplayStep = null;
      holdUsedSinceLastLock = false;
      scheduleTrajectoryRunStart(modeController.getState().mode.id);
    },
    onModelDecision: (decision) => {
      const modeId = modeController.getState().mode.id;
      const replay = pendingTrajectoryReplayStep;
      pendingTrajectoryReplayStep = null;
      trajectoryBuffer.recordDecision({
        modeId,
        modelAxes: getActiveModelAxes(),
        decision,
        replay,
      });
    },
    setLockEffectsSuppressed: (value) => {
      suppressLockEffects = value;
    },
    charcuterie: {
      rows: ROWS,
      defaultSimCount: charcuterieDefaultSimCount,
      scoreWeights: charcuterieScoreWeights,
      holeWeights: charcuterieHoleWeights,
      onDebug: (message) => console.info(message),
    },
  });
  const sessionController = createSessionController({
    settings,
    session,
    onRebuild: () => {
      runtime?.renderNow();
    },
  });
  const runMlParityCheck = async (
    preference: MenuMlBackendPreference,
  ): Promise<MenuMlParityResult> => {
    const model = await modelService.ensureLoaded();
    if (!model) {
      return { ok: false, message: 'Model is not loaded.' };
    }
    const tfjsPreferenceForParity =
      preference === 'tfjs_webgl'
        ? 'webgl'
        : preference === 'tfjs_cpu'
          ? 'cpu'
          : 'auto';
    const nativeRunner = createModelRunner({
      preferredBackend: 'native',
      tfjsBackendPreference: 'auto',
    }).runner;
    const tfjsRunner = createModelRunner({
      preferredBackend: 'tfjs',
      tfjsBackendPreference: tfjsPreferenceForParity,
    }).runner;

    await nativeRunner.prepare(model);
    await tfjsRunner.prepare(model);
    const tfjsInfo = tfjsRunner.getInfo();
    if (tfjsInfo.activeBackend !== 'tfjs') {
      return {
        ok: false,
        message: `TFJS unavailable: ${tfjsInfo.fallbackReason ?? 'unknown error.'}`,
      };
    }

    const state = session.getGame().state;
    const nativeLogits = nativeRunner.predictLogits(
      model,
      state.board,
      state.hold,
    );
    const tfjsLogits = tfjsRunner.predictLogits(model, state.board, state.hold);
    const len = Math.min(nativeLogits.length, tfjsLogits.length);
    if (len === 0) {
      return { ok: false, message: 'Parity check failed: empty logits.' };
    }

    let maxAbsDiff = 0;
    let sumAbsDiff = 0;
    let nativeBestIndex = 0;
    let tfjsBestIndex = 0;
    for (let i = 0; i < len; i++) {
      const absDiff = Math.abs(nativeLogits[i] - tfjsLogits[i]);
      sumAbsDiff += absDiff;
      if (absDiff > maxAbsDiff) maxAbsDiff = absDiff;
      if (nativeLogits[i] > nativeLogits[nativeBestIndex]) nativeBestIndex = i;
      if (tfjsLogits[i] > tfjsLogits[tfjsBestIndex]) tfjsBestIndex = i;
    }
    const meanAbsDiff = sumAbsDiff / len;
    const nativeBestPiece = model.pieces[nativeBestIndex] ?? '?';
    const tfjsBestPiece = model.pieces[tfjsBestIndex] ?? '?';
    const tfBackend = tfjsInfo.runtimeBackend ?? 'tfjs';
    const matchesTop1 = nativeBestIndex === tfjsBestIndex;
    const withinTolerance = maxAbsDiff <= 1e-3;

    return {
      ok: withinTolerance && matchesTop1,
      message:
        `TFJS backend: ${tfBackend}\n` +
        `max|Δ|=${maxAbsDiff.toExponential(3)} mean|Δ|=${meanAbsDiff.toExponential(3)}\n` +
        `top1 native=${nativeBestPiece} tfjs=${tfjsBestPiece}`,
    };
  };

  const gameRenderer = new PixiRenderer(gameGfx);
  const toolRenderer = new PixiRenderer(toolGfx);
  gameRenderer.setGridlineOpacity(settings.graphics.gridlineOpacity);
  gameRenderer.setGhostOpacity(settings.graphics.ghostOpacity);
  gameRenderer.setHighContrast(settings.graphics.highContrast);
  gameRenderer.setColorblindMode(settings.graphics.colorblindMode);
  toolRenderer.setGridlineOpacity(settings.graphics.gridlineOpacity);
  toolRenderer.setGhostOpacity(settings.graphics.ghostOpacity);
  toolRenderer.setHighContrast(settings.graphics.highContrast);
  toolRenderer.setColorblindMode(settings.graphics.colorblindMode);
  const toolCanvas = createToolCanvas(toolRenderer);

  const gameUi: GameScreen = createGameScreen({
    generatorTypes: GENERATOR_TYPES,
    initialGeneratorType: settings.generator.type,
  });
  Object.assign(gameUi.root.style, {
    transform: `translateY(${GAME_SCREEN_Y_OFFSET}px)`,
    transformOrigin: 'top left',
  });
  gameScreen.appendChild(gameUi.root);
  gameUi.setQueueOddsMode(usesModelGenerator(settings.generator.type));
  if (!ENABLE_LEGACY_DATA_TOOLS) {
    gameUi.folderButton.style.display = 'none';
    gameUi.folderStatus.style.display = 'none';
    gameUi.recordRow.style.display = 'none';
    gameUi.recordStatus.style.display = 'none';
    gameUi.manualButton.style.display = 'none';
  }

  const formatSprintTime = (ms: number): string => {
    const totalMs = Math.max(0, Math.floor(ms));
    const minutes = Math.floor(totalMs / 60000);
    const seconds = Math.floor((totalMs % 60000) / 1000);
    const hundredths = Math.floor((totalMs % 1000) / 10);
    return `${minutes}:${String(seconds).padStart(2, '0')}.${String(
      hundredths,
    ).padStart(2, '0')}`;
  };

  const updateSprintHud = (state: GameState) => {
    if (!state.lineGoal) {
      if (gameUi.sprintPanel.style.display !== 'none') {
        gameUi.sprintPanel.style.display = 'none';
      }
      return;
    }
    if (gameUi.sprintPanel.style.display !== 'block') {
      gameUi.sprintPanel.style.display = 'block';
    }
    const linesLeft = Math.max(0, state.lineGoal - state.totalLinesCleared);
    gameUi.sprintTimerValue.textContent = formatSprintTime(state.timeMs);
    gameUi.sprintLinesValue.textContent = String(linesLeft);
  };

  const updateClassicHud = (state: GameState) => {
    if (!state.scoringEnabled) {
      if (gameUi.classicPanel.style.display !== 'none') {
        gameUi.classicPanel.style.display = 'none';
      }
      return;
    }
    if (gameUi.classicPanel.style.display !== 'block') {
      gameUi.classicPanel.style.display = 'block';
    }
    const scoreFormatted = Math.trunc(state.score)
      .toString()
      .replace(/\B(?=(\d{3})+(?!\d))/g, "'");
    gameUi.classicLevelValue.textContent = String(state.level);
    gameUi.classicScoreValue.textContent = scoreFormatted;
  };

  let previousRunEnded = false;
  modelStatusLabel = gameUi.modelStatusLabel;

  runtime = createGameRuntime({
    app,
    session,
    renderer: gameRenderer,
    inputSource: activeInputSource,
    onGameOver: (visible) => {
      gameUi.gameOverLabel.style.display = visible ? 'block' : 'none';
    },
    onFrame: (state) => {
      if (pendingTrajectoryRunModeId && !activeTrajectoryRun) {
        beginTrajectoryRun(pendingTrajectoryRunModeId);
        pendingTrajectoryRunModeId = null;
      }
      updateSprintHud(state);
      updateClassicHud(state);
      gameUi.setQueueOddsMode(
        usesModelGenerator(settingsStore.get().generator.type),
      );
      gameUi.setMlQueueProbabilities(state.mlQueueProbabilities);
      const ended = state.gameOver || state.gameWon;
      if (ended && !previousRunEnded) {
        void snapshotService?.flushRemoteUploads();
        const outcome = state.gameWon ? 'game_won' : 'game_over';
        const terminal = toTrajectoryTerminalStats(state);
        if (authState.authenticated) {
          void finalizeAndUploadTrajectoryRun(outcome, terminal).catch(
            (error) => {
              console.warn('[trajectory] auto upload failed', error);
            },
          );
        } else {
          const finalized = finalizeTrajectoryRun(outcome, terminal);
          if (finalized) {
            lastTrajectoryUploadMessage =
              'Run captured locally. Sign in and use "UPLOAD TRAJECTORY".';
            lastTrajectoryUploadError = null;
          } else if (activeTrajectoryRun == null) {
            lastTrajectoryUploadMessage = `Run ignored (need >= ${lastTrajectoryMinSamplesRequired} samples).`;
            lastTrajectoryUploadError = null;
          }
        }
      }
      previousRunEnded = ended;
    },
  });
  inputService.setOnInputSourceChange((source) => {
    manualInputSource = source;
    runtime?.setInputSource(activeInputSource);
  });
  runtime.setInputSource(activeInputSource);
  updateModelStatusUI(modelService.getStatus());

  let activeToolId = '';
  const uiController = createUiController({
    game: gameUi,
    settingsStore,
    modeController,
    useRemoteUpload,
    getSnapshotState: () => {
      const game = session.getGame();
      const state = game.state;
      const linesLeft =
        state.lineGoal != null
          ? Math.max(0, state.lineGoal - state.totalLinesCleared)
          : undefined;
      return {
        board: state.board,
        active: state.active,
        hold: state.hold,
        next: state.next,
        odds: state.mlQueueProbabilities,
        linesLeft,
        level: state.level,
        score: state.score,
      };
    },
    onPauseInputChange: (paused) => runtime?.setPausedByInput(paused),
    onMenuClick: () => setScreen('menu'),
    onStartGame: () => requestStartGame(),
    onOpenTool: (id) => {
      if (!ENABLE_LEGACY_DATA_TOOLS) return;
      activeToolId = id;
      setScreen('tool');
    },
    onSendFeedback: (feedback, contact) =>
      uploadService.sendFeedback(feedback, contact),
  });
  uiController.bindGameUi();

  const refreshMenuLabelingProgress = async (): Promise<void> => {
    if (!ENABLE_LEGACY_DATA_TOOLS) {
      uiController.setMenuLabelingProgress(null);
      return;
    }
    if (!useRemoteUpload) {
      uiController.setMenuLabelingProgress({
        buildVersion: APP_VERSION,
        labeledBoards: null,
        target: LABELING_PROGRESS_TARGET,
      });
      return;
    }
    const labeledBoards = await uploadService
      .getLabeledBoardCountForBuild(APP_VERSION)
      .catch(() => null);
    uiController.setMenuLabelingProgress({
      buildVersion: APP_VERSION,
      labeledBoards,
      target: LABELING_PROGRESS_TARGET,
    });
  };

  if (ENABLE_LEGACY_DATA_TOOLS) {
    snapshotService = createSnapshotService({
      settingsStore,
      rows: ROWS,
      cols: COLS,
      uploadClient,
      useRemoteUpload,
      identityService,
      buildVersion: APP_VERSION,
      onStateChange: uiController.syncSnapshotUi,
    });
    snapshotService.setModeInfo({
      id: initialModeState.mode.id,
      options: { ...initialModeState.options },
    });
    uiController.attachSnapshotService(snapshotService);
  }

  modeController.setOnModeChange((mode, options) => {
    const previousMode = currentModeForModelSync;
    currentModeForModelSync = mode.id;
    snapshotService?.setModeInfo({
      id: mode.id,
      options: { ...options },
    });
    sessionController.setMode(mode, options);
    void (async () => {
      await autoSavePersonalModelIfDirty(previousMode, 'mode_change');
      await syncActiveModelForContext({
        mode: mode.id,
        reason: 'mode_change',
      });
    })();
  });

  const toolHost: ToolHost = createToolHost(toolScreen);
  const paletteAwareTools: ToolController[] = [];
  if (ENABLE_LEGACY_DATA_TOOLS) {
    const labelingTool = createLabelingTool({
      toolUsesRemote,
      uploadClient,
      uploadBaseUrl,
      buildVersion: APP_VERSION,
      canvas: toolCanvas,
      onBack: () => setScreen('menu'),
    });
    toolHost.register(labelingTool);
    const constructorTool = createConstructorTool({
      canvas: toolCanvas,
      canvasElement: app.canvas,
      uploadClient,
      settingsStore,
      identityService,
      buildVersion: APP_VERSION,
      onBack: () => setScreen('menu'),
    });
    toolHost.register(constructorTool);
    activeToolId = labelingTool.id;
    paletteAwareTools.push(labelingTool, constructorTool);
  }

  const applyToolPalette = () => {
    const palette: PiecePalette = getPiecePalette(settingsStore.get().graphics);
    for (const tool of paletteAwareTools) {
      tool.setPiecePalette?.(palette);
    }
  };
  applyToolPalette();

  menuUi = createMenuScreen({
    settingsStore,
    showExperimentalGameplayControls: SHOW_EXPERIMENTAL_GAMEPLAY_CONTROLS,
    showLegacyDataTools: ENABLE_LEGACY_DATA_TOOLS,
    version: APP_VERSION,
    charcuterieDefaultSimCount,
    tools: ENABLE_LEGACY_DATA_TOOLS ? toolHost.list() : [],
    labelingProgress: ENABLE_LEGACY_DATA_TOOLS
      ? {
          buildVersion: APP_VERSION,
          labeledBoards: null,
          target: LABELING_PROGRESS_TARGET,
        }
      : null,
    authState,
    authInitialResetToken,
    authInitialStatus,
    mlBackendPreference: storedMlBackendPreference,
    getMlRuntimeSummary,
    onMlBackendPreferenceChange: applyMlBackendPreference,
    onMlRunParityCheck: runMlParityCheck,
    getLocalTrainingStats,
    getLocalTrainingPreset,
    onRunLocalBiasTraining: (options) => runLocalHeadTraining(options),
    getTrajectoryUploadSummary,
    onUploadLatestTrajectory: uploadLatestTrajectory,
    onAuthRefresh: refreshAuthState,
    onAuthStartOAuth: (provider) => authService.startOAuth(provider),
    onAuthLogout: logoutAuthState,
    onAuthSendVerifyEmail: sendAuthVerificationEmail,
    onAuthEmailSignup: signupWithEmail,
    onAuthEmailLogin: loginWithEmail,
    onAuthPasswordForgot: sendPasswordReset,
    onAuthPasswordReset: resetPasswordWithToken,
    getModelAxes: getMenuModelAxes,
    onModelAxesChange: setMenuModelAxes,
    modelAxesOptions: {
      arch: [...MODEL_ARCH_OPTIONS],
      rewardProfile: [...REWARD_PROFILE_OPTIONS],
      queuePolicy: [...QUEUE_POLICY_OPTIONS],
    },
    onAuthListGlobalModels: listCurrentModeGlobalModels,
    onAuthResetCurrentModelToGlobal: resetCurrentModeToGlobalModel,
    onAuthLoadCurrentModel: loadCurrentModePersonalModel,
    onAuthSaveCurrentModel: saveCurrentModePersonalModel,
    onAdminListRecordings: listAdminRecordings,
    onAdminLoadRecording: loadAdminRecordingPreview,
    onAdminPublishCurrentModelBaseline: publishCurrentModelAsGlobalBaseline,
    onAdminListGlobalBaselines: listAdminGlobalBaselines,
    onAdminSetGlobalBaselineDefault: setAdminGlobalBaselineDefault,
    onAdminRetireGlobalBaseline: retireAdminGlobalBaseline,
    onAdminPrepareManifest: prepareAdminManifest,
    onAdminTrainGlobalOneShot: runAdminGlobalTrainingOneShot,
    onAdminPublishGlobalCandidate: publishAdminGlobalTrainingCandidate,
    onAdminTrainBotPolicyOneShot: runAdminTrainBotPolicyOneShot,
    onAdminGenerateBotRecordings: runAdminGenerateBotRecordings,
    onAdminRunCapabilityBenchmark: runAdminCapabilityBenchmark,
    onAdminApplyBenchmarkSuggestedArch: applyAdminBenchmarkSuggestedArch,
    onBotLabFetchCurrentPolicy: fetchCurrentAdminBotPolicy,
    onBotLabListPolicies: listAdminBotPolicies,
    onBotLabLoadPolicyById: loadAdminBotPolicyById,
    onBotLabPublishPolicy: (options) =>
      publishAdminBotPolicy({
        pin: options?.pin,
        setCurrent: options?.setCurrent,
        pieceSourceProfile: options?.pieceSourceProfile,
      }),
    onBotLabSelectCurrentPolicy: selectAdminBotPolicyAsCurrent,
    onBotLabPinPolicy: (id) => setAdminBotPolicyPinState(id, true),
    onBotLabUnpinPolicy: (id) => setAdminBotPolicyPinState(id, false),
    onBotLabTrainPolicyOneShot: runAdminTrainBotPolicyOneShot,
    onBotLabRunHeadlessValidate: runAdminHeadlessBotValidation,
    onBotLabStartGuiInspect: startAdminBotGuiInspect,
    onBotLabStopGuiInspect: stopAdminBotGuiInspect,
    onBotLabGenerateRecordings: runAdminGenerateBotRecordings,
    onBotLabRunBenchmark: runAdminCapabilityBenchmark,
    ...uiController.getMenuHandlers(),
  });
  menuScreen.appendChild(menuUi.root);
  uiController.attachMenu(menuUi);
  uiController.setMenuTools(ENABLE_LEGACY_DATA_TOOLS ? toolHost.list() : []);
  void refreshMenuLabelingProgress();
  void refreshAuthState();

  app.renderer.resize(PLAY_WIDTH, PLAY_HEIGHT);
  const settingsController = createSettingsController({
    settingsStore,
    inputService,
    soundService,
    uiController,
    sessionController,
    modelService,
    onModelStatus: updateModelStatusUI,
    onGraphicsChange: (next) => {
      gameRenderer.setGridlineOpacity(next.graphics.gridlineOpacity);
      toolRenderer.setGridlineOpacity(next.graphics.gridlineOpacity);
      gameRenderer.setGhostOpacity(next.graphics.ghostOpacity);
      toolRenderer.setGhostOpacity(next.graphics.ghostOpacity);
      gameRenderer.setHighContrast(next.graphics.highContrast);
      toolRenderer.setHighContrast(next.graphics.highContrast);
      gameRenderer.setColorblindMode(next.graphics.colorblindMode);
      toolRenderer.setColorblindMode(next.graphics.colorblindMode);
      const palette: PiecePalette = getPiecePalette(next.graphics);
      for (const tool of paletteAwareTools) {
        tool.setPiecePalette?.(palette);
      }
    },
  });
  settingsController.start();

  const screenFlow = createScreenFlowController({
    settingsStore,
    runtime: runtime!,
    gameScreen: gameUi,
    menuScreen: menuUi!,
    inputService,
    sessionController,
    toolHost,
    gameGfx,
    toolGfx,
    stopRecording: () => {
      if (!ENABLE_LEGACY_DATA_TOOLS) return;
      void stopRecordingSession();
    },
    startRecording: () => {
      if (!ENABLE_LEGACY_DATA_TOOLS) return;
      startRecordingSession();
    },
  });
  screenManager.register('menu', screenFlow.makeMenuScreen(menuScreen));
  screenManager.register('game', screenFlow.makeGameScreen(gameScreen));
  screenManager.register(
    'tool',
    screenFlow.makeToolScreen(toolScreen, () => activeToolId),
  );

  setScreen = (screen: 'menu' | 'game' | 'tool') => {
    if (screen === 'menu') {
      void refreshMenuLabelingProgress();
    }
    void screenManager.setActive(screen);
  };

  let startingGame = false;
  const startGameWithModelReady = async (): Promise<void> => {
    if (startingGame) return;
    startingGame = true;
    try {
      if (useRemoteUpload) {
        const buildCount = await uploadService
          .getSnapshotCountForBuild(APP_VERSION)
          .catch(() => null);
        snapshotService?.setRemoteSnapshotBankCount(buildCount);
      } else {
        snapshotService?.setRemoteSnapshotBankCount(null);
      }
      const generatorType = settingsStore.get().generator.type;
      if (usesModelGenerator(generatorType)) {
        if (modelService.getStatus() !== 'ready') {
          const loaded = await modelService.ensureLoaded();
          if (!loaded) {
            updateModelStatusUI(modelService.getStatus());
            return;
          }
        }
        await syncActiveModelForContext({
          mode: modeController.getState().mode.id,
          reason: 'start_game',
        });
      }
      activeTrajectoryRun = null;
      previousRunEnded = false;
      scheduleTrajectoryRunStart(modeController.getState().mode.id);
      await screenManager.setActive('game');
    } finally {
      startingGame = false;
    }
  };
  requestStartGame = () => {
    void startGameWithModelReady();
  };

  setScreen('menu');
  if (openPanelOnBoot === 'account') {
    menuUi?.show('account');
  } else if (openPanelOnBoot === 'admin') {
    menuUi?.show('admin');
  }

  const identityConsole = window as Window & {
    wubSetUserId?: (value: string | null) => void;
    wubSetSuperuser?: () => void;
    wubClearUserId?: () => void;
    wubTrainingStats?: () => ReturnType<typeof trajectoryBuffer.getStats>;
    wubTrainingClear?: () => number;
    wubTrainingList?: (options?: {
      modeId?: string;
      limit?: number;
    }) => ReturnType<typeof trajectoryBuffer.listSamples>;
    wubTrainingRun?: (options?: {
      modeId?: string;
      sampleLimit?: number;
      epochs?: number;
      learningRate?: number;
      l2?: number;
      backendPreference?: 'auto' | 'webgl' | 'cpu';
    }) => Promise<{
      ok: boolean;
      message: string;
      samplesUsed: number;
      finalLoss: number | null;
    }>;
  };
  identityConsole.wubSetUserId = (value) => identityService.setUserId(value);
  identityConsole.wubSetSuperuser = () =>
    identityService.setUserId('superuser');
  identityConsole.wubClearUserId = () => identityService.setUserId(null);
  identityConsole.wubTrainingStats = () => trajectoryBuffer.getStats();
  identityConsole.wubTrainingClear = () => trajectoryBuffer.clear();
  identityConsole.wubTrainingList = (options) =>
    trajectoryBuffer.listSamples(options);
  identityConsole.wubTrainingRun = async (options) =>
    await runLocalHeadTraining(options);
}

boot().catch((e) => console.error(e));
