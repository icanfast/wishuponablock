import { Application, Graphics } from 'pixi.js';
import {
  COLS,
  ML_BACKEND_PREFERENCE_STORAGE_KEY,
  ML_MODEL_URL,
  PLAY_HEIGHT,
  PLAY_WIDTH,
  ROWS,
} from './core/constants';
import { GENERATOR_TYPES, usesModelGenerator } from './core/generators';
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
  computeTrajectoryRewards,
  resolveTrajectoryRewardPolicyId,
  type TrajectoryRewardComputation,
  type TrajectoryRewardPolicyId,
  type TrajectoryRewardTerminalStats,
} from './app/trajectoryRewardPolicy';
import { createPersonalTrainerTfjs } from './app/personalTrainerTfjs';
import {
  TRAJECTORY_SESSION_SCHEMA_V1,
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
  type MenuScreen,
} from './ui/screens/menuScreen';
import { createGameScreen, type GameScreen } from './ui/screens/gameScreen';
import {
  createToolHost,
  type ToolController,
  type ToolHost,
} from './ui/tools/toolHost';
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
  const trajectoryRewardPolicyId: TrajectoryRewardPolicyId =
    resolveTrajectoryRewardPolicyId(
      import.meta.env.VITE_TRAJECTORY_REWARD_POLICY as string | undefined,
    );
  const personalTrainer = createPersonalTrainerTfjs();
  let menuUi: MenuScreen | null = null;
  const modeController = createModeController({
    initialModeId: 'practice',
  });
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
  const inputSource = inputService.getInputSource();
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

  const buildModelContextKey = (userId: string | null, mode: string): string =>
    `${userId ?? 'anon'}:${mode}`;

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
      const result = await personalModelService.downloadCurrent(mode);
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
      syncedModelShaByMode.set(mode, resolvedSha);
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

    const bytes = await modelService.ensureModelBytes();
    if (!bytes) return;
    const currentSha = await sha256HexFromBuffer(bytes);
    const syncedSha = syncedModelShaByMode.get(mode) ?? null;
    if (syncedSha && syncedSha === currentSha) return;

    try {
      const result = await personalModelService.uploadCurrent(mode, bytes);
      syncedModelShaByMode.set(mode, result.sha256 ?? currentSha);
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
    const bytes = await modelService.ensureModelBytes();
    if (!bytes) {
      throw new Error('No model is loaded to upload.');
    }
    const currentSha = await sha256HexFromBuffer(bytes);
    const result = await personalModelService.uploadCurrent(mode, bytes);
    syncedModelShaByMode.set(mode, result.sha256 ?? currentSha);
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

  type TrajectoryRunState = {
    sessionId: string;
    modeId: string;
    startedAtMs: number;
  };
  let activeTrajectoryRun: TrajectoryRunState | null = null;
  let pendingTrajectorySession: TrajectorySessionV1 | null = null;
  let lastTrajectoryUploadMessage = 'No uploads yet.';
  let lastTrajectoryUploadAtMs: number | null = null;
  let lastTrajectoryUploadSamples = 0;
  let lastTrajectoryUploadError: string | null = null;

  const toTrajectoryMeta = (
    outcome: string,
    rewards: TrajectoryRewardComputation | null,
  ): TrajectorySessionMetaV1 => {
    const meta: TrajectorySessionMetaV1 = {
      outcome,
      channel: import.meta.env.MODE,
      modelSource: activeModelSource.kind,
    };
    if (activeModelSource.kind === 'personal') {
      meta.modelMode = activeModelSource.mode;
      if (activeModelSource.version != null) {
        meta.modelVersion = activeModelSource.version;
      }
    }
    if (rewards) {
      meta.rewardPolicy = rewards.policyId;
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
      };
    });

  const beginTrajectoryRun = (modeId: string): void => {
    activeTrajectoryRun = {
      sessionId: crypto.randomUUID(),
      modeId,
      startedAtMs: Date.now(),
    };
  };

  const buildTrajectorySession = (
    run: TrajectoryRunState,
    outcome: string,
    terminal: TrajectoryRewardTerminalStats | null = null,
  ): TrajectorySessionV1 | null => {
    const runSamples = trajectoryBuffer
      .listSamples({ modeId: run.modeId })
      .filter((sample) => sample.createdAtMs >= run.startedAtMs);
    if (runSamples.length === 0) return null;
    const rewards = computeTrajectoryRewards(
      runSamples,
      {
        modeId: run.modeId,
        outcome,
        terminal,
      },
      trajectoryRewardPolicyId,
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
      meta: toTrajectoryMeta(outcome, rewards),
    };
  };

  const finalizeTrajectoryRun = (
    outcome: string,
    terminal: TrajectoryRewardTerminalStats | null = null,
  ): TrajectorySessionV1 | null => {
    const run = activeTrajectoryRun;
    activeTrajectoryRun = null;
    if (!run) return null;
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
      const activeSamples = trajectoryBuffer
        .listSamples({ modeId: run.modeId })
        .filter((sample) => sample.createdAtMs >= run.startedAtMs).length;
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

  const runLocalBiasTraining = async (options?: {
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
    const samples = trajectoryBuffer.listSamples({
      modeId,
      limit: options?.sampleLimit,
    });
    const result = await personalTrainer.trainBiasOnly({
      model,
      samples,
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
      'local bias training',
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
  const getLocalTrainingStats = () => {
    const stats = trajectoryBuffer.getStats();
    const currentModeId = modeController.getState().mode.id;
    return {
      currentModeId,
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
      beginTrajectoryRun(modeController.getState().mode.id);
    },
    onModelDecision: (decision) => {
      const modeId = modeController.getState().mode.id;
      trajectoryBuffer.recordDecision({ modeId, decision });
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
    inputSource,
    onGameOver: (visible) => {
      gameUi.gameOverLabel.style.display = visible ? 'block' : 'none';
    },
    onFrame: (state) => {
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
          }
        }
      }
      previousRunEnded = ended;
    },
  });
  inputService.setOnInputSourceChange((source) => {
    runtime?.setInputSource(source);
  });
  runtime.setInputSource(inputService.getInputSource());
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
    onRunLocalBiasTraining: () => runLocalBiasTraining(),
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
    onAuthLoadCurrentModel: loadCurrentModePersonalModel,
    onAuthSaveCurrentModel: saveCurrentModePersonalModel,
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
      beginTrajectoryRun(modeController.getState().mode.id);
      previousRunEnded = false;
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
    await runLocalBiasTraining(options);
}

boot().catch((e) => console.error(e));
