import { Application, Graphics } from 'pixi.js';
import {
  COLS,
  ML_MODEL_URL,
  PLAY_HEIGHT,
  PLAY_WIDTH,
  ROWS,
} from './core/constants';
import { GENERATOR_TYPES, usesModelGenerator } from './core/generators';
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
import { createMenuScreen, type MenuScreen } from './ui/screens/menuScreen';
import { createGameScreen, type GameScreen } from './ui/screens/gameScreen';
import { createToolHost, type ToolHost } from './ui/tools/toolHost';
import { createLabelingTool } from './ui/tools/labelingTool';
import { createConstructorTool } from './ui/tools/constructorTool';
import { createToolCanvas } from './ui/tools/toolCanvas';
import { getPiecePalette } from './core/palette';
import pkg from '../package.json';

function hasWebGL(): boolean {
  const c = document.createElement('canvas');
  return !!(c.getContext('webgl2') || c.getContext('webgl'));
}

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
  const SHOW_DEV_TOOLS =
    import.meta.env.VITE_SHOW_DEV_TOOLS === 'true' || import.meta.env.DEV;
  const uploadService = createUploadService({
    envMode: import.meta.env.VITE_UPLOAD_MODE as string | undefined,
    envBaseUrl: import.meta.env.VITE_UPLOAD_URL as string | undefined,
  });
  const uploadClient = uploadService.uploadClient;
  const uploadBaseUrl = uploadService.baseUrl;
  const useRemoteUpload = uploadService.useRemote;
  const toolUsesRemote = uploadService.toolUsesRemote;
  const LABELING_PROGRESS_TARGET = 2000;
  let modelStatusLabel: HTMLDivElement | null = null;
  let pausedByModel = false;
  let setScreen: (screen: 'menu' | 'game' | 'tool') => void = () => {};
  let requestStartGame: () => void = () => {};
  let runtime: GameRuntime | null = null;
  const modelService = createModelService({ modelUrl: ML_MODEL_URL });
  modelService.setStatusListener((status) => {
    updateModelStatusUI(status);
  });
  const getModelGeneratorLabel = (): string => {
    const type = settingsStore.get().generator.type;
    return type === 'curse' ? 'Curse Upon a Block' : 'Wish Upon a Block';
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
    modelStatusLabel.style.display = 'block';
    let text = `${generatorLabel}: idle (RNG fallback)`;
    let color = '#f4b266';
    let shouldPause = false;
    if (status === 'ready') {
      text = `${generatorLabel}: loaded`;
      color = '#8fd19e';
      shouldPause = false;
    } else if (status === 'loading') {
      text = `${generatorLabel}: loading (RNG fallback)`;
      color = '#f4b266';
      shouldPause = false;
    } else if (status === 'failed') {
      text = `${generatorLabel}: failed to load model`;
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
  void modelService.ensureLoaded();
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
  const session: GameSession = createGameSessionFactory({
    settings,
    initialMode: initialModeState.mode,
    initialModeOptions: initialModeState.options,
    modelService,
    onPieceLock: handlePieceLock,
    onHold: handleHoldSnapshot,
    onLineClear: handleLineClear,
    onBeforeRestart: () => restartRecordingSession(),
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
  gameUi.setQueueOddsMode(settings.generator.type === 'ml');

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
      gameUi.setQueueOddsMode(settingsStore.get().generator.type === 'ml');
      gameUi.setMlQueueProbabilities(state.mlQueueProbabilities);
      const ended = state.gameOver || state.gameWon;
      if (ended && !previousRunEnded) {
        void snapshotService?.flushRemoteUploads();
      }
      previousRunEnded = ended;
    },
  });
  inputService.setOnInputSourceChange((source) => {
    runtime?.setInputSource(source);
  });
  runtime.setInputSource(inputService.getInputSource());
  updateModelStatusUI(modelService.getStatus());

  let activeToolId = 'labeling';
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
      activeToolId = id;
      setScreen('tool');
    },
    onSendFeedback: (feedback, contact) =>
      uploadService.sendFeedback(feedback, contact),
  });
  uiController.bindGameUi();

  const refreshMenuLabelingProgress = async (): Promise<void> => {
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

  modeController.setOnModeChange((mode, options) => {
    snapshotService?.setModeInfo({
      id: mode.id,
      options: { ...options },
    });
    sessionController.setMode(mode, options);
  });

  const toolHost: ToolHost = createToolHost(toolScreen);
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

  const applyToolPalette = () => {
    const palette = getPiecePalette(settingsStore.get().graphics);
    labelingTool.setPiecePalette?.(palette);
    constructorTool.setPiecePalette?.(palette);
  };
  applyToolPalette();

  menuUi = createMenuScreen({
    settingsStore,
    showDevTools: SHOW_DEV_TOOLS,
    version: APP_VERSION,
    charcuterieDefaultSimCount,
    tools: toolHost.list(),
    labelingProgress: {
      buildVersion: APP_VERSION,
      labeledBoards: null,
      target: LABELING_PROGRESS_TARGET,
    },
    ...uiController.getMenuHandlers(),
  });
  menuScreen.appendChild(menuUi.root);
  uiController.attachMenu(menuUi);
  uiController.setMenuTools(toolHost.list());
  void refreshMenuLabelingProgress();

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
      const palette = getPiecePalette(next.graphics);
      labelingTool.setPiecePalette?.(palette);
      constructorTool.setPiecePalette?.(palette);
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
      void stopRecordingSession();
    },
    startRecording: () => {
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
      }
      await screenManager.setActive('game');
    } finally {
      startingGame = false;
    }
  };
  requestStartGame = () => {
    void startGameWithModelReady();
  };

  setScreen('menu');

  const identityConsole = window as Window & {
    wubSetUserId?: (value: string | null) => void;
    wubSetSuperuser?: () => void;
    wubClearUserId?: () => void;
  };
  identityConsole.wubSetUserId = (value) => identityService.setUserId(value);
  identityConsole.wubSetSuperuser = () =>
    identityService.setUserId('superuser');
  identityConsole.wubClearUserId = () => identityService.setUserId(null);
}

boot().catch((e) => console.error(e));
