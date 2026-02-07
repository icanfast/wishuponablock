import { Application, Graphics } from 'pixi.js';
import {
  COLS,
  ML_MODEL_URL,
  PLAY_HEIGHT,
  PLAY_WIDTH,
  ROWS,
} from './core/constants';
import { GENERATOR_TYPES } from './core/generators';
import type { GameSession } from './core/gameSession';
import { createSettingsStore } from './core/settingsStore';
import { createGameRuntime, type GameRuntime } from './app/runtime';
import { createUploadService } from './app/uploadService';
import { createModeController } from './app/modeController';
import { createInputService } from './app/inputService';
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
import { type Board, type PieceKind } from './core/types';
import { createMenuScreen, type MenuScreen } from './ui/screens/menuScreen';
import { createGameScreen, type GameScreen } from './ui/screens/gameScreen';
import { createToolHost, type ToolHost } from './ui/tools/toolHost';
import { createLabelingTool } from './ui/tools/labelingTool';
import { createToolCanvas } from './ui/tools/toolCanvas';
import pkg from '../package.json';

function hasWebGL(): boolean {
  const c = document.createElement('canvas');
  return !!(c.getContext('webgl2') || c.getContext('webgl'));
}

async function boot() {
  const APP_VERSION = pkg.version;
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
  let modelStatusLabel: HTMLDivElement | null = null;
  let pausedByModel = false;
  let setScreen: (screen: 'menu' | 'game' | 'tool') => void = () => {};
  let runtime: GameRuntime | null = null;
  const modelService = createModelService({ modelUrl: ML_MODEL_URL });
  modelService.setStatusListener((status) => {
    updateModelStatusUI(status);
  });
  const updateModelStatusUI = (status: ModelStatus): void => {
    if (!modelStatusLabel) return;
    const isMl = settingsStore.get().generator.type === 'ml';
    if (!isMl) {
      modelStatusLabel.textContent = '';
      modelStatusLabel.style.display = 'none';
      if (pausedByModel) {
        pausedByModel = false;
        runtime?.setPausedByModel(false);
      }
      return;
    }
    modelStatusLabel.style.display = 'block';
    let text = 'ML generator: idle (RNG fallback)';
    let color = '#f4b266';
    let shouldPause = false;
    if (status === 'ready') {
      text = 'ML generator: loaded';
      color = '#8fd19e';
      shouldPause = false;
    } else if (status === 'loading') {
      text = 'ML generator: loading (RNG fallback)';
      color = '#f4b266';
      shouldPause = false;
    } else if (status === 'failed') {
      text = 'ML generator: failed to load model';
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
    initialModeId: 'default',
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

  const handlePieceLock = (board: Board, hold: PieceKind | null) => {
    if (suppressLockEffects) return;
    soundService.playLock();
    snapshotService?.handleLock(board, hold);
  };
  const handleHoldSnapshot = (board: Board, hold: PieceKind | null) => {
    if (suppressLockEffects) return;
    snapshotService?.handleHold(board, hold);
  };

  const initialModeState = modeController.getState();
  const session: GameSession = createGameSessionFactory({
    settings,
    initialMode: initialModeState.mode,
    initialModeOptions: initialModeState.options,
    modelService,
    onPieceLock: handlePieceLock,
    onHold: handleHoldSnapshot,
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
  const toolCanvas = createToolCanvas(toolRenderer);

  const gameUi: GameScreen = createGameScreen({
    generatorTypes: GENERATOR_TYPES,
    initialGeneratorType: settings.generator.type,
  });
  gameScreen.appendChild(gameUi.root);

  modelStatusLabel = gameUi.modelStatusLabel;

  runtime = createGameRuntime({
    app,
    session,
    renderer: gameRenderer,
    inputSource,
    onGameOver: (visible) => {
      gameUi.gameOverLabel.style.display = visible ? 'block' : 'none';
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
    onPauseInputChange: (paused) => runtime?.setPausedByInput(paused),
    onMenuClick: () => setScreen('menu'),
    onStartGame: () => setScreen('game'),
    onOpenTool: (id) => {
      activeToolId = id;
      setScreen('tool');
    },
    onSendFeedback: (feedback, contact) =>
      uploadService.sendFeedback(feedback, contact),
  });
  uiController.bindGameUi();

  snapshotService = createSnapshotService({
    settingsStore,
    rows: ROWS,
    cols: COLS,
    uploadClient,
    useRemoteUpload,
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
    canvas: toolCanvas,
    onBack: () => setScreen('menu'),
  });
  toolHost.register(labelingTool);
  activeToolId = labelingTool.id;

  menuUi = createMenuScreen({
    settingsStore,
    showDevTools: SHOW_DEV_TOOLS,
    version: APP_VERSION,
    charcuterieDefaultSimCount,
    tools: toolHost.list(),
    ...uiController.getMenuHandlers(),
  });
  menuScreen.appendChild(menuUi.root);
  uiController.attachMenu(menuUi);
  uiController.setMenuTools(toolHost.list());

  app.renderer.resize(PLAY_WIDTH, PLAY_HEIGHT);
  const settingsController = createSettingsController({
    settingsStore,
    inputService,
    soundService,
    uiController,
    sessionController,
    modelService,
    onModelStatus: updateModelStatusUI,
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
    void screenManager.setActive(screen);
  };

  void screenManager.setActive('menu');
}

boot().catch((e) => console.error(e));
