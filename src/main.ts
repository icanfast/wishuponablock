import { Application, Graphics } from 'pixi.js';
import {
  BOARD_CELL_PX,
  BOARD_X,
  BOARD_Y,
  BOARD_WIDTH,
  COLS,
  DEFAULT_KEY_BINDINGS,
  GAME_OVER_Y,
  HOLD_X,
  HOLD_Y,
  HOLD_WIDTH,
  ML_MODEL_URL,
  OUTER_MARGIN,
  PANEL_GAP,
  PLAY_HEIGHT,
  PLAY_WIDTH,
  ROWS,
  SETTINGS_X,
  SETTINGS_Y,
  SETTINGS_PANEL_WIDTH,
} from './core/constants';
import { Game } from './core/game';
import {
  createGeneratorFactory,
  GENERATOR_TYPES,
  isGeneratorType,
  type GeneratorType,
} from './core/generators';
import { GameRunner } from './core/runner';
import type { Settings } from './core/settings';
import { createSettingsStore } from './core/settingsStore';
import {
  SnapshotRecorder,
  downloadSnapshotSession,
  saveSnapshotSessionToDirectory,
} from './core/snapshotRecorder';
import { getMode, type GameMode, type ModeOptions } from './core/modes';
import { Keyboard } from './input/keyboard';
import { InputController } from './input/controller';
import { KeyboardInputSource } from './input/keyboardInputSource';
import { ButterfingerInputSource } from './input/butterfingerInputSource';
import { PixiRenderer } from './render/pixiRenderer';
import { PIECES, type Board, type PieceKind } from './core/types';
import type { SnapshotSession } from './core/snapshotRecorder';
import { CharcuterieBot, runBotForPieces } from './bot/charcuterieBot';
import { makeBoard } from './core/board';
import { TETROMINOES } from './core/tetromino';
import { loadWubModel, type LoadedModel } from './core/wubModel';
import { UploadClient, type UploadMode } from './core/uploadClient';

function hasWebGL(): boolean {
  const c = document.createElement('canvas');
  return !!(c.getContext('webgl2') || c.getContext('webgl'));
}

async function boot() {
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

  const gfx = new Graphics();
  app.stage.addChild(gfx);

  const uiLayer = document.createElement('div');
  Object.assign(uiLayer.style, {
    position: 'absolute',
    inset: '0',
    pointerEvents: 'none',
  });
  playWindow.appendChild(uiLayer);

  const spinnerStyle = document.createElement('style');
  spinnerStyle.textContent = `
@keyframes wab-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
`;
  document.head.appendChild(spinnerStyle);

  const charcuterieSpinner = document.createElement('div');
  Object.assign(charcuterieSpinner.style, {
    position: 'absolute',
    inset: '0',
    display: 'none',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'rgba(11, 15, 20, 0.45)',
    zIndex: '10',
    pointerEvents: 'none',
  });
  uiLayer.appendChild(charcuterieSpinner);

  const spinnerPanel = document.createElement('div');
  Object.assign(spinnerPanel.style, {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '10px 12px',
    background: '#0b0f14',
    border: '1px solid #1f2a37',
    borderRadius: '8px',
    color: '#e2e8f0',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    fontSize: '12px',
  });
  charcuterieSpinner.appendChild(spinnerPanel);

  const spinnerRing = document.createElement('div');
  Object.assign(spinnerRing.style, {
    width: '18px',
    height: '18px',
    border: '2px solid #2c3a4a',
    borderTopColor: '#8fa0b8',
    borderRadius: '50%',
    animation: 'wab-spin 0.9s linear infinite',
  });
  spinnerPanel.appendChild(spinnerRing);

  const spinnerText = document.createElement('div');
  spinnerText.textContent = 'Generating board...';
  spinnerPanel.appendChild(spinnerText);

  const holdLabel = document.createElement('div');
  holdLabel.textContent = 'HOLD';
  Object.assign(holdLabel.style, {
    position: 'absolute',
    left: `${HOLD_X + 8}px`,
    top: `${HOLD_Y + 6}px`,
    width: `${HOLD_WIDTH}px`,
    color: '#b6c2d4',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    fontSize: '13px',
    letterSpacing: '0.5px',
    pointerEvents: 'none',
  });
  uiLayer.appendChild(holdLabel);

  const gameOverLabel = document.createElement('div');
  gameOverLabel.textContent = 'GAME OVER';
  Object.assign(gameOverLabel.style, {
    position: 'absolute',
    left: `${HOLD_X}px`,
    top: `${GAME_OVER_Y}px`,
    width: `${HOLD_WIDTH}px`,
    color: '#7f8a9a',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    fontSize: '13px',
    letterSpacing: '0.5px',
    textAlign: 'center',
    pointerEvents: 'none',
    display: 'none',
  });
  uiLayer.appendChild(gameOverLabel);

  const settingsStore = createSettingsStore();
  const settings = settingsStore.get();
  const SHOW_DEV_TOOLS =
    import.meta.env.VITE_SHOW_DEV_TOOLS === 'true' || import.meta.env.DEV;
  let generatorType = settings.generator.type;
  const inferUploadMode = (): UploadMode => {
    const host = window.location.hostname;
    if (!host) return 'local';
    if (host === 'localhost' || host === '127.0.0.1') return 'local';
    return 'remote';
  };
  const rawUploadMode = (
    import.meta.env.VITE_UPLOAD_MODE as string | undefined
  )?.toLowerCase();
  const uploadMode =
    rawUploadMode === 'local' ||
    rawUploadMode === 'remote' ||
    rawUploadMode === 'auto'
      ? (rawUploadMode as UploadMode)
      : inferUploadMode();
  const uploadBaseUrl =
    (import.meta.env.VITE_UPLOAD_URL as string | undefined) ?? '/api';
  console.info(`[Upload] mode=${uploadMode} baseUrl=${uploadBaseUrl}`);
  const uploadClient = new UploadClient({
    mode: uploadMode,
    baseUrl: uploadBaseUrl,
  });
  const useRemoteUpload = uploadClient.isRemote;
  const toolUsesRemote = useRemoteUpload;
  let mlModel: LoadedModel | null = null;
  let mlModelPromise: Promise<LoadedModel | null> | null = null;
  let mlStatus: 'idle' | 'loading' | 'ready' | 'failed' = 'idle';
  let modelStatusLabel: HTMLDivElement | null = null;
  let pausedByModel = false;
  let updatePaused: () => void = () => {};
  function updateModelStatusUI(): void {
    if (!modelStatusLabel) return;
    if (generatorType !== 'ml') {
      modelStatusLabel.textContent = '';
      modelStatusLabel.style.display = 'none';
      if (pausedByModel) {
        pausedByModel = false;
        updatePaused();
      }
      return;
    }
    modelStatusLabel.style.display = 'block';
    let text = 'ML generator: idle (RNG fallback)';
    let color = '#f4b266';
    let shouldPause = false;
    if (mlStatus === 'ready') {
      text = 'ML generator: loaded';
      color = '#8fd19e';
      shouldPause = false;
    } else if (mlStatus === 'loading') {
      text = 'ML generator: loading (RNG fallback)';
      color = '#f4b266';
      shouldPause = false;
    } else if (mlStatus === 'failed') {
      text = 'ML generator: failed to load model';
      color = '#f28b82';
      shouldPause = true;
    }
    modelStatusLabel.textContent = text;
    modelStatusLabel.style.color = color;
    if (pausedByModel !== shouldPause) {
      pausedByModel = shouldPause;
      updatePaused();
    }
  }
  const setMlStatus = (next: 'idle' | 'loading' | 'ready' | 'failed') => {
    if (mlStatus === next) return;
    mlStatus = next;
    updateModelStatusUI();
  };
  const ensureMlModel = (): Promise<LoadedModel | null> => {
    if (mlModel) {
      setMlStatus('ready');
      return Promise.resolve(mlModel);
    }
    if (mlModelPromise) return mlModelPromise;
    setMlStatus('loading');
    mlModelPromise = loadWubModel(ML_MODEL_URL)
      .then((model) => {
        mlModel = model;
        setMlStatus(model ? 'ready' : 'failed');
        return model;
      })
      .catch((err) => {
        console.warn('Failed to load ML model:', err);
        mlModel = null;
        mlModelPromise = null;
        setMlStatus('failed');
        return null;
      });
    return mlModelPromise;
  };
  void ensureMlModel();
  const recorder = new SnapshotRecorder();
  let updateRecorderUI = () => {};
  let snapshotDirHandle: FileSystemDirectoryHandle | null = null;
  let startRecordingSession = () => {};
  let stopRecordingSession: (options?: {
    promptForFolder?: boolean;
  }) => Promise<void> = async () => {};
  let restartRecordingSession = () => {};
  let selectedMode: GameMode = getMode('default');
  let selectedModeOptions: ModeOptions = {};
  const charcuterieDefaultSimCount = 10000;
  const charcuterieScoreWeights = {
    height: 10,
    holes: 20,
    blocks: 0.01,
    clears: 100,
  };
  const charcuterieHoleWeights = {
    bottom: 5,
    mid: 2,
  };

  const kb = new Keyboard();
  const input = new InputController(kb, settings.input);
  const baseInputSource = new KeyboardInputSource(input);
  const butterfingerSource = new ButterfingerInputSource(
    baseInputSource,
    settings.butterfinger,
  );
  let inputSource = settings.butterfinger.enabled
    ? butterfingerSource
    : baseInputSource;

  const lockSound = new Audio('/sfx/lock.ogg');
  lockSound.preload = 'auto';
  lockSound.volume = settings.audio.masterVolume;
  const playLockSound = () => {
    lockSound.currentTime = 0;
    void lockSound.play().catch(() => {
      // Ignore autoplay restrictions and playback errors.
    });
  };
  let suppressLockEffects = false;
  let lastSnapshotKey: string | null = null;
  const buildSnapshotKey = (board: Board, hold: PieceKind | null): string => {
    const rows = board
      .map((row) => row.map((cell) => (cell ? cell : '.')).join(''))
      .join('/');
    return `${rows}|${hold ?? '.'}`;
  };
  const enqueueSnapshotSample = (
    board: Board,
    hold: PieceKind | null,
    reason: 'lock' | 'hold',
  ) => {
    if (!recorder.isRecording) return;
    const key = buildSnapshotKey(board, hold);
    if (key === lastSnapshotKey) {
      console.warn(`[Snapshot] Skipping duplicate (${reason}).`);
      return;
    }
    lastSnapshotKey = key;
    const sample = recorder.record(board, hold, { store: !useRemoteUpload });
    updateRecorderUI();
    if (!sample || !useRemoteUpload) return;
    const session = recorder.sessionMeta;
    if (!session) return;
    const payload = {
      createdAt: new Date().toISOString(),
      meta: {
        session,
        sample: {
          index: sample.index,
          timeMs: sample.timeMs,
          hold: sample.hold,
        },
        trigger: reason,
      },
      board: sample.board,
    };
    console.info(
      `[Snapshot] ${reason} session=${session.id} index=${sample.index}`,
    );
    void uploadClient.enqueueSnapshot(payload);
  };

  const handlePieceLock = (board: Board, hold: PieceKind | null) => {
    if (suppressLockEffects) return;
    playLockSound();
    enqueueSnapshotSample(board, hold, 'lock');
  };
  const handleHoldSnapshot = (board: Board, hold: PieceKind | null) => {
    if (suppressLockEffects) return;
    enqueueSnapshotSample(board, hold, 'hold');
  };

  const getStackHeight = (board: Board): number => {
    for (let y = 0; y < ROWS; y++) {
      if (board[y].some((cell) => cell != null)) {
        return ROWS - y;
      }
    }
    return 0;
  };

  const countBlocks = (board: Board): number =>
    board.reduce((sum, row) => {
      const rowCount = row.reduce((rowSum, cell) => rowSum + (cell ? 1 : 0), 0);
      return sum + rowCount;
    }, 0);

  const getHolePenalty = (board: Board): number => {
    let penalty = 0;
    for (let y = ROWS - 1; y >= 0; y--) {
      const row = board[y];
      let empty = 0;
      let anyFilled = false;
      for (const cell of row) {
        if (cell == null) {
          empty++;
        } else {
          anyFilled = true;
        }
      }
      if (!anyFilled) continue;
      const extraHoles = Math.max(0, empty - 1);
      if (extraHoles === 0) continue;

      const depth = ROWS - 1 - y;
      if (depth < 4) {
        penalty += extraHoles * charcuterieHoleWeights.bottom;
      } else if (depth < 8) {
        penalty += extraHoles * charcuterieHoleWeights.mid;
      }
    }
    return penalty;
  };

  const scoreCharcuterieBoard = (
    board: Board,
    gameOver: boolean,
    clears: number,
  ): {
    score: number;
    height: number;
    holes: number;
    blocks: number;
    clears: number;
  } => {
    const height = getStackHeight(board) + (gameOver ? ROWS : 0);
    const holes = getHolePenalty(board);
    const blocks = countBlocks(board);
    const score =
      height * charcuterieScoreWeights.height +
      holes * charcuterieScoreWeights.holes +
      blocks * charcuterieScoreWeights.blocks -
      clears * charcuterieScoreWeights.clears;
    return { score, height, holes, blocks, clears };
  };

  const applyModeSettings = (base: Settings, mode: GameMode): Settings => {
    if (!mode.settingsPatch) return base;
    return {
      ...base,
      ...mode.settingsPatch,
      game: {
        ...base.game,
        ...mode.settingsPatch.game,
      },
      input: {
        ...base.input,
        ...mode.settingsPatch.input,
      },
      generator: {
        ...base.generator,
        ...mode.settingsPatch.generator,
      },
      audio: {
        ...base.audio,
        ...mode.settingsPatch.audio,
      },
    };
  };

  const runModeStart = (
    game: Game,
    mode: GameMode,
    options: ModeOptions,
  ): void => {
    mode.onStart?.(game, options);
  };

  const buildGame = (cfg: Settings, mode: GameMode, seed: number): Game => {
    const merged = applyModeSettings(cfg, mode);
    return new Game({
      seed,
      ...merged.game,
      lockNudgeRate: cfg.butterfinger.enabled
        ? cfg.butterfinger.lockNudgeRate
        : 0,
      gravityDropRate: cfg.butterfinger.enabled
        ? cfg.butterfinger.gravityDropRate
        : 0,
      lockRotateRate: cfg.butterfinger.enabled
        ? cfg.butterfinger.lockRotateRate
        : 0,
      generatorFactory: createGeneratorFactory(merged.generator, {
        mlModel,
        mlModelPromise: mlModelPromise ?? undefined,
      }),
      onPieceLock: handlePieceLock,
      onHold: handleHoldSnapshot,
    });
  };

  const createCharcuterieGame = (
    cfg: Settings,
    mode: GameMode,
    options: ModeOptions,
  ): Game => {
    const pieces = Math.max(0, Math.trunc(Number(options.pieces ?? 0)));
    const sims = Math.max(
      1,
      Math.trunc(Number(options.simCount ?? charcuterieDefaultSimCount)),
    );
    const simSettings: Settings = {
      ...cfg,
      generator: {
        ...cfg.generator,
        type: 'bag7',
      },
    };
    const seedOverride = options.seed;
    const baseSeed =
      seedOverride !== undefined ? Math.trunc(seedOverride) : Date.now();
    const simStart = performance.now();

    if (pieces === 0 || sims === 1) {
      const game = buildGame(simSettings, mode, baseSeed);
      if (pieces > 0) {
        suppressLockEffects = true;
        try {
          const bot = new CharcuterieBot(baseSeed ^ 0x9e3779b9);
          runBotForPieces(game, bot, pieces);
        } finally {
          suppressLockEffects = false;
        }
      }
      const finalGame = buildGame(cfg, mode, baseSeed);
      if (pieces > 0) {
        finalGame.applyInitialBoard(game.state.board);
      }
      finalGame.markInitialBlocks();
      return finalGame;
    }

    let bestGame: Game | null = null;
    let bestScore = Number.POSITIVE_INFINITY;
    let bestHeight = Number.POSITIVE_INFINITY;
    let bestHoles = Number.POSITIVE_INFINITY;
    let bestBlocks = Number.POSITIVE_INFINITY;
    let bestClears = 0;

    let totalClears = 0;
    let simsWithClears = 0;
    let maxClears = 0;

    suppressLockEffects = true;
    try {
      for (let i = 0; i < sims; i++) {
        const seed = baseSeed + i * 977;
        const game = buildGame(simSettings, mode, seed);
        const bot = new CharcuterieBot(seed ^ 0x9e3779b9);
        runBotForPieces(game, bot, pieces);

        const clears = game.totalLinesCleared;
        const scored = scoreCharcuterieBoard(
          game.state.board,
          game.state.gameOver,
          clears,
        );
        totalClears += clears;
        if (clears > 0) simsWithClears += 1;
        if (clears > maxClears) maxClears = clears;

        if (
          scored.score < bestScore ||
          (scored.score === bestScore && scored.height < bestHeight) ||
          (scored.score === bestScore &&
            scored.height === bestHeight &&
            scored.holes < bestHoles)
        ) {
          bestGame = game;
          bestScore = scored.score;
          bestHeight = scored.height;
          bestHoles = scored.holes;
          bestBlocks = scored.blocks;
          bestClears = clears;
        }
      }
    } finally {
      suppressLockEffects = false;
    }

    if (sims > 1) {
      const simElapsedMs = performance.now() - simStart;
      console.info(
        `[Charcuterie] sims=${sims} pieces=${pieces} ` +
          `bestScore=${bestScore.toFixed(2)} ` +
          `height=${bestHeight} holes=${bestHoles} blocks=${bestBlocks.toFixed(
            0,
          )} clears=${bestClears} ` +
          `clearsTotal=${totalClears} simsWithClears=${simsWithClears} maxClears=${maxClears} ` +
          `seed=${baseSeed} ` +
          `elapsedMs=${simElapsedMs.toFixed(1)}`,
      );
    }

    const finalGame = buildGame(cfg, mode, baseSeed);
    if (bestGame) {
      finalGame.applyInitialBoard(bestGame.state.board);
    }
    finalGame.markInitialBlocks();
    return finalGame;
  };

  const createGame = (
    cfg: Settings,
    mode: GameMode,
    options: ModeOptions,
  ): Game => {
    if (mode.id === 'charcuterie') {
      return createCharcuterieGame(cfg, mode, options);
    }

    const game = buildGame(cfg, mode, Date.now());
    runModeStart(game, mode, options);
    return game;
  };

  const createRunner = (
    g: Game,
    mode: GameMode,
    options: ModeOptions,
  ): GameRunner =>
    new GameRunner(g, {
      fixedStepMs: 1000 / 120,
      onRestart: () => {
        restartRecordingSession();
        if (mode.id === 'charcuterie') {
          const nextSettings = settingsStore.get();
          game = createGame(nextSettings, selectedMode, selectedModeOptions);
          runner = createRunner(game, selectedMode, selectedModeOptions);
          return;
        }
        g.reset(Date.now());
        runModeStart(g, mode, options);
      },
      maxElapsedMs: 250,
      maxStepsPerTick: 10,
    });

  let game = createGame(settings, selectedMode, selectedModeOptions);
  let runner = createRunner(game, selectedMode, selectedModeOptions);

  const renderer = new PixiRenderer(gfx);
  const PIECE_COLORS: Record<PieceKind, string> = {
    I: '#4dd3ff',
    O: '#ffd84d',
    T: '#c77dff',
    S: '#6eea6e',
    Z: '#ff6b6b',
    J: '#4d7cff',
    L: '#ffa94d',
  };

  const settingsPanel = document.createElement('div');
  Object.assign(settingsPanel.style, {
    position: 'absolute',
    left: `${SETTINGS_X}px`,
    top: `${SETTINGS_Y}px`,
    width: `${SETTINGS_PANEL_WIDTH}px`,
    maxHeight: `${PLAY_HEIGHT - SETTINGS_Y - OUTER_MARGIN}px`,
    overflowY: 'auto',
    padding: '8px',
    background: '#121a24',
    color: '#e2e8f0',
    border: '2px solid #0b0f14',
    borderRadius: '6px',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    fontSize: '13px',
    pointerEvents: 'auto',
  });

  const label = document.createElement('div');
  label.textContent = 'Generator';
  Object.assign(label.style, {
    marginBottom: '6px',
    color: '#b6c2d4',
  });

  const select = document.createElement('select');
  Object.assign(select.style, {
    width: '100%',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '13px',
  });

  const labelFor = (type: GeneratorType): string => {
    switch (type) {
      case 'bag7':
        return 'Bag 7';
      case 'bag8i':
        return 'I-Plus Bag';
      case 'inconvenient':
        return 'Inconvenient Bag';
      case 'random':
        return 'Random';
      case 'nes':
        return 'NES';
      case 'ml':
        return 'Wish Upon a Block';
      default:
        return type;
    }
  };

  for (const type of GENERATOR_TYPES) {
    const option = document.createElement('option');
    option.value = type;
    option.textContent = labelFor(type);
    select.appendChild(option);
  }
  select.value = settings.generator.type;

  select.addEventListener('change', () => {
    const value = select.value;
    if (isGeneratorType(value)) {
      settingsStore.apply({ generator: { type: value } });
    }
  });

  settingsPanel.appendChild(label);
  settingsPanel.appendChild(select);

  modelStatusLabel = document.createElement('div');
  Object.assign(modelStatusLabel.style, {
    marginTop: '6px',
    color: '#8fa0b8',
    fontSize: '12px',
  });
  settingsPanel.appendChild(modelStatusLabel);
  updateModelStatusUI();

  const volumeLabel = document.createElement('div');
  volumeLabel.textContent = 'Master Volume';
  Object.assign(volumeLabel.style, {
    marginTop: '12px',
    marginBottom: '6px',
    color: '#b6c2d4',
  });

  const volumeRow = document.createElement('div');
  Object.assign(volumeRow.style, {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  });

  const volumeValue = document.createElement('span');
  Object.assign(volumeValue.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    minWidth: '32px',
    textAlign: 'right',
  });

  const volumeSlider = document.createElement('input');
  volumeSlider.type = 'range';
  volumeSlider.min = '0';
  volumeSlider.max = '100';
  volumeSlider.step = '1';
  volumeSlider.value = String(Math.round(settings.audio.masterVolume * 100));
  Object.assign(volumeSlider.style, {
    flex: '1',
    accentColor: '#6ea8ff',
  });

  const updateVolumeLabel = (value: number) => {
    volumeValue.textContent = String(Math.round(value * 100));
  };
  updateVolumeLabel(settings.audio.masterVolume);

  volumeSlider.addEventListener('input', () => {
    const value = Math.max(0, Math.min(1, Number(volumeSlider.value) / 100));
    updateVolumeLabel(value);
    settingsStore.apply({ audio: { masterVolume: value } });
  });

  volumeRow.appendChild(volumeSlider);
  volumeRow.appendChild(volumeValue);

  const recordLabel = document.createElement('div');
  recordLabel.textContent = 'Snapshots';
  Object.assign(recordLabel.style, {
    marginTop: '12px',
    marginBottom: '6px',
    color: '#b6c2d4',
  });

  const recordRow = document.createElement('div');
  Object.assign(recordRow.style, {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  });

  const commentInput = document.createElement('input');
  commentInput.type = 'text';
  commentInput.placeholder = 'Comment';
  Object.assign(commentInput.style, {
    flex: '1',
    minWidth: '0',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  });

  commentInput.addEventListener('focus', () => {
    pausedByInput = true;
    updatePaused();
  });
  commentInput.addEventListener('blur', () => {
    pausedByInput = false;
    updatePaused();
  });

  const getDirectoryPicker = () =>
    (
      window as Window & {
        showDirectoryPicker?: () => Promise<FileSystemDirectoryHandle>;
      }
    ).showDirectoryPicker;

  const ensureSnapshotDirectory = async (): Promise<boolean> => {
    const picker = getDirectoryPicker();
    if (!picker) {
      updateFolderStatus('Folder access not supported in this browser.');
      snapshotDirHandle = null;
      return false;
    }

    try {
      if (!snapshotDirHandle) {
        snapshotDirHandle = await picker();
      }

      let permission: PermissionState = 'granted';
      if (snapshotDirHandle.queryPermission) {
        permission = await snapshotDirHandle.queryPermission({
          mode: 'readwrite',
        });
      }
      if (permission !== 'granted' && snapshotDirHandle.requestPermission) {
        permission = await snapshotDirHandle.requestPermission({
          mode: 'readwrite',
        });
      }
      if (permission !== 'granted') {
        updateFolderStatus('Folder access denied.');
        snapshotDirHandle = null;
        return false;
      }

      updateFolderStatus(`Folder: ${snapshotDirHandle.name}`);
      updateRecorderUI();
      return true;
    } catch {
      updateFolderStatus('Folder selection cancelled.');
      return false;
    }
  };

  const folderButton = document.createElement('button');
  folderButton.textContent = 'Select Folder';
  Object.assign(folderButton.style, {
    background: '#0b0f14',
    color: '#b6c2d4',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
    cursor: 'pointer',
  });

  const recordButton = document.createElement('button');
  Object.assign(recordButton.style, {
    flex: '1',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
    cursor: 'pointer',
  });

  const discardButton = document.createElement('button');
  discardButton.textContent = 'Discard';
  Object.assign(discardButton.style, {
    background: '#0b0f14',
    color: '#b6c2d4',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
    cursor: 'pointer',
    display: 'none',
  });

  const recordStatus = document.createElement('div');
  Object.assign(recordStatus.style, {
    marginTop: '6px',
    fontSize: '12px',
    color: '#8fa0b8',
  });

  const folderStatus = document.createElement('div');
  Object.assign(folderStatus.style, {
    marginTop: '6px',
    fontSize: '12px',
    color: '#8fa0b8',
  });

  const updateFolderStatus = (text: string) => {
    folderStatus.textContent = text;
  };

  updateRecorderUI = () => {
    if (recorder.isRecording) {
      recordButton.textContent = snapshotDirHandle
        ? 'Stop & Save'
        : 'Stop & Download';
      discardButton.style.display = 'inline-block';
      recordStatus.textContent = `Samples: ${recorder.sampleCount}`;
    } else {
      recordButton.textContent = 'Start Recording';
      discardButton.style.display = 'none';
      recordStatus.textContent = 'Idle';
    }
  };

  startRecordingSession = () => {
    if (recorder.isRecording) return;
    lastSnapshotKey = null;
    recorder.start(settingsStore.get(), ROWS, COLS, commentInput.value, {
      id: selectedMode.id,
      options: { ...selectedModeOptions },
    });
    updateRecorderUI();
  };

  stopRecordingSession = async (
    options: { promptForFolder?: boolean } = {},
  ) => {
    if (!recorder.isRecording) return;
    const session = recorder.stop();
    updateRecorderUI();
    lastSnapshotKey = null;
    if (!session || session.samples.length === 0) return;
    if (useRemoteUpload) return;

    if (options.promptForFolder && !snapshotDirHandle) {
      const ready = await ensureSnapshotDirectory();
      if (!ready) {
        updateFolderStatus('Auto-save unavailable. Downloading instead.');
      }
    }

    if (snapshotDirHandle) {
      void saveSnapshotSessionToDirectory(session, snapshotDirHandle)
        .then(() => updateFolderStatus(`Saved: ${session.meta.id}`))
        .catch(() => {
          updateFolderStatus('Save failed. Downloading instead.');
          downloadSnapshotSession(session);
        });
      return;
    }

    downloadSnapshotSession(session);
    updateFolderStatus(`Downloaded: ${session.meta.id}`);
  };

  restartRecordingSession = () => {
    void stopRecordingSession();
    startRecordingSession();
  };

  commentInput.addEventListener('input', () => {
    if (recorder.isRecording) {
      recorder.setComment(commentInput.value);
    }
  });

  updateFolderStatus('No folder selected.');

  if (useRemoteUpload) {
    folderButton.style.display = 'none';
    folderStatus.style.display = 'none';
    recordRow.style.display = 'none';
    recordStatus.textContent = 'Auto upload enabled.';
  }

  folderButton.addEventListener('click', async () => {
    await ensureSnapshotDirectory();
  });

  recordButton.addEventListener('click', async () => {
    if (useRemoteUpload) return;
    if (!recorder.isRecording) {
      startRecordingSession();
      return;
    }

    await stopRecordingSession({ promptForFolder: true });
  });

  discardButton.addEventListener('click', () => {
    if (useRemoteUpload) return;
    recorder.discard();
    updateRecorderUI();
    lastSnapshotKey = null;
  });

  window.addEventListener('beforeunload', () => {
    if (recorder.isRecording) {
      recorder.stop();
    }
  });

  updateRecorderUI();

  recordRow.appendChild(commentInput);
  recordRow.appendChild(recordButton);
  recordRow.appendChild(discardButton);
  settingsPanel.appendChild(recordLabel);
  settingsPanel.appendChild(folderButton);
  settingsPanel.appendChild(folderStatus);
  settingsPanel.appendChild(recordRow);
  settingsPanel.appendChild(recordStatus);
  uiLayer.appendChild(settingsPanel);

  const toolLayer = document.createElement('div');
  Object.assign(toolLayer.style, {
    position: 'absolute',
    inset: '0',
    pointerEvents: 'none',
    display: 'none',
  });
  uiLayer.appendChild(toolLayer);

  const toolColumn = document.createElement('div');
  Object.assign(toolColumn.style, {
    position: 'absolute',
    left: `${SETTINGS_X}px`,
    top: `${SETTINGS_Y}px`,
    width: `${SETTINGS_PANEL_WIDTH}px`,
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  });
  toolLayer.appendChild(toolColumn);

  const toolPanelMaxHeight = Math.max(
    0,
    PLAY_HEIGHT - SETTINGS_Y - OUTER_MARGIN - 48,
  );
  const toolPanel = document.createElement('div');
  Object.assign(toolPanel.style, {
    width: '100%',
    maxHeight: `${toolPanelMaxHeight}px`,
    overflowY: 'auto',
    padding: '8px',
    background: '#121a24',
    color: '#e2e8f0',
    border: '2px solid #0b0f14',
    borderRadius: '6px',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    fontSize: '13px',
    pointerEvents: 'auto',
  });
  toolColumn.appendChild(toolPanel);

  const toolInfo = document.createElement('div');
  toolInfo.textContent =
    "How to use:\nSelect all the pieces you would want to fall in this situation, then click 'Next'\n\n" +
    'Your answers will be saved and used to train ML piece generator models.';
  Object.assign(toolInfo.style, {
    color: '#b6c2d4',
    fontSize: '13px',
    lineHeight: '1.4',
    textAlign: 'center',
    whiteSpace: 'pre-line',
    pointerEvents: 'none',
  });
  toolColumn.appendChild(toolInfo);

  const toolTitle = document.createElement('div');
  toolTitle.textContent = 'WISH UPON A BLOCK';
  Object.assign(toolTitle.style, {
    marginBottom: '8px',
    color: '#b6c2d4',
    letterSpacing: '0.5px',
  });
  toolPanel.appendChild(toolTitle);

  const toolInputButton = document.createElement('button');
  toolInputButton.textContent = 'Select Snapshot Folder';
  Object.assign(toolInputButton.style, {
    width: '100%',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '8px',
    fontSize: '12px',
    cursor: 'pointer',
  });
  toolPanel.appendChild(toolInputButton);

  const toolInputStatus = document.createElement('div');
  Object.assign(toolInputStatus.style, {
    marginTop: '6px',
    fontSize: '12px',
    color: '#8fa0b8',
    whiteSpace: 'pre-line',
  });
  toolInputStatus.textContent = 'Source: Local (select folder).';
  toolPanel.appendChild(toolInputStatus);

  const toolModeLabel = document.createElement('div');
  toolModeLabel.textContent = 'Mode Filter';
  Object.assign(toolModeLabel.style, {
    marginTop: '10px',
    fontSize: '12px',
    color: '#b6c2d4',
  });
  toolPanel.appendChild(toolModeLabel);

  const toolModeSelect = document.createElement('select');
  Object.assign(toolModeSelect.style, {
    width: '100%',
    marginTop: '6px',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  });
  toolPanel.appendChild(toolModeSelect);

  const toolOutputButton = document.createElement('button');
  toolOutputButton.textContent = 'Select Output Folder';
  Object.assign(toolOutputButton.style, {
    width: '100%',
    marginTop: '10px',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '8px',
    fontSize: '12px',
    cursor: 'pointer',
  });
  toolPanel.appendChild(toolOutputButton);

  const toolOutputStatus = document.createElement('div');
  Object.assign(toolOutputStatus.style, {
    marginTop: '6px',
    fontSize: '12px',
    color: '#8fa0b8',
  });
  toolOutputStatus.textContent = 'No output folder.';
  toolPanel.appendChild(toolOutputStatus);
  if (toolUsesRemote) {
    toolInputButton.style.display = 'none';
    toolOutputButton.style.display = 'none';
    toolOutputStatus.style.display = 'none';
    toolInputStatus.textContent = 'Source: Online';
  }

  const toolSampleStatus = document.createElement('div');
  Object.assign(toolSampleStatus.style, {
    marginTop: '10px',
    fontSize: '12px',
    color: '#8fa0b8',
  });
  toolSampleStatus.textContent = 'Sample: -';
  toolPanel.appendChild(toolSampleStatus);

  const toolActionStatus = document.createElement('div');
  Object.assign(toolActionStatus.style, {
    marginTop: '6px',
    fontSize: '12px',
    color: '#8fa0b8',
  });
  toolActionStatus.textContent = '';
  toolPanel.appendChild(toolActionStatus);

  const toolBackButton = document.createElement('button');
  toolBackButton.textContent = 'BACK TO MENU';
  Object.assign(toolBackButton.style, {
    width: '100%',
    marginTop: '12px',
    background: '#0b0f14',
    color: '#b6c2d4',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '8px',
    fontSize: '12px',
    cursor: 'pointer',
  });
  toolPanel.appendChild(toolBackButton);

  const toolPieceCellPx = Math.max(10, Math.round(BOARD_CELL_PX / 2));
  const toolPiecePreviewPx = toolPieceCellPx * 4;
  const toolPieceButtonHeight = toolPiecePreviewPx + 10;
  const toolPieceButtonWidth = toolPiecePreviewPx + 8;
  const toolPieceGap = 6;
  const toolPieceRowWidth =
    PIECES.length * toolPieceButtonWidth + (PIECES.length - 1) * toolPieceGap;
  const toolPieceRowLeft = BOARD_X + (BOARD_WIDTH - toolPieceRowWidth) / 2;
  const toolPieceRowTop = BOARD_Y + ROWS * BOARD_CELL_PX + PANEL_GAP;

  const toolPieceRow = document.createElement('div');
  Object.assign(toolPieceRow.style, {
    position: 'absolute',
    left: `${toolPieceRowLeft}px`,
    top: `${toolPieceRowTop}px`,
    width: `${toolPieceRowWidth}px`,
    display: 'flex',
    gap: `${toolPieceGap}px`,
    alignItems: 'center',
    pointerEvents: 'auto',
  });
  toolLayer.appendChild(toolPieceRow);

  const toolNextButton = document.createElement('button');
  toolNextButton.textContent = 'NEXT';
  Object.assign(toolNextButton.style, {
    position: 'absolute',
    left: `${BOARD_X}px`,
    top: `${toolPieceRowTop + toolPieceButtonHeight + 8}px`,
    width: `${BOARD_WIDTH}px`,
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '6px',
    padding: '10px 12px',
    fontSize: '13px',
    cursor: 'pointer',
    pointerEvents: 'auto',
  });
  toolLayer.appendChild(toolNextButton);

  const toolPieceButtons = new Map<PieceKind, HTMLButtonElement>();
  const drawPiecePreview = (
    canvas: HTMLCanvasElement,
    piece: PieceKind,
  ): void => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'transparent';
    const color = PIECE_COLORS[piece];
    const pad = Math.max(1, Math.floor(toolPieceCellPx / 6));
    for (const [x, y] of TETROMINOES[piece][0]) {
      const px = x * toolPieceCellPx;
      const py = y * toolPieceCellPx;
      ctx.fillStyle = color;
      ctx.fillRect(
        px + pad,
        py + pad,
        toolPieceCellPx - pad * 2,
        toolPieceCellPx - pad * 2,
      );
    }
  };

  for (const piece of PIECES) {
    const btn = document.createElement('button');
    Object.assign(btn.style, {
      flex: '1',
      minWidth: `${toolPieceButtonWidth}px`,
      height: `${toolPieceButtonHeight}px`,
      background: '#0b0f14',
      color: '#0b0f14',
      border: '2px solid #1f2a37',
      borderRadius: '6px',
      padding: '4px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      cursor: 'pointer',
    });
    const canvas = document.createElement('canvas');
    canvas.width = toolPiecePreviewPx;
    canvas.height = toolPiecePreviewPx;
    canvas.style.display = 'block';
    drawPiecePreview(canvas, piece);
    btn.appendChild(canvas);
    toolPieceRow.appendChild(btn);
    toolPieceButtons.set(piece, btn);
  }

  let toolOutputDirHandle: FileSystemDirectoryHandle | null = null;
  let toolSnapshotsAll: Array<{ name: string; session: SnapshotSession }> = [];
  let toolSnapshots: Array<{ name: string; session: SnapshotSession }> = [];
  let toolTotalSamples = 0;
  let toolTotalSamplesAll = 0;
  let toolSampleOffsets: number[] = [];
  let toolModeFilter = 'all';
  let toolActive = false;
  let currentSample: {
    file: { name: string; session: SnapshotSession };
    index: number;
    board: Board;
    raw: number[][];
    hold: PieceKind | null;
  } | null = null;
  let selectedLabels: PieceKind[] = [];
  let toolBusy = false;
  let labelIndex: Record<string, number> = {};

  const updateLabelButtons = () => {
    for (const [piece, btn] of toolPieceButtons) {
      const selected = selectedLabels.includes(piece);
      btn.style.borderColor = selected ? '#ffffff' : '#1f2a37';
      btn.style.boxShadow = selected
        ? '0 0 0 1px rgba(255,255,255,0.5)'
        : 'none';
    }
  };

  const clearLabelSelection = () => {
    selectedLabels = [];
    updateLabelButtons();
  };

  for (const [piece, btn] of toolPieceButtons) {
    btn.addEventListener('click', () => {
      const idx = selectedLabels.indexOf(piece);
      if (idx >= 0) {
        selectedLabels.splice(idx, 1);
      } else {
        selectedLabels.push(piece);
      }
      updateLabelButtons();
    });
  }

  const decodeBoard = (raw: number[][], order?: readonly string[]): Board => {
    const resolvedOrder = order ?? PIECES;
    return raw.map((row) =>
      row.map((value) => {
        if (value <= 0) return null;
        const piece = resolvedOrder[value - 1] ?? PIECES[value - 1];
        return piece as PieceKind;
      }),
    );
  };

  const decodeHold = (
    hold: number | undefined,
    order?: readonly string[],
  ): PieceKind | null => {
    if (!hold || hold <= 0) return null;
    const resolvedOrder = order ?? PIECES;
    const piece = resolvedOrder[hold - 1] ?? PIECES[hold - 1];
    return (piece as PieceKind) ?? null;
  };

  const encodeBoardString = (raw: number[][]): string =>
    raw.map((row) => row.join('')).join('/');

  const requestDirectoryAccess = async (
    handle: FileSystemDirectoryHandle,
    mode: 'read' | 'readwrite',
  ): Promise<boolean> => {
    let permission: PermissionState = 'granted';
    if (handle.queryPermission) {
      permission = await handle.queryPermission({ mode });
    }
    if (permission !== 'granted' && handle.requestPermission) {
      permission = await handle.requestPermission({ mode });
    }
    return permission === 'granted';
  };

  const writeFileInDir = async (
    dir: FileSystemDirectoryHandle,
    name: string,
    contents: string,
  ): Promise<void> => {
    const handle = await dir.getFileHandle(name, { create: true });
    const writable = await handle.createWritable();
    await writable.write(contents);
    await writable.close();
  };

  const appendJsonl = async (
    dir: FileSystemDirectoryHandle,
    name: string,
    line: string,
  ): Promise<void> => {
    const handle = await dir.getFileHandle(name, { create: true });
    const file = await handle.getFile();
    const existing = await file.text();
    const prefix = existing && !existing.endsWith('\n') ? '\n' : '';
    const next = `${existing}${prefix}${line}\n`;
    await writeFileInDir(dir, name, next);
  };

  const loadLabelIndex = async (): Promise<void> => {
    if (toolUsesRemote) {
      labelIndex = {};
      return;
    }
    if (!toolOutputDirHandle) return;
    try {
      const handle = await toolOutputDirHandle.getFileHandle(
        'labeling_index.json',
        { create: true },
      );
      const file = await handle.getFile();
      const text = await file.text();
      labelIndex = text ? (JSON.parse(text) as Record<string, number>) : {};
    } catch {
      labelIndex = {};
    }
  };

  const saveLabelIndex = async (): Promise<void> => {
    if (toolUsesRemote) return;
    if (!toolOutputDirHandle) return;
    await writeFileInDir(
      toolOutputDirHandle,
      'labeling_index.json',
      JSON.stringify(labelIndex),
    );
  };

  const updateToolSampleStatus = (text: string) => {
    toolSampleStatus.textContent = text;
  };

  const updateToolActionStatus = (text: string) => {
    toolActionStatus.textContent = text;
  };

  const getModeLabel = (id: string): string => {
    if (id === 'unknown') return 'Unknown';
    if (id === 'default' || id === 'cheese' || id === 'charcuterie') {
      return getMode(id).label.toUpperCase();
    }
    return id.toUpperCase();
  };

  const refreshToolModeOptions = () => {
    toolModeSelect.innerHTML = '';
    const addOption = (value: string, label: string) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      toolModeSelect.appendChild(option);
    };

    addOption('all', 'All Modes');
    if (toolUsesRemote) {
      const knownModes = ['default', 'cheese', 'charcuterie'];
      for (const modeId of knownModes) {
        addOption(modeId, getModeLabel(modeId));
      }
      addOption('unknown', 'Unknown');
      toolModeSelect.value = toolModeFilter;
      toolModeSelect.disabled = false;
      return;
    }
    const counts = new Map<string, number>();
    for (const file of toolSnapshotsAll) {
      const modeId = file.session.meta.mode?.id ?? 'unknown';
      counts.set(
        modeId,
        (counts.get(modeId) ?? 0) + file.session.samples.length,
      );
    }

    const modeIds = Array.from(counts.keys()).sort((a, b) => {
      if (a === 'unknown') return 1;
      if (b === 'unknown') return -1;
      return a.localeCompare(b);
    });
    for (const modeId of modeIds) {
      const label = `${getModeLabel(modeId)} (${counts.get(modeId) ?? 0})`;
      addOption(modeId, label);
    }

    if (toolModeFilter !== 'all' && !counts.has(toolModeFilter)) {
      toolModeFilter = 'all';
    }
    toolModeSelect.value = toolModeFilter;
    toolModeSelect.disabled = toolSnapshotsAll.length === 0;
  };

  const rebuildToolSampleOffsets = () => {
    toolSampleOffsets = [];
    let running = 0;
    for (const file of toolSnapshots) {
      toolSampleOffsets.push(running);
      running += file.session.samples.length;
    }
    toolTotalSamples = running;
  };

  const buildToolApiUrl = (path: string): URL => {
    const base = uploadBaseUrl.replace(/\/+$/, '');
    return new URL(`${base}${path}`, window.location.origin);
  };

  const fetchRemoteSample = async (): Promise<SnapshotSession | null> => {
    const url = buildToolApiUrl('/snapshots/random');
    if (toolModeFilter !== 'all') {
      url.searchParams.set('mode', toolModeFilter);
    }
    const res = await fetch(url.toString());
    if (!res.ok) {
      return null;
    }
    const payload = (await res.json()) as {
      id?: number | string;
      createdAt?: string;
      meta?: {
        session?: SnapshotSession['meta'];
        sample?: Record<string, unknown>;
      };
      board?: number[][];
    };
    const sessionMeta = payload.meta?.session;
    if (!sessionMeta || !payload.board) return null;
    const sampleMeta = payload.meta?.sample ?? {};
    const readNumber = (value: unknown, fallback = 0): number => {
      return typeof value === 'number' && Number.isFinite(value)
        ? value
        : fallback;
    };
    const holdValue =
      typeof sampleMeta.hold === 'number' && Number.isFinite(sampleMeta.hold)
        ? sampleMeta.hold
        : undefined;
    const sample = {
      index: readNumber(sampleMeta.index),
      timeMs: readNumber(sampleMeta.timeMs),
      board: payload.board,
      hold: holdValue,
    };
    const session: SnapshotSession = {
      meta: sessionMeta,
      samples: [sample],
    };
    return session;
  };

  const applyToolModeFilter = async () => {
    if (toolUsesRemote) {
      const filterLabel =
        toolModeFilter === 'all' ? 'All Modes' : getModeLabel(toolModeFilter);
      toolInputStatus.textContent = `Source: Online\nFilter: ${filterLabel}`;
      if (!toolActive) return;
      await showNextToolSample();
      return;
    }
    if (toolModeFilter === 'all') {
      toolSnapshots = toolSnapshotsAll;
    } else {
      toolSnapshots = toolSnapshotsAll.filter((file) => {
        const modeId = file.session.meta.mode?.id ?? 'unknown';
        return modeId === toolModeFilter;
      });
    }
    rebuildToolSampleOffsets();
    const filterLabel =
      toolModeFilter === 'all' ? 'All Modes' : getModeLabel(toolModeFilter);
    toolInputStatus.textContent =
      `Source: Local\n` +
      `Loaded ${toolSnapshotsAll.length} files (${toolTotalSamplesAll} samples).\n` +
      `Filter: ${filterLabel} (${toolTotalSamples} samples).`;
    await showNextToolSample();
  };

  const showNextToolSample = async (): Promise<void> => {
    if (toolUsesRemote) {
      const session = await fetchRemoteSample();
      if (!session || session.samples.length === 0) {
        currentSample = null;
        updateToolSampleStatus('Sample: -');
        if (toolActive) {
          renderer.renderBoardOnly(makeBoard(), null);
        }
        return;
      }
      const sample = session.samples[0];
      const rawBoard = sample.board;
      const board = decodeBoard(rawBoard, session.meta.pieceOrder);
      const hold = decodeHold(sample.hold, session.meta.pieceOrder);
      const fileName = `remote:${session.meta.id ?? 'unknown'}`;
      currentSample = {
        file: { name: fileName, session },
        index: sample.index,
        board,
        raw: rawBoard,
        hold,
      };
      const key = `${fileName}#${sample.index}`;
      labelIndex[key] = (labelIndex[key] ?? 0) + 1;
      updateToolSampleStatus(
        `Source: ${fileName} Sample: ${sample.index} Shown: ${labelIndex[key]}`,
      );
      clearLabelSelection();
      if (toolActive) {
        renderer.renderBoardOnly(board, hold);
      }
      return;
    }
    if (toolSnapshots.length === 0 || toolTotalSamples === 0) {
      currentSample = null;
      updateToolSampleStatus('Sample: -');
      renderer.renderBoardOnly(makeBoard(), null);
      return;
    }

    const pick = Math.floor(Math.random() * toolTotalSamples);
    let fileIdx = 0;
    for (let i = 0; i < toolSampleOffsets.length; i++) {
      const start = toolSampleOffsets[i];
      const end = start + toolSnapshots[i].session.samples.length;
      if (pick >= start && pick < end) {
        fileIdx = i;
        break;
      }
    }

    const file = toolSnapshots[fileIdx];
    const sampleIdx = pick - toolSampleOffsets[fileIdx];
    const sample = file.session.samples[sampleIdx];
    const rawBoard = sample.board;
    const board = decodeBoard(rawBoard, file.session.meta.pieceOrder);
    const hold = decodeHold(sample.hold, file.session.meta.pieceOrder);
    currentSample = {
      file,
      index: sample.index,
      board,
      raw: rawBoard,
      hold,
    };
    const key = `${file.name}#${sample.index}`;
    labelIndex[key] = (labelIndex[key] ?? 0) + 1;
    if (toolOutputDirHandle) {
      await saveLabelIndex();
    }
    updateToolSampleStatus(
      `File: ${file.name} Sample: ${sample.index} Shown: ${labelIndex[key]}`,
    );
    clearLabelSelection();
    renderer.renderBoardOnly(board, hold);
  };

  toolInputButton.addEventListener('click', async () => {
    if (toolUsesRemote) return;
    const picker = getDirectoryPicker();
    if (!picker) {
      toolInputStatus.textContent = 'Folder access not supported.';
      return;
    }
    try {
      const handle = await picker();
      const granted = await requestDirectoryAccess(handle, 'read');
      if (!granted) {
        toolInputStatus.textContent = 'Folder access denied.';
        return;
      }
      const files: Array<{ name: string; session: SnapshotSession }> = [];
      for await (const entry of handle.values()) {
        if (entry.kind !== 'file') continue;
        if (!entry.name.endsWith('.json')) continue;
        const fileHandle = entry as FileSystemFileHandle;
        const file = await fileHandle.getFile();
        const text = await file.text();
        const session = JSON.parse(text) as SnapshotSession;
        if (!session?.samples?.length) continue;
        files.push({ name: entry.name, session });
      }
      files.sort((a, b) => a.name.localeCompare(b.name));
      toolSnapshotsAll = files;
      toolTotalSamplesAll = toolSnapshotsAll.reduce(
        (sum, file) => sum + file.session.samples.length,
        0,
      );
      refreshToolModeOptions();
      await applyToolModeFilter();
      updateToolActionStatus(`Loaded ${toolSnapshotsAll.length} files.`);
    } catch {
      toolInputStatus.textContent = 'Folder selection cancelled.';
    }
  });

  toolModeSelect.addEventListener('change', async () => {
    toolModeFilter = toolModeSelect.value;
    await applyToolModeFilter();
  });

  if (toolUsesRemote) {
    refreshToolModeOptions();
  }

  toolOutputButton.addEventListener('click', async () => {
    if (toolUsesRemote) return;
    const picker = getDirectoryPicker();
    if (!picker) {
      toolOutputStatus.textContent = 'Folder access not supported.';
      return;
    }
    try {
      toolOutputDirHandle = await picker();
      const granted = await requestDirectoryAccess(
        toolOutputDirHandle,
        'readwrite',
      );
      if (!granted) {
        toolOutputStatus.textContent = 'Folder access denied.';
        toolOutputDirHandle = null;
        return;
      }
      await loadLabelIndex();
      toolOutputStatus.textContent = `Output: ${toolOutputDirHandle.name}`;
      updateToolActionStatus(`Output set: ${toolOutputDirHandle.name}`);
    } catch {
      toolOutputStatus.textContent = 'Folder selection cancelled.';
    }
  });

  toolNextButton.addEventListener('click', async () => {
    if (toolBusy) return;
    if (!currentSample) {
      updateToolSampleStatus('Sample: -');
      return;
    }
    if (!useRemoteUpload && !toolOutputDirHandle) {
      toolOutputStatus.textContent = 'Select output folder first.';
      return;
    }
    toolBusy = true;
    try {
      const key = `${currentSample.file.name}#${currentSample.index}`;
      const record = {
        createdAt: new Date().toISOString(),
        source: {
          file: currentSample.file.name,
          sessionId: currentSample.file.session.meta.id,
          sampleIndex: currentSample.index,
          shownCount: labelIndex[key] ?? 1,
        },
        board: encodeBoardString(currentSample.raw),
        hold: currentSample.hold,
        labels: [...selectedLabels],
      };
      if (useRemoteUpload) {
        await uploadClient.enqueueLabel({
          createdAt: record.createdAt,
          data: record,
        });
        updateToolActionStatus(
          `Uploaded label for ${currentSample.file.name} #${currentSample.index}`,
        );
      } else {
        await appendJsonl(
          toolOutputDirHandle!,
          'labels.jsonl',
          JSON.stringify(record),
        );
        updateToolActionStatus(
          `Saved label for ${currentSample.file.name} #${currentSample.index}`,
        );
      }
      await showNextToolSample();
    } finally {
      toolBusy = false;
    }
  });

  toolBackButton.addEventListener('click', () => setScreen('menu'));

  const menuLayer = document.createElement('div');
  Object.assign(menuLayer.style, {
    position: 'absolute',
    inset: '0',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'rgba(11, 15, 20, 0.7)',
    pointerEvents: 'auto',
  });

  const makeMenuPanel = () => {
    const panel = document.createElement('div');
    Object.assign(panel.style, {
      width: '240px',
      padding: '16px',
      background: '#121a24',
      color: '#e2e8f0',
      border: '2px solid #0b0f14',
      borderRadius: '8px',
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
      textAlign: 'center',
      fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
      fontSize: '14px',
    });
    return panel;
  };

  const menuMainPanel = makeMenuPanel();
  const playPanel = makeMenuPanel();
  const optionsPanel = makeMenuPanel();
  const aboutPanel = makeMenuPanel();
  const cheesePanel = makeMenuPanel();
  const charcuteriePanel = makeMenuPanel();
  const toolsPanel = makeMenuPanel();
  const feedbackPanel = makeMenuPanel();
  const butterfingerPanel = makeMenuPanel();
  const playMenuRow = document.createElement('div');
  Object.assign(playPanel.style, {
    minHeight: '240px',
    display: 'flex',
  });
  Object.assign(optionsPanel.style, {
    minHeight: '240px',
    display: 'none',
  });
  Object.assign(aboutPanel.style, {
    minHeight: '260px',
    width: '300px',
    display: 'none',
    textAlign: 'left',
  });
  Object.assign(cheesePanel.style, {
    minHeight: '240px',
    display: 'none',
  });
  Object.assign(charcuteriePanel.style, {
    minHeight: '240px',
    display: 'none',
  });
  Object.assign(toolsPanel.style, {
    minHeight: '240px',
    display: 'none',
  });
  Object.assign(feedbackPanel.style, {
    minHeight: '260px',
    width: '320px',
    display: 'none',
    textAlign: 'left',
  });
  Object.assign(butterfingerPanel.style, {
    minHeight: '240px',
    width: '240px',
    display: SHOW_DEV_TOOLS ? 'flex' : 'none',
  });
  Object.assign(playMenuRow.style, {
    display: 'none',
    gap: '16px',
    alignItems: 'flex-start',
  });

  const makeMenuButton = (labelText: string) => {
    const btn = document.createElement('button');
    btn.textContent = labelText;
    Object.assign(btn.style, {
      background: '#0b0f14',
      color: '#e2e8f0',
      border: '1px solid #1f2a37',
      borderRadius: '6px',
      padding: '10px 12px',
      fontSize: '13px',
      cursor: 'pointer',
    });
    return btn;
  };

  const playButton = makeMenuButton('PLAY');
  const optionsButton = makeMenuButton('OPTIONS');
  const toolsButton = makeMenuButton('TOOLS');
  const aboutButton = makeMenuButton('ABOUT');

  menuMainPanel.appendChild(playButton);
  menuMainPanel.appendChild(optionsButton);
  menuMainPanel.appendChild(toolsButton);
  menuMainPanel.appendChild(aboutButton);

  const optionsTitle = document.createElement('div');
  optionsTitle.textContent = 'OPTIONS';
  Object.assign(optionsTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '4px',
  });

  const controlsTitle = document.createElement('div');
  controlsTitle.textContent = 'CONTROLS';
  Object.assign(controlsTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginTop: '12px',
    marginBottom: '4px',
  });

  const controlsList = document.createElement('div');
  Object.assign(controlsList.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  });

  const controlsResetButton = makeMenuButton('RESET CONTROLS');
  Object.assign(controlsResetButton.style, {
    marginTop: '6px',
  });

  const optionsBackButton = makeMenuButton('BACK');
  Object.assign(optionsBackButton.style, {
    marginTop: 'auto',
  });

  optionsPanel.appendChild(optionsTitle);
  optionsPanel.appendChild(volumeLabel);
  optionsPanel.appendChild(volumeRow);
  optionsPanel.appendChild(controlsTitle);
  optionsPanel.appendChild(controlsList);
  optionsPanel.appendChild(controlsResetButton);
  optionsPanel.appendChild(optionsBackButton);

  const aboutTitle = document.createElement('div');
  aboutTitle.textContent = 'ABOUT';
  Object.assign(aboutTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '4px',
  });

  const aboutBody = document.createElement('div');
  aboutBody.textContent =
    'Wish Upon a Block is a lightweight low-latency guideline tetromino game. ' +
    'It is meant to recreate the feeling of "Tetris effect" when every next piece is "just right". ' +
    'Play and try out the ML powered "Wish Upon a Block" piece generator.';
  Object.assign(aboutBody.style, {
    color: '#b6c2d4',
    fontSize: '12px',
    lineHeight: '1.4',
    textAlign: 'center',
  });

  const creditsTitle = document.createElement('div');
  creditsTitle.textContent = 'CREDITS';
  Object.assign(creditsTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginTop: '12px',
    marginBottom: '4px',
  });

  const creditsList = document.createElement('div');
  Object.assign(creditsList.style, {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    color: '#b6c2d4',
    fontSize: '12px',
  });

  const creditsName = document.createElement('div');
  creditsName.textContent = 'Максим Никитин (Maksim Nikitin)';

  const makeCreditsLink = (href: string, text: string) => {
    const link = document.createElement('a');
    link.href = href;
    link.textContent = text;
    link.target = '_blank';
    link.rel = 'noreferrer';
    Object.assign(link.style, {
      color: '#b6c2d4',
      textDecoration: 'none',
      wordBreak: 'break-all',
    });
    return link;
  };

  const githubLinkLine = makeCreditsLink(
    'https://github.com/icanfast/wishuponablock',
    'https://github.com/icanfast/wishuponablock',
  );
  const emailLinkLine = makeCreditsLink(
    'mailto:nikitin.maxim.94@gmail.com',
    'nikitin.maxim.94@gmail.com',
  );
  const telegramLinkLine = makeCreditsLink(
    'https://t.me/icanfast',
    't.me/icanfast',
  );

  creditsList.appendChild(creditsName);
  creditsList.appendChild(githubLinkLine);
  creditsList.appendChild(emailLinkLine);
  creditsList.appendChild(telegramLinkLine);

  const aboutBackButton = makeMenuButton('BACK');
  Object.assign(aboutBackButton.style, {
    marginTop: 'auto',
  });

  aboutPanel.appendChild(aboutTitle);
  aboutPanel.appendChild(aboutBody);
  aboutPanel.appendChild(creditsTitle);
  aboutPanel.appendChild(creditsList);
  aboutPanel.appendChild(aboutBackButton);

  type KeyBindingKey = keyof Settings['input']['bindings'];
  const keybindButtons = new Map<KeyBindingKey, HTMLButtonElement>();
  let rebindingKey: KeyBindingKey | null = null;
  let rebindingButton: HTMLButtonElement | null = null;

  const formatKeyLabel = (code: string): string => {
    if (!code) return '';
    if (code === 'Space') return 'Space';
    if (code.startsWith('Key')) return code.slice(3);
    if (code.startsWith('Digit')) return code.slice(5);
    if (code.startsWith('Arrow')) return code.slice(5);
    if (code.startsWith('Numpad')) return `Num ${code.slice(6)}`;
    return code.replace(/([a-z])([A-Z])/g, '$1 $2');
  };

  const updateKeybindButtons = (bindings: Settings['input']['bindings']) => {
    for (const [key, button] of keybindButtons) {
      if (rebindingKey === key) continue;
      const label = formatKeyLabel(bindings[key]);
      button.textContent = label || '\u00a0';
    }
  };

  const cancelRebind = () => {
    if (rebindingKey && rebindingButton) {
      const current = settingsStore.get().input.bindings;
      rebindingButton.textContent = formatKeyLabel(current[rebindingKey]);
      rebindingButton.blur();
    }
    rebindingKey = null;
    rebindingButton = null;
  };

  const startRebind = (key: KeyBindingKey, button: HTMLButtonElement) => {
    cancelRebind();
    rebindingKey = key;
    rebindingButton = button;
    button.textContent = 'Press a key';
  };

  const reservedCodes = new Set([
    'ControlLeft',
    'ControlRight',
    'AltLeft',
    'AltRight',
    'MetaLeft',
    'MetaRight',
  ]);

  window.addEventListener(
    'keydown',
    (event) => {
      if (!rebindingKey || !rebindingButton) return;
      event.preventDefault();
      event.stopPropagation();
      if (event.code === 'Escape') {
        cancelRebind();
        return;
      }
      if (reservedCodes.has(event.code)) return;
      const currentInput = settingsStore.get().input;
      const nextBindings = { ...currentInput.bindings };
      for (const [action, code] of Object.entries(nextBindings)) {
        if (action === rebindingKey) continue;
        if (code === event.code) {
          nextBindings[action as KeyBindingKey] = '';
        }
      }
      updateKeybindButtons(nextBindings);
      settingsStore.apply({
        input: {
          ...currentInput,
          bindings: { ...nextBindings, [rebindingKey]: event.code },
        },
      });
      cancelRebind();
    },
    true,
  );

  const keybindConfig: Array<{ key: KeyBindingKey; label: string }> = [
    { key: 'moveLeft', label: 'Move Left' },
    { key: 'moveRight', label: 'Move Right' },
    { key: 'softDrop', label: 'Soft Drop' },
    { key: 'hardDrop', label: 'Hard Drop' },
    { key: 'rotateCW', label: 'Rotate CW' },
    { key: 'rotateCCW', label: 'Rotate CCW' },
    { key: 'rotate180', label: 'Rotate 180' },
    { key: 'hold', label: 'Hold' },
    { key: 'restart', label: 'Restart' },
  ];

  for (const item of keybindConfig) {
    const row = document.createElement('div');
    Object.assign(row.style, {
      display: 'grid',
      gridTemplateColumns: '1fr 110px',
      alignItems: 'center',
      columnGap: '8px',
    });

    const label = document.createElement('div');
    label.textContent = item.label;
    Object.assign(label.style, {
      color: '#b6c2d4',
      fontSize: '12px',
      flex: '1',
    });

    const button = document.createElement('button');
    Object.assign(button.style, {
      background: '#0b0f14',
      color: '#e2e8f0',
      border: '1px solid #1f2a37',
      borderRadius: '4px',
      padding: '4px 8px',
      fontSize: '12px',
      lineHeight: '1.2',
      minHeight: '24px',
      width: '110px',
      cursor: 'pointer',
      textAlign: 'center',
    });
    button.addEventListener('click', () => startRebind(item.key, button));

    row.appendChild(label);
    row.appendChild(button);
    controlsList.appendChild(row);
    keybindButtons.set(item.key, button);
  }

  updateKeybindButtons(settings.input.bindings);

  controlsResetButton.addEventListener('click', () => {
    const currentInput = settingsStore.get().input;
    settingsStore.apply({
      input: {
        ...currentInput,
        bindings: { ...DEFAULT_KEY_BINDINGS },
      },
    });
  });

  const dataNotice = document.createElement('div');
  dataNotice.textContent =
    'This game collects anonymized board snapshots to train the piece generator.';
  Object.assign(dataNotice.style, {
    maxWidth: '320px',
    color: '#8fa0b8',
    fontSize: '12px',
    lineHeight: '1.4',
    textAlign: 'center',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
  });

  const menuMainWrapper = document.createElement('div');
  Object.assign(menuMainWrapper.style, {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '12px',
  });
  menuMainWrapper.appendChild(menuMainPanel);
  menuMainWrapper.appendChild(dataNotice);

  const playTitle = document.createElement('div');
  playTitle.textContent = 'PLAY';
  Object.assign(playTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '4px',
  });

  const defaultButton = makeMenuButton('DEFAULT');
  const cheeseModeButton = makeMenuButton('CHEESE');
  const charcuterieModeButton = makeMenuButton('CHARCUTERIE');
  const playBackButton = makeMenuButton('BACK');
  Object.assign(playBackButton.style, {
    marginTop: 'auto',
  });

  playPanel.appendChild(playTitle);
  playPanel.appendChild(defaultButton);
  playPanel.appendChild(cheeseModeButton);
  playPanel.appendChild(charcuterieModeButton);
  playPanel.appendChild(playBackButton);

  let updateButterfingerUI: (cfg: Settings['butterfinger']) => void = () => {};
  if (SHOW_DEV_TOOLS) {
    const butterfingerTitle = document.createElement('div');
    butterfingerTitle.textContent = 'BUTTERFINGER';
    Object.assign(butterfingerTitle.style, {
      color: '#8fa0b8',
      fontSize: '12px',
      letterSpacing: '0.5px',
      marginBottom: '4px',
    });
    butterfingerPanel.appendChild(butterfingerTitle);

    const butterfingerToggleRow = document.createElement('label');
    Object.assign(butterfingerToggleRow.style, {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      fontSize: '12px',
      color: '#e2e8f0',
      cursor: 'pointer',
    });
    const butterfingerToggle = document.createElement('input');
    butterfingerToggle.type = 'checkbox';
    butterfingerToggle.style.cursor = 'pointer';
    const butterfingerToggleText = document.createElement('span');
    butterfingerToggleText.textContent = 'Enable';
    butterfingerToggleRow.appendChild(butterfingerToggle);
    butterfingerToggleRow.appendChild(butterfingerToggleText);
    butterfingerPanel.appendChild(butterfingerToggleRow);

    const makeButterfingerSlider = (
      labelText: string,
      options: { max?: number; step?: number } = {},
    ) => {
      const row = document.createElement('div');
      Object.assign(row.style, {
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
        textAlign: 'left',
        marginTop: '8px',
      });
      const label = document.createElement('div');
      label.textContent = labelText;
      Object.assign(label.style, {
        color: '#b6c2d4',
        fontSize: '11px',
        letterSpacing: '0.3px',
      });
      const value = document.createElement('span');
      Object.assign(value.style, {
        color: '#8fa0b8',
        fontSize: '11px',
        marginLeft: '6px',
      });
      const labelRow = document.createElement('div');
      Object.assign(labelRow.style, {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      });
      labelRow.appendChild(label);
      labelRow.appendChild(value);

      const slider = document.createElement('input');
      slider.type = 'range';
      slider.min = '0';
      slider.max = String(options.max ?? 10);
      slider.step = String(options.step ?? 0.1);
      Object.assign(slider.style, {
        width: '100%',
        accentColor: '#6ea8ff',
      });

      row.appendChild(labelRow);
      row.appendChild(slider);
      butterfingerPanel.appendChild(row);
      return { slider, value };
    };

    const missSlider = makeButterfingerSlider('Miss Rate');
    const wrongDirSlider = makeButterfingerSlider('Wrong Direction');
    const extraTapSlider = makeButterfingerSlider('Extra Tap');
    const lockNudgeSlider = makeButterfingerSlider('Lock Nudge');
    const gravityDropSlider = makeButterfingerSlider('Gravity Drop');
    const lockRotateSlider = makeButterfingerSlider('Lock Rotate', {
      max: 100,
      step: 1,
    });

    const clampRate = (value: number): number =>
      Math.min(1, Math.max(0, value));
    const formatPercent = (value: number): string => {
      const rounded = Math.round(value * 10) / 10;
      return Number.isInteger(rounded)
        ? `${rounded}%`
        : `${rounded.toFixed(1)}%`;
    };
    const updateButterfingerControl = (
      control: { slider: HTMLInputElement; value: HTMLSpanElement },
      rate: number,
    ) => {
      const sliderMax = Number(control.slider.max) || 100;
      const percent = Math.min(sliderMax, clampRate(rate) * 100);
      control.slider.value = String(percent);
      control.value.textContent = formatPercent(percent);
    };

    updateButterfingerUI = (cfg: Settings['butterfinger']) => {
      butterfingerToggle.checked = cfg.enabled;
      updateButterfingerControl(missSlider, cfg.missRate);
      updateButterfingerControl(wrongDirSlider, cfg.wrongDirRate);
      updateButterfingerControl(extraTapSlider, cfg.extraTapRate);
      updateButterfingerControl(lockNudgeSlider, cfg.lockNudgeRate);
      updateButterfingerControl(gravityDropSlider, cfg.gravityDropRate);
      updateButterfingerControl(lockRotateSlider, cfg.lockRotateRate);
    };

    const readButterfingerRate = (control: {
      slider: HTMLInputElement;
      value: HTMLSpanElement;
    }): number => clampRate(Number(control.slider.value) / 100);

    const applyButterfinger = (patch: Partial<Settings['butterfinger']>) => {
      const current = settingsStore.get().butterfinger;
      settingsStore.apply({
        butterfinger: { ...current, ...patch },
      });
    };

    butterfingerToggle.addEventListener('change', () => {
      applyButterfinger({ enabled: butterfingerToggle.checked });
    });

    missSlider.slider.addEventListener('input', () => {
      const rate = readButterfingerRate(missSlider);
      missSlider.value.textContent = formatPercent(rate * 100);
      applyButterfinger({ missRate: rate });
    });

    wrongDirSlider.slider.addEventListener('input', () => {
      const rate = readButterfingerRate(wrongDirSlider);
      wrongDirSlider.value.textContent = formatPercent(rate * 100);
      applyButterfinger({ wrongDirRate: rate });
    });

    extraTapSlider.slider.addEventListener('input', () => {
      const rate = readButterfingerRate(extraTapSlider);
      extraTapSlider.value.textContent = formatPercent(rate * 100);
      applyButterfinger({ extraTapRate: rate });
    });

    lockNudgeSlider.slider.addEventListener('input', () => {
      const rate = readButterfingerRate(lockNudgeSlider);
      lockNudgeSlider.value.textContent = formatPercent(rate * 100);
      applyButterfinger({ lockNudgeRate: rate });
    });

    gravityDropSlider.slider.addEventListener('input', () => {
      const rate = readButterfingerRate(gravityDropSlider);
      gravityDropSlider.value.textContent = formatPercent(rate * 100);
      applyButterfinger({ gravityDropRate: rate });
    });

    lockRotateSlider.slider.addEventListener('input', () => {
      const rate = readButterfingerRate(lockRotateSlider);
      lockRotateSlider.value.textContent = formatPercent(rate * 100);
      applyButterfinger({ lockRotateRate: rate });
    });

    updateButterfingerUI(settings.butterfinger);
  }

  const cheeseTitle = document.createElement('div');
  cheeseTitle.textContent = 'CHEESE';
  Object.assign(cheeseTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '4px',
  });

  const cheese4Button = makeMenuButton('4 LINES');
  const cheese8Button = makeMenuButton('8 LINES');
  const cheese12Button = makeMenuButton('12 LINES');
  const cheeseBackButton = makeMenuButton('BACK');
  Object.assign(cheeseBackButton.style, {
    marginTop: 'auto',
  });

  cheesePanel.appendChild(cheeseTitle);
  cheesePanel.appendChild(cheese4Button);
  cheesePanel.appendChild(cheese8Button);
  cheesePanel.appendChild(cheese12Button);
  cheesePanel.appendChild(cheeseBackButton);

  const charcuterieTitle = document.createElement('div');
  charcuterieTitle.textContent = 'CHARCUTERIE';
  Object.assign(charcuterieTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '4px',
  });

  const charcuterie8Button = makeMenuButton('8 PIECES');
  const charcuterie14Button = makeMenuButton('14 PIECES');
  const charcuterie20Button = makeMenuButton('20 PIECES');
  const charcuterieSimInput = document.createElement('input');
  charcuterieSimInput.type = 'number';
  charcuterieSimInput.min = '1';
  charcuterieSimInput.step = '1';
  charcuterieSimInput.value = String(charcuterieDefaultSimCount);
  Object.assign(charcuterieSimInput.style, {
    width: '100%',
    boxSizing: 'border-box',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  });

  const charcuterieSeedInput = document.createElement('input');
  charcuterieSeedInput.type = 'text';
  charcuterieSeedInput.placeholder = 'Random';
  Object.assign(charcuterieSeedInput.style, {
    width: '100%',
    boxSizing: 'border-box',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  });

  const makeMenuField = (labelText: string, input: HTMLInputElement) => {
    const field = document.createElement('div');
    Object.assign(field.style, {
      display: 'flex',
      flexDirection: 'column',
      gap: '6px',
      textAlign: 'left',
      marginTop: '4px',
    });
    const label = document.createElement('div');
    label.textContent = labelText;
    Object.assign(label.style, {
      color: '#b6c2d4',
      fontSize: '11px',
      letterSpacing: '0.3px',
    });
    field.appendChild(label);
    field.appendChild(input);
    return field;
  };

  const charcuterieSimField = makeMenuField('SIMULATIONS', charcuterieSimInput);
  const charcuterieSeedField = makeMenuField('SEED', charcuterieSeedInput);
  const charcuterieBackButton = makeMenuButton('BACK');
  Object.assign(charcuterieBackButton.style, {
    marginTop: 'auto',
  });

  charcuteriePanel.appendChild(charcuterieTitle);
  charcuteriePanel.appendChild(charcuterie8Button);
  charcuteriePanel.appendChild(charcuterie14Button);
  charcuteriePanel.appendChild(charcuterie20Button);
  if (SHOW_DEV_TOOLS) {
    charcuteriePanel.appendChild(charcuterieSimField);
    charcuteriePanel.appendChild(charcuterieSeedField);
  }
  charcuteriePanel.appendChild(charcuterieBackButton);

  const toolsTitle = document.createElement('div');
  toolsTitle.textContent = 'TOOLS';
  Object.assign(toolsTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '4px',
  });

  const wishButton = makeMenuButton('WISH UPON A BLOCK');
  const toolsBackButton = makeMenuButton('BACK');
  Object.assign(toolsBackButton.style, {
    marginTop: 'auto',
  });

  toolsPanel.appendChild(toolsTitle);
  toolsPanel.appendChild(wishButton);
  toolsPanel.appendChild(toolsBackButton);

  const feedbackTitle = document.createElement('div');
  feedbackTitle.textContent = 'FEEDBACK';
  Object.assign(feedbackTitle.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    letterSpacing: '0.5px',
    marginBottom: '6px',
  });

  const feedbackBody = document.createElement('textarea');
  feedbackBody.placeholder = 'Your feedback and suggestions...';
  Object.assign(feedbackBody.style, {
    width: '100%',
    minHeight: '120px',
    resize: 'vertical',
    boxSizing: 'border-box',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '8px',
    fontSize: '12px',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
  });

  const feedbackContact = document.createElement('input');
  feedbackContact.type = 'text';
  feedbackContact.placeholder = 'Your contact (optional)';
  Object.assign(feedbackContact.style, {
    width: '100%',
    boxSizing: 'border-box',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '8px',
    fontSize: '12px',
  });

  const feedbackButtons = document.createElement('div');
  Object.assign(feedbackButtons.style, {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '8px',
    marginTop: '8px',
  });

  const feedbackBackButton = makeMenuButton('BACK');
  const feedbackSendButton = makeMenuButton('SEND');
  Object.assign(feedbackBackButton.style, {
    flex: '1',
  });
  Object.assign(feedbackSendButton.style, {
    flex: '1',
  });

  const updateFeedbackSendState = () => {
    const canSend = feedbackBody.value.trim().length > 0;
    feedbackSendButton.disabled = !canSend;
    feedbackSendButton.style.opacity = canSend ? '1' : '0.55';
    feedbackSendButton.style.cursor = canSend ? 'pointer' : 'default';
  };
  updateFeedbackSendState();

  feedbackBody.addEventListener('input', () => {
    updateFeedbackSendState();
  });

  feedbackButtons.appendChild(feedbackBackButton);
  feedbackButtons.appendChild(feedbackSendButton);

  feedbackPanel.appendChild(feedbackTitle);
  feedbackPanel.appendChild(feedbackBody);
  feedbackPanel.appendChild(feedbackContact);
  feedbackPanel.appendChild(feedbackButtons);

  menuLayer.appendChild(menuMainWrapper);
  playMenuRow.appendChild(playPanel);
  if (SHOW_DEV_TOOLS) {
    playMenuRow.appendChild(butterfingerPanel);
  }
  menuLayer.appendChild(playMenuRow);
  menuLayer.appendChild(optionsPanel);
  menuLayer.appendChild(aboutPanel);
  menuLayer.appendChild(cheesePanel);
  menuLayer.appendChild(charcuteriePanel);
  menuLayer.appendChild(toolsPanel);
  menuLayer.appendChild(feedbackPanel);
  uiLayer.appendChild(menuLayer);
  uiLayer.appendChild(charcuterieSpinner);

  const footer = document.createElement('div');
  Object.assign(footer.style, {
    position: 'absolute',
    left: '0',
    right: '0',
    bottom: `${OUTER_MARGIN}px`,
    display: 'flex',
    justifyContent: 'center',
    pointerEvents: 'auto',
  });

  const feedbackMenuButton = makeMenuButton('LEAVE FEEDBACK');
  Object.assign(feedbackMenuButton.style, {
    position: 'absolute',
    left: '50%',
    bottom: `${OUTER_MARGIN + 54}px`,
    transform: 'translateX(-50%)',
    width: '200px',
    borderColor: '#bda56a',
    boxShadow: '0 0 0 1px rgba(189, 165, 106, 0.3)',
    pointerEvents: 'auto',
  });
  uiLayer.appendChild(feedbackMenuButton);

  const githubLink = document.createElement('a');
  githubLink.href = 'https://github.com/icanfast/wishuponablock';
  githubLink.target = '_blank';
  githubLink.rel = 'noreferrer';
  Object.assign(githubLink.style, {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    textDecoration: 'none',
    color: '#b6c2d4',
    opacity: '0.85',
  });

  const githubIcon = document.createElement('img');
  githubIcon.src = '/assets/GitHub_Invertocat_White_Clearspace.svg';
  githubIcon.alt = 'GitHub';
  Object.assign(githubIcon.style, {
    width: '36px',
    height: '36px',
    display: 'block',
  });

  githubLink.appendChild(githubIcon);
  footer.appendChild(githubLink);
  uiLayer.appendChild(footer);

  app.renderer.resize(PLAY_WIDTH, PLAY_HEIGHT);

  settingsStore.subscribe((next) => {
    input.setConfig(next.input);
    lockSound.volume = next.audio.masterVolume;
    butterfingerSource.setConfig(next.butterfinger);
    inputSource = next.butterfinger.enabled
      ? butterfingerSource
      : baseInputSource;
    updateButterfingerUI(next.butterfinger);

    if (next.generator.type !== generatorType) {
      generatorType = next.generator.type;
      if (next.generator.type === 'ml') {
        void ensureMlModel();
      }
      updateModelStatusUI();
      game = createGame(next, selectedMode, selectedModeOptions);
      runner = createRunner(game, selectedMode, selectedModeOptions);
      if (select.value !== next.generator.type) {
        select.value = next.generator.type;
      }
      return;
    }

    game.setConfig({
      ...next.game,
      lockNudgeRate: next.butterfinger.enabled
        ? next.butterfinger.lockNudgeRate
        : 0,
      gravityDropRate: next.butterfinger.enabled
        ? next.butterfinger.gravityDropRate
        : 0,
      lockRotateRate: next.butterfinger.enabled
        ? next.butterfinger.lockRotateRate
        : 0,
    });
    if (select.value !== next.generator.type) {
      select.value = next.generator.type;
    }
    const nextVolume = Math.round(next.audio.masterVolume * 100);
    if (Number(volumeSlider.value) !== nextVolume) {
      volumeSlider.value = String(nextVolume);
      updateVolumeLabel(next.audio.masterVolume);
    }
    updateKeybindButtons(next.input.bindings);
  });

  let pausedByVisibility = document.visibilityState !== 'visible';
  let pausedByInput = false;
  let pausedByMenu = true;
  let paused: boolean =
    pausedByVisibility || pausedByInput || pausedByMenu || pausedByModel;
  let resumePending = false;

  const showMenuPanel = (
    panel:
      | 'main'
      | 'play'
      | 'options'
      | 'about'
      | 'cheese'
      | 'charcuterie'
      | 'tools'
      | 'feedback',
  ) => {
    menuMainWrapper.style.display = panel === 'main' ? 'flex' : 'none';
    playMenuRow.style.display = panel === 'play' ? 'flex' : 'none';
    optionsPanel.style.display = panel === 'options' ? 'flex' : 'none';
    aboutPanel.style.display = panel === 'about' ? 'flex' : 'none';
    cheesePanel.style.display = panel === 'cheese' ? 'flex' : 'none';
    charcuteriePanel.style.display = panel === 'charcuterie' ? 'flex' : 'none';
    toolsPanel.style.display = panel === 'tools' ? 'flex' : 'none';
    feedbackPanel.style.display = panel === 'feedback' ? 'flex' : 'none';
    feedbackMenuButton.style.display = panel === 'main' ? 'block' : 'none';
  };

  updatePaused = () => {
    const next =
      pausedByVisibility || pausedByInput || pausedByMenu || pausedByModel;
    if (next === paused) return;
    paused = next;
    if (!paused) resumePending = true;
  };
  updateModelStatusUI();

  document.addEventListener('visibilitychange', () => {
    pausedByVisibility = document.visibilityState !== 'visible';
    updatePaused();
  });
  window.addEventListener('blur', () => {
    pausedByVisibility = true;
    updatePaused();
  });
  window.addEventListener('focus', () => {
    pausedByVisibility = false;
    updatePaused();
  });

  const renderToolSample = () => {
    if (!currentSample) {
      if (toolUsesRemote || toolSnapshots.length > 0) {
        void showNextToolSample();
      } else {
        gfx.clear();
      }
      return;
    }
    renderer.renderBoardOnly(currentSample.board, currentSample.hold);
  };

  const setScreen = (screen: 'menu' | 'game' | 'tool') => {
    const inMenu = screen === 'menu';
    const inGame = screen === 'game';
    const inTool = screen === 'tool';
    toolActive = inTool;
    menuLayer.style.display = inMenu ? 'flex' : 'none';
    settingsPanel.style.display = inGame ? 'block' : 'none';
    menuButton.style.display = inGame ? 'block' : 'none';
    holdLabel.style.display = inGame || inTool ? 'block' : 'none';
    toolLayer.style.display = inTool ? 'block' : 'none';
    footer.style.display = inMenu ? 'flex' : 'none';
    if (!inMenu) {
      feedbackMenuButton.style.display = 'none';
    }
    pausedByMenu = !inGame;
    updatePaused();
    if (inGame) {
      const nextSettings = settingsStore.get();
      input.setConfig(nextSettings.input);
      game = createGame(nextSettings, selectedMode, selectedModeOptions);
      runner = createRunner(game, selectedMode, selectedModeOptions);
      gameOverLabel.style.display = 'none';
      startRecordingSession();
      renderer.render(runner.state);
    } else {
      void stopRecordingSession();
      gfx.clear();
      gameOverLabel.style.display = 'none';
      showMenuPanel('main');
      if (inTool) {
        if (toolUsesRemote) {
          refreshToolModeOptions();
          void applyToolModeFilter();
        }
        renderToolSample();
      }
    }
  };

  playButton.addEventListener('click', () => showMenuPanel('play'));
  defaultButton.addEventListener('click', () => {
    selectedMode = getMode('default');
    selectedModeOptions = {};
    setScreen('game');
  });
  cheeseModeButton.addEventListener('click', () => showMenuPanel('cheese'));
  charcuterieModeButton.addEventListener('click', () =>
    showMenuPanel('charcuterie'),
  );
  optionsButton.addEventListener('click', () => {
    showMenuPanel('options');
  });
  toolsButton.addEventListener('click', () => showMenuPanel('tools'));
  aboutButton.addEventListener('click', () => showMenuPanel('about'));
  feedbackMenuButton.addEventListener('click', () => showMenuPanel('feedback'));

  const startCheese = (lines: number) => {
    selectedMode = getMode('cheese');
    selectedModeOptions = { cheeseLines: lines };
    setScreen('game');
  };

  cheese4Button.addEventListener('click', () => startCheese(4));
  cheese8Button.addEventListener('click', () => startCheese(8));
  cheese12Button.addEventListener('click', () => startCheese(12));
  cheeseBackButton.addEventListener('click', () => showMenuPanel('play'));
  playBackButton.addEventListener('click', () => showMenuPanel('main'));
  optionsBackButton.addEventListener('click', () => showMenuPanel('main'));
  aboutBackButton.addEventListener('click', () => showMenuPanel('main'));

  const readCharcuterieSimCount = (): number => {
    const raw = Number(charcuterieSimInput.value);
    if (!Number.isFinite(raw)) return charcuterieDefaultSimCount;
    return Math.max(1, Math.trunc(raw));
  };

  const readCharcuterieSeed = (): number | undefined => {
    const raw = charcuterieSeedInput.value.trim();
    if (!raw) return undefined;
    const value = Number(raw);
    if (!Number.isFinite(value)) return undefined;
    return Math.trunc(value);
  };

  const startCharcuterie = (pieces: number) => {
    selectedMode = getMode('charcuterie');
    const simCount = readCharcuterieSimCount();
    const seed = readCharcuterieSeed();
    selectedModeOptions = {
      pieces,
      simCount,
      ...(seed !== undefined ? { seed } : {}),
    };
    charcuterieSpinner.style.display = 'flex';
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setScreen('game');
        charcuterieSpinner.style.display = 'none';
      });
    });
  };

  charcuterie8Button.addEventListener('click', () => startCharcuterie(8));
  charcuterie14Button.addEventListener('click', () => startCharcuterie(14));
  charcuterie20Button.addEventListener('click', () => startCharcuterie(20));
  charcuterieBackButton.addEventListener('click', () => showMenuPanel('play'));

  wishButton.addEventListener('click', () => setScreen('tool'));
  toolsBackButton.addEventListener('click', () => showMenuPanel('main'));
  feedbackBackButton.addEventListener('click', () => showMenuPanel('main'));
  feedbackSendButton.addEventListener('click', async () => {
    const feedback = feedbackBody.value.trim();
    if (!feedback) {
      updateFeedbackSendState();
      return;
    }
    if (!useRemoteUpload) {
      console.warn('[Feedback] Remote upload disabled.');
      return;
    }
    feedbackSendButton.disabled = true;
    feedbackSendButton.textContent = 'SENDING...';
    try {
      const res = await fetch(`${uploadBaseUrl}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          createdAt: new Date().toISOString(),
          feedback,
          contact: feedbackContact.value.trim() || null,
        }),
      });
      if (!res.ok) {
        throw new Error(`Feedback failed (${res.status})`);
      }
      feedbackBody.value = '';
      feedbackContact.value = '';
      updateFeedbackSendState();
    } catch (err) {
      console.warn('[Feedback] Upload failed.', err);
      updateFeedbackSendState();
    } finally {
      feedbackSendButton.textContent = 'SEND';
    }
  });

  const menuButton = makeMenuButton('MENU');
  Object.assign(menuButton.style, {
    position: 'absolute',
    left: `${BOARD_X}px`,
    top: `${BOARD_Y + ROWS * BOARD_CELL_PX + PANEL_GAP}px`,
    width: `${COLS * BOARD_CELL_PX}px`,
    pointerEvents: 'auto',
    zIndex: '2',
  });
  uiLayer.appendChild(menuButton);

  menuButton.addEventListener('click', () => setScreen('menu'));

  setScreen('menu');

  const updateGameOverLabel = () => {
    if (pausedByMenu) {
      gameOverLabel.style.display = 'none';
      return;
    }
    gameOverLabel.style.display = runner.state.gameOver ? 'block' : 'none';
  };

  app.ticker.add((t) => {
    if (paused) {
      updateGameOverLabel();
      return;
    }
    if (resumePending) {
      runner.resetTiming();
      resumePending = false;
      updateGameOverLabel();
      return;
    }
    runner.tick(t.elapsedMS, inputSource);
    renderer.render(runner.state);
    updateGameOverLabel();
  });
}

boot().catch((e) => console.error(e));
