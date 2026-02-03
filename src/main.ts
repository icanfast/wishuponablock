import { Application, Graphics } from 'pixi.js';
import {
  BOARD_CELL_PX,
  BOARD_X,
  BOARD_Y,
  COLS,
  GAME_OVER_Y,
  HOLD_X,
  HOLD_Y,
  HOLD_WIDTH,
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
import { PixiRenderer } from './render/pixiRenderer';
import type { Board } from './core/types';
import { CharcuterieBot, runBotForPieces } from './bot/charcuterieBot';

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
  const recorder = new SnapshotRecorder();
  let updateRecorderUI = () => {};
  let snapshotDirHandle: FileSystemDirectoryHandle | null = null;
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
  const inputSource = new KeyboardInputSource(input);

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
  const handlePieceLock = (board: Board) => {
    if (suppressLockEffects) return;
    playLockSound();
    if (recorder.isRecording) {
      recorder.record(board);
      updateRecorderUI();
    }
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
      generatorFactory: createGeneratorFactory(merged.generator),
      onPieceLock: handlePieceLock,
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
    const seedOverride = options.seed;
    const baseSeed =
      seedOverride !== undefined ? Math.trunc(seedOverride) : Date.now();
    const simStart = performance.now();

    if (pieces === 0 || sims === 1) {
      const game = buildGame(cfg, mode, baseSeed);
      if (pieces > 0) {
        suppressLockEffects = true;
        try {
          const bot = new CharcuterieBot(baseSeed ^ 0x9e3779b9);
          runBotForPieces(game, bot, pieces);
        } finally {
          suppressLockEffects = false;
        }
      }
      game.markInitialBlocks();
      return game;
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
        const game = buildGame(cfg, mode, seed);
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

    const finalGame = bestGame ?? buildGame(cfg, mode, baseSeed);
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
    volumeValue.textContent = `${Math.round(value * 100)}%`;
  };
  updateVolumeLabel(settings.audio.masterVolume);

  volumeSlider.addEventListener('input', () => {
    const value = Math.max(0, Math.min(1, Number(volumeSlider.value) / 100));
    settingsStore.apply({ audio: { masterVolume: value } });
  });

  volumeRow.appendChild(volumeSlider);
  volumeRow.appendChild(volumeValue);
  settingsPanel.appendChild(volumeLabel);
  settingsPanel.appendChild(volumeRow);

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

  commentInput.addEventListener('input', () => {
    if (recorder.isRecording) {
      recorder.setComment(commentInput.value);
    }
  });

  updateFolderStatus('No folder selected.');

  folderButton.addEventListener('click', async () => {
    await ensureSnapshotDirectory();
  });

  recordButton.addEventListener('click', async () => {
    if (!recorder.isRecording) {
      recorder.start(settingsStore.get(), ROWS, COLS, commentInput.value, {
        id: selectedMode.id,
        options: { ...selectedModeOptions },
      });
      updateRecorderUI();
      return;
    }

    const session = recorder.stop();
    updateRecorderUI();
    if (!session) return;

    const ready = await ensureSnapshotDirectory();
    if (!ready || !snapshotDirHandle) {
      updateFolderStatus('Auto-save unavailable. Downloading instead.');
      downloadSnapshotSession(session);
      return;
    }

    void saveSnapshotSessionToDirectory(session, snapshotDirHandle)
      .then(() => updateFolderStatus(`Saved: ${session.meta.id}`))
      .catch(() => {
        updateFolderStatus('Save failed. Downloading instead.');
        downloadSnapshotSession(session);
      });
  });

  discardButton.addEventListener('click', () => {
    recorder.discard();
    updateRecorderUI();
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
  const cheesePanel = makeMenuPanel();
  const charcuteriePanel = makeMenuPanel();
  Object.assign(playPanel.style, {
    minHeight: '240px',
    display: 'none',
  });
  Object.assign(cheesePanel.style, {
    minHeight: '240px',
    display: 'none',
  });
  Object.assign(charcuteriePanel.style, {
    minHeight: '240px',
    display: 'none',
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
  const placeholderButton = makeMenuButton('PLACEHOLDER');
  const creditsButton = makeMenuButton('CREDITS');

  menuMainPanel.appendChild(playButton);
  menuMainPanel.appendChild(optionsButton);
  menuMainPanel.appendChild(placeholderButton);
  menuMainPanel.appendChild(creditsButton);

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
  charcuteriePanel.appendChild(charcuterieSimField);
  charcuteriePanel.appendChild(charcuterieSeedField);
  charcuteriePanel.appendChild(charcuterieBackButton);

  menuLayer.appendChild(menuMainPanel);
  menuLayer.appendChild(playPanel);
  menuLayer.appendChild(cheesePanel);
  menuLayer.appendChild(charcuteriePanel);
  uiLayer.appendChild(menuLayer);

  app.renderer.resize(PLAY_WIDTH, PLAY_HEIGHT);

  let generatorType = settings.generator.type;

  settingsStore.subscribe((next) => {
    input.setConfig(next.input);
    lockSound.volume = next.audio.masterVolume;

    if (next.generator.type !== generatorType) {
      generatorType = next.generator.type;
      game = createGame(next, selectedMode, selectedModeOptions);
      runner = createRunner(game, selectedMode, selectedModeOptions);
      if (select.value !== next.generator.type) {
        select.value = next.generator.type;
      }
      return;
    }

    game.setConfig(next.game);
    if (select.value !== next.generator.type) {
      select.value = next.generator.type;
    }
    const nextVolume = Math.round(next.audio.masterVolume * 100);
    if (Number(volumeSlider.value) !== nextVolume) {
      volumeSlider.value = String(nextVolume);
      updateVolumeLabel(next.audio.masterVolume);
    }
  });

  const applySettings = (patch: Partial<Settings>): void => {
    settingsStore.apply(patch);
  };

  // TODO: Wire applySettings to UI when settings controls are added.
  void applySettings;

  let pausedByVisibility = document.visibilityState !== 'visible';
  let pausedByInput = false;
  let pausedByMenu = true;
  let paused: boolean = pausedByVisibility || pausedByInput || pausedByMenu;
  let resumePending = false;

  const showMenuPanel = (panel: 'main' | 'play' | 'cheese' | 'charcuterie') => {
    menuMainPanel.style.display = panel === 'main' ? 'flex' : 'none';
    playPanel.style.display = panel === 'play' ? 'flex' : 'none';
    cheesePanel.style.display = panel === 'cheese' ? 'flex' : 'none';
    charcuteriePanel.style.display = panel === 'charcuterie' ? 'flex' : 'none';
  };

  const updatePaused = () => {
    const next = pausedByVisibility || pausedByInput || pausedByMenu;
    if (next === paused) return;
    paused = next;
    if (!paused) resumePending = true;
  };

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

  const setMode = (mode: 'menu' | 'game') => {
    const inMenu = mode === 'menu';
    menuLayer.style.display = inMenu ? 'flex' : 'none';
    settingsPanel.style.display = inMenu ? 'none' : 'block';
    menuButton.style.display = inMenu ? 'none' : 'block';
    holdLabel.style.display = inMenu ? 'none' : 'block';
    pausedByMenu = inMenu;
    updatePaused();
    if (!inMenu) {
      const nextSettings = settingsStore.get();
      input.setConfig(nextSettings.input);
      game = createGame(nextSettings, selectedMode, selectedModeOptions);
      runner = createRunner(game, selectedMode, selectedModeOptions);
      gameOverLabel.style.display = 'none';
    } else {
      gfx.clear();
      gameOverLabel.style.display = 'none';
      showMenuPanel('main');
    }
  };

  playButton.addEventListener('click', () => showMenuPanel('play'));
  defaultButton.addEventListener('click', () => {
    selectedMode = getMode('default');
    selectedModeOptions = {};
    setMode('game');
  });
  cheeseModeButton.addEventListener('click', () => showMenuPanel('cheese'));
  charcuterieModeButton.addEventListener('click', () =>
    showMenuPanel('charcuterie'),
  );
  optionsButton.addEventListener('click', () => {
    // Placeholder for future options screen
  });
  placeholderButton.addEventListener('click', () => {
    // Placeholder for future mode
  });
  creditsButton.addEventListener('click', () => {
    // Placeholder for future credits screen
  });

  const startCheese = (lines: number) => {
    selectedMode = getMode('cheese');
    selectedModeOptions = { cheeseLines: lines };
    setMode('game');
  };

  cheese4Button.addEventListener('click', () => startCheese(4));
  cheese8Button.addEventListener('click', () => startCheese(8));
  cheese12Button.addEventListener('click', () => startCheese(12));
  cheeseBackButton.addEventListener('click', () => showMenuPanel('play'));
  playBackButton.addEventListener('click', () => showMenuPanel('main'));

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
    setMode('game');
  };

  charcuterie8Button.addEventListener('click', () => startCharcuterie(8));
  charcuterie14Button.addEventListener('click', () => startCharcuterie(14));
  charcuterie20Button.addEventListener('click', () => startCharcuterie(20));
  charcuterieBackButton.addEventListener('click', () => showMenuPanel('play'));

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

  menuButton.addEventListener('click', () => setMode('menu'));

  setMode('menu');

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
