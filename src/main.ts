import { Application, Graphics } from 'pixi.js';
import {
  COLS,
  GAME_OVER_Y,
  HOLD_X,
  HOLD_Y,
  HOLD_WIDTH,
  OUTER_MARGIN,
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
import { Keyboard } from './input/keyboard';
import { InputController } from './input/controller';
import { KeyboardInputSource } from './input/keyboardInputSource';
import { PixiRenderer } from './render/pixiRenderer';
import type { Board } from './core/types';

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
  const handlePieceLock = (board: Board) => {
    playLockSound();
    if (recorder.isRecording) {
      recorder.record(board);
      updateRecorderUI();
    }
  };

  const createGame = (cfg: Settings): Game =>
    new Game({
      seed: Date.now(),
      ...cfg.game,
      generatorFactory: createGeneratorFactory(cfg.generator),
      onPieceLock: handlePieceLock,
    });

  const createRunner = (g: Game): GameRunner =>
    new GameRunner(g, {
      fixedStepMs: 1000 / 120,
      onRestart: () => g.reset(Date.now()),
      maxElapsedMs: 250,
      maxStepsPerTick: 10,
    });

  let game = createGame(settings);
  let runner = createRunner(game);

  const renderer = new PixiRenderer(gfx);

  const settingsPanel = document.createElement('div');
  Object.assign(settingsPanel.style, {
    position: 'absolute',
    left: `${SETTINGS_X}px`,
    top: `${SETTINGS_Y}px`,
    minWidth: `${SETTINGS_PANEL_WIDTH}px`,
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

      const permission = await snapshotDirHandle.requestPermission({
        mode: 'readwrite',
      });
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
      recorder.start(settingsStore.get(), ROWS, COLS, commentInput.value);
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

  const syncPlayWindowSize = () => {
    const settingsWidth = settingsPanel.offsetWidth;
    const settingsBottom = settingsPanel.offsetTop + settingsPanel.offsetHeight;
    const desiredWidth = Math.max(
      PLAY_WIDTH,
      SETTINGS_X + settingsWidth + OUTER_MARGIN,
    );
    const desiredHeight = Math.max(
      PLAY_HEIGHT,
      settingsBottom + OUTER_MARGIN,
    );

    playWindow.style.width = `${desiredWidth}px`;
    playWindow.style.height = `${desiredHeight}px`;
    app.renderer.resize(desiredWidth, desiredHeight);
  };

  requestAnimationFrame(syncPlayWindowSize);

  let generatorType = settings.generator.type;

  settingsStore.subscribe((next) => {
    input.setConfig(next.input);
    lockSound.volume = next.audio.masterVolume;

    if (next.generator.type !== generatorType) {
      generatorType = next.generator.type;
      game = createGame(next);
      runner = createRunner(game);
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
  let paused = pausedByVisibility || pausedByInput;
  let resumePending = false;

  const updatePaused = () => {
    const next = pausedByVisibility || pausedByInput;
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

  const updateGameOverLabel = () => {
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
