import { Application, Graphics } from 'pixi.js';
import {
  GAME_OVER_Y,
  HOLD_X,
  HOLD_Y,
  HOLD_WIDTH,
  PLAY_HEIGHT,
  PLAY_WIDTH,
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
import { Keyboard } from './input/keyboard';
import { InputController } from './input/controller';
import { KeyboardInputSource } from './input/keyboardInputSource';
import { PixiRenderer } from './render/pixiRenderer';

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

  const createGame = (cfg: Settings): Game =>
    new Game({
      seed: Date.now(),
      ...cfg.game,
      generatorFactory: createGeneratorFactory(cfg.generator),
      onPieceLock: playLockSound,
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
      case 'random':
        return 'Random';
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
  uiLayer.appendChild(settingsPanel);

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

  let paused = document.visibilityState !== 'visible';
  let resumePending = false;

  const setPaused = (next: boolean) => {
    if (next === paused) return;
    paused = next;
    if (!paused) resumePending = true;
  };

  document.addEventListener('visibilitychange', () => {
    setPaused(document.visibilityState !== 'visible');
  });
  window.addEventListener('blur', () => setPaused(true));
  window.addEventListener('focus', () => setPaused(false));

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
