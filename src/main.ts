import { Application, Graphics } from 'pixi.js';
import { Game } from './core/game';
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
    width: 420,
    height: 680,
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
  root.appendChild(app.canvas);

  const gfx = new Graphics();
  app.stage.addChild(gfx);

  const settingsStore = createSettingsStore();
  const settings = settingsStore.get();

  const kb = new Keyboard();
  const input = new InputController(kb, settings.input);
  const inputSource = new KeyboardInputSource(input);
  const game = new Game({ seed: Date.now(), ...settings.game });
  const runner = new GameRunner(game, {
    fixedStepMs: 1000 / 120,
    onRestart: (g) => g.reset(Date.now()),
    maxElapsedMs: 250,
    maxStepsPerTick: 10,
  });

  const renderer = new PixiRenderer(gfx);

  settingsStore.subscribe((next) => {
    input.setConfig(next.input);
    game.setConfig(next.game);
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

  app.ticker.add((t) => {
    if (paused) return;
    if (resumePending) {
      runner.resetTiming();
      resumePending = false;
      return;
    }
    runner.tick(t.elapsedMS, inputSource);
    renderer.render(runner.state);
  });
}

boot().catch((e) => console.error(e));
