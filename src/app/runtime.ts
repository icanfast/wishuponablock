import type { Application, Ticker } from 'pixi.js';
import type { GameSession } from '../core/gameSession';
import type { InputSource } from '../core/runner';
import type { PixiRenderer } from '../render/pixiRenderer';

export type GameRuntime = {
  setInputSource: (source: InputSource) => void;
  setPausedByMenu: (paused: boolean) => void;
  setPausedByInput: (paused: boolean) => void;
  setPausedByModel: (paused: boolean) => void;
  renderNow: () => void;
  isPaused: () => boolean;
  destroy: () => void;
};

type GameRuntimeOptions = {
  app: Application;
  session: GameSession;
  renderer: PixiRenderer;
  inputSource: InputSource;
  onGameOver: (visible: boolean) => void;
};

export function createGameRuntime(options: GameRuntimeOptions): GameRuntime {
  const { app, session, renderer, onGameOver } = options;
  let inputSource = options.inputSource;

  let pausedByVisibility = document.visibilityState !== 'visible';
  let pausedByInput = false;
  let pausedByMenu = true;
  let pausedByModel = false;
  let paused: boolean =
    pausedByVisibility || pausedByInput || pausedByMenu || pausedByModel;
  let resumePending = false;

  const updateGameOverLabel = () => {
    if (pausedByMenu) {
      onGameOver(false);
      return;
    }
    onGameOver(session.getRunner().state.gameOver);
  };

  const updatePaused = () => {
    const next =
      pausedByVisibility || pausedByInput || pausedByMenu || pausedByModel;
    if (next === paused) return;
    paused = next;
    if (!paused) resumePending = true;
  };

  const tick = (t: Ticker) => {
    if (paused) {
      updateGameOverLabel();
      return;
    }
    if (resumePending) {
      session.getRunner().resetTiming();
      resumePending = false;
      updateGameOverLabel();
      return;
    }
    const runner = session.getRunner();
    runner.tick(t.elapsedMS, inputSource);
    renderer.render(runner.state);
    updateGameOverLabel();
  };

  app.ticker.add(tick);

  const handleVisibilityChange = () => {
    pausedByVisibility = document.visibilityState !== 'visible';
    updatePaused();
  };

  const handleBlur = () => {
    pausedByVisibility = true;
    updatePaused();
  };

  const handleFocus = () => {
    pausedByVisibility = false;
    updatePaused();
  };

  document.addEventListener('visibilitychange', handleVisibilityChange);
  window.addEventListener('blur', handleBlur);
  window.addEventListener('focus', handleFocus);

  const renderNow = () => {
    const runner = session.getRunner();
    renderer.render(runner.state);
    updateGameOverLabel();
  };

  return {
    setInputSource: (source) => {
      inputSource = source;
    },
    setPausedByMenu: (pausedValue) => {
      pausedByMenu = pausedValue;
      updatePaused();
      updateGameOverLabel();
    },
    setPausedByInput: (pausedValue) => {
      pausedByInput = pausedValue;
      updatePaused();
    },
    setPausedByModel: (pausedValue) => {
      pausedByModel = pausedValue;
      updatePaused();
    },
    renderNow,
    isPaused: () => paused,
    destroy: () => {
      app.ticker.remove(tick);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('blur', handleBlur);
      window.removeEventListener('focus', handleFocus);
    },
  };
}
