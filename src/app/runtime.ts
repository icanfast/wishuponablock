import type { Application, Ticker } from 'pixi.js';
import type { GameSession } from '../core/gameSession';
import type { InputSource, RunnerTickProfile } from '../core/runner';
import type { PixiRenderer } from '../render/pixiRenderer';
import type { GameState } from '../core/types';
import { hasPerfMetricsSink, recordPerfDuration } from '../core/perfMetrics';

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
  onFrame?: (state: GameState) => void;
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
  let prevTickStartMs: number | null = null;

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

  const recordFrameBreakdown = (
    tickStartMs: number,
    runnerProfile: RunnerTickProfile | null,
    runnerMs: number,
    renderMs: number,
    uiMs: number,
  ) => {
    const tickTotalMs = performance.now() - tickStartMs;
    const inputMs = runnerProfile?.inputMs ?? 0;
    const mlMs = runnerProfile?.mlMs ?? 0;
    const mlInputEncodeMs = runnerProfile?.mlInputEncodeMs ?? 0;
    const mlConvStackMs = runnerProfile?.mlConvStackMs ?? 0;
    const mlPoolMs = runnerProfile?.mlPoolMs ?? 0;
    const mlHeadMs = runnerProfile?.mlHeadMs ?? 0;
    const mlPostMs = runnerProfile?.mlPostMs ?? 0;
    const mlOtherMs = runnerProfile?.mlOtherMs ?? 0;
    const simulationMs = runnerProfile
      ? runnerProfile.simulationMs +
        runnerProfile.stepOverheadMs +
        runnerProfile.loopOverheadMs
      : Math.max(0, runnerMs - inputMs - mlMs);
    const knownMs = inputMs + mlMs + simulationMs + renderMs + uiMs;
    const otherMs = Math.max(0, tickTotalMs - knownMs);
    const nowMs = performance.now();

    recordPerfDuration('runtime.stage.input_ms', inputMs, nowMs);
    recordPerfDuration('runtime.stage.ml_ms', mlMs, nowMs);
    recordPerfDuration(
      'runtime.stage.ml_input_encode_ms',
      mlInputEncodeMs,
      nowMs,
    );
    recordPerfDuration('runtime.stage.ml_conv_stack_ms', mlConvStackMs, nowMs);
    recordPerfDuration('runtime.stage.ml_pool_ms', mlPoolMs, nowMs);
    recordPerfDuration('runtime.stage.ml_head_ms', mlHeadMs, nowMs);
    recordPerfDuration('runtime.stage.ml_post_ms', mlPostMs, nowMs);
    recordPerfDuration('runtime.stage.ml_other_ms', mlOtherMs, nowMs);
    recordPerfDuration('runtime.stage.simulation_ms', simulationMs, nowMs);
    recordPerfDuration('runtime.stage.render_ms', renderMs, nowMs);
    recordPerfDuration('runtime.stage.ui_ms', uiMs, nowMs);
    recordPerfDuration('runtime.stage.other_ms', otherMs, nowMs);

    recordPerfDuration('runtime.tick.total_ms', tickTotalMs, nowMs);
    recordPerfDuration('runtime.runner.tick_ms', runnerMs, nowMs);
    recordPerfDuration('runtime.render_ms', renderMs, nowMs);
    recordPerfDuration('runtime.unknown_ms', otherMs, nowMs);
  };

  const tick = (t: Ticker) => {
    const perfEnabled = hasPerfMetricsSink();
    const tickStartMs = perfEnabled ? performance.now() : 0;
    if (perfEnabled && prevTickStartMs != null) {
      recordPerfDuration('frame.interval_ms', tickStartMs - prevTickStartMs);
    }
    if (perfEnabled) {
      prevTickStartMs = tickStartMs;
    } else {
      prevTickStartMs = null;
    }

    if (paused) {
      if (!perfEnabled) {
        updateGameOverLabel();
        return;
      }
      const gameOverUiStartMs = performance.now();
      updateGameOverLabel();
      const uiMs = performance.now() - gameOverUiStartMs;
      recordFrameBreakdown(tickStartMs, null, 0, 0, uiMs);
      recordPerfDuration('runtime.on_frame_ms', 0);
      recordPerfDuration('runtime.game_over_ui_ms', uiMs);
      return;
    }
    if (resumePending) {
      session.getRunner().resetTiming();
      resumePending = false;
      if (!perfEnabled) {
        updateGameOverLabel();
        return;
      }
      const gameOverUiStartMs = performance.now();
      updateGameOverLabel();
      const uiMs = performance.now() - gameOverUiStartMs;
      recordFrameBreakdown(tickStartMs, null, 0, 0, uiMs);
      recordPerfDuration('runtime.on_frame_ms', 0);
      recordPerfDuration('runtime.game_over_ui_ms', uiMs);
      return;
    }
    const runner = session.getRunner();
    if (!perfEnabled) {
      runner.tick(t.elapsedMS, inputSource);
      renderer.render(runner.state);
      options.onFrame?.(runner.state);
      updateGameOverLabel();
      return;
    }
    const runnerStartMs = performance.now();
    const runnerProfile = runner.tick(t.elapsedMS, inputSource);
    const runnerMs = performance.now() - runnerStartMs;

    const renderStartMs = performance.now();
    renderer.render(runner.state);
    const renderMs = performance.now() - renderStartMs;

    const onFrameStartMs = performance.now();
    options.onFrame?.(runner.state);
    const onFrameMs = performance.now() - onFrameStartMs;

    const gameOverUiStartMs = performance.now();
    updateGameOverLabel();
    const gameOverUiMs = performance.now() - gameOverUiStartMs;
    const uiMs = onFrameMs + gameOverUiMs;
    recordFrameBreakdown(tickStartMs, runnerProfile, runnerMs, renderMs, uiMs);
    recordPerfDuration('runtime.on_frame_ms', onFrameMs);
    recordPerfDuration('runtime.game_over_ui_ms', gameOverUiMs);
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
    options.onFrame?.(runner.state);
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
