import type { Game } from './game';
import type { GameState, InputFrame } from './types';
import {
  hasPerfMetricsSink,
  recordPerfCount,
  recordPerfDuration,
  readPerfDurationTotal,
} from './perfMetrics';

export interface InputSource {
  sample(state: GameState, dtMs: number): InputFrame;
  reset?(seed: number): void;
}

export interface GameRunnerOptions {
  fixedStepMs: number;
  onRestart?: (game: Game) => void;
  /**
   * Optional clamp to prevent spiral-of-death after long stalls.
   */
  maxElapsedMs?: number;
  /**
   * Optional cap on steps per tick. Extra accumulated time is dropped.
   */
  maxStepsPerTick?: number;
}

type RunnerStepProfile = {
  totalMs: number;
  inputMs: number;
  gameMs: number;
  mlMs: number;
  mlInputEncodeMs: number;
  mlConvStackMs: number;
  mlPoolMs: number;
  mlHeadMs: number;
  mlPostMs: number;
  mlOtherMs: number;
  overheadMs: number;
};

export interface RunnerTickProfile {
  totalMs: number;
  steps: number;
  inputMs: number;
  mlMs: number;
  mlInputEncodeMs: number;
  mlConvStackMs: number;
  mlPoolMs: number;
  mlHeadMs: number;
  mlPostMs: number;
  mlOtherMs: number;
  simulationMs: number;
  stepOverheadMs: number;
  loopOverheadMs: number;
}

const EMPTY_INPUT: InputFrame = {
  moveX: 0,
  rotate: 0,
  rotate180: false,
  softDrop: false,
  hardDrop: false,
  hold: false,
  restart: false,
};

export class GameRunner {
  private accMs = 0;

  constructor(
    private game: Game,
    private options: GameRunnerOptions,
  ) {}

  get state(): GameState {
    return this.game.state;
  }

  resetTiming(): void {
    this.accMs = 0;
  }

  tick(
    elapsedMs: number,
    input: InputSource = NullInputSource,
  ): RunnerTickProfile | null {
    const perfEnabled = hasPerfMetricsSink();
    const tickStartMs = perfEnabled ? performance.now() : 0;
    const clamped =
      this.options.maxElapsedMs == null
        ? elapsedMs
        : Math.min(elapsedMs, this.options.maxElapsedMs);

    this.accMs += clamped;
    let steps = 0;
    let stepTotalMs = 0;
    let stepMaxMs = 0;
    let stepInputMs = 0;
    let stepGameMs = 0;
    let stepMlMs = 0;
    let stepMlInputEncodeMs = 0;
    let stepMlConvStackMs = 0;
    let stepMlPoolMs = 0;
    let stepMlHeadMs = 0;
    let stepMlPostMs = 0;
    let stepMlOtherMs = 0;
    let stepOverheadMs = 0;

    while (this.accMs >= this.options.fixedStepMs) {
      const stepProfile = this.step(input);
      if (perfEnabled && stepProfile) {
        stepTotalMs += stepProfile.totalMs;
        stepInputMs += stepProfile.inputMs;
        stepGameMs += stepProfile.gameMs;
        stepMlMs += stepProfile.mlMs;
        stepMlInputEncodeMs += stepProfile.mlInputEncodeMs;
        stepMlConvStackMs += stepProfile.mlConvStackMs;
        stepMlPoolMs += stepProfile.mlPoolMs;
        stepMlHeadMs += stepProfile.mlHeadMs;
        stepMlPostMs += stepProfile.mlPostMs;
        stepMlOtherMs += stepProfile.mlOtherMs;
        stepOverheadMs += stepProfile.overheadMs;
        if (stepProfile.totalMs > stepMaxMs) stepMaxMs = stepProfile.totalMs;
      }
      this.accMs -= this.options.fixedStepMs;
      steps++;
      if (
        this.options.maxStepsPerTick != null &&
        steps >= this.options.maxStepsPerTick
      ) {
        this.accMs = 0;
        break;
      }
    }
    if (perfEnabled) {
      const nowMs = performance.now();
      const tickTotalMs = nowMs - tickStartMs;
      const simulationMs = Math.max(0, stepGameMs - stepMlMs);
      const loopOverheadMs = Math.max(0, tickTotalMs - stepTotalMs);
      recordPerfCount('runner.tick.steps', steps, nowMs);
      if (steps > 0) {
        recordPerfDuration('runner.tick.step_total_ms', stepTotalMs, nowMs);
        recordPerfDuration('runner.tick.step_max_ms', stepMaxMs, nowMs);
        recordPerfDuration(
          'runner.tick.step_avg_ms',
          stepTotalMs / Math.max(1, steps),
          nowMs,
        );
      }
      recordPerfDuration('runner.tick.input_ms', stepInputMs, nowMs);
      recordPerfDuration('runner.tick.game_ms', stepGameMs, nowMs);
      recordPerfDuration('runner.tick.ml_ms', stepMlMs, nowMs);
      recordPerfDuration(
        'runner.tick.ml_input_encode_ms',
        stepMlInputEncodeMs,
        nowMs,
      );
      recordPerfDuration(
        'runner.tick.ml_conv_stack_ms',
        stepMlConvStackMs,
        nowMs,
      );
      recordPerfDuration('runner.tick.ml_pool_ms', stepMlPoolMs, nowMs);
      recordPerfDuration('runner.tick.ml_head_ms', stepMlHeadMs, nowMs);
      recordPerfDuration('runner.tick.ml_post_ms', stepMlPostMs, nowMs);
      recordPerfDuration('runner.tick.ml_other_ms', stepMlOtherMs, nowMs);
      recordPerfDuration('runner.tick.simulation_ms', simulationMs, nowMs);
      recordPerfDuration('runner.tick.step_overhead_ms', stepOverheadMs, nowMs);
      recordPerfDuration('runner.tick.loop_overhead_ms', loopOverheadMs, nowMs);
      recordPerfDuration('runner.tick.total_ms', tickTotalMs, nowMs);
      return {
        totalMs: tickTotalMs,
        steps,
        inputMs: stepInputMs,
        mlMs: stepMlMs,
        mlInputEncodeMs: stepMlInputEncodeMs,
        mlConvStackMs: stepMlConvStackMs,
        mlPoolMs: stepMlPoolMs,
        mlHeadMs: stepMlHeadMs,
        mlPostMs: stepMlPostMs,
        mlOtherMs: stepMlOtherMs,
        simulationMs,
        stepOverheadMs,
        loopOverheadMs,
      };
    }
    return null;
  }

  step(input: InputSource = NullInputSource): RunnerStepProfile | null {
    const perfEnabled = hasPerfMetricsSink();
    if (!perfEnabled) {
      const frame = input.sample(this.game.state, this.options.fixedStepMs);
      if (frame.restart) {
        this.options.onRestart?.(this.game);
        return null;
      }
      this.game.step(this.options.fixedStepMs, frame);
      return null;
    }
    const stepStartMs = performance.now();
    const inputStartMs = performance.now();
    const frame = input.sample(this.game.state, this.options.fixedStepMs);
    const inputEndMs = performance.now();
    const inputMs = inputEndMs - inputStartMs;
    recordPerfDuration('runner.input.sample_ms', inputMs, inputEndMs);
    if (frame.restart) {
      this.options.onRestart?.(this.game);
      const totalMs = performance.now() - stepStartMs;
      recordPerfDuration('runner.step.total_ms', totalMs);
      return {
        totalMs,
        inputMs,
        gameMs: 0,
        mlMs: 0,
        mlInputEncodeMs: 0,
        mlConvStackMs: 0,
        mlPoolMs: 0,
        mlHeadMs: 0,
        mlPostMs: 0,
        mlOtherMs: 0,
        overheadMs: Math.max(0, totalMs - inputMs),
      };
    }
    const modelInputBefore = readPerfDurationTotal('ml.model.input_encode_ms');
    const modelConvBefore = readPerfDurationTotal('ml.model.conv_stack_ms');
    const modelPoolBefore = readPerfDurationTotal('ml.model.pool_ms');
    const modelHeadBefore = readPerfDurationTotal('ml.model.head_ms');
    const modelPostBefore =
      readPerfDurationTotal('ml.sample_distribution_ms') +
      readPerfDurationTotal('curse.infer_distribution_ms');
    const mlBefore =
      readPerfDurationTotal('ml.on_lock_ms') +
      readPerfDurationTotal('curse.on_lock_ms');
    const gameStepStartMs = performance.now();
    this.game.step(this.options.fixedStepMs, frame);
    const gameStepEndMs = performance.now();
    const gameMs = gameStepEndMs - gameStepStartMs;
    const mlAfter =
      readPerfDurationTotal('ml.on_lock_ms') +
      readPerfDurationTotal('curse.on_lock_ms');
    const mlMs = Math.max(0, mlAfter - mlBefore);
    const mlInputEncodeMs = Math.max(
      0,
      readPerfDurationTotal('ml.model.input_encode_ms') - modelInputBefore,
    );
    const mlConvStackMs = Math.max(
      0,
      readPerfDurationTotal('ml.model.conv_stack_ms') - modelConvBefore,
    );
    const mlPoolMs = Math.max(
      0,
      readPerfDurationTotal('ml.model.pool_ms') - modelPoolBefore,
    );
    const mlHeadMs = Math.max(
      0,
      readPerfDurationTotal('ml.model.head_ms') - modelHeadBefore,
    );
    const mlPostMs = Math.max(
      0,
      readPerfDurationTotal('ml.sample_distribution_ms') +
        readPerfDurationTotal('curse.infer_distribution_ms') -
        modelPostBefore,
    );
    const mlKnownMs =
      mlInputEncodeMs + mlConvStackMs + mlPoolMs + mlHeadMs + mlPostMs;
    const mlOtherMs = Math.max(0, mlMs - mlKnownMs);
    const totalMs = gameStepEndMs - stepStartMs;
    const overheadMs = Math.max(0, totalMs - inputMs - gameMs);
    recordPerfDuration('runner.game.step_ms', gameMs, gameStepEndMs);
    recordPerfDuration('runner.step.ml_ms', mlMs, gameStepEndMs);
    recordPerfDuration(
      'runner.step.ml_input_encode_ms',
      mlInputEncodeMs,
      gameStepEndMs,
    );
    recordPerfDuration(
      'runner.step.ml_conv_stack_ms',
      mlConvStackMs,
      gameStepEndMs,
    );
    recordPerfDuration('runner.step.ml_pool_ms', mlPoolMs, gameStepEndMs);
    recordPerfDuration('runner.step.ml_head_ms', mlHeadMs, gameStepEndMs);
    recordPerfDuration('runner.step.ml_post_ms', mlPostMs, gameStepEndMs);
    recordPerfDuration('runner.step.ml_other_ms', mlOtherMs, gameStepEndMs);
    recordPerfDuration('runner.step.total_ms', totalMs, gameStepEndMs);
    return {
      totalMs,
      inputMs,
      gameMs,
      mlMs,
      mlInputEncodeMs,
      mlConvStackMs,
      mlPoolMs,
      mlHeadMs,
      mlPostMs,
      mlOtherMs,
      overheadMs,
    };
  }

  runSteps(steps: number, input: InputSource = NullInputSource): void {
    const count = Math.max(0, Math.trunc(steps));
    for (let i = 0; i < count; i++) this.step(input);
  }

  runFor(ms: number, input: InputSource = NullInputSource): void {
    const steps = Math.floor(ms / this.options.fixedStepMs);
    this.runSteps(steps, input);
  }

  runUntil(
    predicate: (state: GameState) => boolean,
    maxSteps: number | undefined,
    input: InputSource = NullInputSource,
  ): void {
    const limit = maxSteps == null ? Infinity : Math.max(0, maxSteps);
    let steps = 0;
    while (!predicate(this.game.state) && steps < limit) {
      this.step(input);
      steps++;
    }
  }
}

export const NullInputSource: InputSource = {
  sample: () => EMPTY_INPUT,
};
