import type { Game } from './game';
import type { GameState, InputFrame } from './types';

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

  tick(elapsedMs: number, input: InputSource = NullInputSource): void {
    const clamped =
      this.options.maxElapsedMs == null
        ? elapsedMs
        : Math.min(elapsedMs, this.options.maxElapsedMs);

    this.accMs += clamped;
    let steps = 0;

    while (this.accMs >= this.options.fixedStepMs) {
      this.step(input);
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
  }

  step(input: InputSource = NullInputSource): void {
    const frame = input.sample(this.game.state, this.options.fixedStepMs);
    if (frame.restart) {
      this.options.onRestart?.(this.game);
      return;
    }
    this.game.step(this.options.fixedStepMs, frame);
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
