import type { InputFrame, GameState, ActivePiece } from '../core/types';
import type { InputSource } from '../core/runner';
import { GameRunner } from '../core/runner';
import type { Game } from '../core/game';
import { XorShift32 } from '../core/rng';

const EMPTY_INPUT: InputFrame = {
  moveX: 0,
  rotate: 0,
  rotate180: false,
  softDrop: false,
  hardDrop: false,
  hold: false,
  restart: false,
};

type RotationPlan = 'none' | 'cw' | 'ccw' | '180';

const HORIZONTAL_STEPS = [-3, -2, -1, 0, 1, 2, 3, 4];

export class CharcuterieBot implements InputSource {
  private rng: XorShift32;
  private activeRef: ActivePiece | null = null;
  private queue: InputFrame[] = [];

  constructor(seed: number) {
    this.rng = new XorShift32(seed);
  }

  reset(seed: number): void {
    this.rng = new XorShift32(seed);
    this.activeRef = null;
    this.queue = [];
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  sample(state: GameState, _dtMs: number): InputFrame {
    if (this.activeRef !== state.active) {
      this.activeRef = state.active;
      this.queue = this.buildPlan();
    }
    return this.queue.shift() ?? EMPTY_INPUT;
  }

  private buildPlan(): InputFrame[] {
    const rotation = this.pickRotation();
    const steps = HORIZONTAL_STEPS[this.rng.nextInt(HORIZONTAL_STEPS.length)];

    const frames: InputFrame[] = [];

    if (rotation !== 'none') {
      frames.push({
        ...EMPTY_INPUT,
        rotate: rotation === 'cw' ? 1 : rotation === 'ccw' ? -1 : 0,
        rotate180: rotation === '180',
      });
    }

    frames.push({
      ...EMPTY_INPUT,
      moveX: steps,
      hardDrop: true,
    });

    return frames;
  }

  private pickRotation(): RotationPlan {
    const roll = this.rng.nextInt(100); // 0..99
    if (roll < 30) return 'none';
    if (roll < 52) return 'cw';
    if (roll < 78) return '180';
    return 'ccw';
  }
}

export function runBotForPieces(
  game: Game,
  bot: InputSource,
  pieces: number,
  fixedStepMs = 1000 / 120,
): number {
  const target = Math.max(0, Math.trunc(pieces));
  if (target === 0) return 0;

  const runner = new GameRunner(game, { fixedStepMs });
  let placed = 0;
  let lastActive = game.state.active;

  while (placed < target && !game.state.gameOver) {
    runner.step(bot);
    if (game.state.active !== lastActive) {
      placed++;
      lastActive = game.state.active;
    }
  }

  return placed;
}
