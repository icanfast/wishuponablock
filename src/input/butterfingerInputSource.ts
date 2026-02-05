import type { InputFrame, GameState } from '../core/types';
import type { InputSource } from '../core/runner';
import type { ButterfingerSettings } from '../core/settings';
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

const clamp01 = (value: number): number =>
  Math.min(1, Math.max(0, value));

export class ButterfingerInputSource implements InputSource {
  private rng: XorShift32;
  private config: ButterfingerSettings;

  constructor(
    private base: InputSource,
    config: ButterfingerSettings,
    seed = Date.now(),
  ) {
    this.config = config;
    this.rng = new XorShift32(seed);
  }

  reset(seed: number): void {
    this.rng = new XorShift32(seed);
    this.base.reset?.(seed);
  }

  setConfig(config: ButterfingerSettings): void {
    this.config = config;
  }

  sample(state: GameState, dtMs: number): InputFrame {
    const frame = this.base.sample(state, dtMs);
    if (!this.config.enabled) return frame;
    return this.apply(frame);
  }

  private apply(frame: InputFrame): InputFrame {
    const out: InputFrame = { ...frame };
    const missRate = clamp01(this.config.missRate);
    const wrongRate = clamp01(this.config.wrongDirRate);
    const extraRate = clamp01(this.config.extraTapRate);

    if (out.moveX !== 0) {
      const allowWrongDir = !frame.moveXFromRepeat;
      if (this.roll(missRate)) {
        out.moveX = 0;
      } else {
        if (allowWrongDir && this.roll(wrongRate)) {
          out.moveX = -out.moveX;
        }
        if (this.roll(extraRate)) {
          const sign = Math.sign(out.moveX);
          if (Number.isFinite(out.moveX)) {
            out.moveX += sign;
          } else if (sign !== 0) {
            out.moveX = sign * Number.POSITIVE_INFINITY;
          }
        }
      }
    }

    return out;
  }

  private roll(rate: number): boolean {
    if (rate <= 0) return false;
    return this.rng.nextU32() / 0xffffffff < rate;
  }
}

export const NullButterfingerConfig: ButterfingerSettings = {
  enabled: false,
  missRate: 0,
  wrongDirRate: 0,
  extraTapRate: 0,
  lockNudgeRate: 0,
};
