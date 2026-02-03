import type { InputFrame } from '../core/types';
import { Keyboard } from './keyboard';

export interface InputConfig {
  dasMs: number;
  /**
   * Horizontal ARR (auto-repeat rate) in ms.
   * Use 0 for "instant ARR" (snap to wall after DAS).
   */
  arrMs: number;
}

export class InputController {
  private kb: Keyboard;
  private cfg: InputConfig;

  private activeDir: -1 | 0 | 1 = 0; // last pressed dir that is currently held
  private heldMs = 0;
  private nextRepeatAt = Infinity;

  constructor(kb: Keyboard, cfg: InputConfig) {
    this.kb = kb;
    this.cfg = cfg;
  }

  setConfig(cfg: Partial<InputConfig>): void {
    this.cfg = { ...this.cfg, ...cfg };
    // Reset repeat timing to avoid odd mid-hold behavior.
    if (this.activeDir === 0) {
      this.heldMs = 0;
      this.nextRepeatAt = Infinity;
    } else {
      this.heldMs = 0;
      this.nextRepeatAt = this.cfg.dasMs;
    }
  }

  sample(dtMs: number): InputFrame {
    const leftHeld = this.kb.isHeld('ArrowLeft');
    const rightHeld = this.kb.isHeld('ArrowRight');

    let moveX = 0;

    const leftPressed = this.kb.consumePressed('ArrowLeft');
    const rightPressed = this.kb.consumePressed('ArrowRight');

    // Fresh presses win and produce an immediate move
    if (leftPressed) {
      this.setActiveDir(-1);
      moveX = -1;
    }
    if (rightPressed) {
      this.setActiveDir(1);
      moveX = 1;
    }

    // If no fresh press, handle repeats
    if (moveX === 0) {
      // Resolve activeDir based on what’s held
      if (!leftHeld && !rightHeld) {
        this.setActiveDir(0);
      } else if (leftHeld && !rightHeld && this.activeDir !== -1) {
        this.setActiveDir(-1);
        moveX = -1; // immediate when switching from “none/opposite”
      } else if (rightHeld && !leftHeld && this.activeDir !== 1) {
        this.setActiveDir(1);
        moveX = 1;
      }

      if (this.activeDir !== 0 && (leftHeld || rightHeld)) {
        this.heldMs += dtMs;
        if (this.heldMs >= this.nextRepeatAt) {
          if (this.cfg.arrMs <= 0) {
            moveX = this.activeDir * Infinity;
          } else {
            const steps =
              1 +
              Math.floor(
                (this.heldMs - this.nextRepeatAt) / this.cfg.arrMs,
              );
            moveX = this.activeDir * steps;
            this.nextRepeatAt += steps * this.cfg.arrMs;
          }
        }
      }
    }

    const rotateCW =
      this.kb.consumePressed('ArrowUp') || this.kb.consumePressed('KeyE');
    const rotateCCW = this.kb.consumePressed('KeyW');
    const hold = this.kb.consumePressed('KeyQ');
    const hardDrop = this.kb.consumePressed('Space');
    const softDrop = this.kb.isHeld('ArrowDown');
    const rotate180 = this.kb.consumePressed('KeyA');
    const restart = this.kb.consumePressed('KeyR');

    const rotate: -1 | 0 | 1 = rotate180
      ? 0
      : rotateCW
        ? 1
        : rotateCCW
          ? -1
          : 0;

    return { moveX, rotate, rotate180, softDrop, hardDrop, hold, restart };
  }

  private setActiveDir(dir: -1 | 0 | 1): void {
    if (dir === this.activeDir) return;
    this.activeDir = dir;
    this.heldMs = 0;
    this.nextRepeatAt = dir === 0 ? Infinity : this.cfg.dasMs;
  }
}
