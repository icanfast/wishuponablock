import type { InputFrame } from '../core/types';
import { DEFAULT_KEY_BINDINGS } from '../core/constants';
import { Keyboard } from './keyboard';

export interface KeyBindings {
  moveLeft: string;
  moveRight: string;
  softDrop: string;
  hardDrop: string;
  rotateCW: string;
  rotateCCW: string;
  rotate180: string;
  hold: string;
  restart: string;
}

export interface InputConfig {
  dasMs: number;
  /**
   * Horizontal ARR (auto-repeat rate) in ms.
   * Use 0 for "instant ARR" (snap to wall after DAS).
   */
  arrMs: number;
  bindings: KeyBindings;
}

export class InputController {
  private kb: Keyboard;
  private cfg: InputConfig;

  private activeDir: -1 | 0 | 1 = 0; // last pressed dir that is currently held
  private heldMs = 0;
  private nextRepeatAt = Infinity;

  constructor(kb: Keyboard, cfg: InputConfig) {
    this.kb = kb;
    this.cfg = {
      ...cfg,
      bindings: {
        ...DEFAULT_KEY_BINDINGS,
        ...cfg.bindings,
      },
    };
  }

  setConfig(cfg: Partial<InputConfig>): void {
    this.cfg = {
      ...this.cfg,
      ...cfg,
      bindings: {
        ...this.cfg.bindings,
        ...cfg.bindings,
      },
    };
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
    const bindings = this.cfg.bindings;
    const leftHeld = this.kb.isHeld(bindings.moveLeft);
    const rightHeld = this.kb.isHeld(bindings.moveRight);

    let moveX = 0;
    let moveXFromRepeat = false;

    const leftPressed = this.kb.consumePressed(bindings.moveLeft);
    const rightPressed = this.kb.consumePressed(bindings.moveRight);

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
          moveXFromRepeat = true;
          if (this.cfg.arrMs <= 0) {
            moveX = this.activeDir * Infinity;
          } else {
            const steps =
              1 +
              Math.floor((this.heldMs - this.nextRepeatAt) / this.cfg.arrMs);
            moveX = this.activeDir * steps;
            this.nextRepeatAt += steps * this.cfg.arrMs;
          }
        }
      }
    }

    const rotateCW =
      this.kb.consumePressed(bindings.rotateCW) ||
      this.kb.consumePressed('ArrowUp');
    const rotateCCW = this.kb.consumePressed(bindings.rotateCCW);
    const hold = this.kb.consumePressed(bindings.hold);
    const hardDrop = this.kb.consumePressed(bindings.hardDrop);
    const softDrop = this.kb.isHeld(bindings.softDrop);
    const rotate180 = this.kb.consumePressed(bindings.rotate180);
    const restart = this.kb.consumePressed(bindings.restart);

    const rotate: -1 | 0 | 1 = rotate180
      ? 0
      : rotateCW
        ? 1
        : rotateCCW
          ? -1
          : 0;

    return {
      moveX,
      moveXFromRepeat,
      rotate,
      rotate180,
      softDrop,
      hardDrop,
      hold,
      restart,
    };
  }

  private setActiveDir(dir: -1 | 0 | 1): void {
    if (dir === this.activeDir) return;
    this.activeDir = dir;
    this.heldMs = 0;
    this.nextRepeatAt = dir === 0 ? Infinity : this.cfg.dasMs;
  }
}
