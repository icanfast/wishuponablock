import {
  DEFAULT_GRAVITY_MS,
  DEFAULT_LOCK_DELAY_MS,
  DEFAULT_SOFT_DROP_MS,
  NEXT_COUNT,
  SPAWN_X,
  SPAWN_Y,
} from './constants';
import { clearLines, makeBoard } from './board';
import { Bag7 } from './bag7';
import type { PieceGenerator } from './generator';
import {
  collides,
  dropDistance,
  merge,
  tryRotateSRS,
  tryRotate180PreferDirect,
} from './piece';

import type { GameState, InputFrame } from './types';

export interface GameConfig {
  seed: number;
  gravityMs?: number;
  softDropMs?: number;
  lockDelayMs?: number;
  generatorFactory?: (seed: number) => PieceGenerator;
}

export class Game {
  readonly state: GameState;

  private generator: PieceGenerator;

  private gravityAcc = 0;
  private dropIntervalMs: number;
  private lockAcc = 0;

  private gravityMs: number;
  private softDropMs: number;
  private lockDelayMs: number;

  constructor(cfg: GameConfig) {
    this.gravityMs = cfg.gravityMs ?? DEFAULT_GRAVITY_MS;
    this.dropIntervalMs = this.gravityMs;
    this.softDropMs = cfg.softDropMs ?? DEFAULT_SOFT_DROP_MS;
    this.lockDelayMs = cfg.lockDelayMs ?? DEFAULT_LOCK_DELAY_MS;

    const makeGenerator = cfg.generatorFactory ?? ((seed) => new Bag7(seed));
    this.generator = makeGenerator(cfg.seed);

    const board = makeBoard();
    const first = this.generator.next();

    this.state = {
      board,
      active: { k: first, r: 0, x: SPAWN_X, y: SPAWN_Y },
      ghostY: 0,
      hold: null,
      canHold: true,
      next: [],
      gameOver: false,
    };

    this.updateNextView();
    this.recomputeGhost();
    if (collides(this.state.board, this.state.active))
      this.state.gameOver = true;
  }

  setConfig(cfg: Partial<GameConfig>): void {
    const prevGravity = this.gravityMs;
    const prevSoftDrop = this.softDropMs;

    if (cfg.gravityMs !== undefined) this.gravityMs = cfg.gravityMs;
    if (cfg.softDropMs !== undefined) this.softDropMs = cfg.softDropMs;
    if (cfg.lockDelayMs !== undefined) this.lockDelayMs = cfg.lockDelayMs;

    // Keep current interval in sync if it was previously matching one of these.
    if (this.dropIntervalMs === prevGravity) {
      this.dropIntervalMs = this.gravityMs;
    } else if (this.dropIntervalMs === prevSoftDrop) {
      this.dropIntervalMs = this.softDropMs;
    }
  }

  step(dtMs: number, input: InputFrame): void {
    if (this.state.gameOver) return;

    if (this.applyInput(input)) return;
    this.applyGravity(dtMs, input);
    this.applyLock(dtMs);
  }

  private applyInput(input: InputFrame): boolean {
    if (input.hold) this.doHold();
    if (input.rotate !== 0) this.doRotate(input.rotate);
    if (input.moveX !== 0) this.doMoveSteps(input.moveX);
    if (input.rotate180) this.doRotate180();
    if (input.hardDrop) {
      this.doHardDrop();
      return true;
    }
    return false;
  }

  private applyGravity(dtMs: number, input: InputFrame): void {
    const softInstant = input.softDrop && this.softDropMs <= 0;

    if (softInstant) {
      this.doSoftDropInstant();
      return;
    }

    const newInterval = input.softDrop ? this.softDropMs : this.gravityMs;

    // adjust accumulator when interval changes to avoid "retroactive" drops
    this.adjustDropInterval(newInterval);

    if (this.dropIntervalMs > 0) {
      this.gravityAcc += dtMs;

      while (this.gravityAcc >= this.dropIntervalMs) {
        this.gravityAcc -= this.dropIntervalMs;

        if (!this.tryMoveDown()) {
          // grounded: stop consuming extra "gravity" this tick
          this.gravityAcc = 0;
          break;
        }
      }
    }
  }

  private applyLock(dtMs: number): void {
    if (
      collides(this.state.board, this.state.active, this.state.active.r, 0, 1)
    ) {
      this.lockAcc += dtMs;
      if (this.lockAcc >= this.lockDelayMs) {
        this.lockPiece();
      }
    } else {
      this.lockAcc = 0;
    }
  }

  reset(seed: number): void {
    this.generator.reset(seed);

    this.state.board = makeBoard();
    this.state.hold = null;
    this.state.canHold = true;
    this.state.next = [];
    this.state.gameOver = false;

    this.gravityAcc = 0;
    this.lockAcc = 0;

    const first = this.generator.next();
    this.state.active = { k: first, r: 0, x: SPAWN_X, y: SPAWN_Y };

    this.updateNextView();
    this.recomputeGhost();

    if (collides(this.state.board, this.state.active)) {
      this.state.gameOver = true;
    }
  }

  private adjustDropInterval(newInterval: number): void {
    if (newInterval === this.dropIntervalMs) return;

    // If either is zero/negative, just reset; we'll handle true "infinite gravity" later if desired.
    if (this.dropIntervalMs <= 0 || newInterval <= 0) {
      this.gravityAcc = 0;
      this.dropIntervalMs = newInterval;
      return;
    }

    const phase = this.gravityAcc / this.dropIntervalMs; // 0..1-ish
    this.gravityAcc = phase * newInterval;

    // safety clamp
    this.gravityAcc = Math.min(this.gravityAcc, newInterval);
    this.dropIntervalMs = newInterval;
  }

  private doMoveSteps(move: number): void {
    if (move === 0) return;

    const dir = move < 0 ? -1 : 1;
    let moved = false;

    if (!Number.isFinite(move)) {
      while (
        !collides(
          this.state.board,
          this.state.active,
          this.state.active.r,
          dir,
          0,
        )
      ) {
        this.state.active.x += dir;
        moved = true;
      }
    } else {
      const steps = Math.abs(Math.trunc(move));
      for (let i = 0; i < steps; i++) {
        if (
          collides(
            this.state.board,
            this.state.active,
            this.state.active.r,
            dir,
            0,
          )
        ) {
          break;
        }
        this.state.active.x += dir;
        moved = true;
      }
    }

    if (moved) {
      this.lockAcc = 0;
      this.recomputeGhost();
    }
  }

  private doRotate(dir: -1 | 1): void {
    const changed = tryRotateSRS(this.state.board, this.state.active, dir);
    if (changed) {
      this.lockAcc = 0;
      this.recomputeGhost();
    }
  }

  private doRotate180(): void {
    const changed = tryRotate180PreferDirect(
      this.state.board,
      this.state.active,
    );
    if (changed) {
      this.lockAcc = 0;
      this.recomputeGhost();
    }
  }

  private tryMoveDown(): boolean {
    if (
      !collides(this.state.board, this.state.active, this.state.active.r, 0, 1)
    ) {
      this.state.active.y += 1;
      this.lockAcc = 0;
      this.recomputeGhost();
      return true;
    }
    return false;
  }

  private doHardDrop(): void {
    const d = dropDistance(this.state.board, this.state.active);
    this.state.active.y += d;
    this.lockPiece();
  }

  private doSoftDropInstant(): void {
    const d = dropDistance(this.state.board, this.state.active);
    if (d > 0) {
      this.state.active.y += d;
      this.lockAcc = 0;
      this.gravityAcc = 0;
      this.recomputeGhost();
    } else {
      this.gravityAcc = 0;
    }
  }

  private lockPiece(): void {
    merge(this.state.board, this.state.active);
    clearLines(this.state.board);

    this.gravityAcc = 0;
    this.lockAcc = 0;
    this.state.canHold = true;

    this.spawnNext();
  }

  private spawnNext(): void {
    const k = this.generator.next();

    this.state.active = { k, r: 0, x: SPAWN_X, y: SPAWN_Y };

    this.updateNextView();
    this.recomputeGhost();

    if (collides(this.state.board, this.state.active)) {
      this.state.gameOver = true;
    }
  }

  private doHold(): void {
    if (!this.state.canHold) return;

    const current = this.state.active.k;
    const held = this.state.hold;

    this.state.canHold = false;

    if (held == null) {
      this.state.hold = current;
      this.spawnNext();
      return;
    }

    this.state.hold = current;
    this.state.active = { k: held, r: 0, x: SPAWN_X, y: SPAWN_Y };
    this.gravityAcc = 0;
    this.lockAcc = 0;

    this.recomputeGhost();

    if (collides(this.state.board, this.state.active)) {
      this.state.gameOver = true;
    }
  }

  private updateNextView(): void {
    this.state.next = this.generator.peek(NEXT_COUNT);
  }

  private recomputeGhost(): void {
    const d = dropDistance(this.state.board, this.state.active);
    this.state.ghostY = this.state.active.y + d;
  }
}
