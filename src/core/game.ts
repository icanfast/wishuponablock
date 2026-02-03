import {
  COLS,
  DEFAULT_GRAVITY_MS,
  DEFAULT_HARD_LOCK_DELAY_MS,
  DEFAULT_LOCK_DELAY_MS,
  DEFAULT_SOFT_DROP_MS,
  NEXT_COUNT,
  ROWS,
  SPAWN_X,
  SPAWN_Y,
} from './constants';
import { clearLines, makeBoard } from './board';
import { Bag7 } from './bag7';
import type { PieceGenerator } from './generator';
import { XorShift32 } from './rng';
import {
  collides,
  cellsOf,
  dropDistance,
  merge,
  tryRotateSRS,
  tryRotate180PreferDirect,
} from './piece';

import type { Board, GameState, InputFrame, PieceKind } from './types';
import { PIECES } from './types';

export interface GameConfig {
  seed: number;
  gravityMs?: number;
  softDropMs?: number;
  lockDelayMs?: number;
  hardLockDelayMs?: number;
  generatorFactory?: (seed: number) => PieceGenerator;
  onPieceLock?: (board: Board) => void;
}

export class Game {
  readonly state: GameState;

  private generator: PieceGenerator;
  private rng: XorShift32;

  private gravityAcc = 0;
  private dropIntervalMs: number;
  private lockAcc = 0;
  private hardLockAcc = 0;
  private cheeseRows: boolean[] = [];
  private cheeseRemaining = 0;
  private cheeseActive = false;

  private gravityMs: number;
  private softDropMs: number;
  private lockDelayMs: number;
  private hardLockDelayMs: number;
  private onPieceLock?: (board: Board) => void;

  constructor(cfg: GameConfig) {
    this.rng = new XorShift32(cfg.seed);
    this.gravityMs = cfg.gravityMs ?? DEFAULT_GRAVITY_MS;
    this.dropIntervalMs = this.gravityMs;
    this.softDropMs = cfg.softDropMs ?? DEFAULT_SOFT_DROP_MS;
    this.lockDelayMs = cfg.lockDelayMs ?? DEFAULT_LOCK_DELAY_MS;
    this.hardLockDelayMs = cfg.hardLockDelayMs ?? DEFAULT_HARD_LOCK_DELAY_MS;
    this.onPieceLock = cfg.onPieceLock;

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
      gameWon: false,
    };

    this.resetCheeseState();
    this.updateNextView();
    this.spawnActive(first);
  }

  setConfig(cfg: Partial<GameConfig>): void {
    const prevGravity = this.gravityMs;
    const prevSoftDrop = this.softDropMs;

    if (cfg.gravityMs !== undefined) this.gravityMs = cfg.gravityMs;
    if (cfg.softDropMs !== undefined) this.softDropMs = cfg.softDropMs;
    if (cfg.lockDelayMs !== undefined) this.lockDelayMs = cfg.lockDelayMs;
    if (cfg.hardLockDelayMs !== undefined)
      this.hardLockDelayMs = cfg.hardLockDelayMs;

    // Keep current interval in sync if it was previously matching one of these.
    if (this.dropIntervalMs === prevGravity) {
      this.dropIntervalMs = this.gravityMs;
    } else if (this.dropIntervalMs === prevSoftDrop) {
      this.dropIntervalMs = this.softDropMs;
    }
  }

  step(dtMs: number, input: InputFrame): void {
    if (this.state.gameOver || this.state.gameWon) return;

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
      this.hardLockAcc += dtMs;
      if (
        this.lockAcc >= this.lockDelayMs ||
        this.hardLockAcc >= this.hardLockDelayMs
      ) {
        this.lockPiece();
      }
    } else {
      this.lockAcc = 0;
      this.hardLockAcc = 0;
    }
  }

  reset(seed: number): void {
    this.rng = new XorShift32(seed);
    this.generator.reset(seed);

    this.state.board = makeBoard();
    this.state.hold = null;
    this.state.canHold = true;
    this.state.next = [];
    this.state.gameOver = false;
    this.state.gameWon = false;
    this.resetCheeseState();

    this.gravityAcc = 0;
    this.lockAcc = 0;
    this.hardLockAcc = 0;

    const first = this.generator.next();
    this.updateNextView();
    this.spawnActive(first);
  }

  applyCheese(lines: number): void {
    const count = Math.max(0, Math.min(ROWS, Math.trunc(lines)));
    this.cheeseActive = count > 0;
    this.cheeseRemaining = count;
    this.cheeseRows = Array(ROWS).fill(false);
    this.state.gameWon = false;

    if (!this.cheeseActive) {
      return;
    }

    for (let i = 0; i < count; i++) {
      const y = ROWS - 1 - i;
      this.cheeseRows[y] = true;
      const hole = this.rng.nextInt(COLS);
      for (let x = 0; x < COLS; x++) {
        if (x === hole) {
          this.state.board[y][x] = null;
        } else {
          const kind = PIECES[this.rng.nextInt(PIECES.length)];
          this.state.board[y][x] = kind;
        }
      }
    }

    this.recomputeGhost();
  }

  private resetCheeseState(): void {
    this.cheeseRows = Array(ROWS).fill(false);
    this.cheeseRemaining = 0;
    this.cheeseActive = false;
    this.state.gameWon = false;
  }

  private applyCheeseClears(clearedRows: number[]): void {
    if (!this.cheeseActive) return;

    for (const y of clearedRows) {
      if (this.cheeseRows[y]) {
        this.cheeseRemaining = Math.max(0, this.cheeseRemaining - 1);
      }
      this.cheeseRows.splice(y, 1);
      this.cheeseRows.unshift(false);
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
    const hasAboveTop = cellsOf(this.state.active).some(([, y]) => y < 0);
    merge(this.state.board, this.state.active);
    const cleared = clearLines(this.state.board);
    if (cleared.length > 0) {
      this.applyCheeseClears(cleared);
    }
    this.onPieceLock?.(this.state.board);

    this.gravityAcc = 0;
    this.lockAcc = 0;
    this.state.canHold = true;
    this.hardLockAcc = 0;

    if (hasAboveTop) {
      this.state.gameOver = true;
      return;
    }

    if (this.cheeseActive && this.cheeseRemaining <= 0) {
      this.state.gameWon = true;
      return;
    }

    this.spawnNext();
  }

  private spawnNext(): void {
    const k = this.generator.next();
    this.updateNextView();
    this.spawnActive(k);
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
    this.gravityAcc = 0;
    this.lockAcc = 0;
    this.hardLockAcc = 0;

    this.spawnActive(held);
  }

  private updateNextView(): void {
    this.state.next = this.generator.peek(NEXT_COUNT);
  }

  private spawnActive(k: PieceKind): void {
    this.state.active = { k, r: 0, x: SPAWN_X, y: SPAWN_Y };
    this.liftSpawnIfBlocked();
    this.recomputeGhost();
    if (collides(this.state.board, this.state.active)) {
      this.state.gameOver = true;
    }
  }

  private liftSpawnIfBlocked(): void {
    for (let i = 0; i < 4; i++) {
      if (!collides(this.state.board, this.state.active)) return;
      this.state.active.y -= 1;
    }
  }

  private recomputeGhost(): void {
    const d = dropDistance(this.state.board, this.state.active);
    this.state.ghostY = this.state.active.y + d;
  }
}
