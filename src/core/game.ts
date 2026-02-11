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
import { getSrsKickTests } from './srs';

import type {
  Board,
  GameState,
  InputFrame,
  PieceKind,
  Rotation,
} from './types';
import { PIECES } from './types';

export interface GameConfig {
  seed: number;
  gravityMs?: number;
  softDropMs?: number;
  lockDelayMs?: number;
  hardLockDelayMs?: number;
  lockNudgeRate?: number;
  gravityDropRate?: number;
  lockRotateRate?: number;
  lineGoal?: number;
  classicStartLevel?: number;
  scoringEnabled?: boolean;
  generatorFactory?: (seed: number) => PieceGenerator;
  onPieceLock?: (board: Board, hold: PieceKind | null) => void;
  onHold?: (board: Board, hold: PieceKind | null) => void;
  onLineClear?: (combo: number, clearedLines: number) => void;
}

export class Game {
  readonly state: GameState;

  private generator: PieceGenerator;
  private rng: XorShift32;
  totalLinesCleared = 0;

  private gravityAcc = 0;
  private dropIntervalMs: number;
  private lockAcc = 0;
  private hardLockAcc = 0;
  private cheeseRows: boolean[] = [];
  private cheeseRemaining = 0;
  private cheeseActive = false;
  private initialBlocks: boolean[][] = [];
  private initialBlocksRemaining = 0;
  private initialBlocksActive = false;
  private lockNudgeRate = 0;
  private gravityDropRate = 0;
  private lockRotateRate = 0;
  private butterfingerRng: XorShift32;
  private lineGoal: number | null = null;
  private classicStartLevel: number | null = null;
  private scoringEnabled = false;
  private softDropHeld = false;
  private softDropSegmentCells = 0;
  private softDropLastSegmentCells = 0;

  private gravityMs: number;
  private softDropMs: number;
  private lockDelayMs: number;
  private hardLockDelayMs: number;
  private onPieceLock?: (board: Board, hold: PieceKind | null) => void;
  private onHold?: (board: Board, hold: PieceKind | null) => void;
  private onLineClear?: (combo: number, clearedLines: number) => void;

  constructor(cfg: GameConfig) {
    this.rng = new XorShift32(cfg.seed);
    this.gravityMs = cfg.gravityMs ?? DEFAULT_GRAVITY_MS;
    this.dropIntervalMs = this.gravityMs;
    this.softDropMs = cfg.softDropMs ?? DEFAULT_SOFT_DROP_MS;
    this.lockDelayMs = cfg.lockDelayMs ?? DEFAULT_LOCK_DELAY_MS;
    this.hardLockDelayMs = cfg.hardLockDelayMs ?? DEFAULT_HARD_LOCK_DELAY_MS;
    this.lockNudgeRate = clamp01(cfg.lockNudgeRate ?? 0);
    this.gravityDropRate = clamp01(cfg.gravityDropRate ?? 0);
    this.lockRotateRate = clamp01(cfg.lockRotateRate ?? 0);
    this.lineGoal =
      cfg.lineGoal != null && Number.isFinite(cfg.lineGoal)
        ? Math.max(1, Math.trunc(cfg.lineGoal))
        : null;
    this.classicStartLevel =
      cfg.classicStartLevel != null && Number.isFinite(cfg.classicStartLevel)
        ? Math.max(0, Math.trunc(cfg.classicStartLevel))
        : null;
    this.scoringEnabled = Boolean(cfg.scoringEnabled);
    this.onPieceLock = cfg.onPieceLock;
    this.onHold = cfg.onHold;
    this.onLineClear = cfg.onLineClear;
    this.butterfingerRng = new XorShift32(cfg.seed ^ 0x6d2b79f5);

    const makeGenerator = cfg.generatorFactory ?? ((seed) => new Bag7(seed));
    this.generator = makeGenerator(cfg.seed);

    const board = makeBoard();
    this.generator.onLock?.(board, null);
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
      combo: 0,
      timeMs: 0,
      totalLinesCleared: 0,
      lineGoal: this.lineGoal,
      level: this.getClassicLevel(0),
      score: 0,
      scoringEnabled: this.scoringEnabled,
    };

    if (this.classicStartLevel != null) {
      this.gravityMs = getClassicGravityMs(this.state.level);
      this.dropIntervalMs = this.gravityMs;
    }

    this.totalLinesCleared = 0;
    this.state.totalLinesCleared = 0;
    this.resetCheeseState();
    this.resetInitialBlocks();
    this.updateNextView();
    this.spawnActive(first);
  }

  applyInitialBoard(board: Board): void {
    this.state.board = board.map((row) => row.slice());
    this.gravityAcc = 0;
    this.lockAcc = 0;
    this.hardLockAcc = 0;
    this.state.gameOver = false;
    this.state.gameWon = false;
    this.state.combo = 0;
    this.state.timeMs = 0;
    this.totalLinesCleared = 0;
    this.state.totalLinesCleared = 0;
    this.state.level = this.getClassicLevel(0);
    this.state.score = 0;
    this.state.scoringEnabled = this.scoringEnabled;
    if (this.generator.onLock) {
      this.generator.onLock(this.state.board, this.state.hold);
      const first = this.generator.next();
      this.updateNextView();
      this.spawnActive(first);
      return;
    }
    this.liftSpawnIfBlocked();
    this.recomputeGhost();
    if (collides(this.state.board, this.state.active)) {
      this.state.gameOver = true;
    }
  }

  setConfig(cfg: Partial<GameConfig>): void {
    const prevGravity = this.gravityMs;
    const prevSoftDrop = this.softDropMs;

    if (cfg.gravityMs !== undefined) this.gravityMs = cfg.gravityMs;
    if (cfg.softDropMs !== undefined) this.softDropMs = cfg.softDropMs;
    if (cfg.lockDelayMs !== undefined) this.lockDelayMs = cfg.lockDelayMs;
    if (cfg.hardLockDelayMs !== undefined)
      this.hardLockDelayMs = cfg.hardLockDelayMs;
    if (cfg.lockNudgeRate !== undefined)
      this.lockNudgeRate = clamp01(cfg.lockNudgeRate);
    if (cfg.gravityDropRate !== undefined)
      this.gravityDropRate = clamp01(cfg.gravityDropRate);
    if (cfg.lockRotateRate !== undefined)
      this.lockRotateRate = clamp01(cfg.lockRotateRate);
    if (cfg.scoringEnabled !== undefined) {
      this.scoringEnabled = Boolean(cfg.scoringEnabled);
      this.state.scoringEnabled = this.scoringEnabled;
    }

    // Keep current interval in sync if it was previously matching one of these.
    if (this.dropIntervalMs === prevGravity) {
      this.dropIntervalMs = this.gravityMs;
    } else if (this.dropIntervalMs === prevSoftDrop) {
      this.dropIntervalMs = this.softDropMs;
    }
  }

  step(dtMs: number, input: InputFrame): void {
    if (this.state.gameOver || this.state.gameWon) return;

    this.state.timeMs += Math.max(0, dtMs);
    this.updateSoftDropState(input.softDrop);
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

        if (!this.tryMoveDown(input.softDrop)) {
          // grounded: stop consuming extra "gravity" this tick
          this.gravityAcc = 0;
          break;
        }

        if (this.tryButterfingerGravityDrop()) {
          this.doHardDrop();
          return;
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
        if (this.tryButterfingerNudge()) {
          this.recomputeGhost();
          if (
            !collides(
              this.state.board,
              this.state.active,
              this.state.active.r,
              0,
              1,
            )
          ) {
            this.lockAcc = 0;
            this.hardLockAcc = 0;
            return;
          }
        }
        this.lockPiece();
      }
    } else {
      this.lockAcc = 0;
      this.hardLockAcc = 0;
    }
  }

  reset(seed: number): void {
    this.rng = new XorShift32(seed);
    this.butterfingerRng = new XorShift32(seed ^ 0x6d2b79f5);
    this.generator.reset(seed);

    this.state.board = makeBoard();
    this.state.hold = null;
    this.state.canHold = true;
    this.state.next = [];
    this.state.gameOver = false;
    this.state.gameWon = false;
    this.state.combo = 0;
    this.state.timeMs = 0;
    this.resetCheeseState();
    this.resetInitialBlocks();
    this.totalLinesCleared = 0;
    this.state.totalLinesCleared = 0;
    this.state.lineGoal = this.lineGoal;
    this.state.level = this.getClassicLevel(0);
    this.state.score = 0;
    this.state.scoringEnabled = this.scoringEnabled;
    this.softDropHeld = false;
    this.softDropSegmentCells = 0;
    this.softDropLastSegmentCells = 0;

    this.gravityAcc = 0;
    this.lockAcc = 0;
    this.hardLockAcc = 0;

    this.generator.onLock?.(this.state.board, null);
    const first = this.generator.next();
    this.updateNextView();
    this.spawnActive(first);
  }

  applyCheese(lines: number): void {
    this.resetInitialBlocks();
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

  markInitialBlocks(): void {
    this.initialBlocks = this.state.board.map((row) =>
      row.map((cell) => cell != null),
    );
    this.initialBlocksRemaining = this.initialBlocks.reduce(
      (sum, row) => sum + row.filter(Boolean).length,
      0,
    );
    this.initialBlocksActive = this.initialBlocksRemaining > 0;
    this.state.gameWon = false;
  }

  private resetInitialBlocks(): void {
    this.initialBlocks = Array.from({ length: ROWS }, () =>
      Array(COLS).fill(false),
    );
    this.initialBlocksRemaining = 0;
    this.initialBlocksActive = false;
  }

  private applyInitialBlockClears(clearedRows: number[]): void {
    if (!this.initialBlocksActive) return;

    for (const y of clearedRows) {
      const row = this.initialBlocks[y];
      for (const cell of row) {
        if (cell) this.initialBlocksRemaining -= 1;
      }
      this.initialBlocks.splice(y, 1);
      this.initialBlocks.unshift(Array(COLS).fill(false));
    }

    if (this.initialBlocksRemaining <= 0) {
      this.initialBlocksActive = false;
      this.state.gameWon = true;
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

  private tryMoveDown(countSoftDrop = false): boolean {
    if (
      !collides(this.state.board, this.state.active, this.state.active.r, 0, 1)
    ) {
      this.state.active.y += 1;
      if (countSoftDrop && this.softDropHeld) {
        this.softDropSegmentCells += 1;
      }
      this.lockAcc = 0;
      this.recomputeGhost();
      return true;
    }
    return false;
  }

  private doHardDrop(): void {
    this.tryButterfingerNudge();
    const d = dropDistance(this.state.board, this.state.active);
    this.state.active.y += d;
    this.lockPiece();
  }

  private doSoftDropInstant(): void {
    const d = dropDistance(this.state.board, this.state.active);
    if (d > 0) {
      this.state.active.y += d;
      if (this.softDropHeld) {
        this.softDropSegmentCells += d;
      }
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
    const prevLevel = this.state.level;
    if (cleared.length > 0) {
      this.applyCheeseClears(cleared);
      this.applyInitialBlockClears(cleared);
      this.state.combo += 1;
      this.onLineClear?.(this.state.combo, cleared.length);
    } else {
      this.state.combo = 0;
    }
    this.totalLinesCleared += cleared.length;
    this.state.totalLinesCleared = this.totalLinesCleared;
    if (this.classicStartLevel != null) {
      const nextLevel = this.getClassicLevel(this.totalLinesCleared);
      if (nextLevel !== prevLevel) {
        this.state.level = nextLevel;
        const prevGravity = this.gravityMs;
        this.gravityMs = getClassicGravityMs(nextLevel);
        if (this.dropIntervalMs === prevGravity) {
          this.adjustDropInterval(this.gravityMs);
        }
      }
    }
    if (this.scoringEnabled) {
      this.state.score += this.getLineClearScore(cleared.length);
    }
    this.consumeSoftDropScore();
    this.onPieceLock?.(this.state.board, this.state.hold);

    this.gravityAcc = 0;
    this.lockAcc = 0;
    this.state.canHold = true;
    this.hardLockAcc = 0;
    this.softDropHeld = false;
    this.softDropSegmentCells = 0;
    this.softDropLastSegmentCells = 0;

    if (hasAboveTop) {
      this.state.gameOver = true;
      return;
    }

    if (this.lineGoal != null && this.totalLinesCleared >= this.lineGoal) {
      this.state.gameWon = true;
      return;
    }

    if (this.cheeseActive && this.cheeseRemaining <= 0) {
      this.state.gameWon = true;
      return;
    }

    if (this.initialBlocksActive && this.initialBlocksRemaining <= 0) {
      this.state.gameWon = true;
      return;
    }

    if (this.state.gameWon) {
      return;
    }

    this.generator.onLock?.(this.state.board, this.state.hold);
    this.spawnNext();
  }

  private spawnNext(): void {
    const k = this.generator.next();
    this.updateNextView();
    this.spawnActive(k);
  }

  private updateSoftDropState(softDrop: boolean): void {
    if (softDrop && !this.softDropHeld) {
      this.softDropHeld = true;
      this.softDropSegmentCells = 0;
      return;
    }
    if (!softDrop && this.softDropHeld) {
      this.softDropHeld = false;
      this.softDropLastSegmentCells = this.softDropSegmentCells;
      this.softDropSegmentCells = 0;
    }
  }

  private consumeSoftDropScore(): number {
    const value = this.softDropHeld
      ? this.softDropSegmentCells
      : this.softDropLastSegmentCells;
    this.softDropSegmentCells = 0;
    this.softDropLastSegmentCells = 0;
    this.softDropHeld = false;
    return value;
  }

  private getClassicLevel(lines: number): number {
    if (this.classicStartLevel == null) return 0;
    const start = this.classicStartLevel;
    const startLines = start * 10;
    const transition = Math.min(
      startLines + 10,
      Math.max(100, startLines - 50),
    );
    if (lines < transition) return start;
    return start + 1 + Math.floor((lines - transition) / 10);
  }

  private getLineClearScore(linesCleared: number): number {
    if (!this.scoringEnabled) return 0;
    if (linesCleared <= 0) return 0;
    const multiplier = this.state.level + 1;
    switch (linesCleared) {
      case 1:
        return 40 * multiplier;
      case 2:
        return 100 * multiplier;
      case 3:
        return 300 * multiplier;
      case 4:
        return 1200 * multiplier;
      default:
        return 0;
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
      this.onHold?.(this.state.board, this.state.hold);
      return;
    }

    this.state.hold = current;
    this.gravityAcc = 0;
    this.lockAcc = 0;
    this.hardLockAcc = 0;

    this.spawnActive(held);
    this.onHold?.(this.state.board, this.state.hold);
  }

  private updateNextView(): void {
    this.state.next = this.generator.peek(NEXT_COUNT);
  }

  private spawnActive(k: PieceKind): void {
    this.state.active = { k, r: 0, x: SPAWN_X, y: SPAWN_Y };
    this.softDropHeld = false;
    this.softDropSegmentCells = 0;
    this.softDropLastSegmentCells = 0;
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

  private tryButterfingerNudge(): boolean {
    if (this.lockNudgeRate <= 0) return false;
    if (!this.rollButterfinger(this.lockNudgeRate)) return false;
    let changed = false;

    if (this.lockRotateRate > 0 && this.rollButterfinger(this.lockRotateRate)) {
      changed = this.tryButterfingerRotate();
      if (
        changed &&
        !collides(
          this.state.board,
          this.state.active,
          this.state.active.r,
          0,
          1,
        )
      ) {
        return true;
      }
    }

    const dir = this.butterfingerRng.nextInt(2) === 0 ? -1 : 1;
    const nudged = this.tryNudgeWithKicks(dir);
    return changed || nudged;
  }

  private tryButterfingerRotate(): boolean {
    const roll = this.butterfingerRng.nextInt(3);
    if (roll === 2) {
      const changed = tryRotate180PreferDirect(
        this.state.board,
        this.state.active,
      );
      if (changed) this.recomputeGhost();
      return changed;
    }
    const dir: -1 | 1 = roll === 0 ? 1 : -1;
    const changed = tryRotateSRS(this.state.board, this.state.active, dir);
    if (changed) this.recomputeGhost();
    return changed;
  }

  private tryButterfingerGravityDrop(): boolean {
    if (this.gravityDropRate <= 0) return false;
    return this.rollButterfinger(this.gravityDropRate);
  }

  private tryNudgeWithKicks(dir: -1 | 1): boolean {
    if (this.tryOffset(dir, 0)) return true;

    const from = this.state.active.r;
    const cw = ((from + 1) % 4) as Rotation;
    const ccw = ((from + 3) % 4) as Rotation;
    const kickTests = [
      ...getSrsKickTests(this.state.active.k, from, cw),
      ...getSrsKickTests(this.state.active.k, from, ccw),
    ];

    for (const [dx, dy] of kickTests) {
      if (this.tryOffset(dir + dx, dy)) {
        return true;
      }
    }

    return false;
  }

  private tryOffset(dx: number, dy: number): boolean {
    if (
      collides(this.state.board, this.state.active, this.state.active.r, dx, dy)
    ) {
      return false;
    }
    this.state.active.x += dx;
    this.state.active.y += dy;
    return true;
  }

  private rollButterfinger(rate: number): boolean {
    return this.butterfingerRng.nextU32() / 0xffffffff < rate;
  }
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

const NES_FPS = 60.0988;
const NES_FRAME_MS = 1000 / NES_FPS;

function getClassicFrames(level: number): number {
  const l = Math.max(0, Math.trunc(level));
  const table = [48, 43, 38, 33, 28, 23, 18, 13, 8, 6];
  if (l <= 9) return table[l];
  if (l <= 12) return 5;
  if (l <= 15) return 4;
  if (l <= 18) return 3;
  if (l <= 28) return 2;
  return 1;
}

function getClassicGravityMs(level: number): number {
  return getClassicFrames(level) * NES_FRAME_MS;
}
