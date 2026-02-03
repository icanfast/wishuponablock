This file is a merged representation of the entire codebase, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
.github/
  workflows/
    ci.yml
public/
  assets/
    bunny.png
    logo.svg
  favicon.png
  style.css
src/
  bot/
    controller.ts
  core/
    bag7.ts
    board.ts
    constants.ts
    game.ts
    generator.ts
    piece.ts
    rng.ts
    runner.ts
    settings.ts
    settingsStore.ts
    srs.ts
    tetromino.ts
    types.ts
  input/
    controller.ts
    keyboard.ts
    keyboardInputSource.ts
  render/
    pixiRenderer.ts
  main.ts
  vite-env.d.ts
.eslintrc.cjs
.gitignore
.prettierrc
eslint.config.mjs
index.html
package.json
tsconfig.json
vite.config.ts
```

# Files

## File: src/bot/controller.ts
```typescript
import type { InputSource } from '../core/runner';

export interface BotController extends InputSource {}
```

## File: src/core/bag7.ts
```typescript
import { PIECES, type PieceKind } from './types';
import { XorShift32, shuffleInPlace } from './rng';
import type { PieceGenerator } from './generator';

export class Bag7 implements PieceGenerator {
  private rng: XorShift32;
  private q: PieceKind[] = [];

  constructor(seed: number) {
    this.rng = new XorShift32(seed);
    this.refill();
  }

  reset(seed: number): void {
    this.rng = new XorShift32(seed);
    this.q = [];
    this.refill();
  }

  next(): PieceKind {
    if (this.q.length === 0) this.refill();
    return this.q.shift()!;
  }

  ensure(n: number): void {
    while (this.q.length < n) this.refill();
  }

  peek(n: number): PieceKind[] {
    this.ensure(n);
    return this.q.slice(0, n);
  }

  private refill(): void {
    const bag = [...PIECES] as PieceKind[];
    shuffleInPlace(bag, this.rng);
    this.q.push(...bag);
  }
}
```

## File: src/core/board.ts
```typescript
import { COLS, ROWS } from './constants';
import type { Board } from './types';

export function makeBoard(): Board {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(null));
}

export function clearLines(board: Board): number {
  let cleared = 0;
  for (let y = ROWS - 1; y >= 0; y--) {
    if (board[y].every((c) => c != null)) {
      board.splice(y, 1);
      board.unshift(Array(COLS).fill(null));
      cleared++;
      y++;
    }
  }
  return cleared;
}
```

## File: src/core/constants.ts
```typescript
export const COLS = 10;
export const ROWS = 20;

export const DEFAULT_GRAVITY_MS = 800;
export const DEFAULT_SOFT_DROP_MS = 0;
export const DEFAULT_LOCK_DELAY_MS = 500;
export const DEFAULT_DAS_MS = 130;
export const DEFAULT_ARR_MS = 0;

export const SETTINGS_STORAGE_KEY = 'wishuponablock.settings';

export const SPAWN_X = 3;
export const SPAWN_Y = -1;
export const NEXT_COUNT = 5;
```

## File: src/core/game.ts
```typescript
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
```

## File: src/core/generator.ts
```typescript
import type { PieceKind } from './types';

export interface PieceGenerator {
  next(): PieceKind;
  peek(n: number): PieceKind[];
  reset(seed: number): void;
}
```

## File: src/core/piece.ts
```typescript
import { COLS, ROWS } from './constants';
import { TETROMINOES, rotAdd } from './tetromino';
import { getSrsKickTests } from './srs';
import type { ActivePiece, Board, Rotation, Vec2 } from './types';

export function cellsOf(
  piece: ActivePiece,
  r: Rotation = piece.r,
  dx = 0,
  dy = 0,
): Vec2[] {
  const shape = TETROMINOES[piece.k][r];
  return shape.map(([x, y]) => [piece.x + x + dx, piece.y + y + dy]);
}

export function collides(
  board: Board,
  piece: ActivePiece,
  r: Rotation = piece.r,
  dx = 0,
  dy = 0,
): boolean {
  for (const [cx, cy] of cellsOf(piece, r, dx, dy)) {
    if (cx < 0 || cx >= COLS || cy >= ROWS) return true;
    if (cy >= 0 && board[cy][cx] != null) return true;
  }
  return false;
}

export function merge(board: Board, piece: ActivePiece): void {
  for (const [cx, cy] of cellsOf(piece)) {
    if (cy >= 0) board[cy][cx] = piece.k;
  }
}

export function dropDistance(board: Board, piece: ActivePiece): number {
  let d = 0;
  while (!collides(board, piece, piece.r, 0, d + 1)) d++;
  return d;
}

export function tryRotateSRS(
  board: Board,
  piece: ActivePiece,
  dir: -1 | 1,
): boolean {
  const from = piece.r;
  const to = rotAdd(from, dir);

  // SRS defines a list of translation tests depending on (from -> to) and piece type. :contentReference[oaicite:5]{index=5}
  const tests = getSrsKickTests(piece.k, from, to);

  for (const [dx, dy] of tests) {
    if (!collides(board, piece, to, dx, dy)) {
      piece.r = to;
      piece.x += dx;
      piece.y += dy;
      return true;
    }
  }
  return false;
}

export function tryRotate180PreferDirect(
  board: Board,
  piece: ActivePiece,
): boolean {
  const to = ((piece.r + 2) % 4) as Rotation;

  // 1) Direct 180, no kicks, no intermediate state
  if (!collides(board, piece, to, 0, 0)) {
    piece.r = to;
    return true;
  }

  // 2) Fallback: two 90-degree SRS rotations (with kicks)
  return tryRotate180ViaSRS(board, piece);
}

/**
 * This implementation tries CW+CW, then CCW+CCW, using SRS kicks for each 90° step.
 */
export function tryRotate180ViaSRS(board: Board, piece: ActivePiece): boolean {
  const attempt = (dir: -1 | 1): boolean => {
    const tmp: ActivePiece = { ...piece };
    if (!tryRotateSRS(board, tmp, dir)) return false;
    if (!tryRotateSRS(board, tmp, dir)) return false;
    piece.x = tmp.x;
    piece.y = tmp.y;
    piece.r = tmp.r;
    return true;
  };

  return attempt(1) || attempt(-1);
}
```

## File: src/core/rng.ts
```typescript
export class XorShift32 {
  private s: number;

  constructor(seed: number) {
    // avoid zero state
    this.s = seed | 0 || 0x12345678;
  }

  nextU32(): number {
    // xorshift32
    let x = this.s | 0;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.s = x | 0;
    return this.s >>> 0;
  }

  nextInt(maxExclusive: number): number {
    return this.nextU32() % maxExclusive;
  }
}

export function shuffleInPlace<T>(arr: T[], rng: XorShift32): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = rng.nextInt(i + 1);
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}
```

## File: src/core/runner.ts
```typescript
import type { Game } from './game';
import type { GameState, InputFrame } from './types';

export interface InputSource {
  sample(state: GameState, dtMs: number): InputFrame;
  reset?(seed: number): void;
}

export interface GameRunnerOptions {
  fixedStepMs: number;
  onRestart?: (game: Game) => void;
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

  tick(elapsedMs: number, input: InputSource = NullInputSource): void {
    this.accMs += elapsedMs;

    while (this.accMs >= this.options.fixedStepMs) {
      this.step(input);
      this.accMs -= this.options.fixedStepMs;
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
```

## File: src/core/settings.ts
```typescript
import {
  DEFAULT_ARR_MS,
  DEFAULT_DAS_MS,
  DEFAULT_GRAVITY_MS,
  DEFAULT_LOCK_DELAY_MS,
  DEFAULT_SOFT_DROP_MS,
  SETTINGS_STORAGE_KEY,
} from './constants';
import type { GameConfig } from './game';
import type { InputConfig } from '../input/controller';

export type GameSettings = Required<
  Pick<GameConfig, 'gravityMs' | 'softDropMs' | 'lockDelayMs'>
>;

export interface Settings {
  game: GameSettings;
  input: InputConfig;
}

export const DEFAULT_SETTINGS: Settings = {
  game: {
    gravityMs: DEFAULT_GRAVITY_MS,
    softDropMs: DEFAULT_SOFT_DROP_MS,
    lockDelayMs: DEFAULT_LOCK_DELAY_MS,
  },
  input: {
    dasMs: DEFAULT_DAS_MS,
    arrMs: DEFAULT_ARR_MS,
  },
};

function num(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

function mergeGame(
  base: GameSettings,
  patch?: Partial<GameSettings>,
): GameSettings {
  return {
    gravityMs: num(patch?.gravityMs) ?? base.gravityMs,
    softDropMs: num(patch?.softDropMs) ?? base.softDropMs,
    lockDelayMs: num(patch?.lockDelayMs) ?? base.lockDelayMs,
  };
}

function mergeInput(
  base: InputConfig,
  patch?: Partial<InputConfig>,
): InputConfig {
  return {
    dasMs: num(patch?.dasMs) ?? base.dasMs,
    arrMs: num(patch?.arrMs) ?? base.arrMs,
  };
}

export function mergeSettings(
  base: Settings,
  patch: Partial<Settings>,
): Settings {
  return {
    game: mergeGame(base.game, patch.game),
    input: mergeInput(base.input, patch.input),
  };
}

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw) as Partial<Settings>;
    return mergeSettings(DEFAULT_SETTINGS, parsed);
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(settings: Settings): void {
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Non-fatal; ignore persistence errors.
  }
}
```

## File: src/core/settingsStore.ts
```typescript
import {
  loadSettings,
  mergeSettings,
  saveSettings,
  type Settings,
} from './settings';

export interface SettingsStore {
  get(): Settings;
  apply(patch: Partial<Settings>): Settings;
  subscribe(listener: (settings: Settings) => void): () => void;
}

export function createSettingsStore(
  initial?: Settings,
): SettingsStore {
  let settings = initial ?? loadSettings();
  const listeners = new Set<(s: Settings) => void>();

  const notify = () => {
    for (const listener of listeners) listener(settings);
  };

  return {
    get() {
      return settings;
    },
    apply(patch) {
      settings = mergeSettings(settings, patch);
      saveSettings(settings);
      notify();
      return settings;
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
```

## File: src/core/srs.ts
```typescript
import type { PieceKind, Rotation, Vec2 } from './types';

// SRS basic-rotation kick data (y is "up" here).
// We convert to our game coords (y down) by flipping dy.

type KickTable = Record<
  Rotation,
  Partial<Record<Rotation, readonly Vec2[]>>
>;

const JLSTZ_KICKS: KickTable = {
  0: {
    1: [
      [0, 0],
      [-1, 0],
      [-1, 1],
      [0, -2],
      [-1, -2],
    ],
    3: [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, -2],
      [1, -2],
    ],
  },
  1: {
    0: [
      [0, 0],
      [1, 0],
      [1, -1],
      [0, 2],
      [1, 2],
    ],
    2: [
      [0, 0],
      [1, 0],
      [1, -1],
      [0, 2],
      [1, 2],
    ],
  },
  2: {
    1: [
      [0, 0],
      [-1, 0],
      [-1, 1],
      [0, -2],
      [-1, -2],
    ],
    3: [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, -2],
      [1, -2],
    ],
  },
  3: {
    2: [
      [0, 0],
      [-1, 0],
      [-1, -1],
      [0, 2],
      [-1, 2],
    ],
    0: [
      [0, 0],
      [-1, 0],
      [-1, -1],
      [0, 2],
      [-1, 2],
    ],
  },
};

const I_KICKS: KickTable = {
  0: {
    1: [
      [0, 0],
      [-2, 0],
      [1, 0],
      [-2, -1],
      [1, 2],
    ],
    3: [
      [0, 0],
      [-1, 0],
      [2, 0],
      [-1, 2],
      [2, -1],
    ],
  },
  1: {
    0: [
      [0, 0],
      [2, 0],
      [-1, 0],
      [2, 1],
      [-1, -2],
    ],
    2: [
      [0, 0],
      [-1, 0],
      [2, 0],
      [-1, 2],
      [2, -1],
    ],
  },
  2: {
    1: [
      [0, 0],
      [1, 0],
      [-2, 0],
      [1, -2],
      [-2, 1],
    ],
    3: [
      [0, 0],
      [2, 0],
      [-1, 0],
      [2, 1],
      [-1, -2],
    ],
  },
  3: {
    2: [
      [0, 0],
      [-2, 0],
      [1, 0],
      [-2, -1],
      [1, 2],
    ],
    0: [
      [0, 0],
      [1, 0],
      [-2, 0],
      [1, -2],
      [-2, 1],
    ],
  },
};

function srsToGame(dx: number, dyUp: number): Vec2 {
  // Our y grows down, SRS y grows up → flip sign.
  return [dx, -dyUp];
}

/**
 * Returns the list of translation tests (dx,dy) to apply *after* basic rotation,
 * in our game coordinate system (y down).
 *
 * For JLSTZ + I: 5 tests. For O: no kicks.
 */
export function getSrsKickTests(
  kind: PieceKind,
  from: Rotation,
  to: Rotation,
): Vec2[] {
  if (kind === 'O') return [[0, 0]];

  const table = kind === 'I' ? I_KICKS : JLSTZ_KICKS;
  const tests = table[from]?.[to];
  if (!tests) return [];

  return tests.map(([x, y]) => srsToGame(x, y));
}
```

## File: src/core/tetromino.ts
```typescript
import type { PieceKind, Rotation, Vec2 } from './types';

export type CellOffset = Vec2;

// Offsets inside a 4x4 box; piece position (x,y) is top-left of the 4x4.
export const TETROMINOES: Record<
  PieceKind,
  readonly (readonly CellOffset[])[]
> = {
/* eslint-disable prettier/prettier */
  I: [
    [[0, 1], [1, 1], [2, 1], [3, 1]],
    [[2, 0], [2, 1], [2, 2], [2, 3]],
    [[0, 2], [1, 2], [2, 2], [3, 2]],
    [[1, 0], [1, 1], [1, 2], [1, 3]],
  ],
  O: [
    [[1, 1], [2, 1], [1, 2], [2, 2]],
    [[1, 1], [2, 1], [1, 2], [2, 2]],
    [[1, 1], [2, 1], [1, 2], [2, 2]],
    [[1, 1], [2, 1], [1, 2], [2, 2]],
  ],
  T: [
    [[1, 1], [0, 2], [1, 2], [2, 2]],
    [[1, 1], [1, 2], [2, 2], [1, 3]],
    [[0, 2], [1, 2], [2, 2], [1, 3]],
    [[1, 1], [0, 2], [1, 2], [1, 3]],
  ],
  S: [
    [[1, 1], [2, 1], [0, 2], [1, 2]],
    [[1, 1], [1, 2], [2, 2], [2, 3]],
    [[1, 2], [2, 2], [0, 3], [1, 3]],
    [[0, 1], [0, 2], [1, 2], [1, 3]],
  ],
  Z: [
    [[0, 1], [1, 1], [1, 2], [2, 2]],
    [[2, 1], [1, 2], [2, 2], [1, 3]],
    [[0, 2], [1, 2], [1, 3], [2, 3]],
    [[1, 1], [0, 2], [1, 2], [0, 3]],
  ],
  J: [
    [[0, 1], [0, 2], [1, 2], [2, 2]],
    [[1, 1], [2, 1], [1, 2], [1, 3]],
    [[0, 2], [1, 2], [2, 2], [2, 3]],
    [[1, 1], [1, 2], [0, 3], [1, 3]],
  ],
  L: [
    [[2, 1], [0, 2], [1, 2], [2, 2]],
    [[1, 1], [1, 2], [1, 3], [2, 3]],
    [[0, 2], [1, 2], [2, 2], [0, 3]],
    [[0, 1], [1, 1], [1, 2], [1, 3]],
  ],
} as const;
/* eslint-enable prettier/prettier */

export function rotAdd(r: Rotation, delta: -1 | 1): Rotation {
  const v = (r + (delta === 1 ? 1 : 3)) % 4;
  return v as Rotation;
}
```

## File: src/core/types.ts
```typescript
export const PIECES = ['I', 'O', 'T', 'S', 'Z', 'J', 'L'] as const;
export type PieceKind = (typeof PIECES)[number];

export type Rotation = 0 | 1 | 2 | 3;
export type Vec2 = readonly [number, number];

export type Cell = PieceKind | null;
export type Board = Cell[][];

export interface ActivePiece {
  k: PieceKind;
  r: Rotation;
  x: number;
  y: number; // can be negative while spawning
}

export interface GameState {
  board: Board;
  active: ActivePiece;
  ghostY: number;
  hold: PieceKind | null;
  canHold: boolean;
  next: PieceKind[]; // preview window (derived from generator)
  gameOver: boolean;
}

export interface InputFrame {
  /**
   * Signed number of horizontal steps to attempt this frame.
   * Can be +/-Infinity to indicate "instant ARR" to the wall.
   */
  moveX: number;
  rotate: -1 | 0 | 1; // -1 = CCW, +1 = CW
  rotate180: boolean;
  softDrop: boolean;
  hardDrop: boolean;
  hold: boolean;
  restart: boolean;
}
```

## File: src/input/controller.ts
```typescript
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
```

## File: src/input/keyboard.ts
```typescript
export class Keyboard {
  private held = new Set<string>();
  private pressed = new Set<string>(); // “went down since last consume”

  constructor() {
    window.addEventListener('keydown', (e) => {
      const code = e.code;
      if (!this.held.has(code)) {
        this.pressed.add(code);
      }
      this.held.add(code);

      if (
        ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(
          code,
        )
      ) {
        e.preventDefault();
      }
    });

    window.addEventListener('keyup', (e) => {
      this.held.delete(e.code);
    });

    window.addEventListener('blur', () => {
      this.held.clear();
      this.pressed.clear();
    });
  }

  isHeld(code: string): boolean {
    return this.held.has(code);
  }

  consumePressed(code: string): boolean {
    if (this.pressed.has(code)) {
      this.pressed.delete(code);
      return true;
    }
    return false;
  }
}
```

## File: src/input/keyboardInputSource.ts
```typescript
import type { InputSource } from '../core/runner';
import type { GameState, InputFrame } from '../core/types';
import { InputController } from './controller';

export class KeyboardInputSource implements InputSource {
  constructor(private controller: InputController) {}

  sample(_state: GameState, dtMs: number): InputFrame {
    return this.controller.sample(dtMs);
  }
}
```

## File: src/render/pixiRenderer.ts
```typescript
import { Graphics } from 'pixi.js';
import { COLS, ROWS } from '../core/constants';
import { cellsOf } from '../core/piece';
import type { GameState, PieceKind } from '../core/types';

const COLORS: Record<PieceKind, number> = {
  I: 0x4dd3ff,
  O: 0xffd84d,
  T: 0xc77dff,
  S: 0x6eea6e,
  Z: 0xff6b6b,
  J: 0x4d7cff,
  L: 0xffa94d,
};

export class PixiRenderer {
  constructor(
    private gfx: Graphics,
    private cell = 28,
    private boardX = 40,
    private boardY = 40,
  ) {}

  render(state: GameState): void {
    const { gfx, cell, boardX, boardY } = this;

    gfx.clear();

    // background + frame
    gfx
      .rect(boardX - 2, boardY - 2, COLS * cell + 4, ROWS * cell + 4)
      .fill(0x121a24);
    gfx.rect(boardX, boardY, COLS * cell, ROWS * cell).fill(0x0b0f14);

    // settled blocks
    for (let y = 0; y < ROWS; y++) {
      for (let x = 0; x < COLS; x++) {
        const k = state.board[y][x];
        if (!k) continue;
        drawCell(gfx, boardX, boardY, cell, x, y, COLORS[k]);
      }
    }

    if (state.gameOver) return;

    // ghost
    const ghostColor = dim(COLORS[state.active.k], 0.35);
    const ghostPiece = { ...state.active, y: state.ghostY };
    for (const [x, y] of cellsOf(ghostPiece)) {
      if (y < 0) continue;
      drawCell(gfx, boardX, boardY, cell, x, y, ghostColor);
    }

    // active
    const color = COLORS[state.active.k];
    for (const [x, y] of cellsOf(state.active)) {
      if (y < 0) continue;
      drawCell(gfx, boardX, boardY, cell, x, y, color);
    }
  }
}

function drawCell(
  gfx: Graphics,
  boardX: number,
  boardY: number,
  cell: number,
  x: number,
  y: number,
  color: number,
): void {
  const px = boardX + x * cell;
  const py = boardY + y * cell;
  const pad = 2;
  gfx.rect(px + pad, py + pad, cell - pad * 2, cell - pad * 2).fill(color);
}

function dim(color: number, factor: number): number {
  const r = ((color >> 16) & 0xff) * factor;
  const g = ((color >> 8) & 0xff) * factor;
  const b = (color & 0xff) * factor;
  return ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
}
```

## File: .github/workflows/ci.yml
```yaml
name: ci

on:
  push:
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - run: npm run lint
      - run: npm run build
```

## File: public/assets/logo.svg
```xml
<svg width="735" height="289" viewBox="0 0 735 289" fill="none" xmlns="http://www.w3.org/2000/svg">
<g clip-path="url(#clip0_7_84)">
<path d="M304.17 288.01V89.99H244.97C244.77 89.99 244.56 89.99 244.34 90C233.7 90.37 225.15 99.12 225.05 109.76C225.05 109.82 225.05 109.88 225.05 109.93V268.4C225.05 268.73 225.07 269.1 225.1 269.51C226 279.71 234.54 287.53 244.77 287.97C245.2 287.99 245.61 288 245.99 288H304.17V288.01Z" fill="#E72264"/>
<path d="M548.21 287.91C559.24 287.08 567.97 278 568.05 266.93C568.05 266.86 568.05 266.79 568.05 266.72V110.16C568.05 110.1 568.05 110.03 568.05 109.97C568 99.03 559.1 90.11 548.16 90C548.08 90 548 90 547.92 90H488.93V288.02H545.74C546.51 288.02 547.34 287.98 548.21 287.92V287.91Z" fill="#E72264"/>
<path d="M293.07 90H355.07C361.15 90 369.75 94.92 374.29 101L505.76 277C510.3 283.08 509.05 288 502.98 288H440.98C434.9 288 426.3 283.08 421.76 277L290.28 101C285.74 94.93 286.99 90 293.06 90H293.07Z" fill="#E72264"/>
<path d="M498.07 90H436.07C429.99 90 421.39 94.92 416.85 101L285.38 277C280.84 283.08 282.09 288 288.16 288H350.16C356.24 288 364.84 283.08 369.38 277L500.85 101C505.39 94.93 504.14 90 498.07 90Z" fill="#E72264"/>
<path d="M529.07 0C550.05 0 567.07 17.01 567.07 38C567.07 58.99 550.06 76 529.07 76C508.08 76 491.07 58.99 491.07 38C491.07 17.01 508.08 0 529.07 0Z" fill="#E72264"/>
<path d="M265.07 0C286.05 0 303.07 17.01 303.07 38C303.07 58.99 286.06 76 265.07 76C244.08 76 227.07 58.99 227.07 38C227.07 17.01 244.08 0 265.07 0Z" fill="#E72264"/>
<path d="M695.07 0H635.07C613.53 0 596.07 17.46 596.07 39V99C596.07 120.54 613.53 138 635.07 138H695.07C716.61 138 734.07 120.54 734.07 99V39C734.07 17.46 716.61 0 695.07 0ZM655.44 39.34V79.84C655.44 85.3 654.35 90.04 652.17 94.05C649.99 98.06 646.88 101.17 642.84 103.38C638.8 105.59 634.01 106.7 628.48 106.7C623.78 106.7 619.72 105.85 616.29 104.16C612.86 102.47 609.98 100.1 607.63 97.06C607.63 97.06 601.08 89.99 607.08 85C613.21 79.9 618.52 86.58 618.52 86.58C619.76 88.31 621.23 89.61 622.93 90.47C624.62 91.34 626.54 91.77 628.69 91.77C630.69 91.77 632.46 91.36 633.98 90.53C635.5 89.7 636.71 88.46 637.61 86.8C638.51 85.14 638.96 83.1 638.96 80.68V39.35C638.96 39.35 638.8 31.01 647.08 31.01C655.36 31.01 655.45 39.35 655.45 39.35L655.44 39.34ZM714.35 98.38C709.82 102.86 703.15 105.56 694.35 106.48C688.3 107.12 683.02 106.7 678.52 105.23C674.02 103.76 669.81 101.21 665.88 97.59C665.88 97.59 659.8 92.35 664.7 85.77C668.38 80.82 675.3 85.96 675.3 85.96C677.92 88.4 680.73 90.2 683.74 91.38C686.75 92.56 690.14 92.95 693.93 92.55C697.3 92.2 699.86 91.28 701.62 89.81C703.38 88.34 704.14 86.5 703.91 84.3C703.71 82.38 702.94 80.86 701.61 79.75C700.28 78.64 698.53 77.75 696.37 77.07C694.21 76.39 691.89 75.79 689.4 75.25C686.91 74.71 684.39 74.02 681.83 73.18C679.27 72.34 676.9 71.23 674.7 69.86C672.5 68.49 670.66 66.64 669.16 64.29C667.66 61.94 666.72 58.95 666.34 55.31C665.86 50.77 666.54 46.77 668.36 43.31C670.19 39.85 672.97 37.07 676.71 34.97C680.45 32.87 684.8 31.56 689.75 31.04C694.91 30.5 699.68 30.9 704.06 32.24C708.44 33.59 712.17 35.63 715.24 38.36C715.24 38.36 719.54 41.97 716.05 48.18C713.34 53.01 705.73 50.11 705.73 50.11C703.29 48.07 700.89 46.62 698.55 45.76C696.2 44.9 693.65 44.61 690.9 44.9C688.08 45.2 685.9 45.97 684.36 47.2C682.82 48.44 682.16 50.06 682.37 52.05C682.56 53.84 683.33 55.23 684.69 56.24C686.05 57.24 687.78 58.05 689.9 58.66C692.01 59.27 694.34 59.86 696.86 60.43C699.39 61 701.93 61.69 704.48 62.5C707.03 63.31 709.4 64.47 711.57 65.98C713.74 67.49 715.6 69.45 717.14 71.86C718.68 74.27 719.65 77.4 720.06 81.25C720.79 88.2 718.89 93.91 714.35 98.38Z" fill="#E72264"/>
<path d="M107.17 0.340088C47.98 0.340088 0 48.3201 0 107.51C0 108.67 0.02 109.82 0.06 110.97V213.96C0.06 213.96 0.06 213.97 0.06 213.98V224.45C0.06 224.45 0.06 224.45 0.07 224.44V250.33C0.07 250.53 0.06 250.72 0.06 250.92C0.06 251.12 0.07 251.32 0.08 251.52V252H0.09C0.66 271.89 16.95 287.84 36.98 287.84C57.01 287.84 73.47 271.89 74.04 252V213.98H104.66C105.49 214 106.33 214.01 107.16 214.01C166.35 214.01 214.33 166.7 214.33 107.51C214.33 48.3201 166.36 0.340088 107.17 0.340088ZM107.17 140.01C100.62 140.01 84.02 140.01 77.13 140.01H74.07V107.52C74.07 89.3101 88.96 74.5401 107.17 74.5401C125.38 74.5401 140.15 89.3101 140.15 107.52C140.15 125.73 125.38 140.02 107.17 140.02V140.01Z" fill="#E72264"/>
<path opacity="0.15" d="M36.05 214H73.99V252.01H73.82C73.25 271.9 56.96 287.85 36.93 287.85C16.9 287.85 0.6 271.9 0.03 252.01H0.02V251.53C0.02 251.33 0 251.13 0 250.93C0 250.73 0.01 250.54 0.01 250.34V247.66C1.15 229.87 17.94 214 36.04 214H36.05Z" fill="#1D1D1B"/>
<g opacity="0.05">
<path opacity="0.5" d="M0.0599976 168.28V213.98C0.0599976 173.72 33.89 140.58 74.05 140.03V125.45C44.2 128.59 17.77 144.66 0.0599976 168.28Z" fill="black"/>
<path d="M0.119995 248.37C8.84 241.97 19.42 238.2 30.84 238.2H74.06V213.98H37.03C17.47 213.98 1.48 229.15 0.119995 248.37Z" fill="black"/>
</g>
<g opacity="0.05">
<path opacity="0.5" d="M74.05 140.03V126.79C43.95 129.74 17.39 146.16 0.0599976 170.27V213.98C0.0599976 173.72 33.89 140.58 74.05 140.03Z" fill="black"/>
<path d="M37.03 213.98C18 213.98 2.34001 228.35 0.26001 246.83C8.97001 240.03 19.73 236.01 31.4 236.01H74.05V213.99H37.02L37.03 213.98Z" fill="black"/>
</g>
<g opacity="0.05">
<path opacity="0.5" d="M74.05 140.03V128.12C43.68 130.87 16.95 147.69 0.0599976 172.34V213.97C0.0599976 173.71 33.89 140.57 74.05 140.02V140.03Z" fill="black"/>
<path d="M37.03 213.98C18.53 213.98 3.22002 227.55 0.460022 245.28C9.12002 238.09 20.06 233.8 31.96 233.8H74.05V213.98H37.02H37.03Z" fill="black"/>
</g>
<g opacity="0.05">
<path opacity="0.5" d="M74.05 140.03V129.46C43.37 132 16.46 149.28 0.0599976 174.53V213.98C0.0599976 173.72 33.89 140.58 74.05 140.03Z" fill="black"/>
<path d="M37.03 213.98C19.06 213.98 4.08998 226.79 0.72998 243.77C9.31998 236.18 20.41 231.6 32.53 231.6H74.06V213.98H37.03Z" fill="black"/>
</g>
<g opacity="0.05">
<path opacity="0.5" d="M74.05 140.03V130.81C43.01 133.11 15.89 150.93 0.0599976 176.84V213.98C0.0599976 173.72 33.89 140.58 74.05 140.03Z" fill="black"/>
<path d="M37.03 213.98C19.6 213.98 5.00001 226.03 1.07001 242.24C9.54001 234.26 20.77 229.39 33.09 229.39H74.05V213.97H37.02L37.03 213.98Z" fill="black"/>
</g>
<g opacity="0.05">
<path opacity="0.5" d="M74.05 140.03V132.16C42.6 134.22 15.22 152.65 0.0599976 179.32V213.98C0.0599976 173.72 33.89 140.58 74.05 140.03Z" fill="black"/>
<path d="M37.03 213.98C20.15 213.98 5.92998 225.28 1.47998 240.71C9.79998 232.34 21.14 227.19 33.66 227.19H74.06V213.98H37.03Z" fill="black"/>
</g>
<g opacity="0.05">
<path opacity="0.5" d="M74.05 140.03V133.52C42.12 135.31 14.43 154.48 0.0599976 182.01V213.98C0.0599976 173.72 33.89 140.58 74.05 140.03Z" fill="black"/>
<path d="M37.03 213.98C20.72 213.98 6.88001 224.54 1.95001 239.19C10.08 230.44 21.53 224.99 34.21 224.99H74.05V213.98H37.02H37.03Z" fill="black"/>
</g>
<g opacity="0.05">
<path opacity="0.5" d="M74.05 140.03V134.89C41.53 136.39 13.47 156.46 0.0599976 185V213.98C0.0599976 173.72 33.89 140.58 74.05 140.03Z" fill="black"/>
<path d="M37.03 213.98C21.28 213.98 7.83999 223.83 2.48999 237.7C10.4 228.56 21.92 222.79 34.78 222.79H74.06V213.98H37.03Z" fill="black"/>
</g>
<g opacity="0.05">
<path opacity="0.5" d="M74.05 140.03V136.29C40.78 137.48 12.24 158.68 0.0599976 188.47V213.98C0.0599976 173.72 33.89 140.58 74.05 140.03Z" fill="black"/>
<path d="M37.03 213.98C21.85 213.98 8.80999 223.13 3.10999 236.21C10.74 226.68 22.34 220.59 35.35 220.59H74.06V213.98H37.03Z" fill="black"/>
</g>
</g>
<defs>
<clipPath id="clip0_7_84">
<rect width="734.07" height="288.01" fill="white"/>
</clipPath>
</defs>
</svg>
```

## File: public/style.css
```css
body {
  margin: 0;
  padding: 0;
  color: rgba(255, 255, 255, 0.87);
  background-color: #000000;
}

#app {
  width: 100%;
  height: 100vh;
  overflow: hidden;
  display: flex;
  justify-content: center;
  align-items: center;
}
```

## File: src/main.ts
```typescript
import { Application, Graphics } from 'pixi.js';
import { Game } from './core/game';
import { GameRunner } from './core/runner';
import type { Settings } from './core/settings';
import { createSettingsStore } from './core/settingsStore';
import { Keyboard } from './input/keyboard';
import { InputController } from './input/controller';
import { KeyboardInputSource } from './input/keyboardInputSource';
import { PixiRenderer } from './render/pixiRenderer';

function hasWebGL(): boolean {
  const c = document.createElement('canvas');
  return !!(c.getContext('webgl2') || c.getContext('webgl'));
}

async function boot() {
  if (!hasWebGL()) {
    document.body.innerHTML = `<div style="padding:16px;color:#fff;background:#000;height:100vh">
      WebGL is disabled/unavailable. Enable hardware acceleration.
    </div>`;
    return;
  }

  const app = new Application();
  await app.init({
    width: 420,
    height: 680,
    backgroundColor: 0x0b0f14,
    preference: 'webgl',
    powerPreference: 'high-performance',
    antialias: false,
  });

  const root = document.getElementById('app') ?? document.body;
  root.innerHTML = '';
  Object.assign(document.body.style, { margin: '0', overflow: 'hidden' });
  Object.assign(root.style, {
    position: 'fixed',
    inset: '0',
    overflow: 'hidden',
  });
  root.appendChild(app.canvas);

  const gfx = new Graphics();
  app.stage.addChild(gfx);

  const settingsStore = createSettingsStore();
  const settings = settingsStore.get();

  const kb = new Keyboard();
  const input = new InputController(kb, settings.input);
  const inputSource = new KeyboardInputSource(input);
  const game = new Game({ seed: Date.now(), ...settings.game });
  const runner = new GameRunner(game, {
    fixedStepMs: 1000 / 120,
    onRestart: (g) => g.reset(Date.now()),
  });

  const renderer = new PixiRenderer(gfx);

  settingsStore.subscribe((next) => {
    input.setConfig(next.input);
    game.setConfig(next.game);
  });

  const applySettings = (patch: Partial<Settings>): void => {
    settingsStore.apply(patch);
  };

  // TODO: Wire applySettings to UI when settings controls are added.
  void applySettings;

  app.ticker.add((t) => {
    runner.tick(t.elapsedMS, inputSource);
    renderer.render(runner.state);
  });
}

boot().catch((e) => console.error(e));
```

## File: src/vite-env.d.ts
```typescript
/// <reference types="vite/client" />
```

## File: .eslintrc.cjs
```javascript
module.exports = {
  root: true,
  env: { browser: true, es2022: true },
  parser: '@typescript-eslint/parser',
  plugins: ['@typescript-eslint'],
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'prettier',
  ],
};
```

## File: .prettierrc
```
{
  "singleQuote": true,
  "semi": true,
  "trailingComma": "all"
}
```

## File: eslint.config.mjs
```javascript
import js from '@eslint/js';
import prettier from 'eslint-plugin-prettier/recommended';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist'] },
  {
    extends: [
      js.configs.recommended,
      ...tseslint.configs.recommended,
      prettier,
    ],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
    },
    rules: {},
  },
);
```

## File: index.html
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/png" href="/favicon.png" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="stylesheet" href="/style.css" />
    <title>PixiJS - Template</title>
  </head>

  <body>
    <div id="app">
      <div id="pixi-container"></div>
    </div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

## File: package.json
```json
{
  "name": "wishuponablock",
  "version": "0.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "start": "npm run dev",
    "build": "npm run lint && tsc && vite build",
    "lint": "eslint .",
    "dev": "vite",
    "format": "prettier . --write"
  },
  "dependencies": {
    "pixi.js": "^8.8.1"
  },
  "devDependencies": {
    "@eslint/js": "^9.21.0",
    "@typescript-eslint/eslint-plugin": "^8.54.0",
    "@typescript-eslint/parser": "^8.54.0",
    "eslint": "^9.39.2",
    "eslint-config-prettier": "^10.1.8",
    "eslint-plugin-prettier": "^5.2.6",
    "prettier": "^3.8.1",
    "typescript": "~5.7.3",
    "typescript-eslint": "^8.25.0",
    "vite": "^6.2.0"
  }
}
```

## File: tsconfig.json
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "module": "ESNext",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "skipLibCheck": true,

    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,

    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedSideEffectImports": true
  },
  "include": ["src"]
}
```

## File: vite.config.ts
```typescript
import { defineConfig } from 'vite';

// https://vite.dev/config/
export default defineConfig({
  server: {
    port: 8080,
    open: true,
  },
});
```

## File: .gitignore
```
# Logs
logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*
lerna-debug.log*

node_modules
dist
dist-ssr
*.local

# Editor directories and files
.vscode/*
!.vscode/extensions.json
.idea
.DS_Store
*.suo
*.ntvs*
*.njsproj
*.sln
*.sw?
```
