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
  sfx/
    lock.ogg
  favicon.png
  style.css
resources/
  lock.ogg
src/
  __tests__/
    core.test.ts
  bot/
    controller.ts
  core/
    bag7.ts
    board.ts
    constants.ts
    game.ts
    generator.ts
    generators.ts
    piece.ts
    randomGenerator.ts
    replay.ts
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
CHANGELOG.md
eslint.config.mjs
index.html
package.json
tsconfig.json
vite.config.ts
vitest.config.ts
```

# Files

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

## File: src/core/generator.ts
```typescript
import type { PieceKind } from './types';

export interface PieceGenerator {
  next(): PieceKind;
  peek(n: number): PieceKind[];
  reset(seed: number): void;
}
```

## File: src/core/generators.ts
```typescript
import type { PieceGenerator } from './generator';
import { Bag7 } from './bag7';
import { RandomGenerator } from './randomGenerator';

export const GENERATOR_TYPES = ['bag7', 'random'] as const;
export type GeneratorType = (typeof GENERATOR_TYPES)[number];

export interface GeneratorSettings {
  type: GeneratorType;
}

export function isGeneratorType(value: unknown): value is GeneratorType {
  return (
    typeof value === 'string' &&
    (GENERATOR_TYPES as readonly string[]).includes(value)
  );
}

export function createGeneratorFactory(
  settings: GeneratorSettings,
): (seed: number) => PieceGenerator {
  switch (settings.type) {
    case 'random':
      return (seed) => new RandomGenerator(seed);
    case 'bag7':
    default:
      return (seed) => new Bag7(seed);
  }
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

## File: src/core/randomGenerator.ts
```typescript
import { PIECES, type PieceKind } from './types';
import { XorShift32 } from './rng';
import type { PieceGenerator } from './generator';

export class RandomGenerator implements PieceGenerator {
  private rng: XorShift32;

  constructor(seed: number) {
    this.rng = new XorShift32(seed);
  }

  reset(seed: number): void {
    this.rng = new XorShift32(seed);
  }

  next(): PieceKind {
    const i = this.rng.nextInt(PIECES.length);
    return PIECES[i];
  }

  peek(n: number): PieceKind[] {
    const out: PieceKind[] = [];
    const rng = this.rng.clone();
    for (let i = 0; i < n; i++) {
      out.push(PIECES[rng.nextInt(PIECES.length)]);
    }
    return out;
  }
}
```

## File: src/core/replay.ts
```typescript
import type { InputFrame } from './types';
import type { Settings } from './settings';

export interface ReplayHeader {
  protocolVersion: number;
  buildVersion?: string;
  seed: number;
  generator: {
    type: string;
    params?: Record<string, unknown>;
  };
  settings: Settings;
  createdAt: string;
}

export interface Replay {
  header: ReplayHeader;
  inputs: InputFrame[];
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

## File: CHANGELOG.md
```markdown
# Changelog

## 0.1.0
- Initial internal prototype.
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

## File: vitest.config.ts
```typescript
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/__tests__/**/*.test.ts'],
  },
});
```

## File: src/__tests__/core.test.ts
```typescript
import { describe, expect, it } from 'vitest';
import { Bag7 } from '../core/bag7';
import { Game } from '../core/game';
import { GameRunner, type InputSource } from '../core/runner';
import { dropDistance } from '../core/piece';
import type { GameState, InputFrame, PieceKind } from '../core/types';
import type { PieceGenerator } from '../core/generator';

const EMPTY_INPUT: InputFrame = {
  moveX: 0,
  rotate: 0,
  rotate180: false,
  softDrop: false,
  hardDrop: false,
  hold: false,
  restart: false,
};

class FixedGenerator implements PieceGenerator {
  private queue: PieceKind[];

  constructor(kind: PieceKind) {
    this.queue = [kind];
  }

  next(): PieceKind {
    return this.queue[0];
  }

  peek(n: number): PieceKind[] {
    return Array.from({ length: n }, () => this.queue[0]);
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  reset(_seed: number): void {
    // fixed generator has no state
  }
}

class CountingInput implements InputSource {
  count = 0;

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  sample(_state: GameState, _dtMs: number): InputFrame {
    this.count++;
    return EMPTY_INPUT;
  }
}

describe('Bag7', () => {
  it('is deterministic for a given seed', () => {
    const a = new Bag7(123456);
    const b = new Bag7(123456);

    const take = (bag: Bag7, n: number): PieceKind[] =>
      Array.from({ length: n }, () => bag.next());

    expect(take(a, 14)).toEqual(take(b, 14));
  });

  it('peek does not consume pieces', () => {
    const bag = new Bag7(42);
    const preview = bag.peek(7);
    const actual = Array.from({ length: 7 }, () => bag.next());
    expect(actual).toEqual(preview);
  });
});

describe('Game', () => {
  it('instant soft drop moves to floor without locking', () => {
    const game = new Game({
      seed: 1,
      gravityMs: 1000,
      softDropMs: 0,
      lockDelayMs: 500,
      generatorFactory: () => new FixedGenerator('O'),
    });

    const startY = game.state.active.y;
    const d = dropDistance(game.state.board, game.state.active);
    expect(d).toBeGreaterThan(0);

    game.step(16, { ...EMPTY_INPUT, softDrop: true });

    expect(game.state.active.y).toBe(startY + d);
    const anyLocked = game.state.board.some((row) =>
      row.some((cell) => cell != null),
    );
    expect(anyLocked).toBe(false);
  });
});

describe('GameRunner', () => {
  it('caps steps per tick', () => {
    const game = new Game({
      seed: 1,
      generatorFactory: () => new FixedGenerator('T'),
    });
    const runner = new GameRunner(game, {
      fixedStepMs: 16,
      maxStepsPerTick: 2,
    });

    const input = new CountingInput();
    runner.tick(1000, input);

    expect(input.count).toBe(2);
  });
});
```

## File: src/bot/controller.ts
```typescript
import type { InputSource } from '../core/runner';

export type BotController = InputSource;
```

## File: src/core/constants.ts
```typescript
export const COLS = 10;
export const ROWS = 20;
export const NEXT_COUNT = 5;

// Layout
export const OUTER_MARGIN = 40;
export const BOARD_CELL_PX = 28;
export const PANEL_GAP = 16;

export const QUEUE_COLS = 6;
export const QUEUE_PREVIEW_ROWS = 4;
export const QUEUE_GAP_ROWS = 1;

export const HOLD_COLS = 6;
export const HOLD_ROWS = 4;
export const HOLD_LABEL_HEIGHT = BOARD_CELL_PX;

export const SETTINGS_PANEL_WIDTH = BOARD_CELL_PX * 5;

export const BOARD_WIDTH = COLS * BOARD_CELL_PX;
export const BOARD_HEIGHT = ROWS * BOARD_CELL_PX;

export const QUEUE_WIDTH = QUEUE_COLS * BOARD_CELL_PX;
export const QUEUE_PREVIEW_HEIGHT =
  QUEUE_PREVIEW_ROWS * BOARD_CELL_PX;
export const QUEUE_GAP_PX = QUEUE_GAP_ROWS * BOARD_CELL_PX;
export const QUEUE_PANEL_HEIGHT =
  NEXT_COUNT * QUEUE_PREVIEW_HEIGHT +
  (NEXT_COUNT - 1) * QUEUE_GAP_PX;

export const HOLD_WIDTH = HOLD_COLS * BOARD_CELL_PX;
export const HOLD_HEIGHT = HOLD_ROWS * BOARD_CELL_PX;
export const HOLD_PANEL_HEIGHT = HOLD_LABEL_HEIGHT + HOLD_HEIGHT;

export const HOLD_X = OUTER_MARGIN;
export const HOLD_Y = OUTER_MARGIN;
export const HOLD_INNER_Y = HOLD_Y + HOLD_LABEL_HEIGHT;
export const GAME_OVER_Y = HOLD_Y + HOLD_PANEL_HEIGHT + PANEL_GAP;

export const BOARD_X = HOLD_X + HOLD_WIDTH + PANEL_GAP;
export const BOARD_Y = OUTER_MARGIN;

export const QUEUE_X = BOARD_X + BOARD_WIDTH + PANEL_GAP;
export const QUEUE_Y = BOARD_Y;

export const SETTINGS_X = QUEUE_X + QUEUE_WIDTH + PANEL_GAP;
export const SETTINGS_Y = BOARD_Y;

export const PLAY_WIDTH = SETTINGS_X + SETTINGS_PANEL_WIDTH + OUTER_MARGIN;
export const PLAY_HEIGHT =
  OUTER_MARGIN +
  Math.max(BOARD_HEIGHT, QUEUE_PANEL_HEIGHT, HOLD_PANEL_HEIGHT) +
  OUTER_MARGIN;

export const DEFAULT_GRAVITY_MS = 800;
export const DEFAULT_SOFT_DROP_MS = 0;
export const DEFAULT_LOCK_DELAY_MS = 500;
export const DEFAULT_HARD_LOCK_DELAY_MS = 2000;
export const DEFAULT_DAS_MS = 130;
export const DEFAULT_ARR_MS = 0;
export const DEFAULT_MASTER_VOLUME = 1;

export const SETTINGS_STORAGE_KEY = 'wishuponablock.settings';

export const GAME_PROTOCOL_VERSION = 1;

export const SPAWN_X = 3;
export const SPAWN_Y = -1;
```

## File: src/core/game.ts
```typescript
import {
  DEFAULT_GRAVITY_MS,
  DEFAULT_HARD_LOCK_DELAY_MS,
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
  cellsOf,
  dropDistance,
  merge,
  tryRotateSRS,
  tryRotate180PreferDirect,
} from './piece';

import type { GameState, InputFrame, PieceKind } from './types';

export interface GameConfig {
  seed: number;
  gravityMs?: number;
  softDropMs?: number;
  lockDelayMs?: number;
  hardLockDelayMs?: number;
  generatorFactory?: (seed: number) => PieceGenerator;
  onPieceLock?: () => void;
}

export class Game {
  readonly state: GameState;

  private generator: PieceGenerator;

  private gravityAcc = 0;
  private dropIntervalMs: number;
  private lockAcc = 0;
  private hardLockAcc = 0;

  private gravityMs: number;
  private softDropMs: number;
  private lockDelayMs: number;
  private hardLockDelayMs: number;
  private onPieceLock?: () => void;

  constructor(cfg: GameConfig) {
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
    };

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
    this.generator.reset(seed);

    this.state.board = makeBoard();
    this.state.hold = null;
    this.state.canHold = true;
    this.state.next = [];
    this.state.gameOver = false;

    this.gravityAcc = 0;
    this.lockAcc = 0;
    this.hardLockAcc = 0;

    const first = this.generator.next();
    this.updateNextView();
    this.spawnActive(first);
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
    this.onPieceLock?.();
    const hasAboveTop = cellsOf(this.state.active).some(([, y]) => y < 0);
    merge(this.state.board, this.state.active);
    clearLines(this.state.board);

    this.gravityAcc = 0;
    this.lockAcc = 0;
    this.state.canHold = true;
    this.hardLockAcc = 0;

    if (hasAboveTop) {
      this.state.gameOver = true;
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

  clone(): XorShift32 {
    const copy = new XorShift32(1);
    copy.s = this.s;
    return copy;
  }
}

export function shuffleInPlace<T>(arr: T[], rng: XorShift32): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = rng.nextInt(i + 1);
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}
```

## File: src/core/settings.ts
```typescript
import {
  DEFAULT_ARR_MS,
  DEFAULT_DAS_MS,
  DEFAULT_GRAVITY_MS,
  DEFAULT_HARD_LOCK_DELAY_MS,
  DEFAULT_LOCK_DELAY_MS,
  DEFAULT_MASTER_VOLUME,
  DEFAULT_SOFT_DROP_MS,
  SETTINGS_STORAGE_KEY,
} from './constants';
import {
  type GeneratorSettings,
  isGeneratorType,
} from './generators';
import type { GameConfig } from './game';
import type { InputConfig } from '../input/controller';

export type GameSettings = Required<
  Pick<
    GameConfig,
    'gravityMs' | 'softDropMs' | 'lockDelayMs' | 'hardLockDelayMs'
  >
>;

export interface Settings {
  game: GameSettings;
  input: InputConfig;
  generator: GeneratorSettings;
  audio: AudioSettings;
}

export interface AudioSettings {
  masterVolume: number;
}

export const DEFAULT_SETTINGS: Settings = {
  game: {
    gravityMs: DEFAULT_GRAVITY_MS,
    softDropMs: DEFAULT_SOFT_DROP_MS,
    lockDelayMs: DEFAULT_LOCK_DELAY_MS,
    hardLockDelayMs: DEFAULT_HARD_LOCK_DELAY_MS,
  },
  input: {
    dasMs: DEFAULT_DAS_MS,
    arrMs: DEFAULT_ARR_MS,
  },
  generator: {
    type: 'bag7',
  },
  audio: {
    masterVolume: DEFAULT_MASTER_VOLUME,
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
    hardLockDelayMs:
      num(patch?.hardLockDelayMs) ?? base.hardLockDelayMs,
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

function mergeGenerator(
  base: GeneratorSettings,
  patch?: Partial<GeneratorSettings>,
): GeneratorSettings {
  return {
    type: isGeneratorType(patch?.type) ? patch.type : base.type,
  };
}

function mergeAudio(
  base: AudioSettings,
  patch?: Partial<AudioSettings>,
): AudioSettings {
  return {
    masterVolume: num(patch?.masterVolume) ?? base.masterVolume,
  };
}

export function mergeSettings(
  base: Settings,
  patch: Partial<Settings>,
): Settings {
  return {
    game: mergeGame(base.game, patch.game),
    input: mergeInput(base.input, patch.input),
    generator: mergeGenerator(base.generator, patch.generator),
    audio: mergeAudio(base.audio, patch.audio),
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

export function createSettingsStore(initial?: Settings): SettingsStore {
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

type KickTable = Record<Rotation, Partial<Record<Rotation, readonly Vec2[]>>>;

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
              Math.floor((this.heldMs - this.nextRepeatAt) / this.cfg.arrMs);
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

## File: src/render/pixiRenderer.ts
```typescript
import { Graphics } from 'pixi.js';
import {
  BOARD_CELL_PX,
  BOARD_X,
  BOARD_Y,
  COLS,
  HOLD_COLS,
  HOLD_INNER_Y,
  HOLD_LABEL_HEIGHT,
  HOLD_PANEL_HEIGHT,
  HOLD_WIDTH,
  HOLD_X,
  HOLD_Y,
  NEXT_COUNT,
  QUEUE_COLS,
  QUEUE_GAP_PX,
  QUEUE_PREVIEW_HEIGHT,
  QUEUE_X,
  QUEUE_Y,
  ROWS,
} from '../core/constants';
import { cellsOf } from '../core/piece';
import { TETROMINOES } from '../core/tetromino';
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
    private cell = BOARD_CELL_PX,
    private boardX = BOARD_X,
    private boardY = BOARD_Y,
  ) {}

  render(state: GameState): void {
    const { gfx, cell, boardX, boardY } = this;

    gfx.clear();

    // board background + frame
    gfx
      .rect(boardX - 2, boardY - 2, COLS * cell + 4, ROWS * cell + 4)
      .fill(0x121a24);
    gfx.rect(boardX, boardY, COLS * cell, ROWS * cell).fill(0x0b0f14);

    this.renderHold(state);
    this.renderQueue(state);

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

  private renderQueue(state: GameState): void {
    const { gfx, cell } = this;
    const count = Math.min(state.next.length, NEXT_COUNT);
    if (count === 0) return;

    const panelHeight =
      count * QUEUE_PREVIEW_HEIGHT + (count - 1) * QUEUE_GAP_PX;
    const panelWidth = QUEUE_COLS * cell;

    gfx
      .rect(QUEUE_X - 2, QUEUE_Y - 2, panelWidth + 4, panelHeight + 4)
      .fill(0x121a24);
    gfx.rect(QUEUE_X, QUEUE_Y, panelWidth, panelHeight).fill(0x0b0f14);

    const offsetX = Math.floor((QUEUE_COLS - 4) / 2) * cell;

    for (let i = 0; i < count; i++) {
      const kind = state.next[i];
      const boxY = QUEUE_Y + i * (QUEUE_PREVIEW_HEIGHT + QUEUE_GAP_PX);
      this.drawPreviewPiece(kind, QUEUE_X + offsetX, boxY, cell);
    }
  }

  private renderHold(state: GameState): void {
    const { gfx, cell } = this;

    gfx
      .rect(HOLD_X - 2, HOLD_Y - 2, HOLD_WIDTH + 4, HOLD_PANEL_HEIGHT + 4)
      .fill(0x121a24);
    gfx.rect(HOLD_X, HOLD_Y, HOLD_WIDTH, HOLD_PANEL_HEIGHT).fill(0x0b0f14);
    gfx.rect(HOLD_X, HOLD_Y, HOLD_WIDTH, HOLD_LABEL_HEIGHT).fill(0x121a24);

    if (!state.hold) return;

    const offsetX = Math.floor((HOLD_COLS - 4) / 2) * cell;
    this.drawPreviewPiece(state.hold, HOLD_X + offsetX, HOLD_INNER_Y, cell);
  }

  private drawPreviewPiece(
    kind: PieceKind,
    originX: number,
    originY: number,
    cell: number,
  ): void {
    const color = COLORS[kind];
    const shape = TETROMINOES[kind][0];
    for (const [x, y] of shape) {
      drawCell(this.gfx, originX, originY, cell, x, y, color);
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

## File: package.json
```json
{
  "name": "wishuponablock",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "start": "npm run dev",
    "build": "npm run lint && tsc && vite build",
    "lint": "eslint .",
    "dev": "vite",
    "format": "prettier . --write",
    "test": "vitest run"
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
    "vite": "^6.2.0",
    "vitest": "^3.2.4"
  }
}
```

## File: src/main.ts
```typescript
import { Application, Graphics } from 'pixi.js';
import {
  GAME_OVER_Y,
  HOLD_X,
  HOLD_Y,
  HOLD_WIDTH,
  PLAY_HEIGHT,
  PLAY_WIDTH,
  SETTINGS_X,
  SETTINGS_Y,
  SETTINGS_PANEL_WIDTH,
} from './core/constants';
import { Game } from './core/game';
import {
  createGeneratorFactory,
  GENERATOR_TYPES,
  isGeneratorType,
  type GeneratorType,
} from './core/generators';
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
    width: PLAY_WIDTH,
    height: PLAY_HEIGHT,
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

  const playWindow = document.createElement('div');
  Object.assign(playWindow.style, {
    position: 'relative',
    width: `${PLAY_WIDTH}px`,
    height: `${PLAY_HEIGHT}px`,
  });
  root.appendChild(playWindow);
  playWindow.appendChild(app.canvas);

  const gfx = new Graphics();
  app.stage.addChild(gfx);

  const uiLayer = document.createElement('div');
  Object.assign(uiLayer.style, {
    position: 'absolute',
    inset: '0',
    pointerEvents: 'none',
  });
  playWindow.appendChild(uiLayer);

  const holdLabel = document.createElement('div');
  holdLabel.textContent = 'HOLD';
  Object.assign(holdLabel.style, {
    position: 'absolute',
    left: `${HOLD_X + 8}px`,
    top: `${HOLD_Y + 6}px`,
    width: `${HOLD_WIDTH}px`,
    color: '#b6c2d4',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    fontSize: '13px',
    letterSpacing: '0.5px',
    pointerEvents: 'none',
  });
  uiLayer.appendChild(holdLabel);

  const gameOverLabel = document.createElement('div');
  gameOverLabel.textContent = 'GAME OVER';
  Object.assign(gameOverLabel.style, {
    position: 'absolute',
    left: `${HOLD_X}px`,
    top: `${GAME_OVER_Y}px`,
    width: `${HOLD_WIDTH}px`,
    color: '#7f8a9a',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    fontSize: '13px',
    letterSpacing: '0.5px',
    textAlign: 'center',
    pointerEvents: 'none',
    display: 'none',
  });
  uiLayer.appendChild(gameOverLabel);

  const settingsStore = createSettingsStore();
  const settings = settingsStore.get();

  const kb = new Keyboard();
  const input = new InputController(kb, settings.input);
  const inputSource = new KeyboardInputSource(input);

  const lockSound = new Audio('/sfx/lock.ogg');
  lockSound.preload = 'auto';
  lockSound.volume = settings.audio.masterVolume;
  const playLockSound = () => {
    lockSound.currentTime = 0;
    void lockSound.play().catch(() => {
      // Ignore autoplay restrictions and playback errors.
    });
  };

  const createGame = (cfg: Settings): Game =>
    new Game({
      seed: Date.now(),
      ...cfg.game,
      generatorFactory: createGeneratorFactory(cfg.generator),
      onPieceLock: playLockSound,
    });

  const createRunner = (g: Game): GameRunner =>
    new GameRunner(g, {
      fixedStepMs: 1000 / 120,
      onRestart: () => g.reset(Date.now()),
      maxElapsedMs: 250,
      maxStepsPerTick: 10,
    });

  let game = createGame(settings);
  let runner = createRunner(game);

  const renderer = new PixiRenderer(gfx);

  const settingsPanel = document.createElement('div');
  Object.assign(settingsPanel.style, {
    position: 'absolute',
    left: `${SETTINGS_X}px`,
    top: `${SETTINGS_Y}px`,
    minWidth: `${SETTINGS_PANEL_WIDTH}px`,
    padding: '8px',
    background: '#121a24',
    color: '#e2e8f0',
    border: '2px solid #0b0f14',
    borderRadius: '6px',
    fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif',
    fontSize: '13px',
    pointerEvents: 'auto',
  });

  const label = document.createElement('div');
  label.textContent = 'Generator';
  Object.assign(label.style, {
    marginBottom: '6px',
    color: '#b6c2d4',
  });

  const select = document.createElement('select');
  Object.assign(select.style, {
    width: '100%',
    background: '#0b0f14',
    color: '#e2e8f0',
    border: '1px solid #1f2a37',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '13px',
  });

  const labelFor = (type: GeneratorType): string => {
    switch (type) {
      case 'bag7':
        return 'Bag 7';
      case 'random':
        return 'Random';
      default:
        return type;
    }
  };

  for (const type of GENERATOR_TYPES) {
    const option = document.createElement('option');
    option.value = type;
    option.textContent = labelFor(type);
    select.appendChild(option);
  }
  select.value = settings.generator.type;

  select.addEventListener('change', () => {
    const value = select.value;
    if (isGeneratorType(value)) {
      settingsStore.apply({ generator: { type: value } });
    }
  });

  settingsPanel.appendChild(label);
  settingsPanel.appendChild(select);

  const volumeLabel = document.createElement('div');
  volumeLabel.textContent = 'Master Volume';
  Object.assign(volumeLabel.style, {
    marginTop: '12px',
    marginBottom: '6px',
    color: '#b6c2d4',
  });

  const volumeRow = document.createElement('div');
  Object.assign(volumeRow.style, {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  });

  const volumeValue = document.createElement('span');
  Object.assign(volumeValue.style, {
    color: '#8fa0b8',
    fontSize: '12px',
    minWidth: '32px',
    textAlign: 'right',
  });

  const volumeSlider = document.createElement('input');
  volumeSlider.type = 'range';
  volumeSlider.min = '0';
  volumeSlider.max = '100';
  volumeSlider.step = '1';
  volumeSlider.value = String(Math.round(settings.audio.masterVolume * 100));
  Object.assign(volumeSlider.style, {
    flex: '1',
    accentColor: '#6ea8ff',
  });

  const updateVolumeLabel = (value: number) => {
    volumeValue.textContent = `${Math.round(value * 100)}%`;
  };
  updateVolumeLabel(settings.audio.masterVolume);

  volumeSlider.addEventListener('input', () => {
    const value = Math.max(0, Math.min(1, Number(volumeSlider.value) / 100));
    settingsStore.apply({ audio: { masterVolume: value } });
  });

  volumeRow.appendChild(volumeSlider);
  volumeRow.appendChild(volumeValue);
  settingsPanel.appendChild(volumeLabel);
  settingsPanel.appendChild(volumeRow);
  uiLayer.appendChild(settingsPanel);

  let generatorType = settings.generator.type;

  settingsStore.subscribe((next) => {
    input.setConfig(next.input);
    lockSound.volume = next.audio.masterVolume;

    if (next.generator.type !== generatorType) {
      generatorType = next.generator.type;
      game = createGame(next);
      runner = createRunner(game);
      if (select.value !== next.generator.type) {
        select.value = next.generator.type;
      }
      return;
    }

    game.setConfig(next.game);
    if (select.value !== next.generator.type) {
      select.value = next.generator.type;
    }
    const nextVolume = Math.round(next.audio.masterVolume * 100);
    if (Number(volumeSlider.value) !== nextVolume) {
      volumeSlider.value = String(nextVolume);
      updateVolumeLabel(next.audio.masterVolume);
    }
  });

  const applySettings = (patch: Partial<Settings>): void => {
    settingsStore.apply(patch);
  };

  // TODO: Wire applySettings to UI when settings controls are added.
  void applySettings;

  let paused = document.visibilityState !== 'visible';
  let resumePending = false;

  const setPaused = (next: boolean) => {
    if (next === paused) return;
    paused = next;
    if (!paused) resumePending = true;
  };

  document.addEventListener('visibilitychange', () => {
    setPaused(document.visibilityState !== 'visible');
  });
  window.addEventListener('blur', () => setPaused(true));
  window.addEventListener('focus', () => setPaused(false));

  const updateGameOverLabel = () => {
    gameOverLabel.style.display = runner.state.gameOver ? 'block' : 'none';
  };

  app.ticker.add((t) => {
    if (paused) {
      updateGameOverLabel();
      return;
    }
    if (resumePending) {
      runner.resetTiming();
      resumePending = false;
      updateGameOverLabel();
      return;
    }
    runner.tick(t.elapsedMS, inputSource);
    renderer.render(runner.state);
    updateGameOverLabel();
  });
}

boot().catch((e) => console.error(e));
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
