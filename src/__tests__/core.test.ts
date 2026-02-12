import { describe, expect, it } from 'vitest';
import { Bag7 } from '../core/bag7';
import { Game } from '../core/game';
import { GameRunner, type InputSource } from '../core/runner';
import { dropDistance } from '../core/piece';
import type { GameState, InputFrame, PieceKind } from '../core/types';
import type { PieceGenerator } from '../core/generator';
import { inferCurseDistribution } from '../core/curseInference';
import { softmax } from '../core/wubModel';

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

  it('timed soft drop increases gravity instead of hard dropping', () => {
    const game = new Game({
      seed: 2,
      gravityMs: 1000,
      softDropMs: 200,
      lockDelayMs: 500,
      generatorFactory: () => new FixedGenerator('O'),
    });

    const startY = game.state.active.y;
    const d = dropDistance(game.state.board, game.state.active);
    expect(d).toBeGreaterThan(1);

    game.step(199, { ...EMPTY_INPUT, softDrop: true });
    expect(game.state.active.y).toBe(startY);

    game.step(1, { ...EMPTY_INPUT, softDrop: true });
    expect(game.state.active.y).toBe(startY + 1);
  });

  it('soft drop cannot be slower than base gravity', () => {
    const game = new Game({
      seed: 3,
      gravityMs: 100,
      softDropMs: 400,
      lockDelayMs: 500,
      generatorFactory: () => new FixedGenerator('I'),
    });

    const startY = game.state.active.y;
    game.step(100, { ...EMPTY_INPUT, softDrop: true });

    expect(game.state.active.y).toBe(startY + 1);
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

describe('Curse inference', () => {
  it('flips softmax probabilities and renormalizes', () => {
    const logits = new Float32Array([4, 1, -2]);
    const base = softmax(logits);
    const cursed = inferCurseDistribution(logits);

    const cursedSum = cursed[0] + cursed[1] + cursed[2];
    expect(cursedSum).toBeCloseTo(1, 6);
    expect(cursed[0]).toBeLessThan(cursed[1]);
    expect(cursed[1]).toBeLessThan(cursed[2]);
    expect(base[0]).toBeGreaterThan(base[1]);
  });

  it('keeps uniform logits uniform after inversion', () => {
    const logits = new Float32Array([0, 0, 0, 0]);
    const cursed = inferCurseDistribution(logits);
    expect(Array.from(cursed)).toEqual([0.25, 0.25, 0.25, 0.25]);
  });
});
