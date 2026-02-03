import { describe, expect, it } from 'vitest';
import { Bag7 } from '../core/bag7';
import { Game } from '../core/game';
import { GameRunner, type InputSource } from '../core/runner';
import { dropDistance } from '../core/piece';
import type {
  GameState,
  InputFrame,
  PieceKind,
} from '../core/types';
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

  reset(_seed: number): void {
    // fixed generator has no state
  }
}

class CountingInput implements InputSource {
  count = 0;

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
