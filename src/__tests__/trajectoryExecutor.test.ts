import { describe, expect, it } from 'vitest';
import { COLS, ROWS, SPAWN_X, SPAWN_Y } from '../core/constants';
import { dropDistance } from '../core/piece';
import type { ActivePiece, Board, PieceKind } from '../core/types';
import {
  planTrajectoryLockExecution,
  simulateTrajectoryExecutorCommands,
} from '../core/trajectoryExecutor';

const emptyBoard = (): Board =>
  Array.from({ length: ROWS }, () => Array.from({ length: COLS }, () => null));

const makeTarget = (
  board: Board,
  piece: ActivePiece,
  holdUsed = false,
): {
  lockPiece: PieceKind;
  lockRotation: number;
  lockX: number;
  lockY: number;
  holdUsed: boolean;
  gameTimeMs: number;
  totalLinesCleared: number;
  score: number;
} => ({
  lockPiece: piece.k,
  lockRotation: piece.r,
  lockX: piece.x,
  lockY: piece.y + dropDistance(board, piece),
  holdUsed,
  gameTimeMs: 0,
  totalLinesCleared: 0,
  score: 0,
});

describe('trajectory executor', () => {
  it('finds direct hard-drop path when already aligned', () => {
    const board = emptyBoard();
    const active: ActivePiece = {
      k: 'I',
      r: 0,
      x: SPAWN_X,
      y: SPAWN_Y,
    };
    const target = makeTarget(board, active, false);
    const planned = planTrajectoryLockExecution({
      board,
      active,
      hold: null,
      canHold: true,
      target,
    });
    expect(planned.ok).toBe(true);
    if (!planned.ok) return;
    expect(planned.commands).toEqual(['hard_drop']);
  });

  it('finds deterministic path for rotated placement', () => {
    const board = emptyBoard();
    const active: ActivePiece = {
      k: 'T',
      r: 0,
      x: SPAWN_X,
      y: SPAWN_Y,
    };
    const targetPiece: ActivePiece = {
      k: 'T',
      r: 1,
      x: SPAWN_X + 2,
      y: SPAWN_Y,
    };
    const target = makeTarget(board, targetPiece, false);
    const planned = planTrajectoryLockExecution({
      board,
      active,
      hold: null,
      canHold: true,
      target,
    });
    expect(planned.ok).toBe(true);
    if (!planned.ok) return;
    const simulation = simulateTrajectoryExecutorCommands({
      board,
      active,
      hold: null,
      canHold: true,
      commands: planned.commands,
    });
    expect(simulation.ok).toBe(true);
    expect(simulation.locked).toBe(true);
    expect(simulation.finalPiece.k).toBe(target.lockPiece);
    expect(simulation.finalPiece.r).toBe(target.lockRotation);
    expect(simulation.finalPiece.x).toBe(target.lockX);
    expect(simulation.finalPiece.y).toBe(target.lockY);
  });

  it('supports hold-first plan when hold piece matches target', () => {
    const board = emptyBoard();
    const active: ActivePiece = {
      k: 'I',
      r: 0,
      x: SPAWN_X,
      y: SPAWN_Y,
    };
    const hold: PieceKind = 'T';
    const holdSpawn: ActivePiece = {
      k: hold,
      r: 0,
      x: SPAWN_X,
      y: SPAWN_Y,
    };
    const target = makeTarget(board, holdSpawn, true);
    const planned = planTrajectoryLockExecution({
      board,
      active,
      hold,
      canHold: true,
      target,
    });
    expect(planned.ok).toBe(true);
    if (!planned.ok) return;
    expect(planned.commands[0]).toBe('hold');
    const simulation = simulateTrajectoryExecutorCommands({
      board,
      active,
      hold,
      canHold: true,
      commands: planned.commands,
    });
    expect(simulation.ok).toBe(true);
    expect(simulation.finalPiece.k).toBe('T');
    expect(simulation.finalPiece.y).toBe(target.lockY);
    expect(simulation.locked).toBe(true);
  });

  it('fails when hold is required but hold piece is unknown', () => {
    const board = emptyBoard();
    const active: ActivePiece = {
      k: 'I',
      r: 0,
      x: SPAWN_X,
      y: SPAWN_Y,
    };
    const target = makeTarget(
      board,
      { k: 'T', r: 0, x: SPAWN_X, y: SPAWN_Y },
      true,
    );
    const planned = planTrajectoryLockExecution({
      board,
      active,
      hold: null,
      canHold: true,
      target,
    });
    expect(planned.ok).toBe(false);
    if (planned.ok) return;
    expect(planned.reason.toLowerCase()).toContain('hold');
  });
});
