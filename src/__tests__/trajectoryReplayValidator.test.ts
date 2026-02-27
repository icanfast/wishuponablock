import { describe, expect, it } from 'vitest';
import { clearLines } from '../core/board';
import { COLS, ROWS, SPAWN_X, SPAWN_Y } from '../core/constants';
import { dropDistance, merge } from '../core/piece';
import {
  planTrajectoryLockExecution,
  simulateTrajectoryExecutorCommands,
} from '../core/trajectoryExecutor';
import type {
  TrajectoryReplayStepV1,
  TrajectorySessionV1,
} from '../core/trajectoryProtocol';
import type { ActivePiece, Board, PieceKind } from '../core/types';
import { PIECES } from '../core/types';
import { validateTrajectoryReplaySessions } from '../app/trajectoryReplayValidator';

const emptyBoard = (): Board =>
  Array.from({ length: ROWS }, () => Array.from({ length: COLS }, () => null));

const cloneBoard = (board: Board): Board => board.map((row) => row.slice());

const boardToOccupancy = (board: Board): number[][] =>
  board.map((row) => row.map((cell) => (cell ? 1 : 0)));

const makeReplay = (
  board: Board,
  piece: ActivePiece,
  holdUsed = false,
  gameTimeMs = 0,
): TrajectoryReplayStepV1 => ({
  lockPiece: piece.k,
  lockRotation: piece.r,
  lockX: piece.x,
  lockY: piece.y + dropDistance(board, piece),
  holdUsed,
  gameTimeMs,
  totalLinesCleared: 0,
  score: 0,
});

const makeSample = (input: {
  id: string;
  createdAtMs: number;
  board: Board;
  action: PieceKind;
  hold: PieceKind | null;
  replay: TrajectoryReplayStepV1;
}): TrajectorySessionV1['samples'][number] => ({
  id: input.id,
  createdAtMs: input.createdAtMs,
  deliberationMs: 100,
  boardOccupancy: boardToOccupancy(input.board),
  hold: input.hold,
  action: input.action,
  actionIndex: Math.max(0, PIECES.indexOf(input.action)),
  pieces: [...PIECES],
  logits: Array.from({ length: PIECES.length }, () => 0),
  probabilities: Array.from({ length: PIECES.length }, () => 1 / PIECES.length),
  inferenceMs: 0,
  samplingMs: 0,
  totalDecisionMs: 0,
  reward: null,
  replay: input.replay,
});

const buildTwoStepSession = (): TrajectorySessionV1 => {
  const board0 = emptyBoard();
  const active0: ActivePiece = { k: 'I', r: 0, x: SPAWN_X, y: SPAWN_Y };
  const replay0 = makeReplay(board0, active0, false, 1000);
  const plan0 = planTrajectoryLockExecution({
    board: board0,
    active: active0,
    hold: null,
    canHold: true,
    target: replay0,
  });
  if (!plan0.ok) {
    throw new Error(`plan0 failed in test setup: ${plan0.reason}`);
  }
  const sim0 = simulateTrajectoryExecutorCommands({
    board: board0,
    active: active0,
    hold: null,
    canHold: true,
    commands: plan0.commands,
  });
  if (!sim0.ok || !sim0.locked) {
    throw new Error('sim0 failed in test setup');
  }
  const board1 = cloneBoard(board0);
  merge(board1, sim0.finalPiece);
  clearLines(board1);
  const sample0 = makeSample({
    id: 's0',
    createdAtMs: 1_000,
    board: board1,
    action: 'O',
    hold: null,
    replay: replay0,
  });

  const active1: ActivePiece = { k: 'O', r: 0, x: SPAWN_X + 1, y: SPAWN_Y };
  const replay1 = makeReplay(board1, active1, false, 2000);
  const plan1 = planTrajectoryLockExecution({
    board: board1,
    active: { k: 'O', r: 0, x: SPAWN_X, y: SPAWN_Y },
    hold: null,
    canHold: true,
    target: replay1,
  });
  if (!plan1.ok) {
    throw new Error(`plan1 failed in test setup: ${plan1.reason}`);
  }
  const sim1 = simulateTrajectoryExecutorCommands({
    board: board1,
    active: { k: 'O', r: 0, x: SPAWN_X, y: SPAWN_Y },
    hold: null,
    canHold: true,
    commands: plan1.commands,
  });
  if (!sim1.ok || !sim1.locked) {
    throw new Error('sim1 failed in test setup');
  }
  const board2 = cloneBoard(board1);
  merge(board2, sim1.finalPiece);
  clearLines(board2);
  const sample1 = makeSample({
    id: 's1',
    createdAtMs: 2_000,
    board: board2,
    action: 'T',
    hold: null,
    replay: replay1,
  });

  return {
    schema: 'wishuponablock.trajectory_session.v1',
    sessionId: 'session_replay_validator',
    modeId: 'practice',
    buildVersion: '0.3.0-dev',
    startedAtMs: 1_000,
    endedAtMs: 3_000,
    durationMs: 2_000,
    initialState: {
      boardOccupancy: boardToOccupancy(board0),
      hold: null,
      active: { ...active0 },
      next: ['O', 'T', 'S'],
      canHold: true,
      timeMs: 0,
      totalLinesCleared: 0,
      score: 0,
    },
    samples: [sample0, sample1],
    meta: null,
  };
};

describe('trajectory replay validator', () => {
  it('validates a deterministic two-step session with full parity', () => {
    const session = buildTwoStepSession();
    const summary = validateTrajectoryReplaySessions([session]);
    expect(summary.totalSessions).toBe(1);
    expect(summary.replaySteps).toBe(2);
    expect(summary.validatedSteps).toBe(2);
    expect(summary.passedSteps).toBe(2);
    expect(summary.successRate).toBe(1);
    expect(summary.issues.length).toBe(0);
    expect(summary.byModeBuild).toHaveLength(1);
    expect(summary.byModeBuild[0].modeId).toBe('practice');
    expect(summary.byModeBuild[0].buildVersion).toBe('0.3.0-dev');
  });

  it('reports board mismatches when replay parity fails', () => {
    const session = buildTwoStepSession();
    session.samples[1].boardOccupancy = session.samples[0].boardOccupancy.map(
      (row) => row.slice(),
    );
    const summary = validateTrajectoryReplaySessions([session]);
    expect(summary.validatedSteps).toBe(2);
    expect(summary.passedSteps).toBe(1);
    expect(summary.failureCounts.board_mismatch).toBe(1);
    expect(
      summary.issues.some((issue) => issue.code === 'board_mismatch'),
    ).toBe(true);
  });
});
