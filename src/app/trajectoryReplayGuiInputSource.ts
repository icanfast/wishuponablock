import { SPAWN_X, SPAWN_Y } from '../core/constants';
import { collides } from '../core/piece';
import {
  planTrajectoryLockExecution,
  trajectoryExecutorCommandToInputFrame,
} from '../core/trajectoryExecutor';
import type {
  TrajectoryReplayStepV1,
  TrajectorySessionV1,
} from '../core/trajectoryProtocol';
import type { InputSource } from '../core/runner';
import type { GameState, InputFrame, PieceKind } from '../core/types';

const EMPTY_INPUT: InputFrame = {
  moveX: 0,
  rotate: 0,
  rotate180: false,
  softDrop: false,
  hardDrop: false,
  hold: false,
  restart: false,
};

type ReplayEntry = {
  index: number;
  sample: TrajectorySessionV1['samples'][number];
};

export type TrajectoryReplayGuiRunStats = {
  totalReplaySteps: number;
  plannedSteps: number;
  passedBoardChecks: number;
  failedBoardChecks: number;
  failedPlans: number;
  skippedSteps: number;
  commandsEmitted: number;
};

export type TrajectoryReplayGuiInputSourceConfig = {
  session: TrajectorySessionV1;
  apmInput: number;
  allowSoftDrop?: boolean;
  maxNodes?: number;
  onLog?: (line: string) => void;
  onComplete?: (stats: TrajectoryReplayGuiRunStats) => void;
};

const boardToOccupancy = (board: GameState['board']): number[][] =>
  board.map((row) => row.map((cell) => (cell ? 1 : 0)));

const sameOccupancy = (left: number[][], right: number[][]): boolean => {
  if (left.length !== right.length) return false;
  for (let y = 0; y < left.length; y += 1) {
    const lRow = left[y];
    const rRow = right[y];
    if (lRow.length !== rRow.length) return false;
    for (let x = 0; x < lRow.length; x += 1) {
      const l = lRow[x] > 0 ? 1 : 0;
      const r = rRow[x] > 0 ? 1 : 0;
      if (l !== r) return false;
    }
  }
  return true;
};

const summarizeBoard = (board: GameState['board']): string => {
  let occupied = 0;
  const heights = new Array<number>(board[0]?.length ?? 0).fill(0);
  for (let y = 0; y < board.length; y += 1) {
    for (let x = 0; x < board[y].length; x += 1) {
      if (!board[y][x]) continue;
      occupied += 1;
      if (heights[x] === 0) {
        heights[x] = board.length - y;
      }
    }
  }
  const heightsSummary =
    heights.length > 0 ? `h=[${heights.join(',')}]` : 'h=[]';
  return `occupied=${occupied}, ${heightsSummary}`;
};

const normalizeRotation = (value: number): 0 | 1 | 2 | 3 => {
  const normalized = Math.trunc(value) % 4;
  if (normalized === 1) return 1;
  if (normalized === 2) return 2;
  if (normalized === 3 || normalized === -1) return 3;
  return 0;
};

const coerceActiveForTarget = (
  state: GameState,
  lockPiece: PieceKind,
  log: (line: string) => void,
): void => {
  if (state.active.k === lockPiece) return;
  const forced = {
    k: lockPiece,
    r: 0 as const,
    x: SPAWN_X,
    y: SPAWN_Y,
  };
  for (let i = 0; i < 4; i += 1) {
    if (!collides(state.board, forced)) break;
    forced.y -= 1;
  }
  state.active = forced;
  log(`forced active piece to ${lockPiece} for replay alignment.`);
};

type PlanVariant = {
  label: string;
  target: TrajectoryReplayStepV1;
  maxNodesScale: number;
};

type PlanAttempt = {
  label: string;
  reason: string;
  visitedNodes: number;
};

type SuccessfulPlan = {
  label: string;
  result: Extract<ReturnType<typeof planTrajectoryLockExecution>, { ok: true }>;
};

const buildPlanVariants = (replay: TrajectoryReplayStepV1): PlanVariant[] => {
  if (replay.holdUsed) {
    return [
      { label: 'as_recorded', target: replay, maxNodesScale: 1 },
      {
        label: 'fallback_hold_used_false',
        target: { ...replay, holdUsed: false },
        maxNodesScale: 2,
      },
    ];
  }
  return [
    { label: 'as_recorded', target: replay, maxNodesScale: 1 },
    {
      label: 'fallback_hold_used_true',
      target: { ...replay, holdUsed: true },
      maxNodesScale: 2,
    },
  ];
};

export const createTrajectoryReplayGuiInputSource = (
  config: TrajectoryReplayGuiInputSourceConfig,
): InputSource => {
  const replayEntries: ReplayEntry[] = config.session.samples
    .map((sample, index) => ({ sample, index }))
    .filter((entry) => Boolean(entry.sample.replay));
  const clampApm = Math.min(1200, Math.max(20, Math.trunc(config.apmInput)));
  const actionIntervalMs = 60_000 / clampApm;
  const log = (line: string) => config.onLog?.(`[replay-exec] ${line}`);
  const maxNodes = Math.max(1, Math.trunc(config.maxNodes ?? 30_000));
  const allowSoftDrop = config.allowSoftDrop !== false;

  let activeRef: GameState['active'] | null = null;
  let queue: InputFrame[] = [];
  let cooldownMs = 0;
  let replayCursor = 0;
  let pendingCheck: ReplayEntry | null = null;
  let finished = false;
  const stats: TrajectoryReplayGuiRunStats = {
    totalReplaySteps: replayEntries.length,
    plannedSteps: 0,
    passedBoardChecks: 0,
    failedBoardChecks: 0,
    failedPlans: 0,
    skippedSteps: config.session.samples.length - replayEntries.length,
    commandsEmitted: 0,
  };

  const complete = (): void => {
    if (finished) return;
    finished = true;
    const summary =
      `complete: planned=${stats.plannedSteps}/${stats.totalReplaySteps}, ` +
      `board_checks_ok=${stats.passedBoardChecks}, board_checks_failed=${stats.failedBoardChecks}, ` +
      `plan_failed=${stats.failedPlans}, skipped=${stats.skippedSteps}, ` +
      `commands=${stats.commandsEmitted}.`;
    log(summary);
    config.onComplete?.(stats);
  };

  const planNextForActive = (state: GameState): void => {
    if (replayCursor >= replayEntries.length) {
      complete();
      return;
    }
    const entry = replayEntries[replayCursor];
    const replay = entry.sample.replay;
    if (!replay) {
      replayCursor += 1;
      planNextForActive(state);
      return;
    }

    if (pendingCheck) {
      const boardMatches = sameOccupancy(
        boardToOccupancy(state.board),
        pendingCheck.sample.boardOccupancy,
      );
      const holdMatches = state.hold === pendingCheck.sample.hold;
      if (boardMatches && holdMatches) {
        stats.passedBoardChecks += 1;
      } else {
        stats.failedBoardChecks += 1;
        log(
          `parity mismatch after sample #${pendingCheck.index}: ` +
            `board=${boardMatches ? 'ok' : 'mismatch'} hold=${holdMatches ? 'ok' : 'mismatch'}.`,
        );
      }
      pendingCheck = null;
    }

    if (!replay.holdUsed && state.active.k !== replay.lockPiece) {
      coerceActiveForTarget(state, replay.lockPiece, log);
    }
    state.active.r = normalizeRotation(state.active.r);
    const variants = buildPlanVariants(replay);
    const attempts: PlanAttempt[] = [];
    let planned: SuccessfulPlan | null = null;
    for (const variant of variants) {
      const result = planTrajectoryLockExecution({
        board: state.board,
        active: state.active,
        hold: state.hold,
        canHold: state.canHold,
        target: variant.target,
        maxNodes: Math.max(1, Math.trunc(maxNodes * variant.maxNodesScale)),
        allowSoftDrop,
      });
      if (result.ok) {
        planned = { label: variant.label, result };
        break;
      }
      attempts.push({
        label: variant.label,
        reason: result.reason,
        visitedNodes: result.visitedNodes,
      });
    }
    replayCursor += 1;
    if (!planned) {
      stats.failedPlans += 1;
      const attemptsSummary = attempts
        .map(
          (attempt) =>
            `${attempt.label}=>${attempt.reason} (visited=${attempt.visitedNodes})`,
        )
        .join(' | ');
      log(
        `plan failed for sample #${entry.index}. ` +
          `target={piece:${replay.lockPiece},rot:${replay.lockRotation},x:${replay.lockX},y:${replay.lockY},holdUsed:${replay.holdUsed}} ` +
          `state={active:${state.active.k}@${state.active.x},${state.active.y},r${state.active.r};hold:${state.hold ?? 'null'};canHold:${state.canHold}} ` +
          `board={${summarizeBoard(state.board)}} ` +
          `attempts=${attemptsSummary}`,
      );
      complete();
      return;
    }
    stats.plannedSteps += 1;
    queue = planned.result.commands.map((command) =>
      trajectoryExecutorCommandToInputFrame(command),
    );
    pendingCheck = entry;
    log(
      `planned sample #${entry.index}: commands=${queue.length}, ` +
        `depth=${planned.result.searchDepth}, visited=${planned.result.visitedNodes}, ` +
        `variant=${planned.label}.`,
    );
  };

  return {
    sample: (state, dtMs) => {
      if (finished) return EMPTY_INPUT;
      cooldownMs = Math.max(0, cooldownMs - Math.max(0, dtMs));
      if (state.active !== activeRef) {
        activeRef = state.active;
        queue = [];
        planNextForActive(state);
      }
      if (finished || queue.length === 0) {
        if (pendingCheck && (state.gameOver || state.gameWon)) {
          const boardMatches = sameOccupancy(
            boardToOccupancy(state.board),
            pendingCheck.sample.boardOccupancy,
          );
          const holdMatches = state.hold === pendingCheck.sample.hold;
          if (boardMatches && holdMatches) {
            stats.passedBoardChecks += 1;
          } else {
            stats.failedBoardChecks += 1;
          }
          pendingCheck = null;
          complete();
        }
        return EMPTY_INPUT;
      }
      if (cooldownMs > 0) return EMPTY_INPUT;
      cooldownMs = actionIntervalMs;
      const frame = queue.shift() ?? EMPTY_INPUT;
      stats.commandsEmitted += 1;
      return frame;
    },
    reset: () => {
      activeRef = null;
      queue = [];
      cooldownMs = 0;
      replayCursor = 0;
      pendingCheck = null;
      finished = false;
      stats.plannedSteps = 0;
      stats.passedBoardChecks = 0;
      stats.failedBoardChecks = 0;
      stats.failedPlans = 0;
      stats.commandsEmitted = 0;
    },
  };
};
