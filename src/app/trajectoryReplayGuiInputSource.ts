import { SPAWN_X, SPAWN_Y } from '../core/constants';
import { collides } from '../core/piece';
import {
  planTrajectoryLockExecution,
  trajectoryExecutorCommandToInputFrame,
} from '../core/trajectoryExecutor';
import type { TrajectoryExecutorCommand } from '../core/trajectoryExecutor';
import type {
  TrajectoryReplayStepV1,
  TrajectorySessionV1,
} from '../core/trajectoryProtocol';
import type { InputSource } from '../core/runner';
import type { GameState, InputFrame, PieceKind } from '../core/types';

export type ReplayExecutorTargetGhost = {
  sampleIndex: number;
  ghost: {
    k: PieceKind;
    r: 0 | 1 | 2 | 3;
    x: number;
    y: number;
  };
};

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
  executionMode?: 'apm' | 'step';
  allowSoftDrop?: boolean;
  maxNodes?: number;
  stopOnParityMismatch?: boolean;
  debugTrace?: boolean;
  onLog?: (line: string) => void;
  onTargetGhostChange?: (target: ReplayExecutorTargetGhost | null) => void;
  onComplete?: (stats: TrajectoryReplayGuiRunStats) => void;
};

export type TrajectoryReplayGuiInputSource = InputSource & {
  isStepMode: boolean;
  requestStep: () => void;
};

const boardToOccupancy = (board: GameState['board']): number[][] =>
  board.map((row) => row.map((cell) => (cell ? 1 : 0)));

const occupancyToBoard = (occupancy: number[][]): GameState['board'] =>
  occupancy.map((row) => row.map((cell) => (cell > 0 ? 'I' : null)));

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

const summarizeOccupancyDiff = (
  actual: number[][],
  expected: number[][],
): { diffCount: number; firstDiffs: Array<{ x: number; y: number }> } => {
  const firstDiffs: Array<{ x: number; y: number }> = [];
  let diffCount = 0;
  const rows = Math.max(actual.length, expected.length);
  for (let y = 0; y < rows; y += 1) {
    const aRow = actual[y] ?? [];
    const eRow = expected[y] ?? [];
    const cols = Math.max(aRow.length, eRow.length);
    for (let x = 0; x < cols; x += 1) {
      const a = (aRow[x] ?? 0) > 0 ? 1 : 0;
      const e = (eRow[x] ?? 0) > 0 ? 1 : 0;
      if (a === e) continue;
      diffCount += 1;
      if (firstDiffs.length < 12) {
        firstDiffs.push({ x, y });
      }
    }
  }
  return { diffCount, firstDiffs };
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

const commandToDebugString = (command: TrajectoryExecutorCommand): string => {
  switch (command) {
    case 'left':
      return 'L';
    case 'right':
      return 'R';
    case 'rotate_cw':
      return 'CW';
    case 'rotate_ccw':
      return 'CCW';
    case 'rotate_180':
      return 'R180';
    case 'soft_drop':
      return 'SD';
    case 'hard_drop':
      return 'HD';
    case 'hold':
      return 'HOLD';
    default:
      return String(command);
  }
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
  // Mutate in-place so active object identity stays stable for the
  // active-change detector; replacing the object here causes queue resets.
  state.active.k = forced.k;
  state.active.r = forced.r;
  state.active.x = forced.x;
  state.active.y = forced.y;
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
): TrajectoryReplayGuiInputSource => {
  const allEntries: ReplayEntry[] = config.session.samples.map(
    (sample, index) => ({ sample, index }),
  );
  const replayStepsTotal = allEntries.reduce(
    (count, entry) => count + (entry.sample.replay ? 1 : 0),
    0,
  );
  const clampApm = Math.min(1200, Math.max(20, Math.trunc(config.apmInput)));
  const actionIntervalMs = 60_000 / clampApm;
  const stepMode = config.executionMode === 'step';
  const log = (line: string) => config.onLog?.(`[replay-exec] ${line}`);
  const maxNodes = Math.max(1, Math.trunc(config.maxNodes ?? 30_000));
  const allowSoftDrop = config.allowSoftDrop !== false;
  const stopOnParityMismatch = config.stopOnParityMismatch !== false;
  const debugTrace = config.debugTrace !== false;

  let activeRef: GameState['active'] | null = null;
  let queue: InputFrame[] = [];
  let commandQueue: TrajectoryExecutorCommand[] = [];
  let cooldownMs = 0;
  let manualStepBudget = 0;
  let sampleCursor = 0;
  let pendingCheck: ReplayEntry | null = null;
  let finished = false;
  const stats: TrajectoryReplayGuiRunStats = {
    totalReplaySteps: replayStepsTotal,
    plannedSteps: 0,
    passedBoardChecks: 0,
    failedBoardChecks: 0,
    failedPlans: 0,
    skippedSteps: config.session.samples.length - replayStepsTotal,
    commandsEmitted: 0,
  };

  const complete = (): void => {
    if (finished) return;
    finished = true;
    config.onTargetGhostChange?.(null);
    const summary =
      `complete: planned=${stats.plannedSteps}/${stats.totalReplaySteps}, ` +
      `board_checks_ok=${stats.passedBoardChecks}, board_checks_failed=${stats.failedBoardChecks}, ` +
      `plan_failed=${stats.failedPlans}, skipped=${stats.skippedSteps}, ` +
      `commands=${stats.commandsEmitted}.`;
    log(summary);
    config.onComplete?.(stats);
  };

  const planNextForActive = (state: GameState): void => {
    if (pendingCheck) {
      const boardMatches = sameOccupancy(
        boardToOccupancy(state.board),
        pendingCheck.sample.boardOccupancy,
      );
      const holdMatches = state.hold === pendingCheck.sample.hold;
      let activeMatches = state.active.k === pendingCheck.sample.action;
      if (!activeMatches) {
        const before = state.active.k;
        coerceActiveForTarget(state, pendingCheck.sample.action, log);
        activeMatches = state.active.k === pendingCheck.sample.action;
        log(
          `active drift after sample #${pendingCheck.index}: ` +
            `actual_active=${before} expected_active=${pendingCheck.sample.action} ` +
            `-> corrected_active=${state.active.k}.`,
        );
      }
      if (boardMatches && holdMatches && activeMatches) {
        stats.passedBoardChecks += 1;
      } else {
        stats.failedBoardChecks += 1;
        const occupancyNow = boardToOccupancy(state.board);
        const occupancyDiff = summarizeOccupancyDiff(
          occupancyNow,
          pendingCheck.sample.boardOccupancy,
        );
        log(
          `parity mismatch after sample #${pendingCheck.index}: ` +
            `board=${boardMatches ? 'ok' : 'mismatch'} hold=${holdMatches ? 'ok' : 'mismatch'} active=${activeMatches ? 'ok' : 'mismatch'} ` +
            `diff_cells=${occupancyDiff.diffCount} ` +
            `first_diff=${JSON.stringify(occupancyDiff.firstDiffs)} ` +
            `actual_hold=${state.hold ?? 'null'} expected_hold=${pendingCheck.sample.hold ?? 'null'} ` +
            `actual_active=${state.active.k} expected_active=${pendingCheck.sample.action}.`,
        );
        if (stopOnParityMismatch) {
          log(
            `stopping on first parity mismatch (sample #${pendingCheck.index}) to avoid cascading divergence.`,
          );
          pendingCheck = null;
          complete();
          return;
        }
      }
      pendingCheck = null;
    }

    while (sampleCursor < allEntries.length) {
      const entry = allEntries[sampleCursor];
      const replay = entry.sample.replay;
      if (!replay) {
        sampleCursor += 1;
        state.board = occupancyToBoard(entry.sample.boardOccupancy);
        if (entry.sample.hold !== state.hold) {
          state.canHold = false;
        }
        state.hold = entry.sample.hold;
        coerceActiveForTarget(state, entry.sample.action, log);
        state.active.r = normalizeRotation(state.active.r);
        log(
          `applied non-replay checkpoint sample #${entry.index}: ` +
            `active=${state.active.k}, hold=${state.hold ?? 'null'}, canHold=${state.canHold}.`,
        );
        continue;
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
      sampleCursor += 1;
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
            `note=if holdUsed=true and hold is empty, runtime first-hold semantics may differ from static plan assumptions. ` +
            `attempts=${attemptsSummary}`,
        );
        complete();
        return;
      }
      stats.plannedSteps += 1;
      queue = planned.result.commands.map((command) =>
        trajectoryExecutorCommandToInputFrame(command),
      );
      commandQueue = [...planned.result.commands];
      config.onTargetGhostChange?.({
        sampleIndex: entry.index,
        ghost: {
          k: replay.lockPiece,
          r: normalizeRotation(replay.lockRotation),
          x: Math.trunc(replay.lockX),
          y: Math.trunc(replay.lockY),
        },
      });
      pendingCheck = entry;
      log(
        `planned sample #${entry.index}: commands=${queue.length}, ` +
          `depth=${planned.result.searchDepth}, visited=${planned.result.visitedNodes}, ` +
          `variant=${planned.label}.`,
      );
      if (debugTrace) {
        const commandTrace =
          planned.result.commands.map(commandToDebugString).join(' -> ') ||
          '(none)';
        log(
          `debug sample #${entry.index}: target={piece:${replay.lockPiece},rot:${replay.lockRotation},x:${replay.lockX},y:${replay.lockY},holdUsed:${replay.holdUsed}} ` +
            `plan=[${commandTrace}]`,
        );
      }
      return;
    }
    complete();
  };

  return {
    sample: (state, dtMs) => {
      if (finished) return EMPTY_INPUT;
      cooldownMs = Math.max(0, cooldownMs - Math.max(0, dtMs));
      if (state.active !== activeRef) {
        activeRef = state.active;
        queue = [];
        commandQueue = [];
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
      if (stepMode) {
        if (manualStepBudget <= 0) return EMPTY_INPUT;
        manualStepBudget -= 1;
      } else {
        if (cooldownMs > 0) return EMPTY_INPUT;
        cooldownMs = actionIntervalMs;
      }
      const frame = queue.shift() ?? EMPTY_INPUT;
      const command = commandQueue.shift() ?? null;
      if (debugTrace && command) {
        const targetSampleIndex = pendingCheck?.index ?? null;
        log(
          `debug emit sample #${targetSampleIndex ?? 'n/a'}: command=${commandToDebugString(command)} frame={moveX:${frame.moveX},rotate:${frame.rotate},rotate180:${frame.rotate180 ? 1 : 0},softDrop:${frame.softDrop ? 1 : 0},hardDrop:${frame.hardDrop ? 1 : 0},hold:${frame.hold ? 1 : 0}} remaining=${commandQueue.length}.`,
        );
      }
      stats.commandsEmitted += 1;
      return frame;
    },
    reset: () => {
      activeRef = null;
      queue = [];
      commandQueue = [];
      cooldownMs = 0;
      manualStepBudget = 0;
      sampleCursor = 0;
      pendingCheck = null;
      finished = false;
      config.onTargetGhostChange?.(null);
      stats.plannedSteps = 0;
      stats.passedBoardChecks = 0;
      stats.failedBoardChecks = 0;
      stats.failedPlans = 0;
      stats.commandsEmitted = 0;
    },
    isStepMode: stepMode,
    requestStep: () => {
      if (!stepMode || finished) return;
      manualStepBudget += 1;
      if (debugTrace) {
        log(
          `debug manual step granted: budget=${manualStepBudget}, pending_commands=${commandQueue.length}.`,
        );
      }
    },
  };
};
