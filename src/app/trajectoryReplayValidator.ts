import { clearLines } from '../core/board';
import { SPAWN_X, SPAWN_Y } from '../core/constants';
import { collides, merge } from '../core/piece';
import {
  planTrajectoryLockExecution,
  simulateTrajectoryExecutorCommands,
} from '../core/trajectoryExecutor';
import type { TrajectorySessionV1 } from '../core/trajectoryProtocol';
import type { ActivePiece, Board, PieceKind } from '../core/types';

export type ReplayValidationFailureCode =
  | 'missing_initial_state'
  | 'missing_replay'
  | 'plan_failed'
  | 'simulation_failed'
  | 'lock_mismatch'
  | 'board_mismatch'
  | 'hold_mismatch';

export type ReplayValidationIssue = {
  code: ReplayValidationFailureCode;
  sessionId: string;
  modeId: string;
  buildVersion: string;
  sampleId: string | null;
  sampleIndex: number | null;
  detail: string;
};

export type ReplaySessionValidationSummary = {
  sessionId: string;
  modeId: string;
  buildVersion: string;
  totalSamples: number;
  replaySteps: number;
  validatedSteps: number;
  passedSteps: number;
  skippedSteps: number;
  successRate: number;
  failureCounts: Record<string, number>;
};

export type ReplayModeBuildSummary = {
  modeId: string;
  buildVersion: string;
  sessions: number;
  replaySteps: number;
  validatedSteps: number;
  passedSteps: number;
  successRate: number;
};

export type ReplayValidationSummary = {
  totalSessions: number;
  totalSamples: number;
  replaySteps: number;
  validatedSteps: number;
  passedSteps: number;
  skippedSteps: number;
  successRate: number;
  failureCounts: Record<string, number>;
  byModeBuild: ReplayModeBuildSummary[];
  sessions: ReplaySessionValidationSummary[];
  issues: ReplayValidationIssue[];
};

export type ReplayValidationOptions = {
  maxNodes?: number;
  allowSoftDrop?: boolean;
  maxReportedIssues?: number;
};

type ReplayState = {
  boardOccupancy: number[][];
  hold: PieceKind | null;
  active: ActivePiece;
  canHold: boolean;
};

const cloneBoardOccupancy = (occupancy: number[][]): number[][] =>
  occupancy.map((row) => row.map((cell) => (cell > 0 ? 1 : 0)));

const occupancyToBoard = (occupancy: number[][]): Board =>
  occupancy.map((row) => row.map((cell) => (cell > 0 ? 'I' : null)));

const boardToOccupancy = (board: Board): number[][] =>
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

const increment = (map: Record<string, number>, key: string): void => {
  map[key] = (map[key] ?? 0) + 1;
};

const normalizeRotation = (value: number): 0 | 1 | 2 | 3 => {
  const normalized = Math.trunc(value) % 4;
  if (normalized === 1) return 1;
  if (normalized === 2) return 2;
  if (normalized === 3 || normalized === -1) return 3;
  return 0;
};

const pushIssue = (
  issues: ReplayValidationIssue[],
  issue: ReplayValidationIssue,
  maxReportedIssues: number,
): void => {
  if (issues.length >= maxReportedIssues) return;
  issues.push(issue);
};

const spawnActiveForBoard = (board: Board, kind: PieceKind): ActivePiece => {
  const active: ActivePiece = {
    k: kind,
    r: 0,
    x: SPAWN_X,
    y: SPAWN_Y,
  };
  for (let i = 0; i < 4; i += 1) {
    if (!collides(board, active)) return active;
    active.y -= 1;
  }
  return active;
};

const failStep = (
  sessionSummary: ReplaySessionValidationSummary,
  globalFailureCounts: Record<string, number>,
  issues: ReplayValidationIssue[],
  maxReportedIssues: number,
  input: {
    code: ReplayValidationFailureCode;
    detail: string;
    sessionId: string;
    modeId: string;
    buildVersion: string;
    sampleId: string | null;
    sampleIndex: number | null;
  },
): void => {
  increment(sessionSummary.failureCounts, input.code);
  increment(globalFailureCounts, input.code);
  pushIssue(
    issues,
    {
      code: input.code,
      detail: input.detail,
      sessionId: input.sessionId,
      modeId: input.modeId,
      buildVersion: input.buildVersion,
      sampleId: input.sampleId,
      sampleIndex: input.sampleIndex,
    },
    maxReportedIssues,
  );
};

const finalizeSessionSummary = (
  summary: ReplaySessionValidationSummary,
): ReplaySessionValidationSummary => {
  const successRate =
    summary.validatedSteps > 0
      ? summary.passedSteps / summary.validatedSteps
      : 0;
  return {
    ...summary,
    successRate,
  };
};

export const validateTrajectoryReplaySessions = (
  sessions: TrajectorySessionV1[],
  options: ReplayValidationOptions = {},
): ReplayValidationSummary => {
  const maxReportedIssues = Math.max(
    0,
    Math.trunc(options.maxReportedIssues ?? 200),
  );
  const sessionSummaries: ReplaySessionValidationSummary[] = [];
  const issues: ReplayValidationIssue[] = [];
  const failureCounts: Record<string, number> = {};

  for (const session of sessions) {
    const sessionSummary: ReplaySessionValidationSummary = {
      sessionId: session.sessionId,
      modeId: session.modeId,
      buildVersion: session.buildVersion,
      totalSamples: session.samples.length,
      replaySteps: 0,
      validatedSteps: 0,
      passedSteps: 0,
      skippedSteps: 0,
      successRate: 0,
      failureCounts: {},
    };

    let state: ReplayState | null = null;
    let startIndex = 0;
    if (session.initialState) {
      state = {
        boardOccupancy: cloneBoardOccupancy(
          session.initialState.boardOccupancy,
        ),
        hold: session.initialState.hold,
        active: {
          k: session.initialState.active.k,
          r: normalizeRotation(session.initialState.active.r),
          x: session.initialState.active.x,
          y: session.initialState.active.y,
        },
        canHold: session.initialState.canHold,
      };
    } else if (session.samples.length > 0) {
      const first = session.samples[0];
      const firstBoard = occupancyToBoard(first.boardOccupancy);
      state = {
        boardOccupancy: cloneBoardOccupancy(first.boardOccupancy),
        hold: first.hold,
        active: spawnActiveForBoard(firstBoard, first.action),
        canHold: true,
      };
      startIndex = 1;
      sessionSummary.skippedSteps += 1;
      failStep(sessionSummary, failureCounts, issues, maxReportedIssues, {
        code: 'missing_initial_state',
        detail: 'Session has no initialState; first replay step is skipped.',
        sessionId: session.sessionId,
        modeId: session.modeId,
        buildVersion: session.buildVersion,
        sampleId: first.id,
        sampleIndex: 0,
      });
    }

    if (!state) {
      sessionSummaries.push(finalizeSessionSummary(sessionSummary));
      continue;
    }

    for (let index = startIndex; index < session.samples.length; index += 1) {
      const sample = session.samples[index];
      if (!sample.replay) {
        sessionSummary.skippedSteps += 1;
        failStep(sessionSummary, failureCounts, issues, maxReportedIssues, {
          code: 'missing_replay',
          detail: 'Sample has no replay lock metadata.',
          sessionId: session.sessionId,
          modeId: session.modeId,
          buildVersion: session.buildVersion,
          sampleId: sample.id,
          sampleIndex: index,
        });
      } else {
        sessionSummary.replaySteps += 1;
        sessionSummary.validatedSteps += 1;

        const boardBefore = occupancyToBoard(state.boardOccupancy);
        const plan = planTrajectoryLockExecution({
          board: boardBefore,
          active: state.active,
          hold: state.hold,
          canHold: state.canHold,
          target: sample.replay,
          maxNodes: options.maxNodes,
          allowSoftDrop: options.allowSoftDrop,
        });
        if (!plan.ok) {
          failStep(sessionSummary, failureCounts, issues, maxReportedIssues, {
            code: 'plan_failed',
            detail: plan.reason,
            sessionId: session.sessionId,
            modeId: session.modeId,
            buildVersion: session.buildVersion,
            sampleId: sample.id,
            sampleIndex: index,
          });
        } else {
          const simulation = simulateTrajectoryExecutorCommands({
            board: boardBefore,
            active: state.active,
            hold: state.hold,
            canHold: state.canHold,
            commands: plan.commands,
          });
          if (!simulation.ok || !simulation.locked) {
            const detail =
              simulation.failedAt != null
                ? `Simulation failed at command index ${simulation.failedAt}.`
                : 'Simulation failed before lock.';
            failStep(sessionSummary, failureCounts, issues, maxReportedIssues, {
              code: 'simulation_failed',
              detail,
              sessionId: session.sessionId,
              modeId: session.modeId,
              buildVersion: session.buildVersion,
              sampleId: sample.id,
              sampleIndex: index,
            });
          } else if (
            simulation.finalPiece.k !== sample.replay.lockPiece ||
            simulation.finalPiece.r !== sample.replay.lockRotation ||
            simulation.finalPiece.x !== sample.replay.lockX ||
            simulation.finalPiece.y !== sample.replay.lockY
          ) {
            failStep(sessionSummary, failureCounts, issues, maxReportedIssues, {
              code: 'lock_mismatch',
              detail:
                `Simulated lock (${simulation.finalPiece.k},r=${simulation.finalPiece.r},` +
                `x=${simulation.finalPiece.x},y=${simulation.finalPiece.y})` +
                ` != replay (${sample.replay.lockPiece},r=${sample.replay.lockRotation},` +
                `x=${sample.replay.lockX},y=${sample.replay.lockY}).`,
              sessionId: session.sessionId,
              modeId: session.modeId,
              buildVersion: session.buildVersion,
              sampleId: sample.id,
              sampleIndex: index,
            });
          } else {
            const boardAfter = occupancyToBoard(state.boardOccupancy);
            merge(boardAfter, simulation.finalPiece);
            clearLines(boardAfter);
            const simulatedOccupancy = boardToOccupancy(boardAfter);

            if (!sameOccupancy(simulatedOccupancy, sample.boardOccupancy)) {
              failStep(
                sessionSummary,
                failureCounts,
                issues,
                maxReportedIssues,
                {
                  code: 'board_mismatch',
                  detail: 'Post-lock board occupancy does not match sample.',
                  sessionId: session.sessionId,
                  modeId: session.modeId,
                  buildVersion: session.buildVersion,
                  sampleId: sample.id,
                  sampleIndex: index,
                },
              );
            } else if (simulation.hold !== sample.hold) {
              failStep(
                sessionSummary,
                failureCounts,
                issues,
                maxReportedIssues,
                {
                  code: 'hold_mismatch',
                  detail:
                    `Post-lock hold mismatch (simulated=${simulation.hold ?? 'null'},` +
                    ` sample=${sample.hold ?? 'null'}).`,
                  sessionId: session.sessionId,
                  modeId: session.modeId,
                  buildVersion: session.buildVersion,
                  sampleId: sample.id,
                  sampleIndex: index,
                },
              );
            } else {
              sessionSummary.passedSteps += 1;
            }
          }
        }
      }

      const nextBoardOccupancy = cloneBoardOccupancy(sample.boardOccupancy);
      const nextBoard = occupancyToBoard(nextBoardOccupancy);
      state = {
        boardOccupancy: nextBoardOccupancy,
        hold: sample.hold,
        active: spawnActiveForBoard(nextBoard, sample.action),
        canHold: true,
      };
    }

    sessionSummaries.push(finalizeSessionSummary(sessionSummary));
  }

  const byModeBuildMap = new Map<string, ReplayModeBuildSummary>();
  for (const sessionSummary of sessionSummaries) {
    const key = `${sessionSummary.modeId}\u0000${sessionSummary.buildVersion}`;
    const current = byModeBuildMap.get(key);
    if (!current) {
      byModeBuildMap.set(key, {
        modeId: sessionSummary.modeId,
        buildVersion: sessionSummary.buildVersion,
        sessions: 1,
        replaySteps: sessionSummary.replaySteps,
        validatedSteps: sessionSummary.validatedSteps,
        passedSteps: sessionSummary.passedSteps,
        successRate:
          sessionSummary.validatedSteps > 0
            ? sessionSummary.passedSteps / sessionSummary.validatedSteps
            : 0,
      });
      continue;
    }
    current.sessions += 1;
    current.replaySteps += sessionSummary.replaySteps;
    current.validatedSteps += sessionSummary.validatedSteps;
    current.passedSteps += sessionSummary.passedSteps;
    current.successRate =
      current.validatedSteps > 0
        ? current.passedSteps / current.validatedSteps
        : 0;
  }

  const replaySteps = sessionSummaries.reduce(
    (sum, entry) => sum + entry.replaySteps,
    0,
  );
  const validatedSteps = sessionSummaries.reduce(
    (sum, entry) => sum + entry.validatedSteps,
    0,
  );
  const passedSteps = sessionSummaries.reduce(
    (sum, entry) => sum + entry.passedSteps,
    0,
  );
  const skippedSteps = sessionSummaries.reduce(
    (sum, entry) => sum + entry.skippedSteps,
    0,
  );
  const totalSamples = sessionSummaries.reduce(
    (sum, entry) => sum + entry.totalSamples,
    0,
  );

  return {
    totalSessions: sessionSummaries.length,
    totalSamples,
    replaySteps,
    validatedSteps,
    passedSteps,
    skippedSteps,
    successRate: validatedSteps > 0 ? passedSteps / validatedSteps : 0,
    failureCounts,
    byModeBuild: [...byModeBuildMap.values()].sort((left, right) => {
      if (left.modeId !== right.modeId) {
        return left.modeId.localeCompare(right.modeId);
      }
      return left.buildVersion.localeCompare(right.buildVersion);
    }),
    sessions: sessionSummaries,
    issues,
  };
};
