import { SPAWN_X, SPAWN_Y } from './constants';
import { collides, dropDistance } from './piece';
import { getSrsKickTests } from './srs';
import { rotAdd } from './tetromino';
import {
  PIECES,
  type ActivePiece,
  type Board,
  type InputFrame,
  type PieceKind,
  type Rotation,
} from './types';
import type { TrajectoryReplayStepV1 } from './trajectoryProtocol';

const EMPTY_INPUT: InputFrame = {
  moveX: 0,
  rotate: 0,
  rotate180: false,
  softDrop: false,
  hardDrop: false,
  hold: false,
  restart: false,
};

const SEARCH_ACTION_ORDER = [
  'left',
  'right',
  'rotate_cw',
  'rotate_ccw',
  'rotate_180',
  'soft_drop',
] as const;

type SearchAction = (typeof SEARCH_ACTION_ORDER)[number];
export type TrajectoryExecutorSearchAction = SearchAction;

export type TrajectoryExecutorCommand = SearchAction | 'hold' | 'hard_drop';

type SearchActionOrderOptions = {
  allowSoftDrop?: boolean;
  shuffleSearchActions?: boolean;
  random?: () => number;
};

export type TrajectoryExecutorPlanInput = {
  board: Board;
  active: ActivePiece;
  hold: PieceKind | null;
  canHold: boolean;
  target: TrajectoryReplayStepV1;
  maxNodes?: number;
  allowSoftDrop?: boolean;
  shuffleSearchActions?: boolean;
  random?: () => number;
};

export type TrajectoryExecutorPlanResult =
  | {
      ok: true;
      commands: TrajectoryExecutorCommand[];
      visitedNodes: number;
      searchDepth: number;
      holdsConsumed: boolean;
    }
  | {
      ok: false;
      reason: string;
      visitedNodes: number;
      holdsConsumed: boolean;
    };

export type TrajectoryExecutorReachablePlacement = {
  lockPiece: PieceKind;
  lockRotation: number;
  lockX: number;
  lockY: number;
  holdUsed: boolean;
  srsKickCount?: number;
  commands: TrajectoryExecutorCommand[];
  searchDepth: number;
};

export type TrajectoryExecutorEnumerateInput = {
  board: Board;
  active: ActivePiece;
  hold: PieceKind | null;
  canHold: boolean;
  nextPieceOnFirstHold?: PieceKind | null;
  maxNodesPerBranch?: number;
  allowSoftDrop?: boolean;
  shuffleSearchActions?: boolean;
  random?: () => number;
};

export type TrajectoryExecutorSimulationResult = {
  ok: boolean;
  finalPiece: ActivePiece;
  hold: PieceKind | null;
  canHold: boolean;
  locked: boolean;
  failedAt: number | null;
};

type SearchNode = {
  piece: ActivePiece;
  commands: SearchAction[];
  srsKickCount: number;
};

type AppliedSearchAction = {
  piece: ActivePiece;
  srsKicksUsed: number;
};

type ApplySearchActionOptions = {
  onlyWhenTwoQuarterTurnsFail?: boolean;
};

const clonePiece = (piece: ActivePiece): ActivePiece => ({ ...piece });

const stateKey = (piece: ActivePiece): string =>
  `${piece.k}:${piece.r}:${piece.x}:${piece.y}`;

const rotateTo = (value: number): number => {
  const normalized = Math.trunc(value) % 4;
  return normalized < 0 ? normalized + 4 : normalized;
};

const PIECE_ORDER = new Map(PIECES.map((piece, index) => [piece, index]));

const defaultRandom = (): number => Math.random();

const clampRandom01 = (value: number): number => {
  if (!Number.isFinite(value)) return 0;
  if (value <= 0) return 0;
  if (value >= 1) return 0.999999999999;
  return value;
};

const shuffleSearchActions = (
  values: readonly SearchAction[],
  random: () => number,
): SearchAction[] => {
  const out = [...values];
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(clampRandom01(random()) * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
};

const resolveSearchActionOrder = (
  options: SearchActionOrderOptions,
): SearchAction[] => {
  const allowSoftDrop = options.allowSoftDrop !== false;
  const base = allowSoftDrop
    ? SEARCH_ACTION_ORDER
    : SEARCH_ACTION_ORDER.filter((action) => action !== 'soft_drop');
  if (options.shuffleSearchActions !== true) {
    return [...base];
  }
  return shuffleSearchActions(base, options.random ?? defaultRandom);
};

const toInputFrame = (command: TrajectoryExecutorCommand): InputFrame => {
  switch (command) {
    case 'left':
      return { ...EMPTY_INPUT, moveX: -1 };
    case 'right':
      return { ...EMPTY_INPUT, moveX: 1 };
    case 'rotate_cw':
      return { ...EMPTY_INPUT, rotate: 1 };
    case 'rotate_ccw':
      return { ...EMPTY_INPUT, rotate: -1 };
    case 'rotate_180':
      return { ...EMPTY_INPUT, rotate180: true };
    case 'soft_drop':
      return { ...EMPTY_INPUT, softDrop: true };
    case 'hard_drop':
      return { ...EMPTY_INPUT, hardDrop: true };
    case 'hold':
      return { ...EMPTY_INPUT, hold: true };
    default:
      return { ...EMPTY_INPUT };
  }
};

const tryRotateSrsAction = (
  board: Board,
  piece: ActivePiece,
  dir: -1 | 1,
): AppliedSearchAction | null => {
  const from = piece.r;
  const to = rotAdd(from, dir);
  const tests = getSrsKickTests(piece.k, from, to);
  for (const [dx, dy] of tests) {
    if (!collides(board, piece, to, dx, dy)) {
      const next = clonePiece(piece);
      next.r = to;
      next.x += dx;
      next.y += dy;
      return {
        piece: next,
        srsKicksUsed: dx === 0 && dy === 0 ? 0 : 1,
      };
    }
  }
  return null;
};

const tryRotate180ViaTwoQuarterTurnsAction = (
  board: Board,
  piece: ActivePiece,
): AppliedSearchAction | null => {
  const attemptViaDir = (dir: -1 | 1): AppliedSearchAction | null => {
    const first = tryRotateSrsAction(board, piece, dir);
    if (!first) return null;
    const second = tryRotateSrsAction(board, first.piece, dir);
    if (!second) return null;
    return {
      piece: second.piece,
      srsKicksUsed: first.srsKicksUsed + second.srsKicksUsed,
    };
  };

  return attemptViaDir(1) ?? attemptViaDir(-1);
};

const tryRotate180Action = (
  board: Board,
  piece: ActivePiece,
  options: ApplySearchActionOptions = {},
): AppliedSearchAction | null => {
  const viaTwoQuarterTurns = tryRotate180ViaTwoQuarterTurnsAction(board, piece);
  if (options.onlyWhenTwoQuarterTurnsFail === true && viaTwoQuarterTurns) {
    // In planning/enumeration, emit rotate_180 only when both two-step SRS
    // variants fail, so the executor keeps richer quarter-turn signal.
    return null;
  }

  const to = ((piece.r + 2) % 4) as Rotation;
  if (!collides(board, piece, to, 0, 0)) {
    const next = clonePiece(piece);
    next.r = to;
    return {
      piece: next,
      srsKicksUsed: 0,
    };
  }

  return viaTwoQuarterTurns;
};

export const trajectoryExecutorCommandToInputFrame = (
  command: TrajectoryExecutorCommand,
): InputFrame => toInputFrame(command);

const applySearchAction = (
  board: Board,
  piece: ActivePiece,
  action: SearchAction,
  options: ApplySearchActionOptions = {},
): AppliedSearchAction | null => {
  const next = clonePiece(piece);
  switch (action) {
    case 'left':
      if (collides(board, next, next.r, -1, 0)) return null;
      next.x -= 1;
      return {
        piece: next,
        srsKicksUsed: 0,
      };
    case 'right':
      if (collides(board, next, next.r, 1, 0)) return null;
      next.x += 1;
      return {
        piece: next,
        srsKicksUsed: 0,
      };
    case 'soft_drop':
      if (collides(board, next, next.r, 0, 1)) return null;
      next.y += 1;
      return {
        piece: next,
        srsKicksUsed: 0,
      };
    case 'rotate_cw':
      return tryRotateSrsAction(board, next, 1);
    case 'rotate_ccw':
      return tryRotateSrsAction(board, next, -1);
    case 'rotate_180':
      return tryRotate180Action(board, next, options);
    default:
      return null;
  }
};

const spawnPieceForExecutor = (
  board: Board,
  piece: PieceKind,
): ActivePiece | null => {
  const spawned: ActivePiece = {
    k: piece,
    r: 0,
    x: SPAWN_X,
    y: SPAWN_Y,
  };
  if (collides(board, spawned, spawned.r, 0, 0)) {
    return null;
  }
  return spawned;
};

const isTargetLockReachable = (
  board: Board,
  piece: ActivePiece,
  target: TrajectoryReplayStepV1,
): boolean => {
  if (piece.k !== target.lockPiece) return false;
  if (rotateTo(piece.r) !== rotateTo(target.lockRotation)) return false;
  if (piece.x !== target.lockX) return false;
  const lockY = piece.y + dropDistance(board, piece);
  return lockY === target.lockY;
};

const resolveStartState = (
  input: TrajectoryExecutorPlanInput,
):
  | {
      ok: true;
      piece: ActivePiece;
      prefix: TrajectoryExecutorCommand[];
      holdsConsumed: boolean;
    }
  | {
      ok: false;
      reason: string;
      holdsConsumed: boolean;
    } => {
  if (input.target.holdUsed) {
    if (!input.canHold) {
      return {
        ok: false,
        reason: 'Target requires hold, but canHold=false.',
        holdsConsumed: false,
      };
    }
    if (!input.hold) {
      return {
        ok: false,
        reason: 'Target requires hold, but hold piece is unknown.',
        holdsConsumed: false,
      };
    }
    if (input.hold !== input.target.lockPiece) {
      return {
        ok: false,
        reason:
          'Target requires hold, but hold piece does not match target lock piece.',
        holdsConsumed: false,
      };
    }
    return {
      ok: true,
      piece: {
        k: input.hold,
        r: 0,
        x: SPAWN_X,
        y: SPAWN_Y,
      },
      prefix: ['hold'],
      holdsConsumed: true,
    };
  }

  if (input.active.k !== input.target.lockPiece) {
    return {
      ok: false,
      reason:
        'Target lock piece does not match active piece and holdUsed=false.',
      holdsConsumed: false,
    };
  }
  return {
    ok: true,
    piece: clonePiece(input.active),
    prefix: [],
    holdsConsumed: false,
  };
};

export const planTrajectoryLockExecution = (
  input: TrajectoryExecutorPlanInput,
): TrajectoryExecutorPlanResult => {
  const maxNodes = Math.max(1, Math.trunc(input.maxNodes ?? 30_000));
  const startResolved = resolveStartState(input);
  if (!startResolved.ok) {
    return {
      ok: false,
      reason: startResolved.reason,
      visitedNodes: 0,
      holdsConsumed: startResolved.holdsConsumed,
    };
  }

  const actionOrder = resolveSearchActionOrder({
    allowSoftDrop: input.allowSoftDrop,
    shuffleSearchActions: input.shuffleSearchActions,
    random: input.random,
  });
  const queue: SearchNode[] = [
    {
      piece: startResolved.piece,
      commands: [],
      srsKickCount: 0,
    },
  ];
  const visited = new Set<string>([stateKey(startResolved.piece)]);
  let head = 0;

  while (head < queue.length) {
    const node = queue[head++];
    if (isTargetLockReachable(input.board, node.piece, input.target)) {
      return {
        ok: true,
        commands: [...startResolved.prefix, ...node.commands, 'hard_drop'],
        visitedNodes: visited.size,
        searchDepth: node.commands.length,
        holdsConsumed: startResolved.holdsConsumed,
      };
    }
    if (visited.size >= maxNodes) {
      break;
    }
    for (const action of actionOrder) {
      const next = applySearchAction(input.board, node.piece, action, {
        onlyWhenTwoQuarterTurnsFail: true,
      });
      if (!next) continue;
      const key = stateKey(next.piece);
      if (visited.has(key)) continue;
      visited.add(key);
      queue.push({
        piece: next.piece,
        commands: [...node.commands, action],
        srsKickCount: node.srsKickCount + next.srsKicksUsed,
      });
    }
  }

  return {
    ok: false,
    reason: `No deterministic path found within ${maxNodes} explored states.`,
    visitedNodes: visited.size,
    holdsConsumed: startResolved.holdsConsumed,
  };
};

const enumerateReachableLockPlacementsForStart = (input: {
  board: Board;
  startPiece: ActivePiece;
  prefix: TrajectoryExecutorCommand[];
  holdUsed: boolean;
  maxNodes: number;
  allowSoftDrop?: boolean;
  shuffleSearchActions?: boolean;
  random?: () => number;
}): TrajectoryExecutorReachablePlacement[] => {
  const actionOrder = resolveSearchActionOrder({
    allowSoftDrop: input.allowSoftDrop,
    shuffleSearchActions: input.shuffleSearchActions,
    random: input.random,
  });
  const queue: SearchNode[] = [
    {
      piece: clonePiece(input.startPiece),
      commands: [],
      srsKickCount: 0,
    },
  ];
  const visited = new Set<string>([stateKey(input.startPiece)]);
  const byLockKey = new Map<string, TrajectoryExecutorReachablePlacement>();
  let head = 0;

  while (head < queue.length) {
    const node = queue[head++];
    const lockY = node.piece.y + dropDistance(input.board, node.piece);
    const lockKey =
      `${node.piece.k}:` +
      `${rotateTo(node.piece.r)}:` +
      `${node.piece.x}:` +
      `${lockY}:` +
      `${input.holdUsed ? 1 : 0}`;
    if (!byLockKey.has(lockKey)) {
      byLockKey.set(lockKey, {
        lockPiece: node.piece.k,
        lockRotation: rotateTo(node.piece.r),
        lockX: node.piece.x,
        lockY,
        holdUsed: input.holdUsed,
        srsKickCount: node.srsKickCount,
        commands: [...input.prefix, ...node.commands, 'hard_drop'],
        searchDepth: node.commands.length,
      });
    }
    if (visited.size >= input.maxNodes) {
      break;
    }
    for (const action of actionOrder) {
      const next = applySearchAction(input.board, node.piece, action, {
        onlyWhenTwoQuarterTurnsFail: true,
      });
      if (!next) continue;
      const key = stateKey(next.piece);
      if (visited.has(key)) continue;
      visited.add(key);
      queue.push({
        piece: next.piece,
        commands: [...node.commands, action],
        srsKickCount: node.srsKickCount + next.srsKicksUsed,
      });
    }
  }

  return Array.from(byLockKey.values());
};

export const enumerateTrajectoryExecutorPlacements = (
  input: TrajectoryExecutorEnumerateInput,
): TrajectoryExecutorReachablePlacement[] => {
  const maxNodesPerBranch = Math.max(
    256,
    Math.trunc(input.maxNodesPerBranch ?? 12_000),
  );
  const allowSoftDrop = input.allowSoftDrop !== false;
  const placements: TrajectoryExecutorReachablePlacement[] = [];

  placements.push(
    ...enumerateReachableLockPlacementsForStart({
      board: input.board,
      startPiece: input.active,
      prefix: [],
      holdUsed: false,
      maxNodes: maxNodesPerBranch,
      allowSoftDrop,
      shuffleSearchActions: input.shuffleSearchActions,
      random: input.random,
    }),
  );

  if (input.canHold) {
    const holdTarget = input.hold ?? input.nextPieceOnFirstHold ?? null;
    if (holdTarget) {
      const spawned = spawnPieceForExecutor(input.board, holdTarget);
      if (spawned) {
        placements.push(
          ...enumerateReachableLockPlacementsForStart({
            board: input.board,
            startPiece: spawned,
            prefix: ['hold'],
            holdUsed: true,
            maxNodes: maxNodesPerBranch,
            allowSoftDrop,
            shuffleSearchActions: input.shuffleSearchActions,
            random: input.random,
          }),
        );
      }
    }
  }

  placements.sort((left, right) => {
    if (left.holdUsed !== right.holdUsed) {
      return left.holdUsed ? 1 : -1;
    }
    const pieceCompare =
      (PIECE_ORDER.get(left.lockPiece) ?? 99) -
      (PIECE_ORDER.get(right.lockPiece) ?? 99);
    if (pieceCompare !== 0) return pieceCompare;
    if (left.lockY !== right.lockY) return right.lockY - left.lockY;
    if (left.lockX !== right.lockX) return left.lockX - right.lockX;
    if (left.lockRotation !== right.lockRotation) {
      return left.lockRotation - right.lockRotation;
    }
    if (left.searchDepth !== right.searchDepth) {
      return left.searchDepth - right.searchDepth;
    }
    return left.commands.length - right.commands.length;
  });
  return placements;
};

export const simulateTrajectoryExecutorCommands = (input: {
  board: Board;
  active: ActivePiece;
  hold: PieceKind | null;
  canHold: boolean;
  commands: TrajectoryExecutorCommand[];
}): TrajectoryExecutorSimulationResult => {
  let piece = clonePiece(input.active);
  let hold = input.hold;
  let canHold = input.canHold;
  let locked = false;
  for (let i = 0; i < input.commands.length; i += 1) {
    const command = input.commands[i];
    if (command === 'hold') {
      if (!canHold || hold == null) {
        return {
          ok: false,
          finalPiece: piece,
          hold,
          canHold,
          locked,
          failedAt: i,
        };
      }
      const previousActive = piece.k;
      piece = {
        k: hold,
        r: 0,
        x: SPAWN_X,
        y: SPAWN_Y,
      };
      hold = previousActive;
      canHold = false;
      continue;
    }
    if (command === 'hard_drop') {
      piece.y += dropDistance(input.board, piece);
      locked = true;
      return {
        ok: true,
        finalPiece: piece,
        hold,
        canHold,
        locked,
        failedAt: null,
      };
    }
    const next = applySearchAction(input.board, piece, command);
    if (!next) {
      return {
        ok: false,
        finalPiece: piece,
        hold,
        canHold,
        locked,
        failedAt: i,
      };
    }
    piece = next.piece;
  }
  return {
    ok: true,
    finalPiece: piece,
    hold,
    canHold,
    locked,
    failedAt: null,
  };
};
