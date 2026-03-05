import { readFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';

import {
  applyModeSettings,
  runModeStart,
} from '../../../src/app/modeService.ts';
import { createGeneratorFactory } from '../../../src/core/generators.ts';
import { Game } from '../../../src/core/game.ts';
import { getMode } from '../../../src/core/modes.ts';
import { createModelRunner } from '../../../src/core/modelRunner.ts';
import { clearLines } from '../../../src/core/board.ts';
import { collides, dropDistance } from '../../../src/core/piece.ts';
import {
  PLACEMENT_ACTION_DIM,
  placementActionIndexFromPlacement,
} from '../../../src/core/placementActionSpace.ts';
import { XorShift32 } from '../../../src/core/rng.ts';
import { GameRunner, type InputSource } from '../../../src/core/runner.ts';
import { DEFAULT_SETTINGS } from '../../../src/core/settings.ts';
import {
  PIECES,
  type Board,
  type GameState,
  type InputFrame,
  type PieceKind,
} from '../../../src/core/types.ts';
import { TETROMINOES } from '../../../src/core/tetromino.ts';
import {
  enumerateTrajectoryExecutorPlacements,
  type TrajectoryExecutorReachablePlacement,
  trajectoryExecutorCommandToInputFrame,
} from '../../../src/core/trajectoryExecutor.ts';
import {
  encodeBotObservation,
  normalizeBotObservationSpace,
  type BotObservationSpace,
} from '../../../src/core/botObservation.ts';
import {
  parseWubModelFromJsonText,
  type LoadedModel,
} from '../../../src/core/wubModel.ts';
import type {
  InitPayload,
  JsonObject,
  PlacementExecutionMode,
  PieceSourceProfile,
  SetCurriculumPayload,
  StepBatchResult,
} from './protocol.ts';

const FIXED_STEP_MS = 1000 / 120;
const DEFAULT_ACTION_DIM = PLACEMENT_ACTION_DIM;
const DEFAULT_MAX_PIECES = 512;
const STEP_MAX_TICKS = 120;
const OFFLINE_GRAVITY_MS = Number.POSITIVE_INFINITY;
const OFFLINE_SOFT_DROP_MS = 0;
const TOP_OUT_PENALTY = 100;
const TRAJECTORY_SCHEMA = 'wishuponablock.trajectory_session.v1';
const TRAJECTORY_BUILD_VERSION = 'offline_ppo_py';
const TRAJECTORY_PIECES = [...PIECES];
const TRAJECTORY_MIN_SAMPLE_ID = 8;
const MAX_COMPLETED_TRAJECTORY_QUEUE = 1;
const DEFAULT_CURRICULUM_DANGER_HEIGHT = 14;

type ActionCurriculumConfig = {
  topK: number;
  biasStrength: number;
  dangerHeight: number;
};

const DEFAULT_ACTION_CURRICULUM: ActionCurriculumConfig = {
  topK: 0,
  biasStrength: 0,
  dangerHeight: DEFAULT_CURRICULUM_DANGER_HEIGHT,
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

const clampInt = (
  value: unknown,
  fallback: number,
  min: number,
  max: number,
): number => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
  return Math.max(min, Math.min(max, Math.trunc(value)));
};

const normalizePieceSource = (value: unknown): PieceSourceProfile =>
  value === 'active_generator' ? 'active_generator' : 'bag7';

const normalizeObservationSpace = (value: unknown): BotObservationSpace =>
  normalizeBotObservationSpace(typeof value === 'string' ? value : null);

const normalizePlacementExecutionMode = (
  value: unknown,
): PlacementExecutionMode => (value === 'commands' ? 'commands' : 'teleport');

const normalizeActionCurriculum = (
  payload: SetCurriculumPayload | null | undefined,
): ActionCurriculumConfig => {
  const topK = clampInt(payload?.topK, 0, 0, DEFAULT_ACTION_DIM);
  const biasStrengthRaw =
    typeof payload?.biasStrength === 'number' &&
    Number.isFinite(payload.biasStrength)
      ? payload.biasStrength
      : 0;
  const biasStrength = Math.max(0, Math.min(1, biasStrengthRaw));
  const dangerHeight = clampInt(
    payload?.dangerHeight,
    DEFAULT_CURRICULUM_DANGER_HEIGHT,
    1,
    64,
  );
  return {
    topK,
    biasStrength,
    dangerHeight,
  };
};

const normalizeModeId = (value: unknown): string => {
  if (typeof value !== 'string') return 'practice';
  const v = value.trim().toLowerCase();
  if (
    v === 'practice' ||
    v === 'sprint' ||
    v === 'classic' ||
    v === 'cheese' ||
    v === 'charcuterie'
  ) {
    return v;
  }
  return 'practice';
};

const countBoardHoles = (board: Board): number => {
  if (board.length === 0 || board[0].length === 0) return 0;
  let holes = 0;
  const cols = board[0].length;
  for (let x = 0; x < cols; x += 1) {
    let seenBlock = false;
    for (let y = 0; y < board.length; y += 1) {
      const filled = board[y][x] != null;
      if (filled) {
        seenBlock = true;
      } else if (seenBlock) {
        holes += 1;
      }
    }
  }
  return holes;
};

const computeColumnHeights = (board: Board): number[] => {
  if (board.length === 0 || board[0].length === 0) return [];
  const rows = board.length;
  const cols = board[0].length;
  const heights = new Array<number>(cols).fill(0);
  for (let x = 0; x < cols; x += 1) {
    for (let y = 0; y < rows; y += 1) {
      if (board[y][x] != null) {
        heights[x] = rows - y;
        break;
      }
    }
  }
  return heights;
};

const computeBoardBumpiness = (board: Board): number => {
  const heights = computeColumnHeights(board);
  if (heights.length <= 1) return 0;
  let bumpiness = 0;
  for (let i = 0; i < heights.length - 1; i += 1) {
    bumpiness += Math.abs(heights[i] - heights[i + 1]);
  }
  return bumpiness;
};

const countBoardBlocks = (board: Board): number => {
  let blocks = 0;
  for (const row of board) {
    for (const cell of row) {
      if (cell != null) blocks += 1;
    }
  }
  return blocks;
};

const getStackHeight = (board: Board): number => {
  const rows = board.length;
  for (let y = 0; y < rows; y += 1) {
    if (board[y].some((cell) => cell != null)) {
      return rows - y;
    }
  }
  return 0;
};

const CHARCUTERIE_SCORE_WEIGHTS = {
  height: 10,
  holes: 20,
  blocks: 0.01,
  clears: 100,
} as const;

const CHARCUTERIE_HOLE_WEIGHTS = {
  bottom: 5,
  mid: 2,
} as const;
const HOLE_DELTA_REWARD_WEIGHT = 0.05;
const BUMPINESS_DELTA_REWARD_WEIGHT = 0.03;
const HEIGHT_DELTA_REWARD_WEIGHT = 0.04;
const PRACTICE_TIME_DELTA_REWARD_WEIGHT = 1e-5;
const BOARD_QUALITY_DELTA_REWARD_WEIGHT = 0.1;

const getCharcuterieHolePenalty = (board: Board): number => {
  const rows = board.length;
  let penalty = 0;
  for (let y = rows - 1; y >= 0; y -= 1) {
    const row = board[y];
    let empty = 0;
    let anyFilled = false;
    for (const cell of row) {
      if (cell == null) {
        empty += 1;
      } else {
        anyFilled = true;
      }
    }
    if (!anyFilled) continue;
    const extraHoles = Math.max(0, empty - 1);
    if (extraHoles === 0) continue;
    const depth = rows - 1 - y;
    if (depth < 4) {
      penalty += extraHoles * CHARCUTERIE_HOLE_WEIGHTS.bottom;
    } else if (depth < 8) {
      penalty += extraHoles * CHARCUTERIE_HOLE_WEIGHTS.mid;
    }
  }
  return penalty;
};

const scoreCharcuterieBoard = (
  board: Board,
  gameOver: boolean,
  clears: number,
): number => {
  const height = getStackHeight(board) + (gameOver ? board.length : 0);
  const holes = getCharcuterieHolePenalty(board);
  const blocks = countBoardBlocks(board);
  return (
    height * CHARCUTERIE_SCORE_WEIGHTS.height +
    holes * CHARCUTERIE_SCORE_WEIGHTS.holes +
    blocks * CHARCUTERIE_SCORE_WEIGHTS.blocks -
    clears * CHARCUTERIE_SCORE_WEIGHTS.clears
  );
};

type PieceRewardBreakdown = {
  linesTerm: number;
  scoreTerm: number;
  timeTerm: number;
  heightTerm: number;
  holeDeltaTerm: number;
  bumpinessDeltaTerm: number;
  boardScoreTerm: number;
  boardQualityDeltaTerm: number;
};

const computePieceReward = (options: {
  modeId: string;
  linesDelta: number;
  scoreDelta: number;
  timeDeltaMs: number;
  heightDelta: number;
  holesDelta: number;
  bumpinessDelta: number;
  boardScoreDelta: number;
  boardQualityDelta: number;
}): {
  reward: number;
  breakdown: PieceRewardBreakdown;
} => {
  const {
    modeId,
    linesDelta,
    scoreDelta,
    timeDeltaMs,
    heightDelta,
    holesDelta,
    bumpinessDelta,
    boardScoreDelta,
    boardQualityDelta,
  } = options;
  let linesTerm = 0;
  let scoreTerm = 0;
  let timeTerm = 0;
  let boardScoreTerm = 0;
  const heightDeltaTerm = -heightDelta * HEIGHT_DELTA_REWARD_WEIGHT;
  const holeDeltaTerm = -holesDelta * HOLE_DELTA_REWARD_WEIGHT;
  const bumpinessDeltaTerm = -bumpinessDelta * BUMPINESS_DELTA_REWARD_WEIGHT;
  const boardQualityDeltaTerm =
    boardQualityDelta * BOARD_QUALITY_DELTA_REWARD_WEIGHT;
  if (modeId === 'sprint') {
    linesTerm = linesDelta * 1.2;
    scoreTerm = scoreDelta * 0.001;
    timeTerm = -(timeDeltaMs / 4000);
  } else if (modeId === 'classic') {
    linesTerm = linesDelta * 0.6;
    scoreTerm = scoreDelta * 0.002;
  } else if (modeId === 'charcuterie') {
    boardScoreTerm = boardScoreDelta * 0.12;
    linesTerm = linesDelta * 0.15;
    timeTerm = -(timeDeltaMs / 25000);
  } else if (modeId === 'cheese') {
    linesTerm = linesDelta * 0.7;
  } else {
    linesTerm = linesDelta * 0.5;
    scoreTerm = scoreDelta * 0.0008;
    timeTerm = -(timeDeltaMs * PRACTICE_TIME_DELTA_REWARD_WEIGHT);
  }
  const reward =
    linesTerm +
    scoreTerm +
    timeTerm +
    boardScoreTerm +
    boardQualityDeltaTerm +
    heightDeltaTerm +
    holeDeltaTerm +
    bumpinessDeltaTerm;
  return {
    reward,
    breakdown: {
      linesTerm,
      scoreTerm,
      timeTerm,
      heightTerm: heightDeltaTerm,
      holeDeltaTerm,
      bumpinessDeltaTerm,
      boardScoreTerm,
      boardQualityDeltaTerm,
    },
  };
};

type BoardQualityMetrics = {
  aggregateHeight: number;
  maxHeight: number;
  bumpiness: number;
  openHoles: number;
  enclosedHoles: number;
  holeCoverDepth: number;
  quality: number;
};

const BOARD_QUALITY_WEIGHTS = {
  aggregateHeight: 0.03,
  bumpiness: 0.12,
  openHoles: 1.4,
  enclosedHoles: 2.2,
  holeCoverDepth: 0.15,
  dangerQuadratic: 0.35,
} as const;

const BOARD_QUALITY_DANGER_HEIGHT = 12;
const PLACEMENT_LINE_CLEAR_BONUS = 3.0;
const PLACEMENT_COMPLEXITY_CMD_WEIGHT = 0.01;
const PLACEMENT_COMPLEXITY_ROT_WEIGHT = 0.03;
const PLACEMENT_COMPLEXITY_SOFT_DROP_WEIGHT = 0.005;
const PLACEMENT_HOLD_COMPLEXITY_PENALTY = 0.05;
const PLACEMENT_TOP_OUT_PENALTY = 5.0;

const cloneBoard = (board: Board): Board => board.map((row) => [...row]);

const evaluateBoardQuality = (board: Board): BoardQualityMetrics => {
  if (board.length === 0 || board[0].length === 0) {
    return {
      aggregateHeight: 0,
      maxHeight: 0,
      bumpiness: 0,
      openHoles: 0,
      enclosedHoles: 0,
      holeCoverDepth: 0,
      quality: 0,
    };
  }
  const rows = board.length;
  const cols = board[0].length;
  const heights = computeColumnHeights(board);
  const aggregateHeight = heights.reduce((sum, h) => sum + h, 0);
  const maxHeight = heights.reduce((maxH, h) => Math.max(maxH, h), 0);
  const bumpiness = computeBoardBumpiness(board);

  const reachable: boolean[][] = Array.from({ length: rows }, () =>
    Array(cols).fill(false),
  );
  const queue: Array<[number, number]> = [];
  for (let x = 0; x < cols; x += 1) {
    if (board[0][x] == null) {
      reachable[0][x] = true;
      queue.push([x, 0]);
    }
  }
  let qi = 0;
  while (qi < queue.length) {
    const [x, y] = queue[qi] ?? [0, 0];
    qi += 1;
    const neighbors: Array<[number, number]> = [
      [x + 1, y],
      [x - 1, y],
      [x, y + 1],
      [x, y - 1],
    ];
    for (const [nx, ny] of neighbors) {
      if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;
      if (reachable[ny][nx]) continue;
      if (board[ny][nx] != null) continue;
      reachable[ny][nx] = true;
      queue.push([nx, ny]);
    }
  }

  let openHoles = 0;
  let enclosedHoles = 0;
  let holeCoverDepth = 0;
  for (let x = 0; x < cols; x += 1) {
    let seenBlock = false;
    let filledAbove = 0;
    for (let y = 0; y < rows; y += 1) {
      const filled = board[y][x] != null;
      if (filled) {
        seenBlock = true;
        filledAbove += 1;
      } else if (seenBlock) {
        if (reachable[y][x]) openHoles += 1;
        else enclosedHoles += 1;
        holeCoverDepth += filledAbove;
      }
    }
  }

  const danger = Math.max(0, maxHeight - BOARD_QUALITY_DANGER_HEIGHT);
  const quality =
    aggregateHeight * BOARD_QUALITY_WEIGHTS.aggregateHeight +
    bumpiness * BOARD_QUALITY_WEIGHTS.bumpiness +
    openHoles * BOARD_QUALITY_WEIGHTS.openHoles +
    enclosedHoles * BOARD_QUALITY_WEIGHTS.enclosedHoles +
    holeCoverDepth * BOARD_QUALITY_WEIGHTS.holeCoverDepth +
    danger * danger * BOARD_QUALITY_WEIGHTS.dangerQuadratic;

  return {
    aggregateHeight,
    maxHeight,
    bumpiness,
    openHoles,
    enclosedHoles,
    holeCoverDepth,
    quality,
  };
};

const applyPlacementToBoard = (
  board: Board,
  placement: TrajectoryExecutorReachablePlacement,
): {
  boardAfter: Board;
  linesCleared: number;
  hasAboveTop: boolean;
  invalid: boolean;
} => {
  const boardAfter = cloneBoard(board);
  const rows = boardAfter.length;
  const cols = rows > 0 ? boardAfter[0].length : 0;
  const pieceKind = placement.lockPiece as PieceKind;
  const shape = TETROMINOES[pieceKind][clampRotation(placement.lockRotation)];

  let hasAboveTop = false;
  for (const [ox, oy] of shape) {
    const x = Math.trunc(placement.lockX + ox);
    const y = Math.trunc(placement.lockY + oy);
    if (x < 0 || x >= cols || y >= rows) {
      return {
        boardAfter,
        linesCleared: 0,
        hasAboveTop: false,
        invalid: true,
      };
    }
    if (y < 0) {
      hasAboveTop = true;
      continue;
    }
    if (boardAfter[y][x] != null) {
      return {
        boardAfter,
        linesCleared: 0,
        hasAboveTop: false,
        invalid: true,
      };
    }
    boardAfter[y][x] = placement.lockPiece;
  }
  const linesCleared = clearLines(boardAfter).length;
  return {
    boardAfter,
    linesCleared,
    hasAboveTop,
    invalid: false,
  };
};

const scorePlacementCandidate = (options: {
  beforeMetrics: BoardQualityMetrics;
  placement: TrajectoryExecutorReachablePlacement;
  board: Board;
}): number => {
  const { beforeMetrics, placement, board } = options;
  const applied = applyPlacementToBoard(board, placement);
  if (applied.invalid) return -1e9;
  const afterMetrics = evaluateBoardQuality(applied.boardAfter);
  const improvement = beforeMetrics.quality - afterMetrics.quality;

  let rotateCount = 0;
  let softDropCount = 0;
  for (const command of placement.commands) {
    if (
      command === 'rotate_cw' ||
      command === 'rotate_ccw' ||
      command === 'rotate_180'
    ) {
      rotateCount += 1;
    } else if (command === 'soft_drop') {
      softDropCount += 1;
    }
  }
  const complexityPenalty =
    placement.commands.length * PLACEMENT_COMPLEXITY_CMD_WEIGHT +
    rotateCount * PLACEMENT_COMPLEXITY_ROT_WEIGHT +
    softDropCount * PLACEMENT_COMPLEXITY_SOFT_DROP_WEIGHT +
    (placement.holdUsed ? PLACEMENT_HOLD_COMPLEXITY_PENALTY : 0);

  let score =
    improvement +
    applied.linesCleared * PLACEMENT_LINE_CLEAR_BONUS -
    complexityPenalty;
  if (applied.hasAboveTop) score -= PLACEMENT_TOP_OUT_PENALTY;
  if (!Number.isFinite(score)) score = -1e9;
  return score;
};

const encodeObservation = (
  observationSpace: BotObservationSpace,
  model: LoadedModel,
  state: GameState,
): number[] =>
  Array.from(
    encodeBotObservation({
      observationSpace,
      model,
      state,
    }),
  );

const buildPlacementChoices = (
  state: GameState,
  actionDim: number,
  curriculum: ActionCurriculumConfig,
  random: (() => number) | null = null,
): {
  commandsBySlot: Array<InputFrame[] | null>;
  placementsBySlot: Array<TrajectoryExecutorReachablePlacement | null>;
  actionMask: number[];
  actionBiases: number[];
} => {
  const placements = enumerateTrajectoryExecutorPlacements({
    board: state.board,
    active: state.active,
    hold: state.hold,
    canHold: state.canHold,
    nextPieceOnFirstHold: state.next[0] ?? null,
    maxNodesPerBranch: 20_000,
    allowSoftDrop: true,
    shuffleSearchActions: random != null,
    random: random ?? undefined,
  });
  const actionMask = new Array<number>(actionDim).fill(0);
  const commandsBySlot: Array<InputFrame[] | null> = new Array(actionDim).fill(
    null,
  );
  const placementsBySlot: Array<TrajectoryExecutorReachablePlacement | null> =
    new Array(actionDim).fill(null);
  const actionBiases = new Array<number>(actionDim).fill(0);
  const scoresBySlot = new Array<number>(actionDim).fill(
    Number.NEGATIVE_INFINITY,
  );
  const beforeMetrics = evaluateBoardQuality(state.board);
  for (const placement of placements) {
    const actionIndex = placementActionIndexFromPlacement(placement);
    if (actionIndex == null || actionIndex < 0 || actionIndex >= actionDim) {
      continue;
    }
    const score = scorePlacementCandidate({
      beforeMetrics,
      placement,
      board: state.board,
    });
    if (
      actionMask[actionIndex] > 0 &&
      commandsBySlot[actionIndex] &&
      score <= scoresBySlot[actionIndex]
    ) {
      continue;
    }
    actionMask[actionIndex] = 1;
    scoresBySlot[actionIndex] = score;
    placementsBySlot[actionIndex] = {
      ...placement,
      commands: [...placement.commands],
    };
    commandsBySlot[actionIndex] = placement.commands.map((command) =>
      trajectoryExecutorCommandToInputFrame(command),
    );
  }
  if (actionMask.every((value) => value <= 0)) {
    const fallbackLockY =
      state.active.y + dropDistance(state.board, state.active);
    actionMask[0] = 1;
    scoresBySlot[0] = 0;
    placementsBySlot[0] = {
      lockPiece: state.active.k,
      lockRotation: Math.max(0, Math.min(3, Math.trunc(state.active.r))),
      lockX: Math.trunc(state.active.x),
      lockY: Math.trunc(fallbackLockY),
      holdUsed: false,
      commands: ['hard_drop'],
      searchDepth: 0,
    };
    commandsBySlot[0] = [trajectoryExecutorCommandToInputFrame('hard_drop')];
  }

  const legalIndices: number[] = [];
  for (let i = 0; i < actionMask.length; i += 1) {
    if (actionMask[i] > 0) legalIndices.push(i);
  }
  const dangerBypass = beforeMetrics.maxHeight >= curriculum.dangerHeight;

  if (
    curriculum.topK > 0 &&
    !dangerBypass &&
    legalIndices.length > curriculum.topK
  ) {
    const ranked = [...legalIndices].sort((a, b) => {
      const sa = Number.isFinite(scoresBySlot[a]) ? scoresBySlot[a] : -1e12;
      const sb = Number.isFinite(scoresBySlot[b]) ? scoresBySlot[b] : -1e12;
      return sb - sa;
    });
    const keptSet = new Set<number>(ranked.slice(0, curriculum.topK));
    for (const index of legalIndices) {
      if (!keptSet.has(index)) actionMask[index] = 0;
    }
  }

  const kept = actionMask
    .map((value, index) => (value > 0 ? index : -1))
    .filter((value) => value >= 0);
  if (kept.length > 0) {
    const finiteScorePairs = kept
      .map((index) => ({ index, score: scoresBySlot[index] }))
      .filter((entry) => Number.isFinite(entry.score));
    if (finiteScorePairs.length > 0) {
      let minScore = Number.POSITIVE_INFINITY;
      let maxScore = Number.NEGATIVE_INFINITY;
      for (const entry of finiteScorePairs) {
        if (entry.score < minScore) minScore = entry.score;
        if (entry.score > maxScore) maxScore = entry.score;
      }
      const span = maxScore - minScore;
      if (span > 1e-9) {
        for (const entry of finiteScorePairs) {
          actionBiases[entry.index] = Math.max(
            0,
            Math.min(1, (entry.score - minScore) / span),
          );
        }
      } else {
        for (const index of kept) actionBiases[index] = 1;
      }
    } else {
      for (const index of kept) actionBiases[index] = 1;
    }
  }

  if (actionMask.every((value) => value <= 0)) {
    const fallbackIndex = legalIndices.length > 0 ? legalIndices[0] : 0;
    actionMask[fallbackIndex] = 1;
    actionBiases[fallbackIndex] = 1;
  }
  return { commandsBySlot, placementsBySlot, actionMask, actionBiases };
};

const boardToOccupancy = (board: Board): number[][] =>
  board.map((row) => row.map((cell) => (cell != null ? 1 : 0)));

const clampRotation = (value: number): 0 | 1 | 2 | 3 => {
  const normalized = Math.trunc(value) % 4;
  if (normalized === 1) return 1;
  if (normalized === 2) return 2;
  if (normalized === 3 || normalized === -1) return 3;
  return 0;
};

const asPieceOrNull = (value: unknown): string | null => {
  if (typeof value !== 'string' || value.length === 0) return null;
  return value;
};

const pieceIndex = (piece: string | null): number => {
  if (!piece) return 0;
  const index = TRAJECTORY_PIECES.indexOf(
    piece as (typeof TRAJECTORY_PIECES)[number],
  );
  return index >= 0 ? index : 0;
};

const nextFloat = (rng: XorShift32): number => rng.nextU32() / 0xffffffff;

class OneFrameInputSource implements InputSource {
  constructor(private frame: InputFrame) {}
  sample(): InputFrame {
    const out = this.frame;
    this.frame = EMPTY_INPUT;
    return out;
  }
}

type BotEnvResetProfile = {
  total_s: number;
  obs_s: number;
  choices_s: number;
};

type BotEnvStepProfile = {
  total_s: number;
  choices_current_s: number;
  runner_s: number;
  reward_s: number;
  obs_s: number;
  choices_next_s: number;
};

type BotEnvStepResult = {
  obs: number[];
  actionMask: number[];
  actionBias: number[];
  reward: number;
  done: boolean;
  info: JsonObject;
  profile: BotEnvStepProfile;
  completedSession: JsonObject | null;
};

class BotEnv {
  private game: Game;
  private runner: GameRunner;
  private pieceSource: PieceSourceProfile;
  private done = false;
  private piecesPlaced = 0;
  private lockCount = 0;
  private episodeStartWallMs = 0;
  private episodeSampleSerial = 0;
  private episodeSessionSerial = 0;
  private episodeInitialState: JsonObject | null = null;
  private episodeSamples: JsonObject[] = [];
  private planningRng: XorShift32;
  private actionCurriculum: ActionCurriculumConfig = {
    ...DEFAULT_ACTION_CURRICULUM,
  };
  private cachedChoices: {
    commandsBySlot: Array<InputFrame[] | null>;
    placementsBySlot: Array<TrajectoryExecutorReachablePlacement | null>;
    actionMask: number[];
    actionBiases: number[];
  } | null = null;

  constructor(
    private readonly envId: number,
    private readonly modeId: string,
    private readonly model: LoadedModel,
    private readonly observationSpace: BotObservationSpace,
    pieceSource: PieceSourceProfile,
    private readonly queuePolicyId: string,
    private readonly maxPiecesPerEpisode: number,
    private readonly placementExecutionMode: PlacementExecutionMode,
    seed: number,
  ) {
    this.pieceSource = pieceSource;
    this.planningRng = new XorShift32(seed ^ 0x71e9135b);
    const built = this.buildGame(seed);
    this.game = built.game;
    this.runner = built.runner;
    this.cachedChoices = buildPlacementChoices(
      this.game.state,
      DEFAULT_ACTION_DIM,
      this.actionCurriculum,
      () => nextFloat(this.planningRng),
    );
    this.startEpisodeCapture();
    this.syncPrevMetrics();
  }

  reset(seed: number): {
    obs: number[];
    actionMask: number[];
    actionBias: number[];
    info: JsonObject;
    profile: BotEnvResetProfile;
  } {
    const resetStart = performance.now();
    const built = this.buildGame(seed);
    this.game = built.game;
    this.runner = built.runner;
    this.done = false;
    this.piecesPlaced = 0;
    this.lockCount = 0;
    this.planningRng = new XorShift32(seed ^ 0x71e9135b);
    this.startEpisodeCapture();
    const choicesStart = performance.now();
    this.cachedChoices = buildPlacementChoices(
      this.game.state,
      DEFAULT_ACTION_DIM,
      this.actionCurriculum,
      () => nextFloat(this.planningRng),
    );
    const choicesElapsedS = (performance.now() - choicesStart) / 1000;
    this.syncPrevMetrics();
    const obsStart = performance.now();
    const obs = encodeObservation(
      this.observationSpace,
      this.model,
      this.game.state,
    );
    const obsElapsedS = (performance.now() - obsStart) / 1000;
    const choices = this.cachedChoices;
    const totalElapsedS = (performance.now() - resetStart) / 1000;
    return {
      obs,
      actionMask: choices?.actionMask ?? [],
      actionBias: choices?.actionBiases ?? [],
      info: {
        modeId: this.modeId,
        seed,
      },
      profile: {
        total_s: totalElapsedS,
        obs_s: obsElapsedS,
        choices_s: choicesElapsedS,
      },
    };
  }

  setPieceSource(pieceSource: PieceSourceProfile): void {
    this.pieceSource = pieceSource;
  }

  setActionCurriculum(curriculum: ActionCurriculumConfig): void {
    this.actionCurriculum = {
      ...curriculum,
    };
    this.cachedChoices = buildPlacementChoices(
      this.game.state,
      DEFAULT_ACTION_DIM,
      this.actionCurriculum,
      () => nextFloat(this.planningRng),
    );
  }

  step(actionIndexRaw: number): BotEnvStepResult {
    const stepStart = performance.now();
    if (this.done) {
      const doneChoicesStart = performance.now();
      const obs = encodeObservation(
        this.observationSpace,
        this.model,
        this.game.state,
      );
      const choices =
        this.cachedChoices ??
        buildPlacementChoices(
          this.game.state,
          DEFAULT_ACTION_DIM,
          this.actionCurriculum,
          () => nextFloat(this.planningRng),
        );
      this.cachedChoices = choices;
      const doneChoicesElapsedS = (performance.now() - doneChoicesStart) / 1000;
      const totalElapsedS = (performance.now() - stepStart) / 1000;
      return {
        obs,
        actionMask: choices.actionMask,
        actionBias: choices.actionBiases,
        reward: 0,
        done: true,
        info: { alreadyDone: true, piecesPlaced: this.piecesPlaced },
        completedSession: null,
        profile: {
          total_s: totalElapsedS,
          choices_current_s: doneChoicesElapsedS,
          runner_s: 0,
          reward_s: 0,
          obs_s: 0,
          choices_next_s: 0,
        },
      };
    }

    const beforeLockCount = this.lockCount;
    const before = this.snapshotMetrics();
    const choicesCurrentStart = performance.now();
    const choices =
      this.cachedChoices ??
      buildPlacementChoices(
        this.game.state,
        DEFAULT_ACTION_DIM,
        this.actionCurriculum,
        () => nextFloat(this.planningRng),
      );
    this.cachedChoices = choices;
    const choicesCurrentElapsedS =
      (performance.now() - choicesCurrentStart) / 1000;
    const actionIndex = clampInt(actionIndexRaw, 0, 0, DEFAULT_ACTION_DIM - 1);
    const hasAction = choices.actionMask[actionIndex] > 0;
    const resolvedActionIndex = hasAction
      ? actionIndex
      : Math.max(
          0,
          choices.actionMask.findIndex((x) => x > 0),
        );
    const selectedPlacement = choices.placementsBySlot[resolvedActionIndex];
    const commands = choices.commandsBySlot[resolvedActionIndex] ?? [
      trajectoryExecutorCommandToInputFrame('hard_drop'),
    ];

    let ticks = 0;
    let executionPath: 'commands' | 'teleport' = 'commands';
    const runnerStart = performance.now();
    if (
      this.placementExecutionMode === 'teleport' &&
      selectedPlacement != null
    ) {
      const teleport = this.executePlacementByTeleport(
        selectedPlacement,
        beforeLockCount,
      );
      if (teleport.applied) {
        ticks += teleport.ticks;
        executionPath = 'teleport';
      } else {
        ticks += this.executePlacementByCommands(commands, beforeLockCount);
      }
    } else {
      ticks += this.executePlacementByCommands(commands, beforeLockCount);
    }

    if (this.lockCount === beforeLockCount && !this.isTerminal()) {
      // Emergency fallback to prevent deadlocks on invalid plans.
      this.runner.step(
        new OneFrameInputSource({
          ...EMPTY_INPUT,
          hardDrop: true,
        }),
      );
      ticks += 1;
    }
    const runnerElapsedS = (performance.now() - runnerStart) / 1000;

    if (this.lockCount > beforeLockCount) {
      this.piecesPlaced += 1;
    }

    const rewardStart = performance.now();
    const after = this.snapshotMetrics();
    const linesDelta = after.lines - before.lines;
    const scoreDelta = after.score - before.score;
    const timeDeltaMs = after.timeMs - before.timeMs;
    const heightDelta = after.height - before.height;
    const holesDelta = after.holes - before.holes;
    const bumpinessDelta = after.bumpiness - before.bumpiness;
    const boardQualityDelta = before.boardQuality - after.boardQuality;
    const rewardResult = computePieceReward({
      modeId: this.modeId,
      linesDelta,
      scoreDelta,
      timeDeltaMs,
      heightDelta,
      holesDelta,
      bumpinessDelta,
      boardScoreDelta: before.boardScore - after.boardScore,
      boardQualityDelta,
    });
    const reward = rewardResult.reward;
    const topOutPenalty = this.game.state.gameOver ? TOP_OUT_PENALTY : 0;
    const finalReward = reward - topOutPenalty;
    const rewardElapsedS = (performance.now() - rewardStart) / 1000;

    if (this.lockCount > beforeLockCount) {
      this.captureEpisodeStep({
        reward: finalReward,
        after,
        selectedPlacement,
      });
    }

    this.syncPrevMetrics();
    const becameDone =
      this.isTerminal() || this.piecesPlaced >= this.maxPiecesPerEpisode;
    let completedSession: JsonObject | null = null;
    if (becameDone) {
      this.done = true;
      completedSession = this.finalizeEpisodeCapture();
    }

    const obsStart = performance.now();
    const obs = encodeObservation(
      this.observationSpace,
      this.model,
      this.game.state,
    );
    const obsElapsedS = (performance.now() - obsStart) / 1000;
    const choicesNextStart = performance.now();
    const nextChoices = buildPlacementChoices(
      this.game.state,
      DEFAULT_ACTION_DIM,
      this.actionCurriculum,
      () => nextFloat(this.planningRng),
    );
    const choicesNextElapsedS = (performance.now() - choicesNextStart) / 1000;
    this.cachedChoices = nextChoices;
    const totalElapsedS = (performance.now() - stepStart) / 1000;
    return {
      obs,
      actionMask: nextChoices.actionMask,
      actionBias: nextChoices.actionBiases,
      reward: Number.isFinite(finalReward) ? finalReward : 0,
      done: this.done,
      info: {
        modeId: this.modeId,
        piecesPlaced: this.piecesPlaced,
        lockObserved: this.lockCount > beforeLockCount,
        placementExecutionMode: this.placementExecutionMode,
        executionPath,
        ticks,
        heightDelta,
        holesDelta,
        bumpinessDelta,
        linesDelta,
        scoreDelta,
        timeDeltaMs,
        boardScoreDelta: before.boardScore - after.boardScore,
        rewardBase: reward,
        rewardFinal: finalReward,
        rewardTermLines: rewardResult.breakdown.linesTerm,
        rewardTermScore: rewardResult.breakdown.scoreTerm,
        rewardTermTime: rewardResult.breakdown.timeTerm,
        rewardTermHeight: rewardResult.breakdown.heightTerm,
        rewardTermHoles: rewardResult.breakdown.holeDeltaTerm,
        rewardTermBumpiness: rewardResult.breakdown.bumpinessDeltaTerm,
        rewardTermBoardScore: rewardResult.breakdown.boardScoreTerm,
        rewardTermBoardQuality: rewardResult.breakdown.boardQualityDeltaTerm,
        topOutPenalty,
        gameWon: this.game.state.gameWon,
        gameOver: this.game.state.gameOver,
      },
      completedSession,
      profile: {
        total_s: totalElapsedS,
        choices_current_s: choicesCurrentElapsedS,
        runner_s: runnerElapsedS,
        reward_s: rewardElapsedS,
        obs_s: obsElapsedS,
        choices_next_s: choicesNextElapsedS,
      },
    };
  }

  private executePlacementByCommands(
    commands: InputFrame[],
    beforeLockCount: number,
  ): number {
    let ticks = 0;
    const stepFrames = [...commands];
    while (
      !this.isTerminal() &&
      this.lockCount === beforeLockCount &&
      ticks < STEP_MAX_TICKS
    ) {
      const frame =
        stepFrames.length > 0
          ? (stepFrames.shift() ?? EMPTY_INPUT)
          : EMPTY_INPUT;
      this.runner.step(new OneFrameInputSource(frame));
      ticks += 1;
    }
    return ticks;
  }

  private executePlacementByTeleport(
    placement: TrajectoryExecutorReachablePlacement,
    beforeLockCount: number,
  ): {
    ticks: number;
    applied: boolean;
  } {
    let ticks = 0;

    if (placement.holdUsed) {
      if (!this.game.state.canHold) {
        return { ticks: 0, applied: false };
      }
      this.runner.step(
        new OneFrameInputSource({
          ...EMPTY_INPUT,
          hold: true,
        }),
      );
      ticks += 1;
      if (this.isTerminal() || this.lockCount > beforeLockCount) {
        return { ticks, applied: true };
      }
    }

    const target: GameState['active'] = {
      k: placement.lockPiece,
      r: clampRotation(placement.lockRotation),
      x: Math.trunc(placement.lockX),
      y: Math.trunc(placement.lockY),
    };

    if (collides(this.game.state.board, target, target.r, 0, 0)) {
      return { ticks, applied: false };
    }

    this.game.state.active = target;
    this.runner.step(
      new OneFrameInputSource({
        ...EMPTY_INPUT,
        hardDrop: true,
      }),
    );
    ticks += 1;
    return { ticks, applied: true };
  }

  private buildGame(seed: number): { game: Game; runner: GameRunner } {
    const mode = getMode(this.modeId);
    const merged = applyModeSettings(
      {
        ...DEFAULT_SETTINGS,
        butterfinger: {
          ...DEFAULT_SETTINGS.butterfinger,
          enabled: false,
        },
      },
      mode,
    );
    const generatorSettings =
      this.pieceSource === 'bag7'
        ? {
            ...merged.generator,
            type: 'bag7' as const,
          }
        : merged.generator;
    const modelRunner = createModelRunner({
      preferredBackend: 'native',
    }).runner;
    const game = new Game({
      seed,
      ...merged.game,
      // Offline PPO uses replay-style timing semantics: no passive gravity
      // progression and instant soft-drop action.
      gravityMs: OFFLINE_GRAVITY_MS,
      softDropMs: OFFLINE_SOFT_DROP_MS,
      lockNudgeRate: 0,
      gravityDropRate: 0,
      lockRotateRate: 0,
      lineGoal: mode.lineGoal,
      classicStartLevel: mode.classicStartLevel,
      scoringEnabled: mode.scoringEnabled,
      generatorFactory: createGeneratorFactory(generatorSettings, {
        mlModel: this.model,
        mlRunner: modelRunner,
        queuePolicyId: this.queuePolicyId,
      }),
      onPieceLock: () => {
        this.lockCount += 1;
      },
    });
    runModeStart(game, mode, {});
    return {
      game,
      runner: new GameRunner(game, { fixedStepMs: FIXED_STEP_MS }),
    };
  }

  private startEpisodeCapture(): void {
    this.episodeStartWallMs = Date.now();
    this.episodeSampleSerial = 0;
    this.episodeInitialState = {
      boardOccupancy: boardToOccupancy(this.game.state.board),
      hold: asPieceOrNull(this.game.state.hold),
      active: {
        k: this.game.state.active.k,
        r: clampRotation(this.game.state.active.r),
        x: Math.trunc(this.game.state.active.x),
        y: Math.trunc(this.game.state.active.y),
      },
      next: this.game.state.next.map((piece) => piece),
      canHold: this.game.state.canHold,
      timeMs: Math.max(0, Math.trunc(this.game.state.timeMs)),
      totalLinesCleared: Math.max(
        0,
        Math.trunc(this.game.state.totalLinesCleared),
      ),
      score: Math.max(0, Math.trunc(this.game.state.score)),
    };
    this.episodeSamples = [];
  }

  private captureEpisodeStep(input: {
    reward: number;
    after: {
      lines: number;
      score: number;
      timeMs: number;
      holes: number;
      boardScore: number;
    };
    selectedPlacement: TrajectoryExecutorReachablePlacement | null;
  }): void {
    const placement = input.selectedPlacement;
    if (!placement) return;
    const actionPiece = this.game.state.active.k;
    const actionIndex = pieceIndex(actionPiece);
    const probabilities = TRAJECTORY_PIECES.map((_, index) =>
      index === actionIndex ? 1 : 0,
    );
    const logits = probabilities.map((value) => (value > 0 ? 1 : 0));
    const createdAtMs =
      this.episodeStartWallMs + Math.max(0, Math.trunc(input.after.timeMs));
    const rewardValue = Number.isFinite(input.reward)
      ? Math.max(-1e9, Math.min(1e9, input.reward))
      : null;
    this.episodeSampleSerial += 1;
    const sampleIdCore = `${this.envId}_${this.episodeSampleSerial.toString().padStart(6, '0')}`;
    const sampleId =
      sampleIdCore.length >= TRAJECTORY_MIN_SAMPLE_ID
        ? sampleIdCore
        : sampleIdCore.padEnd(TRAJECTORY_MIN_SAMPLE_ID, '0');
    this.episodeSamples.push({
      id: sampleId,
      createdAtMs,
      deliberationMs: null,
      boardOccupancy: boardToOccupancy(this.game.state.board),
      hold: asPieceOrNull(this.game.state.hold),
      action: actionPiece,
      actionIndex,
      pieces: [...TRAJECTORY_PIECES],
      logits,
      probabilities,
      inferenceMs: 0,
      samplingMs: 0,
      totalDecisionMs: 0,
      reward: rewardValue,
      replay: {
        lockPiece: placement.lockPiece,
        lockRotation: clampRotation(placement.lockRotation),
        lockX: Math.trunc(placement.lockX),
        lockY: Math.trunc(placement.lockY),
        holdUsed: placement.holdUsed,
        gameTimeMs: Math.max(0, Math.trunc(input.after.timeMs)),
        totalLinesCleared: Math.max(0, Math.trunc(input.after.lines)),
        score: Math.max(0, Math.trunc(input.after.score)),
      },
    });
  }

  private finalizeEpisodeCapture(): JsonObject | null {
    if (!this.episodeInitialState || this.episodeSamples.length <= 0) {
      return null;
    }
    this.episodeSessionSerial += 1;
    const sessionCore =
      `bot_offline_env${this.envId}_` +
      `${this.episodeSessionSerial.toString().padStart(8, '0')}`;
    const sessionId =
      sessionCore.length >= 8 ? sessionCore : sessionCore.padEnd(8, '0');
    const durationMs = Math.max(0, Math.trunc(this.game.state.timeMs));
    const endedAtMs = this.episodeStartWallMs + durationMs;
    const outcome = this.game.state.gameWon
      ? 'game_won'
      : this.game.state.gameOver
        ? 'game_over'
        : this.piecesPlaced >= this.maxPiecesPerEpisode
          ? 'max_pieces'
          : 'manual';
    const session: JsonObject = {
      schema: TRAJECTORY_SCHEMA,
      sessionId,
      modeId: this.modeId,
      buildVersion: TRAJECTORY_BUILD_VERSION,
      startedAtMs: this.episodeStartWallMs,
      endedAtMs,
      durationMs,
      initialState: this.episodeInitialState,
      samples: this.episodeSamples.map((sample) => ({ ...sample })),
      meta: {
        outcome,
        actorType: 'bot',
        trainingIntent: 'offline_ppo_rollout',
        queuePolicyId: this.queuePolicyId,
        pieceSourceProfile: this.pieceSource,
        pipelineId: 'offline_ppo_v1',
        pipelineMode: this.modeId,
        generatorType: this.pieceSource === 'bag7' ? 'bag7' : 'ml',
      },
    };
    return session;
  }

  private snapshotMetrics(): {
    lines: number;
    score: number;
    timeMs: number;
    height: number;
    holes: number;
    bumpiness: number;
    boardScore: number;
    boardQuality: number;
  } {
    const state = this.game.state;
    const quality = evaluateBoardQuality(state.board);
    return {
      lines: Math.max(0, Math.trunc(state.totalLinesCleared)),
      score: Math.max(0, Math.trunc(state.score)),
      timeMs: Math.max(0, Math.trunc(state.timeMs)),
      height: getStackHeight(state.board),
      holes: countBoardHoles(state.board),
      bumpiness: computeBoardBumpiness(state.board),
      boardScore: scoreCharcuterieBoard(
        state.board,
        state.gameOver,
        state.totalLinesCleared,
      ),
      boardQuality: quality.quality,
    };
  }

  private syncPrevMetrics(): void {
    // Keeping a dedicated sync point allows future metrics extensions.
    this.snapshotMetrics();
  }

  private isTerminal(): boolean {
    return this.game.state.gameOver || this.game.state.gameWon;
  }
}

export class BotEnvPool {
  private readonly envs = new Map<number, BotEnv>();
  private readonly completedTrajectorySessions: JsonObject[] = [];
  private actionCurriculum: ActionCurriculumConfig = {
    ...DEFAULT_ACTION_CURRICULUM,
  };

  static async create(payload: InitPayload): Promise<BotEnvPool> {
    const modeId = normalizeModeId(payload.modeId);
    const numEnvs = clampInt(payload.numEnvs, 1, 1, 4096);
    const observationSpace = normalizeObservationSpace(
      payload.observationSpace,
    );
    const placementExecutionMode = normalizePlacementExecutionMode(
      payload.placementExecutionMode,
    );
    const pieceSourceProfile = normalizePieceSource(payload.pieceSourceProfile);
    const queuePolicyId =
      typeof payload.queuePolicyId === 'string' && payload.queuePolicyId.trim()
        ? payload.queuePolicyId.trim().toLowerCase()
        : 'next_piece_v1';
    const maxPiecesPerEpisode = clampInt(
      payload.maxPiecesPerEpisode,
      DEFAULT_MAX_PIECES,
      1,
      1_000_000,
    );
    const baseSeed = clampInt(payload.seed, Date.now(), 1, 0x7fffffff);
    const modelPath =
      typeof payload.modelPath === 'string' &&
      payload.modelPath.trim().length > 0
        ? payload.modelPath
        : 'public/models/model_v4.json';
    const jsonText = await readFile(modelPath, 'utf8');
    const model = parseWubModelFromJsonText(jsonText);

    const pool = new BotEnvPool();
    for (let i = 0; i < numEnvs; i += 1) {
      const seed = Math.max(1, baseSeed + i * 997);
      pool.envs.set(
        i,
        new BotEnv(
          i,
          modeId,
          model,
          observationSpace,
          pieceSourceProfile,
          queuePolicyId,
          maxPiecesPerEpisode,
          placementExecutionMode,
          seed,
        ),
      );
    }
    pool.setCurriculum(null);
    return pool;
  }

  resetMany(envIds: number[], seeds: number[]): StepBatchResult {
    const batchStart = performance.now();
    const obs: number[][] = [];
    const actionMasks: number[][] = [];
    const actionBiases: number[][] = [];
    const rewards: number[] = [];
    const dones: boolean[] = [];
    const infos: JsonObject[] = [];
    let resetEnvTotalS = 0;
    let resetObsS = 0;
    let resetChoicesS = 0;
    for (let i = 0; i < envIds.length; i += 1) {
      const envId = envIds[i];
      const env = this.requireEnv(envId);
      const seed = clampInt(seeds[i], Date.now() + envId * 911, 1, 0x7fffffff);
      const out = env.reset(seed);
      obs.push(out.obs);
      actionMasks.push(out.actionMask);
      actionBiases.push(out.actionBias);
      rewards.push(0);
      dones.push(false);
      infos.push(out.info);
      resetEnvTotalS += out.profile.total_s;
      resetObsS += out.profile.obs_s;
      resetChoicesS += out.profile.choices_s;
    }
    return {
      obs,
      action_masks: actionMasks,
      action_biases: actionBiases,
      rewards,
      dones,
      infos,
      profile: {
        batch_total_s: (performance.now() - batchStart) / 1000,
        env_count: envIds.length,
        reset_env_total_s: resetEnvTotalS,
        reset_obs_s: resetObsS,
        reset_choices_s: resetChoicesS,
      },
    };
  }

  stepMany(envIds: number[], actions: number[]): StepBatchResult {
    const batchStart = performance.now();
    const obs: number[][] = [];
    const actionMasks: number[][] = [];
    const actionBiases: number[][] = [];
    const rewards: number[] = [];
    const dones: boolean[] = [];
    const infos: JsonObject[] = [];
    let stepEnvTotalS = 0;
    let stepChoicesCurrentS = 0;
    let stepRunnerS = 0;
    let stepRewardS = 0;
    let stepObsS = 0;
    let stepChoicesNextS = 0;
    for (let i = 0; i < envIds.length; i += 1) {
      const envId = envIds[i];
      const env = this.requireEnv(envId);
      const action = clampInt(actions[i], 0, 0, DEFAULT_ACTION_DIM - 1);
      const out = env.step(action);
      obs.push(out.obs);
      actionMasks.push(out.actionMask);
      actionBiases.push(out.actionBias);
      rewards.push(out.reward);
      dones.push(out.done);
      infos.push(out.info);
      if (out.completedSession) {
        if (
          this.completedTrajectorySessions.length >=
          MAX_COMPLETED_TRAJECTORY_QUEUE
        ) {
          this.completedTrajectorySessions.shift();
        }
        this.completedTrajectorySessions.push(out.completedSession);
      }
      stepEnvTotalS += out.profile.total_s;
      stepChoicesCurrentS += out.profile.choices_current_s;
      stepRunnerS += out.profile.runner_s;
      stepRewardS += out.profile.reward_s;
      stepObsS += out.profile.obs_s;
      stepChoicesNextS += out.profile.choices_next_s;
    }
    return {
      obs,
      action_masks: actionMasks,
      action_biases: actionBiases,
      rewards,
      dones,
      infos,
      profile: {
        batch_total_s: (performance.now() - batchStart) / 1000,
        env_count: envIds.length,
        step_env_total_s: stepEnvTotalS,
        step_choices_current_s: stepChoicesCurrentS,
        step_runner_s: stepRunnerS,
        step_reward_s: stepRewardS,
        step_obs_s: stepObsS,
        step_choices_next_s: stepChoicesNextS,
      },
    };
  }

  listEnvIds(): number[] {
    return Array.from(this.envs.keys()).sort((a, b) => a - b);
  }

  setPieceSource(pieceSourceProfile: unknown): PieceSourceProfile {
    const normalized = normalizePieceSource(pieceSourceProfile);
    for (const env of this.envs.values()) {
      env.setPieceSource(normalized);
    }
    return normalized;
  }

  setCurriculum(
    payload: SetCurriculumPayload | null | undefined,
  ): ActionCurriculumConfig {
    this.actionCurriculum = normalizeActionCurriculum(payload);
    for (const env of this.envs.values()) {
      env.setActionCurriculum(this.actionCurriculum);
    }
    return { ...this.actionCurriculum };
  }

  popTrajectorySession(): JsonObject | null {
    return this.completedTrajectorySessions.shift() ?? null;
  }

  private requireEnv(id: number): BotEnv {
    const env = this.envs.get(id);
    if (!env) {
      throw new Error(`Unknown env id: ${id}`);
    }
    return env;
  }
}
