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
import { SPAWN_X, SPAWN_Y } from '../../../src/core/constants.ts';
import { collides, dropDistance } from '../../../src/core/piece.ts';
import {
  PLACEMENT_ACTION_DIM,
  PLACEMENT_ACTION_HOLD_STEP_DIM,
  PLACEMENT_ACTION_HOLD_STEP_INDEX,
  placementActionIndexFromFields,
  placementActionIndexFromNoHoldPlacement,
  placementActionIndexFromPlacement,
} from '../../../src/core/placementActionSpace.ts';
import { XorShift32 } from '../../../src/core/rng.ts';
import { GameRunner, type InputSource } from '../../../src/core/runner.ts';
import { DEFAULT_SETTINGS } from '../../../src/core/settings.ts';
import {
  PIECES,
  type ActivePiece,
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
  BotActionSpaceKind,
  InitPayload,
  JsonObject,
  PlacementExecutionMode,
  PieceSourceProfile,
  RewardFunctionId,
  SetCurriculumPayload,
  StepBatchResult,
} from './protocol.ts';

const FIXED_STEP_MS = 1000 / 120;
const DEFAULT_MAX_PIECES = 512;
const STEP_MAX_TICKS = 120;
const OFFLINE_GRAVITY_MS = Number.POSITIVE_INFINITY;
const OFFLINE_SOFT_DROP_MS = 0;
const TOP_OUT_PENALTY = 100;
const DEFAULT_REWARD_BLEND_TIMESTEPS = 10_000_000;
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
  value === 'active_generator'
    ? 'active_generator'
    : value === 'random'
      ? 'random'
      : 'bag7';

const normalizeObservationSpace = (value: unknown): BotObservationSpace =>
  normalizeBotObservationSpace(typeof value === 'string' ? value : null);

const normalizePlacementExecutionMode = (
  value: unknown,
): PlacementExecutionMode => (value === 'commands' ? 'commands' : 'teleport');

const normalizeActionSpaceKind = (value: unknown): BotActionSpaceKind =>
  value === 'placement_hold_step_v2'
    ? 'placement_hold_step_v2'
    : 'placement_full_v1';

const actionDimForActionSpaceKind = (kind: BotActionSpaceKind): number =>
  kind === 'placement_hold_step_v2'
    ? PLACEMENT_ACTION_HOLD_STEP_DIM
    : PLACEMENT_ACTION_DIM;

const normalizeActionCurriculum = (
  payload: SetCurriculumPayload | null | undefined,
): ActionCurriculumConfig => {
  const topK = clampInt(payload?.topK, 0, 0, PLACEMENT_ACTION_DIM);
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

const normalizeRewardFunctionId = (
  value: unknown,
  fallback: RewardFunctionId,
): RewardFunctionId => {
  if (value === 'v1' || value === 'v2' || value === 'v3') return value;
  return fallback;
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
  boardQualityAbsoluteTerm: number;
  fullClearTerm: number;
  topOutTerm: number;
};

type PieceRewardInputs = {
  modeId: string;
  linesDelta: number;
  scoreDelta: number;
  timeDeltaMs: number;
  heightDelta: number;
  holesDelta: number;
  bumpinessDelta: number;
  boardScoreDelta: number;
  boardQualityDelta: number;
  topOut: boolean;
};

type PieceRewardResult = {
  reward: number;
  breakdown: PieceRewardBreakdown;
};

type RewardBlendWeights = {
  legacyWeight: number;
  targetWeight: number;
  t: number;
  transitionStep: number;
  transitionTotalSteps: number;
};

type RewardBlendUnit = 'timesteps' | 'updates';

const normalizeRewardBlendWeights = (
  input: RewardBlendWeights | null | undefined,
): RewardBlendWeights => {
  const rawLegacy =
    typeof input?.legacyWeight === 'number' &&
    Number.isFinite(input.legacyWeight)
      ? input.legacyWeight
      : 1;
  const rawTarget =
    typeof input?.targetWeight === 'number' &&
    Number.isFinite(input.targetWeight)
      ? input.targetWeight
      : 0;
  const sum = rawLegacy + rawTarget;
  const legacyRatio =
    !Number.isFinite(sum) || sum <= 1e-9 ? 1 : rawLegacy / sum;
  const t = Math.max(0, Math.min(1, legacyRatio));
  return {
    legacyWeight: t,
    targetWeight: 1 - t,
    t,
    transitionStep: Math.max(0, Math.trunc(input?.transitionStep ?? 0)),
    transitionTotalSteps: Math.max(
      1,
      Math.trunc(input?.transitionTotalSteps ?? DEFAULT_REWARD_BLEND_TIMESTEPS),
    ),
  };
};

const blendPieceReward = (options: {
  legacy: PieceRewardResult;
  target: PieceRewardResult;
  weights: RewardBlendWeights;
}): PieceRewardResult => {
  const { legacy, target, weights } = options;
  const legacyW = weights.legacyWeight;
  const targetW = weights.targetWeight;
  const mix = (a: number, b: number): number => a * legacyW + b * targetW;
  return {
    reward: mix(legacy.reward, target.reward),
    breakdown: {
      linesTerm: mix(legacy.breakdown.linesTerm, target.breakdown.linesTerm),
      scoreTerm: mix(legacy.breakdown.scoreTerm, target.breakdown.scoreTerm),
      timeTerm: mix(legacy.breakdown.timeTerm, target.breakdown.timeTerm),
      heightTerm: mix(legacy.breakdown.heightTerm, target.breakdown.heightTerm),
      holeDeltaTerm: mix(
        legacy.breakdown.holeDeltaTerm,
        target.breakdown.holeDeltaTerm,
      ),
      bumpinessDeltaTerm: mix(
        legacy.breakdown.bumpinessDeltaTerm,
        target.breakdown.bumpinessDeltaTerm,
      ),
      boardScoreTerm: mix(
        legacy.breakdown.boardScoreTerm,
        target.breakdown.boardScoreTerm,
      ),
      boardQualityDeltaTerm: mix(
        legacy.breakdown.boardQualityDeltaTerm,
        target.breakdown.boardQualityDeltaTerm,
      ),
      boardQualityAbsoluteTerm: mix(
        legacy.breakdown.boardQualityAbsoluteTerm,
        target.breakdown.boardQualityAbsoluteTerm,
      ),
      fullClearTerm: mix(
        legacy.breakdown.fullClearTerm,
        target.breakdown.fullClearTerm,
      ),
      topOutTerm: mix(legacy.breakdown.topOutTerm, target.breakdown.topOutTerm),
    },
  };
};

const computePieceRewardV1 = (
  options: PieceRewardInputs,
): PieceRewardResult => {
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
    topOut,
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
  const topOutTerm = topOut ? -TOP_OUT_PENALTY : 0;
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
    bumpinessDeltaTerm +
    topOutTerm;
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
      boardQualityAbsoluteTerm: 0,
      fullClearTerm: 0,
      topOutTerm,
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
const PLACEMENT_COMPLEXITY_ROT_CW_CCW_WEIGHT = 0.03;
const PLACEMENT_COMPLEXITY_ROT_180_WEIGHT = 0.1;
const PLACEMENT_COMPLEXITY_SRS_KICK_WEIGHT = 0.1;
const PLACEMENT_COMPLEXITY_SOFT_DROP_WEIGHT = 0.005;
const PLACEMENT_HOLD_COMPLEXITY_PENALTY = 0.08;
const PLACEMENT_TOP_OUT_PENALTY = 5.0;
const REWARD_V2_LINE_WEIGHT = 0.5;
const REWARD_V2_COMPLEXITY_WEIGHT = 0.5;
const REWARD_V2_BOARD_QUALITY_WEIGHT = 0.2;
const REWARD_V2_HOLE_REMOVE_WEIGHT = 0.05;
const REWARD_V2_HOLE_CREATE_WEIGHT = REWARD_V2_HOLE_REMOVE_WEIGHT * 5;

const REWARD_V3_LINE_WEIGHT = 0.2;
const REWARD_V3_COMPLEXITY_WEIGHT = 0.35;
const REWARD_V3_BOARD_DELTA_WEIGHT = 0.15;
const REWARD_V3_BOARD_ABSOLUTE_WEIGHT = 0.005;
const REWARD_V3_HOLE_CREATE_WEIGHT = 0.3;
const REWARD_V3_HOLE_REMOVE_WEIGHT = 0.0;
const REWARD_V3_DANGER_HEIGHT = 14;
const REWARD_V3_DANGER_WEIGHT = 0.04;
const REWARD_V3_FULL_CLEAR_BONUS = 8.0;

const computePlacementComplexityPenalty = (
  placement: Pick<
    TrajectoryExecutorReachablePlacement,
    'commands' | 'holdUsed' | 'srsKickCount'
  > | null,
): number => {
  if (!placement) return 0;
  let rotateCwCcwCount = 0;
  let rotate180Count = 0;
  let softDropCount = 0;
  for (const command of placement.commands) {
    if (command === 'rotate_cw' || command === 'rotate_ccw') {
      rotateCwCcwCount += 1;
    } else if (command === 'rotate_180') {
      rotate180Count += 1;
    } else if (command === 'soft_drop') {
      softDropCount += 1;
    }
  }
  const srsKickCount = Math.max(0, Math.trunc(placement.srsKickCount ?? 0));
  return (
    placement.commands.length * PLACEMENT_COMPLEXITY_CMD_WEIGHT +
    rotateCwCcwCount * PLACEMENT_COMPLEXITY_ROT_CW_CCW_WEIGHT +
    rotate180Count * PLACEMENT_COMPLEXITY_ROT_180_WEIGHT +
    srsKickCount * PLACEMENT_COMPLEXITY_SRS_KICK_WEIGHT +
    softDropCount * PLACEMENT_COMPLEXITY_SOFT_DROP_WEIGHT +
    (placement.holdUsed ? PLACEMENT_HOLD_COMPLEXITY_PENALTY : 0)
  );
};

const computePlacementComplexityPenaltyWithoutHoldTax = (
  placement: Pick<
    TrajectoryExecutorReachablePlacement,
    'commands' | 'holdUsed' | 'srsKickCount'
  > | null,
): number => {
  const penalty = computePlacementComplexityPenalty(placement);
  if (!placement?.holdUsed) return penalty;
  return Math.max(0, penalty - PLACEMENT_HOLD_COMPLEXITY_PENALTY);
};

const computeLineClearScoreDelta = (
  linesCleared: number,
  level: number,
  scoringEnabled: boolean,
): number => {
  if (!scoringEnabled || linesCleared <= 0) return 0;
  const multiplier = Math.max(0, Math.trunc(level)) + 1;
  switch (linesCleared) {
    case 1:
      return 40 * multiplier;
    case 2:
      return 100 * multiplier;
    case 3:
      return 300 * multiplier;
    case 4:
      return 1200 * multiplier;
    default:
      return 0;
  }
};

const zeroRewardBreakdown = (): PieceRewardResult['breakdown'] => ({
  linesTerm: 0,
  scoreTerm: 0,
  timeTerm: 0,
  heightTerm: 0,
  holeDeltaTerm: 0,
  bumpinessDeltaTerm: 0,
  boardScoreTerm: 0,
  boardQualityDeltaTerm: 0,
  boardQualityAbsoluteTerm: 0,
  fullClearTerm: 0,
  topOutTerm: 0,
});

const computeHoldStepRewardV1 = (
  modeId: string,
  timeDeltaMs: number,
): PieceRewardResult => {
  let timeTerm = 0;
  if (modeId === 'sprint') {
    timeTerm = -(timeDeltaMs / 4000);
  } else if (modeId === 'charcuterie') {
    timeTerm = -(timeDeltaMs / 25000);
  } else if (modeId === 'practice') {
    timeTerm = -(timeDeltaMs * PRACTICE_TIME_DELTA_REWARD_WEIGHT);
  }
  return {
    reward: timeTerm,
    breakdown: {
      ...zeroRewardBreakdown(),
      timeTerm,
    },
  };
};

const computeHoldStepRewardV2 = (): PieceRewardResult => {
  const timeTerm =
    -PLACEMENT_HOLD_COMPLEXITY_PENALTY * REWARD_V2_COMPLEXITY_WEIGHT;
  return {
    reward: timeTerm,
    breakdown: {
      ...zeroRewardBreakdown(),
      timeTerm,
    },
  };
};

const computeHoldStepRewardV3 = (): PieceRewardResult => {
  const timeTerm =
    -PLACEMENT_HOLD_COMPLEXITY_PENALTY * REWARD_V3_COMPLEXITY_WEIGHT;
  return {
    reward: timeTerm,
    breakdown: {
      ...zeroRewardBreakdown(),
      timeTerm,
    },
  };
};

const computePieceRewardV2 = (
  options: PieceRewardInputs & {
    placementComplexityPenalty: number;
  },
): PieceRewardResult => {
  const {
    linesDelta,
    boardQualityDelta,
    holesDelta,
    placementComplexityPenalty,
    topOut,
  } = options;
  const clampedLines = Math.max(0, linesDelta);
  const linesTerm = Math.pow(clampedLines, 1.5) * REWARD_V2_LINE_WEIGHT;
  const boardQualityDeltaTerm =
    boardQualityDelta * REWARD_V2_BOARD_QUALITY_WEIGHT;
  const timeTerm = -placementComplexityPenalty * REWARD_V2_COMPLEXITY_WEIGHT;
  const holeDeltaTerm =
    holesDelta >= 0
      ? -holesDelta * REWARD_V2_HOLE_CREATE_WEIGHT
      : -holesDelta * REWARD_V2_HOLE_REMOVE_WEIGHT;
  const topOutTerm = topOut ? -TOP_OUT_PENALTY : 0;
  const reward =
    linesTerm + boardQualityDeltaTerm + timeTerm + holeDeltaTerm + topOutTerm;
  return {
    reward,
    breakdown: {
      linesTerm,
      scoreTerm: 0,
      timeTerm,
      heightTerm: 0,
      holeDeltaTerm,
      bumpinessDeltaTerm: 0,
      boardScoreTerm: 0,
      boardQualityDeltaTerm,
      boardQualityAbsoluteTerm: 0,
      fullClearTerm: 0,
      topOutTerm,
    },
  };
};

const computePieceRewardV3 = (
  options: PieceRewardInputs & {
    placementComplexityPenalty: number;
    afterMaxHeight: number;
    afterBoardQuality: number;
    isFullClear: boolean;
  },
): PieceRewardResult => {
  const {
    linesDelta,
    boardQualityDelta,
    holesDelta,
    placementComplexityPenalty,
    afterMaxHeight,
    afterBoardQuality,
    isFullClear,
    topOut,
  } = options;
  const clampedLines = Math.max(0, linesDelta);
  const linesTerm = Math.pow(clampedLines, 2) * REWARD_V3_LINE_WEIGHT;
  const boardQualityDeltaTerm =
    boardQualityDelta * REWARD_V3_BOARD_DELTA_WEIGHT;
  const boardQualityAbsoluteTerm =
    -Math.max(0, afterBoardQuality) * REWARD_V3_BOARD_ABSOLUTE_WEIGHT;
  const holeDeltaTerm =
    holesDelta > 0
      ? -holesDelta * REWARD_V3_HOLE_CREATE_WEIGHT
      : -Math.min(-holesDelta, 1) * REWARD_V3_HOLE_REMOVE_WEIGHT;
  const danger = Math.max(0, afterMaxHeight - REWARD_V3_DANGER_HEIGHT);
  const heightTerm = -(danger * danger) * REWARD_V3_DANGER_WEIGHT;
  const timeTerm = -placementComplexityPenalty * REWARD_V3_COMPLEXITY_WEIGHT;
  const fullClearTerm = isFullClear ? REWARD_V3_FULL_CLEAR_BONUS : 0;
  const topOutTerm = topOut ? -TOP_OUT_PENALTY : 0;
  const reward =
    linesTerm +
    boardQualityDeltaTerm +
    boardQualityAbsoluteTerm +
    holeDeltaTerm +
    heightTerm +
    timeTerm +
    fullClearTerm +
    topOutTerm;
  return {
    reward,
    breakdown: {
      linesTerm,
      scoreTerm: 0,
      timeTerm,
      heightTerm,
      holeDeltaTerm,
      bumpinessDeltaTerm: 0,
      boardScoreTerm: 0,
      boardQualityDeltaTerm,
      boardQualityAbsoluteTerm,
      fullClearTerm,
      topOutTerm,
    },
  };
};

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
  const complexityPenalty = computePlacementComplexityPenalty(placement);

  let score =
    improvement +
    applied.linesCleared * PLACEMENT_LINE_CLEAR_BONUS -
    complexityPenalty;
  if (applied.hasAboveTop) score -= PLACEMENT_TOP_OUT_PENALTY;
  if (!Number.isFinite(score)) score = -1e9;
  return score;
};

const padVisibleQueue = (
  pieces: Array<PieceKind | null>,
  length: number = 5,
): Array<PieceKind | null> => {
  const out = pieces.slice(0, length);
  while (out.length < length) out.push(null);
  return out;
};

const buildPostLockStateForPlacement = (options: {
  state: GameState;
  boardAfter: Board;
  placement: TrajectoryExecutorReachablePlacement;
  linesCleared: number;
  topOut: boolean;
}): {
  done: boolean;
  topOut: boolean;
  state: GameState | null;
  scoreDelta: number;
  totalLinesClearedAfter: number;
} => {
  const { state, boardAfter, placement, linesCleared } = options;
  let topOut = options.topOut;
  const totalLinesClearedAfter = Math.max(
    0,
    Math.trunc(state.totalLinesCleared) + Math.max(0, Math.trunc(linesCleared)),
  );
  const scoreDelta = computeLineClearScoreDelta(
    linesCleared,
    Math.max(0, Math.trunc(state.level)),
    Boolean(state.scoringEnabled),
  );
  const lineGoal =
    state.lineGoal != null ? Math.max(1, Math.trunc(state.lineGoal)) : null;
  const gameWon = lineGoal != null && totalLinesClearedAfter >= lineGoal;
  if (topOut || gameWon) {
    return {
      done: true,
      topOut,
      state: null,
      scoreDelta,
      totalLinesClearedAfter,
    };
  }

  let holdAfter = state.hold;
  let activeAfter: PieceKind | null = null;
  let previewAfter: Array<PieceKind | null> = [];
  if (placement.holdUsed) {
    holdAfter = state.active.k;
    if (state.hold == null) {
      activeAfter = state.next[1] ?? null;
      previewAfter = padVisibleQueue(state.next.slice(2, 7));
    } else {
      activeAfter = state.next[0] ?? null;
      previewAfter = padVisibleQueue(state.next.slice(1, 6));
    }
  } else {
    holdAfter = state.hold;
    activeAfter = state.next[0] ?? null;
    previewAfter = padVisibleQueue(state.next.slice(1, 6));
  }

  if (activeAfter == null) {
    return {
      done: true,
      topOut,
      state: null,
      scoreDelta,
      totalLinesClearedAfter,
    };
  }

  const spawnedActive: ActivePiece = {
    k: activeAfter,
    r: 0,
    x: SPAWN_X,
    y: SPAWN_Y,
  };
  if (collides(boardAfter, spawnedActive)) {
    topOut = true;
    return {
      done: true,
      topOut,
      state: null,
      scoreDelta,
      totalLinesClearedAfter,
    };
  }

  return {
    done: false,
    topOut,
    scoreDelta,
    totalLinesClearedAfter,
    state: {
      board: cloneBoard(boardAfter),
      active: spawnedActive,
      ghostY: spawnedActive.y + dropDistance(boardAfter, spawnedActive),
      hold: holdAfter,
      canHold: true,
      next: previewAfter
        .filter((piece): piece is PieceKind => piece != null)
        .slice(0, 5),
      mlQueueProbabilities: [],
      gameOver: false,
      gameWon: false,
      combo: linesCleared > 0 ? state.combo + 1 : 0,
      timeMs: Math.max(0, Math.trunc(state.timeMs)),
      totalLinesCleared: totalLinesClearedAfter,
      lineGoal,
      level: Math.max(1, Math.trunc(state.level)),
      score: Math.max(0, Math.trunc(state.score) + scoreDelta),
      scoringEnabled: Boolean(state.scoringEnabled),
    },
  };
};

const encodeObservation = (
  observationSpace: BotObservationSpace,
  model: LoadedModel,
  state: GameState,
  includePhaseContext: boolean,
): number[] =>
  Array.from(
    encodeBotObservation({
      observationSpace,
      model,
      state,
      includePhaseContext,
    }),
  );

const buildPlacementChoices = (
  state: GameState,
  actionDim: number,
  actionSpaceKind: BotActionSpaceKind,
  curriculum: ActionCurriculumConfig,
  random: (() => number) | null = null,
): {
  commandsBySlot: Array<InputFrame[] | null>;
  placementsBySlot: Array<TrajectoryExecutorReachablePlacement | null>;
  actionMask: number[];
  actionBiases: number[];
  actionScores: number[];
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
  const actionScores = new Array<number>(actionDim).fill(0);
  const beforeMetrics = evaluateBoardQuality(state.board);
  let bestHoldActionScore = Number.NEGATIVE_INFINITY;
  for (const placement of placements) {
    const actionIndex =
      actionSpaceKind === 'placement_hold_step_v2'
        ? placement.holdUsed
          ? null
          : placementActionIndexFromNoHoldPlacement(placement)
        : placementActionIndexFromPlacement(placement);
    if (actionIndex == null || actionIndex < 0 || actionIndex >= actionDim) {
      if (actionSpaceKind === 'placement_hold_step_v2' && placement.holdUsed) {
        const score = scorePlacementCandidate({
          beforeMetrics,
          placement,
          board: state.board,
        });
        if (score > bestHoldActionScore) {
          bestHoldActionScore = score;
        }
      }
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
    actionScores[actionIndex] = Number.isFinite(score) ? score : 0;
    placementsBySlot[actionIndex] = {
      ...placement,
      commands: [...placement.commands],
    };
    commandsBySlot[actionIndex] = placement.commands.map((command) =>
      trajectoryExecutorCommandToInputFrame(command),
    );
  }
  if (
    actionSpaceKind === 'placement_hold_step_v2' &&
    state.canHold &&
    actionDim > PLACEMENT_ACTION_HOLD_STEP_INDEX &&
    Number.isFinite(bestHoldActionScore)
  ) {
    actionMask[PLACEMENT_ACTION_HOLD_STEP_INDEX] = 1;
    scoresBySlot[PLACEMENT_ACTION_HOLD_STEP_INDEX] = bestHoldActionScore;
    actionScores[PLACEMENT_ACTION_HOLD_STEP_INDEX] = bestHoldActionScore;
    placementsBySlot[PLACEMENT_ACTION_HOLD_STEP_INDEX] = null;
    commandsBySlot[PLACEMENT_ACTION_HOLD_STEP_INDEX] = [
      trajectoryExecutorCommandToInputFrame('hold'),
    ];
  }
  if (actionMask.every((value) => value <= 0)) {
    const fallbackLockY =
      state.active.y + dropDistance(state.board, state.active);
    const fallbackIndex =
      actionSpaceKind === 'placement_hold_step_v2'
        ? (placementActionIndexFromFields({
            holdUsed: false,
            lockRotation: Math.max(0, Math.min(3, Math.trunc(state.active.r))),
            lockX: Math.trunc(state.active.x),
            lockY: Math.trunc(fallbackLockY),
          }) ?? 0)
        : 0;
    const clampedFallbackIndex = Math.max(
      0,
      Math.min(actionDim - 1, fallbackIndex),
    );
    actionMask[clampedFallbackIndex] = 1;
    scoresBySlot[clampedFallbackIndex] = 0;
    actionScores[clampedFallbackIndex] = 0;
    placementsBySlot[clampedFallbackIndex] = {
      lockPiece: state.active.k,
      lockRotation: Math.max(0, Math.min(3, Math.trunc(state.active.r))),
      lockX: Math.trunc(state.active.x),
      lockY: Math.trunc(fallbackLockY),
      holdUsed: false,
      srsKickCount: 0,
      commands: ['hard_drop'],
      searchDepth: 0,
    };
    commandsBySlot[clampedFallbackIndex] = [
      trajectoryExecutorCommandToInputFrame('hard_drop'),
    ];
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
    actionScores[fallbackIndex] = Number.isFinite(scoresBySlot[fallbackIndex])
      ? scoresBySlot[fallbackIndex]
      : 0;
  }
  return {
    commandsBySlot,
    placementsBySlot,
    actionMask,
    actionBiases,
    actionScores,
  };
};

const boardToOccupancy = (board: Board): number[][] =>
  board.map((row) => row.map((cell) => (cell != null ? 1 : 0)));

const occupancyToBoard = (occupancy: number[][]): Board =>
  occupancy.map((row) => row.map((cell) => (cell > 0 ? 'I' : null)));

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
  actionScores: number[];
  reward: number;
  done: boolean;
  info: JsonObject;
  profile: BotEnvStepProfile;
  completedSession: JsonObject | null;
};

type HoldCandidateEvaluation = {
  action_index: number;
  hold_used: boolean;
  immediate_reward_no_hold_tax: number;
  done: boolean;
  obs: number[];
};

class BotEnv {
  private game: Game;
  private runner: GameRunner;
  private pieceSource: PieceSourceProfile;
  private done = false;
  private piecesPlaced = 0;
  private lockCount = 0;
  private pendingHoldUsedForCurrentPiece = false;
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
    actionScores: number[];
  } | null = null;

  constructor(
    private readonly envId: number,
    private readonly modeId: string,
    private readonly model: LoadedModel,
    private readonly observationSpace: BotObservationSpace,
    private readonly includePhaseContext: boolean,
    private readonly actionSpaceKind: BotActionSpaceKind,
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
      this.actionDim,
      this.actionSpaceKind,
      this.actionCurriculum,
      () => nextFloat(this.planningRng),
    );
    this.startEpisodeCapture();
    this.syncPrevMetrics();
  }

  reset(
    seed: number,
    initialBoardOccupancy?: number[][] | null,
  ): {
    obs: number[];
    actionMask: number[];
    actionBias: number[];
    actionScores: number[];
    info: JsonObject;
    profile: BotEnvResetProfile;
  } {
    const resetStart = performance.now();
    this.planningRng = new XorShift32(seed ^ 0x71e9135b);
    let built = this.buildGame(seed);
    let initialBoardAugmented = false;
    let initialBoardFilledCells = 0;
    const boardRows = built.game.state.board.length;
    const boardCols = built.game.state.board[0]?.length ?? 0;
    if (
      Array.isArray(initialBoardOccupancy) &&
      initialBoardOccupancy.length === boardRows &&
      initialBoardOccupancy.every(
        (row) => Array.isArray(row) && row.length === boardCols,
      )
    ) {
      const occupancy = initialBoardOccupancy.map((row) =>
        row.map((cell) => (cell > 0 ? 1 : 0)),
      );
      const board = occupancyToBoard(occupancy);
      const filledCells = countBoardBlocks(board);
      built.game.applyInitialBoard(board);
      const augmentedChoices = buildPlacementChoices(
        built.game.state,
        this.actionDim,
        this.actionSpaceKind,
        this.actionCurriculum,
        () => nextFloat(this.planningRng),
      );
      if (
        !built.game.state.gameOver &&
        augmentedChoices.actionMask.some((value) => value > 0)
      ) {
        this.cachedChoices = augmentedChoices;
        initialBoardAugmented = true;
        initialBoardFilledCells = filledCells;
      } else {
        built = this.buildGame(seed);
        this.cachedChoices = null;
      }
    } else {
      this.cachedChoices = null;
    }
    this.game = built.game;
    this.runner = built.runner;
    this.done = false;
    this.piecesPlaced = 0;
    this.lockCount = 0;
    this.pendingHoldUsedForCurrentPiece = false;
    this.startEpisodeCapture();
    const choicesStart = performance.now();
    if (this.cachedChoices == null) {
      this.cachedChoices = buildPlacementChoices(
        this.game.state,
        this.actionDim,
        this.actionSpaceKind,
        this.actionCurriculum,
        () => nextFloat(this.planningRng),
      );
    }
    const choicesElapsedS = (performance.now() - choicesStart) / 1000;
    this.syncPrevMetrics();
    const obsStart = performance.now();
    const obs = encodeObservation(
      this.observationSpace,
      this.model,
      this.game.state,
      this.includePhaseContext,
    );
    const obsElapsedS = (performance.now() - obsStart) / 1000;
    const choices = this.cachedChoices;
    const totalElapsedS = (performance.now() - resetStart) / 1000;
    return {
      obs,
      actionMask: choices?.actionMask ?? [],
      actionBias: choices?.actionBiases ?? [],
      actionScores: choices?.actionScores ?? [],
      info: {
        modeId: this.modeId,
        seed,
        initialBoardAugmented,
        initialBoardFilledCells,
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
      this.actionDim,
      this.actionSpaceKind,
      this.actionCurriculum,
      () => nextFloat(this.planningRng),
    );
  }

  private get actionDim(): number {
    return actionDimForActionSpaceKind(this.actionSpaceKind);
  }

  evaluateHoldCandidates(
    rewardBlend: RewardBlendWeights,
    rewardFunctionFrom: RewardFunctionId,
    rewardFunctionTo: RewardFunctionId,
  ): HoldCandidateEvaluation[] {
    if (this.done) {
      return [];
    }
    const choices = buildPlacementChoices(
      this.game.state,
      PLACEMENT_ACTION_DIM,
      'placement_full_v1',
      this.actionCurriculum,
      () => nextFloat(this.planningRng),
    );
    const blendWeights = normalizeRewardBlendWeights(rewardBlend);
    const before = this.snapshotMetrics();
    const candidates: HoldCandidateEvaluation[] = [];
    for (
      let actionIndex = 0;
      actionIndex < choices.actionMask.length;
      actionIndex += 1
    ) {
      if (choices.actionMask[actionIndex] <= 0) continue;
      const placement = choices.placementsBySlot[actionIndex];
      if (!placement) continue;
      const applied = applyPlacementToBoard(this.game.state.board, placement);
      if (applied.invalid) continue;
      const postLock = buildPostLockStateForPlacement({
        state: this.game.state,
        boardAfter: applied.boardAfter,
        placement,
        linesCleared: applied.linesCleared,
        topOut: applied.hasAboveTop,
      });
      const afterQuality = evaluateBoardQuality(applied.boardAfter);
      const afterHeight = getStackHeight(applied.boardAfter);
      const afterHoles = countBoardHoles(applied.boardAfter);
      const afterBumpiness = computeBoardBumpiness(applied.boardAfter);
      const afterBlocks = countBoardBlocks(applied.boardAfter);
      const placementComplexityPenalty =
        computePlacementComplexityPenaltyWithoutHoldTax(placement);
      const timeDeltaMs = Math.max(
        0,
        Math.trunc(placement.commands.length * FIXED_STEP_MS),
      );
      const linesDelta = applied.linesCleared;
      const scoreDelta = postLock.scoreDelta;
      const heightDelta = afterHeight - before.height;
      const holesDelta = afterHoles - before.holes;
      const bumpinessDelta = afterBumpiness - before.bumpiness;
      const boardQualityDelta = before.boardQuality - afterQuality.quality;
      const boardScoreDelta =
        before.boardScore -
        scoreCharcuterieBoard(
          applied.boardAfter,
          postLock.topOut,
          postLock.totalLinesClearedAfter,
        );
      const isFullClear = afterBlocks === 0 && before.blocks > 0;
      const rewardV1 = computePieceRewardV1({
        modeId: this.modeId,
        linesDelta,
        scoreDelta,
        timeDeltaMs,
        heightDelta,
        holesDelta,
        bumpinessDelta,
        boardScoreDelta,
        boardQualityDelta,
        topOut: postLock.topOut,
      });
      const rewardV2 = computePieceRewardV2({
        modeId: this.modeId,
        linesDelta,
        scoreDelta,
        timeDeltaMs,
        heightDelta,
        holesDelta,
        bumpinessDelta,
        boardScoreDelta,
        boardQualityDelta,
        topOut: postLock.topOut,
        placementComplexityPenalty,
      });
      const rewardV3 = computePieceRewardV3({
        modeId: this.modeId,
        linesDelta,
        scoreDelta,
        timeDeltaMs,
        heightDelta,
        holesDelta,
        bumpinessDelta,
        boardScoreDelta,
        boardQualityDelta,
        topOut: postLock.topOut,
        placementComplexityPenalty,
        afterMaxHeight: afterHeight,
        afterBoardQuality: afterQuality.quality,
        isFullClear,
      });
      const rewardById: Record<RewardFunctionId, PieceRewardResult> = {
        v1: rewardV1,
        v2: rewardV2,
        v3: rewardV3,
      };
      const rewardResult = blendPieceReward({
        legacy: rewardById[rewardFunctionFrom],
        target: rewardById[rewardFunctionTo],
        weights: blendWeights,
      });
      candidates.push({
        action_index: actionIndex,
        hold_used: placement.holdUsed,
        immediate_reward_no_hold_tax: Number.isFinite(rewardResult.reward)
          ? rewardResult.reward
          : 0,
        done: postLock.done,
        obs: postLock.state
          ? encodeObservation(
              this.observationSpace,
              this.model,
              postLock.state,
              this.includePhaseContext,
            )
          : [],
      });
    }
    return candidates;
  }

  step(
    actionIndexRaw: number,
    rewardBlend: RewardBlendWeights | null = null,
    rewardFunctionFrom: RewardFunctionId = 'v1',
    rewardFunctionTo: RewardFunctionId = 'v2',
  ): BotEnvStepResult {
    const stepStart = performance.now();
    const blendWeights = normalizeRewardBlendWeights(rewardBlend);
    if (this.done) {
      const doneChoicesStart = performance.now();
      const obs = encodeObservation(
        this.observationSpace,
        this.model,
        this.game.state,
        this.includePhaseContext,
      );
      const choices =
        this.cachedChoices ??
        buildPlacementChoices(
          this.game.state,
          this.actionDim,
          this.actionSpaceKind,
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
        actionScores: choices.actionScores,
        reward: 0,
        done: true,
        info: {
          alreadyDone: true,
          piecesPlaced: this.piecesPlaced,
          rewardLegacyFunctionId: rewardFunctionFrom,
          rewardTargetFunctionId: rewardFunctionTo,
          rewardBlendT: blendWeights.t,
          rewardBlendLegacyWeight: blendWeights.legacyWeight,
          rewardBlendTargetWeight: blendWeights.targetWeight,
          rewardBlendTransitionStep: blendWeights.transitionStep,
          rewardBlendTransitionTotalSteps: blendWeights.transitionTotalSteps,
        },
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
        this.actionDim,
        this.actionSpaceKind,
        this.actionCurriculum,
        () => nextFloat(this.planningRng),
      );
    this.cachedChoices = choices;
    const choicesCurrentElapsedS =
      (performance.now() - choicesCurrentStart) / 1000;
    const actionIndex = clampInt(actionIndexRaw, 0, 0, this.actionDim - 1);
    const hasAction = choices.actionMask[actionIndex] > 0;
    const resolvedActionIndex = hasAction
      ? actionIndex
      : Math.max(
          0,
          choices.actionMask.findIndex((x) => x > 0),
        );
    const selectedPlacement = choices.placementsBySlot[resolvedActionIndex];
    const isHoldStepAction =
      this.actionSpaceKind === 'placement_hold_step_v2' &&
      resolvedActionIndex === PLACEMENT_ACTION_HOLD_STEP_INDEX &&
      choices.actionMask[resolvedActionIndex] > 0;
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
        ticks += this.executePlacementByCommands(
          commands,
          beforeLockCount,
          isHoldStepAction,
        );
      }
    } else {
      ticks += this.executePlacementByCommands(
        commands,
        beforeLockCount,
        isHoldStepAction,
      );
    }

    if (
      !isHoldStepAction &&
      this.lockCount === beforeLockCount &&
      !this.isTerminal()
    ) {
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
    } else if (isHoldStepAction) {
      this.pendingHoldUsedForCurrentPiece = true;
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
    const isFullClear = after.blocks === 0 && before.blocks > 0;
    const rewardPlacement =
      isHoldStepAction && selectedPlacement == null
        ? ({
            commands: ['hold'],
            holdUsed: true,
            srsKickCount: 0,
          } satisfies Pick<
            TrajectoryExecutorReachablePlacement,
            'commands' | 'holdUsed' | 'srsKickCount'
          >)
        : selectedPlacement;
    const placementComplexityPenalty =
      computePlacementComplexityPenalty(rewardPlacement);
    const rewardV1 = isHoldStepAction
      ? computeHoldStepRewardV1(this.modeId, timeDeltaMs)
      : computePieceRewardV1({
          modeId: this.modeId,
          linesDelta,
          scoreDelta,
          timeDeltaMs,
          heightDelta,
          holesDelta,
          bumpinessDelta,
          boardScoreDelta: before.boardScore - after.boardScore,
          boardQualityDelta,
          topOut: this.game.state.gameOver,
        });
    const rewardV2 = isHoldStepAction
      ? computeHoldStepRewardV2()
      : computePieceRewardV2({
          modeId: this.modeId,
          linesDelta,
          scoreDelta,
          timeDeltaMs,
          heightDelta,
          holesDelta,
          bumpinessDelta,
          boardScoreDelta: before.boardScore - after.boardScore,
          boardQualityDelta,
          topOut: this.game.state.gameOver,
          placementComplexityPenalty,
        });
    const rewardV3 = isHoldStepAction
      ? computeHoldStepRewardV3()
      : computePieceRewardV3({
          modeId: this.modeId,
          linesDelta,
          scoreDelta,
          timeDeltaMs,
          heightDelta,
          holesDelta,
          bumpinessDelta,
          boardScoreDelta: before.boardScore - after.boardScore,
          boardQualityDelta,
          topOut: this.game.state.gameOver,
          placementComplexityPenalty,
          afterMaxHeight: after.height,
          afterBoardQuality: after.boardQuality,
          isFullClear,
        });
    const rewardById: Record<RewardFunctionId, PieceRewardResult> = {
      v1: rewardV1,
      v2: rewardV2,
      v3: rewardV3,
    };
    const rewardLegacy = rewardById[rewardFunctionFrom];
    const rewardTarget = rewardById[rewardFunctionTo];
    const rewardResult = blendPieceReward({
      legacy: rewardLegacy,
      target: rewardTarget,
      weights: blendWeights,
    });
    const rewardLegacyContribution =
      rewardLegacy.reward * blendWeights.legacyWeight;
    const rewardTargetContribution =
      rewardTarget.reward * blendWeights.targetWeight;
    const rewardLegacyContributionBreakdown = {
      linesTerm: rewardLegacy.breakdown.linesTerm * blendWeights.legacyWeight,
      scoreTerm: rewardLegacy.breakdown.scoreTerm * blendWeights.legacyWeight,
      timeTerm: rewardLegacy.breakdown.timeTerm * blendWeights.legacyWeight,
      heightTerm: rewardLegacy.breakdown.heightTerm * blendWeights.legacyWeight,
      holeDeltaTerm:
        rewardLegacy.breakdown.holeDeltaTerm * blendWeights.legacyWeight,
      bumpinessDeltaTerm:
        rewardLegacy.breakdown.bumpinessDeltaTerm * blendWeights.legacyWeight,
      boardScoreTerm:
        rewardLegacy.breakdown.boardScoreTerm * blendWeights.legacyWeight,
      boardQualityDeltaTerm:
        rewardLegacy.breakdown.boardQualityDeltaTerm *
        blendWeights.legacyWeight,
      boardQualityAbsoluteTerm:
        rewardLegacy.breakdown.boardQualityAbsoluteTerm *
        blendWeights.legacyWeight,
      fullClearTerm:
        rewardLegacy.breakdown.fullClearTerm * blendWeights.legacyWeight,
      topOutTerm: rewardLegacy.breakdown.topOutTerm * blendWeights.legacyWeight,
    };
    const rewardTargetContributionBreakdown = {
      linesTerm: rewardTarget.breakdown.linesTerm * blendWeights.targetWeight,
      scoreTerm: rewardTarget.breakdown.scoreTerm * blendWeights.targetWeight,
      timeTerm: rewardTarget.breakdown.timeTerm * blendWeights.targetWeight,
      heightTerm: rewardTarget.breakdown.heightTerm * blendWeights.targetWeight,
      holeDeltaTerm:
        rewardTarget.breakdown.holeDeltaTerm * blendWeights.targetWeight,
      bumpinessDeltaTerm:
        rewardTarget.breakdown.bumpinessDeltaTerm * blendWeights.targetWeight,
      boardScoreTerm:
        rewardTarget.breakdown.boardScoreTerm * blendWeights.targetWeight,
      boardQualityDeltaTerm:
        rewardTarget.breakdown.boardQualityDeltaTerm *
        blendWeights.targetWeight,
      boardQualityAbsoluteTerm:
        rewardTarget.breakdown.boardQualityAbsoluteTerm *
        blendWeights.targetWeight,
      fullClearTerm:
        rewardTarget.breakdown.fullClearTerm * blendWeights.targetWeight,
      topOutTerm: rewardTarget.breakdown.topOutTerm * blendWeights.targetWeight,
    };
    const reward = rewardResult.reward;
    const finalReward = reward;
    const rewardElapsedS = (performance.now() - rewardStart) / 1000;

    if (this.lockCount > beforeLockCount) {
      this.captureEpisodeStep({
        reward: finalReward,
        after,
        selectedPlacement,
        holdUsedOnTurn:
          this.pendingHoldUsedForCurrentPiece ||
          Boolean(selectedPlacement?.holdUsed),
      });
      this.pendingHoldUsedForCurrentPiece = false;
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
      this.includePhaseContext,
    );
    const obsElapsedS = (performance.now() - obsStart) / 1000;
    const choicesNextStart = performance.now();
    const nextChoices = buildPlacementChoices(
      this.game.state,
      this.actionDim,
      this.actionSpaceKind,
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
      actionScores: nextChoices.actionScores,
      reward: Number.isFinite(finalReward) ? finalReward : 0,
      done: this.done,
      info: {
        modeId: this.modeId,
        piecesPlaced: this.piecesPlaced,
        actionSpaceKind: this.actionSpaceKind,
        lockObserved: this.lockCount > beforeLockCount,
        holdStepObserved: isHoldStepAction,
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
        stackHeightAfter: after.height,
        rewardLegacyFunctionId: rewardFunctionFrom,
        rewardTargetFunctionId: rewardFunctionTo,
        rewardLegacyBase: rewardLegacy.reward,
        rewardTargetBase: rewardTarget.reward,
        rewardLegacyContribution,
        rewardTargetContribution,
        rewardBlendT: blendWeights.t,
        rewardBlendLegacyWeight: blendWeights.legacyWeight,
        rewardBlendTargetWeight: blendWeights.targetWeight,
        rewardBlendTransitionStep: blendWeights.transitionStep,
        rewardBlendTransitionTotalSteps: blendWeights.transitionTotalSteps,
        rewardBase: reward,
        rewardFinal: finalReward,
        topOutTerm: rewardResult.breakdown.topOutTerm,
        rewardLegacyTermLines: rewardLegacy.breakdown.linesTerm,
        rewardLegacyTermScore: rewardLegacy.breakdown.scoreTerm,
        rewardLegacyTermTime: rewardLegacy.breakdown.timeTerm,
        rewardLegacyTermHeight: rewardLegacy.breakdown.heightTerm,
        rewardLegacyTermHoles: rewardLegacy.breakdown.holeDeltaTerm,
        rewardLegacyTermBumpiness: rewardLegacy.breakdown.bumpinessDeltaTerm,
        rewardLegacyTermBoardScore: rewardLegacy.breakdown.boardScoreTerm,
        rewardLegacyTermBoardQuality:
          rewardLegacy.breakdown.boardQualityDeltaTerm,
        rewardLegacyTermBoardQualityAbsolute:
          rewardLegacy.breakdown.boardQualityAbsoluteTerm,
        rewardLegacyTermFullClear: rewardLegacy.breakdown.fullClearTerm,
        rewardLegacyTermTopOut: rewardLegacy.breakdown.topOutTerm,
        rewardTargetTermLines: rewardTarget.breakdown.linesTerm,
        rewardTargetTermScore: rewardTarget.breakdown.scoreTerm,
        rewardTargetTermTime: rewardTarget.breakdown.timeTerm,
        rewardTargetTermHeight: rewardTarget.breakdown.heightTerm,
        rewardTargetTermHoles: rewardTarget.breakdown.holeDeltaTerm,
        rewardTargetTermBumpiness: rewardTarget.breakdown.bumpinessDeltaTerm,
        rewardTargetTermBoardScore: rewardTarget.breakdown.boardScoreTerm,
        rewardTargetTermBoardQuality:
          rewardTarget.breakdown.boardQualityDeltaTerm,
        rewardTargetTermBoardQualityAbsolute:
          rewardTarget.breakdown.boardQualityAbsoluteTerm,
        rewardTargetTermFullClear: rewardTarget.breakdown.fullClearTerm,
        rewardTargetTermTopOut: rewardTarget.breakdown.topOutTerm,
        rewardLegacyContributionTermLines:
          rewardLegacyContributionBreakdown.linesTerm,
        rewardLegacyContributionTermScore:
          rewardLegacyContributionBreakdown.scoreTerm,
        rewardLegacyContributionTermTime:
          rewardLegacyContributionBreakdown.timeTerm,
        rewardLegacyContributionTermHeight:
          rewardLegacyContributionBreakdown.heightTerm,
        rewardLegacyContributionTermHoles:
          rewardLegacyContributionBreakdown.holeDeltaTerm,
        rewardLegacyContributionTermBumpiness:
          rewardLegacyContributionBreakdown.bumpinessDeltaTerm,
        rewardLegacyContributionTermBoardScore:
          rewardLegacyContributionBreakdown.boardScoreTerm,
        rewardLegacyContributionTermBoardQuality:
          rewardLegacyContributionBreakdown.boardQualityDeltaTerm,
        rewardLegacyContributionTermBoardQualityAbsolute:
          rewardLegacyContributionBreakdown.boardQualityAbsoluteTerm,
        rewardLegacyContributionTermFullClear:
          rewardLegacyContributionBreakdown.fullClearTerm,
        rewardLegacyContributionTermTopOut:
          rewardLegacyContributionBreakdown.topOutTerm,
        rewardTargetContributionTermLines:
          rewardTargetContributionBreakdown.linesTerm,
        rewardTargetContributionTermScore:
          rewardTargetContributionBreakdown.scoreTerm,
        rewardTargetContributionTermTime:
          rewardTargetContributionBreakdown.timeTerm,
        rewardTargetContributionTermHeight:
          rewardTargetContributionBreakdown.heightTerm,
        rewardTargetContributionTermHoles:
          rewardTargetContributionBreakdown.holeDeltaTerm,
        rewardTargetContributionTermBumpiness:
          rewardTargetContributionBreakdown.bumpinessDeltaTerm,
        rewardTargetContributionTermBoardScore:
          rewardTargetContributionBreakdown.boardScoreTerm,
        rewardTargetContributionTermBoardQuality:
          rewardTargetContributionBreakdown.boardQualityDeltaTerm,
        rewardTargetContributionTermBoardQualityAbsolute:
          rewardTargetContributionBreakdown.boardQualityAbsoluteTerm,
        rewardTargetContributionTermFullClear:
          rewardTargetContributionBreakdown.fullClearTerm,
        rewardTargetContributionTermTopOut:
          rewardTargetContributionBreakdown.topOutTerm,
        rewardTermLines: rewardResult.breakdown.linesTerm,
        rewardTermScore: rewardResult.breakdown.scoreTerm,
        rewardTermTime: rewardResult.breakdown.timeTerm,
        rewardTermHeight: rewardResult.breakdown.heightTerm,
        rewardTermHoles: rewardResult.breakdown.holeDeltaTerm,
        rewardTermBumpiness: rewardResult.breakdown.bumpinessDeltaTerm,
        rewardTermBoardScore: rewardResult.breakdown.boardScoreTerm,
        rewardTermBoardQuality: rewardResult.breakdown.boardQualityDeltaTerm,
        rewardTermBoardQualityAbsolute:
          rewardResult.breakdown.boardQualityAbsoluteTerm,
        rewardTermFullClear: rewardResult.breakdown.fullClearTerm,
        rewardTermTopOut: rewardResult.breakdown.topOutTerm,
        placementHoldUsed:
          isHoldStepAction || selectedPlacement?.holdUsed ? 1 : 0,
        placementSrsKickCount: Math.max(
          0,
          Math.trunc(selectedPlacement?.srsKickCount ?? 0),
        ),
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
    stopAfterCommandDrain = false,
  ): number {
    let ticks = 0;
    const stepFrames = [...commands];
    while (
      !this.isTerminal() &&
      this.lockCount === beforeLockCount &&
      (!stopAfterCommandDrain || stepFrames.length > 0) &&
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
        : this.pieceSource === 'random'
          ? {
              ...merged.generator,
              type: 'random' as const,
            }
          : {
              ...merged.generator,
              type: 'ml' as const,
            };
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
    holdUsedOnTurn: boolean;
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
        holdUsed: input.holdUsedOnTurn,
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
        generatorType:
          this.pieceSource === 'bag7'
            ? 'bag7'
            : this.pieceSource === 'random'
              ? 'random'
              : 'ml',
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
    blocks: number;
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
      blocks: countBoardBlocks(state.board),
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
  private rewardBlendTimesteps = DEFAULT_REWARD_BLEND_TIMESTEPS;
  private rewardBlendTransitionStep = 0;
  private rewardBlendUnit: RewardBlendUnit = 'updates';
  private rewardFunctionFrom: RewardFunctionId = 'v1';
  private rewardFunctionTo: RewardFunctionId = 'v2';

  static async create(payload: InitPayload): Promise<BotEnvPool> {
    const modeId = normalizeModeId(payload.modeId);
    const numEnvs = clampInt(payload.numEnvs, 1, 1, 4096);
    const observationSpace = normalizeObservationSpace(
      payload.observationSpace,
    );
    const includePhaseContext = payload.phaseContextEnabled === true;
    const placementExecutionMode = normalizePlacementExecutionMode(
      payload.placementExecutionMode,
    );
    const actionSpaceKind = normalizeActionSpaceKind(payload.actionSpaceKind);
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
    const rewardBlendTimesteps = clampInt(
      payload.rewardBlendTimesteps,
      DEFAULT_REWARD_BLEND_TIMESTEPS,
      1,
      1_000_000_000,
    );
    const rewardBlendUnit: RewardBlendUnit =
      payload.rewardBlendUnit === 'timesteps' ? 'timesteps' : 'updates';
    const rewardBlendStartStep = clampInt(
      payload.rewardBlendStartStep,
      0,
      0,
      1_000_000_000,
    );
    const rewardFunctionFrom = normalizeRewardFunctionId(
      payload.rewardFunctionFrom,
      'v1',
    );
    const rewardFunctionTo = normalizeRewardFunctionId(
      payload.rewardFunctionTo,
      payload.rewardFunctionTo == null
        ? payload.rewardFunctionFrom == null
          ? 'v2'
          : rewardFunctionFrom
        : 'v2',
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
          includePhaseContext,
          actionSpaceKind,
          pieceSourceProfile,
          queuePolicyId,
          maxPiecesPerEpisode,
          placementExecutionMode,
          seed,
        ),
      );
    }
    pool.rewardBlendTimesteps = rewardBlendTimesteps;
    pool.rewardBlendTransitionStep = rewardBlendStartStep;
    pool.rewardBlendUnit = rewardBlendUnit;
    pool.rewardFunctionFrom = rewardFunctionFrom;
    pool.rewardFunctionTo = rewardFunctionTo;
    pool.setCurriculum(null);
    return pool;
  }

  resetMany(
    envIds: number[],
    seeds: number[],
    initialBoards?: Array<number[][] | null>,
  ): StepBatchResult {
    const batchStart = performance.now();
    const obs: number[][] = [];
    const actionMasks: number[][] = [];
    const actionBiases: number[][] = [];
    const actionScores: number[][] = [];
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
      const initialBoard =
        Array.isArray(initialBoards) && i < initialBoards.length
          ? initialBoards[i]
          : null;
      const out = env.reset(seed, initialBoard);
      obs.push(out.obs);
      actionMasks.push(out.actionMask);
      actionBiases.push(out.actionBias);
      actionScores.push(out.actionScores);
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
      action_scores: actionScores,
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

  evaluateHoldCandidatesMany(envIds: number[]): {
    candidates: HoldCandidateEvaluation[][];
  } {
    const batchBlendWeights = this.nextRewardBlendWeights(0);
    const candidates: HoldCandidateEvaluation[][] = [];
    for (let i = 0; i < envIds.length; i += 1) {
      const env = this.requireEnv(envIds[i]);
      candidates.push(
        env.evaluateHoldCandidates(
          batchBlendWeights,
          this.rewardFunctionFrom,
          this.rewardFunctionTo,
        ),
      );
    }
    return {
      candidates,
    };
  }

  stepMany(envIds: number[], actions: number[]): StepBatchResult {
    const batchStart = performance.now();
    const obs: number[][] = [];
    const actionMasks: number[][] = [];
    const actionBiases: number[][] = [];
    const actionScores: number[][] = [];
    const rewards: number[] = [];
    const dones: boolean[] = [];
    const infos: JsonObject[] = [];
    let stepEnvTotalS = 0;
    let stepChoicesCurrentS = 0;
    let stepRunnerS = 0;
    let stepRewardS = 0;
    let stepObsS = 0;
    let stepChoicesNextS = 0;
    const blendIncrement =
      this.rewardBlendUnit === 'timesteps' ? envIds.length : 0;
    const batchBlendWeights = this.nextRewardBlendWeights(blendIncrement);
    for (let i = 0; i < envIds.length; i += 1) {
      const envId = envIds[i];
      const env = this.requireEnv(envId);
      const out = env.step(
        Math.trunc(actions[i] ?? 0),
        batchBlendWeights,
        this.rewardFunctionFrom,
        this.rewardFunctionTo,
      );
      obs.push(out.obs);
      actionMasks.push(out.actionMask);
      actionBiases.push(out.actionBias);
      actionScores.push(out.actionScores);
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
      action_scores: actionScores,
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

  setPieceSources(
    envIds: number[],
    pieceSourceProfiles: unknown[],
  ): { assigned: number; counts: Record<string, number> } {
    const assigned = Math.min(envIds.length, pieceSourceProfiles.length);
    const counts: Record<string, number> = {};
    for (let i = 0; i < assigned; i += 1) {
      const env = this.requireEnv(envIds[i]);
      const normalized = normalizePieceSource(pieceSourceProfiles[i]);
      env.setPieceSource(normalized);
      counts[normalized] = (counts[normalized] ?? 0) + 1;
    }
    return {
      assigned,
      counts,
    };
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

  setRewardBlendTransitionStep(value: unknown): number {
    const next = clampInt(value, 0, 0, 1_000_000_000);
    this.rewardBlendTransitionStep = next;
    return next;
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

  private nextRewardBlendWeights(
    transitionIncrement: number = 1,
  ): RewardBlendWeights {
    const transitionTotalSteps = Math.max(1, this.rewardBlendTimesteps);
    const transitionStep = Math.max(0, this.rewardBlendTransitionStep);
    const progress = Math.max(
      0,
      Math.min(1, transitionStep / transitionTotalSteps),
    );
    const legacyWeight = 1 - progress;
    const targetWeight = progress;
    const increment = Math.max(0, Math.trunc(transitionIncrement));
    this.rewardBlendTransitionStep = transitionStep + increment;
    return {
      legacyWeight,
      targetWeight,
      t: legacyWeight,
      transitionStep,
      transitionTotalSteps,
    };
  }
}
