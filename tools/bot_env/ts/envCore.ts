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
import { dropDistance } from '../../../src/core/piece.ts';
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
} from '../../../src/core/types.ts';
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
  PieceSourceProfile,
  StepBatchResult,
} from './protocol.ts';

const FIXED_STEP_MS = 1000 / 120;
const DEFAULT_ACTION_DIM = PLACEMENT_ACTION_DIM;
const DEFAULT_MAX_PIECES = 512;
const STEP_MAX_TICKS = 120;
const OFFLINE_GRAVITY_MS = Number.POSITIVE_INFINITY;
const OFFLINE_SOFT_DROP_MS = 0;
const TRAJECTORY_SCHEMA = 'wishuponablock.trajectory_session.v1';
const TRAJECTORY_BUILD_VERSION = 'offline_ppo_py';
const TRAJECTORY_PIECES = [...PIECES];
const TRAJECTORY_MIN_SAMPLE_ID = 8;

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

const computePieceReward = (options: {
  modeId: string;
  linesDelta: number;
  scoreDelta: number;
  timeDeltaMs: number;
  holesDelta: number;
  bumpinessDelta: number;
  boardScoreDelta: number;
}): number => {
  const {
    modeId,
    linesDelta,
    scoreDelta,
    timeDeltaMs,
    holesDelta,
    bumpinessDelta,
    boardScoreDelta,
  } = options;
  const holeDeltaTerm = -holesDelta * HOLE_DELTA_REWARD_WEIGHT;
  const bumpinessDeltaTerm = -bumpinessDelta * BUMPINESS_DELTA_REWARD_WEIGHT;
  if (modeId === 'sprint') {
    return (
      linesDelta * 1.2 +
      scoreDelta * 0.001 +
      holeDeltaTerm +
      bumpinessDeltaTerm -
      timeDeltaMs / 4000
    );
  }
  if (modeId === 'classic') {
    return (
      linesDelta * 0.6 + scoreDelta * 0.002 + holeDeltaTerm + bumpinessDeltaTerm
    );
  }
  if (modeId === 'charcuterie') {
    return (
      boardScoreDelta * 0.12 +
      linesDelta * 0.15 -
      timeDeltaMs / 25000 +
      holeDeltaTerm +
      bumpinessDeltaTerm
    );
  }
  if (modeId === 'cheese') {
    return linesDelta * 0.7 + holeDeltaTerm + bumpinessDeltaTerm;
  }
  return (
    linesDelta * 0.5 +
    scoreDelta * 0.0008 +
    holeDeltaTerm +
    bumpinessDeltaTerm -
    timeDeltaMs / 6000
  );
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
  random: (() => number) | null = null,
): {
  commandsBySlot: Array<InputFrame[] | null>;
  placementsBySlot: Array<TrajectoryExecutorReachablePlacement | null>;
  actionMask: number[];
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
  for (const placement of placements) {
    const actionIndex = placementActionIndexFromPlacement(placement);
    if (actionIndex == null || actionIndex < 0 || actionIndex >= actionDim) {
      continue;
    }
    if (actionMask[actionIndex] > 0 && commandsBySlot[actionIndex]) {
      continue;
    }
    actionMask[actionIndex] = 1;
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
  return { commandsBySlot, placementsBySlot, actionMask };
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
  private cachedChoices: {
    commandsBySlot: Array<InputFrame[] | null>;
    placementsBySlot: Array<TrajectoryExecutorReachablePlacement | null>;
    actionMask: number[];
  } | null = null;

  constructor(
    private readonly envId: number,
    private readonly modeId: string,
    private readonly model: LoadedModel,
    private readonly observationSpace: BotObservationSpace,
    pieceSource: PieceSourceProfile,
    private readonly queuePolicyId: string,
    private readonly maxPiecesPerEpisode: number,
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
      () => nextFloat(this.planningRng),
    );
    this.startEpisodeCapture();
    this.syncPrevMetrics();
  }

  reset(seed: number): {
    obs: number[];
    actionMask: number[];
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
        buildPlacementChoices(this.game.state, DEFAULT_ACTION_DIM, () =>
          nextFloat(this.planningRng),
        );
      this.cachedChoices = choices;
      const doneChoicesElapsedS = (performance.now() - doneChoicesStart) / 1000;
      const totalElapsedS = (performance.now() - stepStart) / 1000;
      return {
        obs,
        actionMask: choices.actionMask,
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
      buildPlacementChoices(this.game.state, DEFAULT_ACTION_DIM, () =>
        nextFloat(this.planningRng),
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
    const stepFrames = [...commands];
    const runnerStart = performance.now();
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
    const reward = computePieceReward({
      modeId: this.modeId,
      linesDelta: after.lines - before.lines,
      scoreDelta: after.score - before.score,
      timeDeltaMs: after.timeMs - before.timeMs,
      holesDelta: after.holes - before.holes,
      bumpinessDelta: after.bumpiness - before.bumpiness,
      boardScoreDelta: before.boardScore - after.boardScore,
    });
    const rewardElapsedS = (performance.now() - rewardStart) / 1000;

    if (this.lockCount > beforeLockCount) {
      this.captureEpisodeStep({
        reward,
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
      () => nextFloat(this.planningRng),
    );
    const choicesNextElapsedS = (performance.now() - choicesNextStart) / 1000;
    this.cachedChoices = nextChoices;
    const totalElapsedS = (performance.now() - stepStart) / 1000;
    return {
      obs,
      actionMask: nextChoices.actionMask,
      reward: Number.isFinite(reward) ? reward : 0,
      done: this.done,
      info: {
        modeId: this.modeId,
        piecesPlaced: this.piecesPlaced,
        lockObserved: this.lockCount > beforeLockCount,
        ticks,
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
    holes: number;
    bumpiness: number;
    boardScore: number;
  } {
    const state = this.game.state;
    return {
      lines: Math.max(0, Math.trunc(state.totalLinesCleared)),
      score: Math.max(0, Math.trunc(state.score)),
      timeMs: Math.max(0, Math.trunc(state.timeMs)),
      holes: countBoardHoles(state.board),
      bumpiness: computeBoardBumpiness(state.board),
      boardScore: scoreCharcuterieBoard(
        state.board,
        state.gameOver,
        state.totalLinesCleared,
      ),
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

  static async create(payload: InitPayload): Promise<BotEnvPool> {
    const modeId = normalizeModeId(payload.modeId);
    const numEnvs = clampInt(payload.numEnvs, 1, 1, 4096);
    const observationSpace = normalizeObservationSpace(
      payload.observationSpace,
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
          seed,
        ),
      );
    }
    return pool;
  }

  resetMany(envIds: number[], seeds: number[]): StepBatchResult {
    const batchStart = performance.now();
    const obs: number[][] = [];
    const actionMasks: number[][] = [];
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
      rewards.push(out.reward);
      dones.push(out.done);
      infos.push(out.info);
      if (out.completedSession) {
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
