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
import { GameRunner, type InputSource } from '../../../src/core/runner.ts';
import { DEFAULT_SETTINGS } from '../../../src/core/settings.ts';
import type { Board, GameState, InputFrame } from '../../../src/core/types.ts';
import { PIECES } from '../../../src/core/types.ts';
import {
  enumerateTrajectoryExecutorPlacements,
  trajectoryExecutorCommandToInputFrame,
} from '../../../src/core/trajectoryExecutor.ts';
import {
  buildModelHeadInput,
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
const DEFAULT_ACTION_DIM = 192;
const DEFAULT_MAX_PIECES = 512;
const STEP_MAX_TICKS = 120;
const PIECE_INDEX = new Map(PIECES.map((piece, idx) => [piece, idx]));

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

const normalizeModeId = (value: unknown): string => {
  if (typeof value !== 'string') return 'charcuterie';
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
  return 'charcuterie';
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

const CHARCUTERIE_WARMUP_TARGET_BLOCKS = 56;

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

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

const computePieceReward = (options: {
  modeId: string;
  linesDelta: number;
  scoreDelta: number;
  timeDeltaMs: number;
  holesDelta: number;
  boardScoreDelta: number;
  charcuterieWarmupProgress?: number;
}): number => {
  const {
    modeId,
    linesDelta,
    scoreDelta,
    timeDeltaMs,
    holesDelta,
    boardScoreDelta,
    charcuterieWarmupProgress,
  } = options;
  const survivalBonus = 0.02;
  if (modeId === 'sprint') {
    return (
      linesDelta * 1.2 + scoreDelta * 0.001 - timeDeltaMs / 4000 + survivalBonus
    );
  }
  if (modeId === 'classic') {
    return linesDelta * 0.6 + scoreDelta * 0.002 + survivalBonus;
  }
  if (modeId === 'charcuterie') {
    const warmupProgress = clamp(charcuterieWarmupProgress ?? 1, 0, 1);
    const adjustedBoardScoreDelta =
      boardScoreDelta >= 0 ? boardScoreDelta : boardScoreDelta * warmupProgress;
    return (
      adjustedBoardScoreDelta * 0.12 +
      linesDelta * 0.15 -
      timeDeltaMs / 25000 +
      survivalBonus
    );
  }
  if (modeId === 'cheese') {
    return linesDelta * 0.7 - Math.max(0, holesDelta) * 0.03 + survivalBonus;
  }
  return (
    linesDelta * 0.5 + scoreDelta * 0.0008 - timeDeltaMs / 6000 + survivalBonus
  );
};

const encodeObservation = (model: LoadedModel, state: GameState): number[] => {
  const headInput = buildModelHeadInput(model, state.board, state.hold);
  const contextDim = PIECES.length + PIECES.length + 5;
  const out = new Float32Array(headInput.length + contextDim);
  out.set(headInput, 0);

  let offset = headInput.length;
  const activeIdx = PIECE_INDEX.get(state.active.k);
  if (activeIdx != null) out[offset + activeIdx] = 1;
  offset += PIECES.length;

  const nextPiece = state.next[0] ?? null;
  const nextIdx = nextPiece == null ? null : PIECE_INDEX.get(nextPiece);
  if (nextIdx != null) out[offset + nextIdx] = 1;
  offset += PIECES.length;

  const lineGoal =
    state.lineGoal != null && state.lineGoal > 0 ? state.lineGoal : null;
  const progress =
    lineGoal != null
      ? clamp(state.totalLinesCleared / lineGoal, 0, 2)
      : clamp(state.totalLinesCleared / 80, 0, 2);
  out[offset++] = progress;
  out[offset++] = clamp(state.timeMs / 180_000, 0, 2);
  out[offset++] = clamp(state.level / 20, 0, 2);
  out[offset++] = clamp(state.score / 200_000, 0, 2);
  out[offset++] = state.canHold ? 1 : 0;

  return Array.from(out);
};

const buildPlacementChoices = (
  state: GameState,
  actionDim: number,
): { commandsBySlot: InputFrame[][]; actionMask: number[] } => {
  const placements = enumerateTrajectoryExecutorPlacements({
    board: state.board,
    active: state.active,
    hold: state.hold,
    canHold: state.canHold,
    nextPieceOnFirstHold: state.next[0] ?? null,
    maxNodesPerBranch: 20_000,
    allowSoftDrop: true,
  });
  const actionMask = new Array<number>(actionDim).fill(0);
  const commandsBySlot: InputFrame[][] = [];
  const maxChoices = Math.min(actionDim, placements.length);
  for (let i = 0; i < maxChoices; i += 1) {
    actionMask[i] = 1;
    commandsBySlot.push(
      placements[i].commands.map((command) =>
        trajectoryExecutorCommandToInputFrame(command),
      ),
    );
  }
  if (commandsBySlot.length === 0) {
    actionMask[0] = 1;
    commandsBySlot.push([trajectoryExecutorCommandToInputFrame('hard_drop')]);
  }
  return { commandsBySlot, actionMask };
};

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

class BotEnv {
  private game: Game;
  private runner: GameRunner;
  private done = false;
  private piecesPlaced = 0;
  private lockCount = 0;
  private cachedChoices: {
    commandsBySlot: InputFrame[][];
    actionMask: number[];
  } | null = null;

  constructor(
    private readonly modeId: string,
    private readonly model: LoadedModel,
    private readonly pieceSource: PieceSourceProfile,
    private readonly queuePolicyId: string,
    private readonly maxPiecesPerEpisode: number,
    seed: number,
  ) {
    const built = this.buildGame(seed);
    this.game = built.game;
    this.runner = built.runner;
    this.cachedChoices = buildPlacementChoices(
      this.game.state,
      DEFAULT_ACTION_DIM,
    );
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
    const choicesStart = performance.now();
    this.cachedChoices = buildPlacementChoices(
      this.game.state,
      DEFAULT_ACTION_DIM,
    );
    const choicesElapsedS = (performance.now() - choicesStart) / 1000;
    this.syncPrevMetrics();
    const obsStart = performance.now();
    const obs = encodeObservation(this.model, this.game.state);
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

  step(actionIndexRaw: number): {
    obs: number[];
    actionMask: number[];
    reward: number;
    done: boolean;
    info: JsonObject;
    profile: BotEnvStepProfile;
  } {
    const stepStart = performance.now();
    if (this.done) {
      const doneChoicesStart = performance.now();
      const obs = encodeObservation(this.model, this.game.state);
      const choices =
        this.cachedChoices ??
        buildPlacementChoices(this.game.state, DEFAULT_ACTION_DIM);
      this.cachedChoices = choices;
      const doneChoicesElapsedS = (performance.now() - doneChoicesStart) / 1000;
      const totalElapsedS = (performance.now() - stepStart) / 1000;
      return {
        obs,
        actionMask: choices.actionMask,
        reward: 0,
        done: true,
        info: { alreadyDone: true, piecesPlaced: this.piecesPlaced },
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
      buildPlacementChoices(this.game.state, DEFAULT_ACTION_DIM);
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
    const blocks = countBoardBlocks(this.game.state.board);
    const warmupProgress = clamp(
      blocks / CHARCUTERIE_WARMUP_TARGET_BLOCKS,
      0,
      1,
    );
    const reward = computePieceReward({
      modeId: this.modeId,
      linesDelta: after.lines - before.lines,
      scoreDelta: after.score - before.score,
      timeDeltaMs: after.timeMs - before.timeMs,
      holesDelta: after.holes - before.holes,
      boardScoreDelta: before.boardScore - after.boardScore,
      charcuterieWarmupProgress: warmupProgress,
    });
    const rewardElapsedS = (performance.now() - rewardStart) / 1000;

    this.syncPrevMetrics();
    if (this.isTerminal() || this.piecesPlaced >= this.maxPiecesPerEpisode) {
      this.done = true;
    }

    const obsStart = performance.now();
    const obs = encodeObservation(this.model, this.game.state);
    const obsElapsedS = (performance.now() - obsStart) / 1000;
    const choicesNextStart = performance.now();
    const nextChoices = buildPlacementChoices(
      this.game.state,
      DEFAULT_ACTION_DIM,
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

  private snapshotMetrics(): {
    lines: number;
    score: number;
    timeMs: number;
    holes: number;
    boardScore: number;
  } {
    const state = this.game.state;
    return {
      lines: Math.max(0, Math.trunc(state.totalLinesCleared)),
      score: Math.max(0, Math.trunc(state.score)),
      timeMs: Math.max(0, Math.trunc(state.timeMs)),
      holes: countBoardHoles(state.board),
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

  static async create(payload: InitPayload): Promise<BotEnvPool> {
    const modeId = normalizeModeId(payload.modeId);
    const numEnvs = clampInt(payload.numEnvs, 1, 1, 4096);
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
          modeId,
          model,
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

  private requireEnv(id: number): BotEnv {
    const env = this.envs.get(id);
    if (!env) {
      throw new Error(`Unknown env id: ${id}`);
    }
    return env;
  }
}
