import { Game } from '../core/game';
import { createGeneratorFactory } from '../core/generators';
import type { ModelGeneratorDecisionEvent } from '../core/modelGenerator';
import type { ModelRunner } from '../core/modelRunner';
import { getMode } from '../core/modes';
import { XorShift32 } from '../core/rng';
import { GameRunner, type InputSource } from '../core/runner';
import type { Settings } from '../core/settings';
import type { Board, GameState, InputFrame, PieceKind } from '../core/types';
import { PIECES } from '../core/types';
import type { LoadedModel } from '../core/wubModel';
import type { ModelAxes } from '../core/modelAxes';
import { applyModeSettings, runModeStart } from './modeService';

const TFJS_CDN_URL = 'https://esm.sh/@tensorflow/tfjs@4.22.0';
const FIXED_STEP_MS = 1000 / 120;
const EMPTY_INPUT: InputFrame = {
  moveX: 0,
  rotate: 0,
  rotate180: false,
  softDrop: false,
  hardDrop: false,
  hold: false,
  restart: false,
};
const ACTION_ROTATIONS = ['none', 'cw', 'ccw', '180'] as const;
const ACTION_MOVE_X = [-3, -2, -1, 0, 1, 2, 3, 4] as const;
const PIECE_INDEX = new Map(PIECES.map((piece, idx) => [piece, idx]));
const HOLD_NONE_INDEX = PIECES.length;

type TfTensor = {
  dataSync: () => Float32Array | Int32Array | Uint8Array;
  dispose: () => void;
};

type TfOptimizer = {
  minimize: (fn: () => TfTensor, returnCost?: boolean) => TfTensor | null;
};

type TfjsModule = {
  ready: () => Promise<void>;
  setBackend: (backend: string) => Promise<boolean>;
  tensor2d: (values: Float32Array, shape: [number, number]) => TfTensor;
  tensor1d: (values: Float32Array | Int32Array, dtype?: string) => TfTensor;
  oneHot: (indices: TfTensor, depth: number) => TfTensor;
  variable: (value: TfTensor) => TfTensor;
  randomNormal: (
    shape: [number, number],
    mean?: number,
    stdDev?: number,
  ) => TfTensor;
  scalar: (value: number) => TfTensor;
  relu: (x: TfTensor) => TfTensor;
  add: (a: TfTensor, b: TfTensor) => TfTensor;
  matMul: (a: TfTensor, b: TfTensor) => TfTensor;
  logSoftmax: (x: TfTensor, axis?: number) => TfTensor;
  softmax: (x: TfTensor, axis?: number) => TfTensor;
  sum: (x: TfTensor, axis?: number | number[]) => TfTensor;
  mul: (a: TfTensor, b: TfTensor) => TfTensor;
  squeeze: (x: TfTensor, axis?: number[]) => TfTensor;
  sub: (a: TfTensor, b: TfTensor) => TfTensor;
  mean: (x: TfTensor, axis?: number | number[]) => TfTensor;
  square: (x: TfTensor) => TfTensor;
  sqrt: (x: TfTensor) => TfTensor;
  div: (a: TfTensor, b: TfTensor) => TfTensor;
  stopGradient: (x: TfTensor) => TfTensor;
  neg: (x: TfTensor) => TfTensor;
  train: {
    adam: (learningRate: number) => TfOptimizer;
  };
  dispose: (values: unknown[]) => void;
};

let tfPromise: Promise<TfjsModule> | null = null;

const loadTf = async (): Promise<TfjsModule> => {
  if (!tfPromise) {
    tfPromise = import(/* @vite-ignore */ TFJS_CDN_URL)
      .then((mod) => (mod.default ?? mod) as TfjsModule)
      .catch((error) => {
        tfPromise = null;
        throw error;
      });
  }
  return tfPromise;
};

export type BotMacroRotation = (typeof ACTION_ROTATIONS)[number];

export type BotMacroAction = {
  rotation: BotMacroRotation;
  moveX: number;
};

export type BotPieceSourceProfile = 'bag7' | 'active_generator';

export type BotPolicyArtifact = {
  id: string;
  modeId: string;
  archId?: string;
  queuePolicyId?: string;
  pipelineId?: string;
  pieceSourceProfile?: BotPieceSourceProfile;
  createdAtMs: number;
  inputDim: number;
  hiddenDim: number;
  actionDim: number;
  actions: BotMacroAction[];
  weights: {
    w1: number[];
    b1: number[];
    wp: number[];
    bp: number[];
  };
};

export type BotTrainOneShotConfig = {
  modeId: string;
  settings: Settings;
  model: LoadedModel;
  modelRunner: ModelRunner;
  modelAxes: ModelAxes;
  episodes?: number;
  maxPiecesPerEpisode?: number;
  gamma?: number;
  learningRate?: number;
  entropyBeta?: number;
  valueWeight?: number;
  epochs?: number;
  seed?: number;
  pieceSourceProfile?: BotPieceSourceProfile;
};

export type BotTrainOneShotResult = {
  ok: boolean;
  message: string;
  episodes: number;
  meanReturn: number;
  finalLoss: number | null;
  policyArtifact: BotPolicyArtifact | null;
};

export type BotTrajectoryDecisionSample = {
  id: string;
  createdAtMs: number;
  deliberationMs: number | null;
  boardOccupancy: number[][];
  hold: PieceKind | null;
  action: PieceKind;
  actionIndex: number;
  pieces: PieceKind[];
  logits: number[];
  probabilities: number[];
  inferenceMs: number;
  samplingMs: number;
  totalDecisionMs: number;
  reward: number | null;
};

export type BotTrajectoryDraft = {
  modeId: string;
  startedAtMs: number;
  endedAtMs: number;
  durationMs: number;
  outcome: 'game_over' | 'game_won' | 'manual';
  terminal: {
    totalLinesCleared: number;
    score: number;
    timeMs: number;
  };
  samples: BotTrajectoryDecisionSample[];
};

export type BotGenerateBatchConfig = {
  modeId: string;
  settings: Settings;
  model: LoadedModel;
  modelRunner: ModelRunner;
  modelAxes: ModelAxes;
  policy: BotPolicyArtifact;
  sessions: number;
  maxPiecesPerEpisode?: number;
  seed?: number;
  trainingIntent?: string;
  pieceSourceProfile?: BotPieceSourceProfile;
};

export type BotGenerateBatchResult = {
  sessionsGenerated: number;
  samplesGenerated: number;
  drafts: BotTrajectoryDraft[];
  trainingIntent: string | null;
};

export type CapabilityBenchmarkConfig = {
  modeId: string;
  settings: Settings;
  model: LoadedModel;
  modelRunner: ModelRunner;
  modelAxes: ModelAxes;
  policy: BotPolicyArtifact;
  episodes?: number;
  maxPiecesPerEpisode?: number;
  seed?: number;
  pieceSourceProfile?: BotPieceSourceProfile;
};

export type CapabilityBenchmarkResult = {
  verdict: 'excellent' | 'good' | 'constrained';
  recommendedArch: 'full' | 'lean';
  metrics: {
    meanDecisionMs: number;
    p95DecisionMs: number;
    meanTickMs: number;
    p95TickMs: number;
    simThroughput: number;
    targetSimFps: number;
  };
};

export type BotHeadlessValidateConfig = {
  modeId: string;
  settings: Settings;
  model: LoadedModel;
  modelRunner: ModelRunner;
  modelAxes: ModelAxes;
  policy: BotPolicyArtifact;
  maxPieces?: number;
  seed?: number;
  pieceSourceProfile?: BotPieceSourceProfile;
};

export type BotHeadlessValidateResult = {
  piecesSurvived: number;
  outcome: 'game_over' | 'game_won' | 'manual';
  meanDecisionMs: number;
  p95DecisionMs: number;
  meanTickMs: number;
  p95TickMs: number;
  meanBoardScoreDelta: number;
  totalReward: number;
  startedAtMs: number;
  endedAtMs: number;
  durationMs: number;
};

type PolicyParams = {
  inputDim: number;
  hiddenDim: number;
  actionDim: number;
  w1: Float32Array;
  b1: Float32Array;
  wp: Float32Array;
  bp: Float32Array;
};

type Transition = {
  observation: Float32Array;
  actionIndex: number;
  reward: number | null;
};

type RolloutResult = {
  transitions: Transition[];
  episodeReturn: number;
  decisions: ModelGeneratorDecisionEvent[];
  tickDurationsMs: number[];
  steps: number;
  piecesPlaced: number;
  boardScoreDeltas: number[];
  boardScoreStart: number;
  boardScoreEnd: number;
  startedAtMs: number;
  endedAtMs: number;
  outcome: 'game_over' | 'game_won' | 'manual';
  terminal: {
    totalLinesCleared: number;
    score: number;
    timeMs: number;
  };
};

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

const mean = (values: number[]): number => {
  if (values.length === 0) return 0;
  let sum = 0;
  for (const value of values) sum += value;
  return sum / values.length;
};

const percentile = (values: number[], p: number): number => {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.floor((sorted.length - 1) * clamp(p, 0, 1))),
  );
  return sorted[index];
};

const boardOccupancy = (board: Board): number[][] =>
  board.map((row) => row.map((cell) => (cell == null ? 0 : 1)));

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

const softmax = (logits: Float32Array): Float32Array => {
  if (logits.length === 0) return new Float32Array();
  let maxValue = logits[0];
  for (let i = 1; i < logits.length; i += 1) {
    if (logits[i] > maxValue) maxValue = logits[i];
  }
  const out = new Float32Array(logits.length);
  let total = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const value = Math.exp(logits[i] - maxValue);
    out[i] = value;
    total += value;
  }
  if (total <= 0 || !Number.isFinite(total)) {
    const uniform = 1 / out.length;
    for (let i = 0; i < out.length; i += 1) out[i] = uniform;
    return out;
  }
  for (let i = 0; i < out.length; i += 1) out[i] /= total;
  return out;
};

const nextFloat = (rng: XorShift32): number => rng.nextU32() / 0xffffffff;

const sampleIndex = (probabilities: Float32Array, rng: XorShift32): number => {
  if (probabilities.length === 0) return 0;
  const roll = nextFloat(rng);
  let cumulative = 0;
  for (let i = 0; i < probabilities.length; i += 1) {
    const weight = probabilities[i];
    if (!Number.isFinite(weight) || weight <= 0) continue;
    cumulative += weight;
    if (roll <= cumulative) return i;
  }
  return probabilities.length - 1;
};

const actionSpace: BotMacroAction[] = ACTION_ROTATIONS.flatMap((rotation) =>
  ACTION_MOVE_X.map((moveX) => ({ rotation, moveX })),
);

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

const normalizePieceSourceProfile = (
  value: BotPieceSourceProfile | undefined,
): BotPieceSourceProfile =>
  value === 'active_generator' ? 'active_generator' : 'bag7';

const asInputFrame = (action: BotMacroAction): InputFrame => ({
  ...EMPTY_INPUT,
  moveX: action.moveX,
  rotate: action.rotation === 'cw' ? 1 : action.rotation === 'ccw' ? -1 : 0,
  rotate180: action.rotation === '180',
  hardDrop: true,
});

const randomizeParams = (
  inputDim: number,
  hiddenDim: number,
  actionDim: number,
  rng: XorShift32,
): PolicyParams => {
  const w1 = new Float32Array(inputDim * hiddenDim);
  const b1 = new Float32Array(hiddenDim);
  const wp = new Float32Array(hiddenDim * actionDim);
  const bp = new Float32Array(actionDim);
  const scaleIn = Math.sqrt(2 / Math.max(1, inputDim));
  const scaleHidden = Math.sqrt(2 / Math.max(1, hiddenDim));
  for (let i = 0; i < w1.length; i += 1) {
    w1[i] = (nextFloat(rng) * 2 - 1) * scaleIn;
  }
  for (let i = 0; i < wp.length; i += 1) {
    wp[i] = (nextFloat(rng) * 2 - 1) * scaleHidden;
  }
  return { inputDim, hiddenDim, actionDim, w1, b1, wp, bp };
};

const forwardPolicy = (
  params: PolicyParams,
  observation: Float32Array,
): {
  logits: Float32Array;
  probabilities: Float32Array;
} => {
  const hidden = new Float32Array(params.hiddenDim);
  for (let h = 0; h < params.hiddenDim; h += 1) {
    let sum = params.b1[h];
    for (let i = 0; i < params.inputDim; i += 1) {
      sum += observation[i] * params.w1[i * params.hiddenDim + h];
    }
    hidden[h] = sum > 0 ? sum : 0;
  }
  const logits = new Float32Array(params.actionDim);
  for (let a = 0; a < params.actionDim; a += 1) {
    let sum = params.bp[a];
    for (let h = 0; h < params.hiddenDim; h += 1) {
      sum += hidden[h] * params.wp[h * params.actionDim + a];
    }
    logits[a] = sum;
  }
  return { logits, probabilities: softmax(logits) };
};

const encodeObservation = (state: GameState): Float32Array => {
  const rows = state.board.length;
  const cols = rows > 0 ? state.board[0].length : 0;
  const boardSize = rows * cols;
  const inputDim =
    boardSize + PIECES.length + (PIECES.length + 1) + PIECES.length + 5;
  const out = new Float32Array(inputDim);
  let offset = 0;
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      out[offset++] = state.board[y][x] == null ? 0 : 1;
    }
  }

  const activeIdx = PIECE_INDEX.get(state.active.k);
  if (activeIdx != null) out[offset + activeIdx] = 1;
  offset += PIECES.length;

  const holdIdx =
    state.hold == null
      ? HOLD_NONE_INDEX
      : (PIECE_INDEX.get(state.hold) ?? HOLD_NONE_INDEX);
  out[offset + holdIdx] = 1;
  offset += PIECES.length + 1;

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
  return out;
};

class MacroPolicyBot implements InputSource {
  private activeRef: GameState['active'] | null = null;
  private queue: InputFrame[] = [];
  private transitions: Transition[] = [];
  private rewardCursor = 0;

  constructor(
    private decide: (state: GameState) => {
      actionIndex: number;
      observation: Float32Array;
    },
  ) {}

  sample(state: GameState): InputFrame {
    if (state.active !== this.activeRef) {
      this.activeRef = state.active;
      const decision = this.decide(state);
      const action = actionSpace[decision.actionIndex] ?? actionSpace[0];
      this.queue = [asInputFrame(action)];
      this.transitions.push({
        observation: decision.observation,
        actionIndex: decision.actionIndex,
        reward: null,
      });
    }
    return this.queue.shift() ?? EMPTY_INPUT;
  }

  onPiecePlaced(reward: number): void {
    const target = this.transitions[this.rewardCursor];
    if (target) {
      target.reward = reward;
      this.rewardCursor += 1;
    }
  }

  drainTransitions(): Transition[] {
    for (const transition of this.transitions) {
      if (transition.reward == null || !Number.isFinite(transition.reward)) {
        transition.reward = 0;
      }
    }
    return this.transitions.map((transition) => ({
      observation: new Float32Array(transition.observation),
      actionIndex: transition.actionIndex,
      reward: transition.reward,
    }));
  }
}

const cloneDecision = (
  event: ModelGeneratorDecisionEvent,
): ModelGeneratorDecisionEvent => ({
  board: event.board.map((row) => row.slice()),
  hold: event.hold,
  action: event.action,
  pieces: [...event.pieces],
  logits: new Float32Array(event.logits),
  probabilities: new Float32Array(event.probabilities),
  inferenceMs: event.inferenceMs,
  samplingMs: event.samplingMs,
  totalMs: event.totalMs,
  wallTimeMs: event.wallTimeMs,
});

const buildHeadlessGame = (config: {
  modeId: string;
  settings: Settings;
  model: LoadedModel;
  modelRunner: ModelRunner;
  modelAxes: ModelAxes;
  pieceSourceProfile: BotPieceSourceProfile;
  seed: number;
  onDecision: (event: ModelGeneratorDecisionEvent) => void;
}): { game: Game; runner: GameRunner } => {
  const mode = getMode(config.modeId);
  const merged = applyModeSettings(
    {
      ...config.settings,
      butterfinger: {
        ...config.settings.butterfinger,
        enabled: false,
      },
    },
    mode,
  );
  const generatorSettings =
    config.pieceSourceProfile === 'bag7'
      ? {
          ...merged.generator,
          type: 'bag7' as const,
        }
      : merged.generator;
  const game = new Game({
    seed: config.seed,
    ...merged.game,
    lockNudgeRate: 0,
    gravityDropRate: 0,
    lockRotateRate: 0,
    lineGoal: mode.lineGoal,
    classicStartLevel: mode.classicStartLevel,
    scoringEnabled: mode.scoringEnabled,
    generatorFactory: createGeneratorFactory(generatorSettings, {
      mlModel: config.model,
      mlRunner: config.modelRunner,
      queuePolicyId: config.modelAxes.queuePolicyId,
      onModelDecision: config.onDecision,
    }),
  });
  runModeStart(game, mode, {});
  return {
    game,
    runner: new GameRunner(game, { fixedStepMs: FIXED_STEP_MS }),
  };
};

const computePieceReward = (options: {
  modeId: string;
  linesDelta: number;
  scoreDelta: number;
  timeDeltaMs: number;
  holesDelta: number;
  boardScoreDelta: number;
}): number => {
  const {
    modeId,
    linesDelta,
    scoreDelta,
    timeDeltaMs,
    holesDelta,
    boardScoreDelta,
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
    return (
      boardScoreDelta * 0.12 +
      linesDelta * 0.15 -
      timeDeltaMs / 2500 +
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

const runRollout = (config: {
  modeId: string;
  settings: Settings;
  model: LoadedModel;
  modelRunner: ModelRunner;
  modelAxes: ModelAxes;
  pieceSourceProfile: BotPieceSourceProfile;
  policyParams: PolicyParams;
  seed: number;
  maxPieces: number;
  greedy: boolean;
}): RolloutResult => {
  const decisions: ModelGeneratorDecisionEvent[] = [];
  const { game, runner } = buildHeadlessGame({
    modeId: config.modeId,
    settings: config.settings,
    model: config.model,
    modelRunner: config.modelRunner,
    modelAxes: config.modelAxes,
    pieceSourceProfile: config.pieceSourceProfile,
    seed: config.seed,
    onDecision: (event) => {
      decisions.push(cloneDecision(event));
    },
  });
  const rng = new XorShift32(config.seed ^ 0x9e3779b9);
  const bot = new MacroPolicyBot((state) => {
    const observation = encodeObservation(state);
    const forward = forwardPolicy(config.policyParams, observation);
    let actionIndex = 0;
    if (config.greedy) {
      let best = forward.logits[0] ?? Number.NEGATIVE_INFINITY;
      for (let i = 1; i < forward.logits.length; i += 1) {
        if (forward.logits[i] > best) {
          best = forward.logits[i];
          actionIndex = i;
        }
      }
    } else {
      actionIndex = sampleIndex(forward.probabilities, rng);
    }
    return { actionIndex, observation };
  });

  const maxPieces = Math.max(1, Math.trunc(config.maxPieces));
  const maxSteps = Math.max(maxPieces * 90, 600);
  const startedAtMs = Date.now();
  let lastActive = game.state.active;
  let piecesPlaced = 0;
  let steps = 0;
  let previousLines = game.state.totalLinesCleared;
  let previousScore = game.state.score;
  let previousTimeMs = game.state.timeMs;
  let previousHoles = countBoardHoles(game.state.board);
  let previousBoardScore = scoreCharcuterieBoard(
    game.state.board,
    game.state.gameOver,
    game.state.totalLinesCleared,
  );
  const boardScoreStart = previousBoardScore;
  const boardScoreDeltas: number[] = [];
  const tickDurationsMs: number[] = [];

  while (
    !game.state.gameOver &&
    !game.state.gameWon &&
    piecesPlaced < maxPieces &&
    steps < maxSteps
  ) {
    const stepStart = performance.now();
    runner.step(bot);
    const stepDuration = performance.now() - stepStart;
    tickDurationsMs.push(stepDuration);
    steps += 1;

    if (game.state.active !== lastActive) {
      const holes = countBoardHoles(game.state.board);
      const nextBoardScore = scoreCharcuterieBoard(
        game.state.board,
        game.state.gameOver,
        game.state.totalLinesCleared,
      );
      const boardScoreDelta = previousBoardScore - nextBoardScore;
      boardScoreDeltas.push(boardScoreDelta);
      const reward = computePieceReward({
        modeId: config.modeId,
        linesDelta: game.state.totalLinesCleared - previousLines,
        scoreDelta: game.state.score - previousScore,
        timeDeltaMs: game.state.timeMs - previousTimeMs,
        holesDelta: holes - previousHoles,
        boardScoreDelta,
      });
      bot.onPiecePlaced(reward);
      previousLines = game.state.totalLinesCleared;
      previousScore = game.state.score;
      previousTimeMs = game.state.timeMs;
      previousHoles = holes;
      previousBoardScore = nextBoardScore;
      piecesPlaced += 1;
      lastActive = game.state.active;
    }
  }

  const transitions = bot.drainTransitions();
  const outcome: RolloutResult['outcome'] = game.state.gameWon
    ? 'game_won'
    : game.state.gameOver
      ? 'game_over'
      : 'manual';
  const terminalBonus = game.state.gameWon ? 1 : game.state.gameOver ? -1 : 0;
  if (transitions.length > 0) {
    const tail = transitions[transitions.length - 1];
    tail.reward = (tail.reward ?? 0) + terminalBonus;
  }
  const episodeReturn = transitions.reduce(
    (sum, item) => sum + (item.reward ?? 0),
    0,
  );
  return {
    transitions,
    episodeReturn,
    decisions,
    tickDurationsMs,
    steps,
    piecesPlaced,
    boardScoreDeltas,
    boardScoreStart,
    boardScoreEnd: previousBoardScore,
    startedAtMs,
    endedAtMs: Date.now(),
    outcome,
    terminal: {
      totalLinesCleared: Math.max(0, Math.trunc(game.state.totalLinesCleared)),
      score: Math.max(0, Math.trunc(game.state.score)),
      timeMs: Math.max(0, Math.trunc(game.state.timeMs)),
    },
  };
};

const computeDiscountedReturns = (
  rewards: number[],
  gamma: number,
): Float32Array => {
  const out = new Float32Array(rewards.length);
  let running = 0;
  for (let i = rewards.length - 1; i >= 0; i -= 1) {
    running = rewards[i] + gamma * running;
    out[i] = running;
  }
  return out;
};

const trainWithTfjs = async (options: {
  params: PolicyParams;
  observations: Float32Array;
  actions: Int32Array;
  returns: Float32Array;
  learningRate: number;
  entropyBeta: number;
  valueWeight: number;
  epochs: number;
}): Promise<{ params: PolicyParams; finalLoss: number | null }> => {
  const tf = await loadTf();
  await tf.ready();
  try {
    await tf.setBackend('webgl');
    await tf.ready();
  } catch {
    await tf.setBackend('cpu');
    await tf.ready();
  }

  const count = options.actions.length;
  if (count <= 0) {
    return { params: options.params, finalLoss: null };
  }

  const inputTensor = tf.tensor2d(options.observations, [
    count,
    options.params.inputDim,
  ]);
  const actionTensor = tf.tensor1d(options.actions, 'int32');
  const returnsTensor = tf.tensor1d(options.returns);
  const oneHot = tf.oneHot(actionTensor, options.params.actionDim);

  const w1 = tf.variable(
    tf.tensor2d(options.params.w1, [
      options.params.inputDim,
      options.params.hiddenDim,
    ]),
  );
  const b1 = tf.variable(tf.tensor1d(options.params.b1));
  const wp = tf.variable(
    tf.tensor2d(options.params.wp, [
      options.params.hiddenDim,
      options.params.actionDim,
    ]),
  );
  const bp = tf.variable(tf.tensor1d(options.params.bp));
  const wv = tf.variable(
    tf.randomNormal([options.params.hiddenDim, 1], 0, 0.02),
  );
  const bv = tf.variable(tf.scalar(0));

  const optimizer = tf.train.adam(options.learningRate);
  let finalLoss: number | null = null;
  for (let epoch = 0; epoch < options.epochs; epoch += 1) {
    const lossTensor = optimizer.minimize(() => {
      const hidden = tf.relu(tf.add(tf.matMul(inputTensor, w1), b1));
      const logits = tf.add(tf.matMul(hidden, wp), bp);
      const logProbs = tf.logSoftmax(logits, 1);
      const probs = tf.softmax(logits, 1);
      const selectedLogProb = tf.sum(tf.mul(logProbs, oneHot), 1);
      const values = tf.squeeze(tf.add(tf.matMul(hidden, wv), bv), [1]);
      const rawAdvantage = tf.sub(returnsTensor, values);
      const advMean = tf.mean(rawAdvantage);
      const centeredAdv = tf.sub(rawAdvantage, advMean);
      const advVar = tf.mean(tf.square(centeredAdv));
      const advStd = tf.sqrt(tf.add(advVar, tf.scalar(1e-8)));
      const normAdv = tf.div(centeredAdv, advStd);
      const policyLoss = tf.neg(
        tf.mean(tf.mul(selectedLogProb, tf.stopGradient(normAdv))),
      );
      const entropy = tf.neg(tf.mean(tf.sum(tf.mul(probs, logProbs), 1)));
      const valueLoss = tf.mean(tf.square(rawAdvantage));
      return tf.add(
        tf.add(policyLoss, tf.mul(valueLoss, tf.scalar(options.valueWeight))),
        tf.neg(tf.mul(entropy, tf.scalar(options.entropyBeta))),
      );
    }, true);
    if (lossTensor) {
      const data = lossTensor.dataSync() as Float32Array;
      finalLoss = Number(data[0]);
      lossTensor.dispose();
    }
  }

  const trained: PolicyParams = {
    inputDim: options.params.inputDim,
    hiddenDim: options.params.hiddenDim,
    actionDim: options.params.actionDim,
    w1: new Float32Array(w1.dataSync() as Float32Array),
    b1: new Float32Array(b1.dataSync() as Float32Array),
    wp: new Float32Array(wp.dataSync() as Float32Array),
    bp: new Float32Array(bp.dataSync() as Float32Array),
  };

  tf.dispose([
    inputTensor,
    actionTensor,
    returnsTensor,
    oneHot,
    w1,
    b1,
    wp,
    bp,
    wv,
    bv,
  ]);
  return { params: trained, finalLoss };
};

const toArtifact = (
  modeId: string,
  params: PolicyParams,
): BotPolicyArtifact => ({
  id: `bot_policy_${modeId}_${Date.now()}`,
  modeId,
  createdAtMs: Date.now(),
  inputDim: params.inputDim,
  hiddenDim: params.hiddenDim,
  actionDim: params.actionDim,
  actions: actionSpace.map((action) => ({ ...action })),
  weights: {
    w1: Array.from(params.w1),
    b1: Array.from(params.b1),
    wp: Array.from(params.wp),
    bp: Array.from(params.bp),
  },
});

const fromArtifact = (policy: BotPolicyArtifact): PolicyParams => ({
  inputDim: policy.inputDim,
  hiddenDim: policy.hiddenDim,
  actionDim: policy.actionDim,
  w1: new Float32Array(policy.weights.w1),
  b1: new Float32Array(policy.weights.b1),
  wp: new Float32Array(policy.weights.wp),
  bp: new Float32Array(policy.weights.bp),
});

const toDraftSamples = (
  decisions: ModelGeneratorDecisionEvent[],
  startedAtMs: number,
): BotTrajectoryDecisionSample[] => {
  const startedPerfMs = decisions.length > 0 ? decisions[0].wallTimeMs : 0;
  return decisions.map((decision, index) => {
    const createdAtMs =
      startedAtMs +
      Math.max(0, Math.trunc(decision.wallTimeMs - startedPerfMs));
    const pieces = [...decision.pieces];
    const actionIndex = Math.max(0, pieces.indexOf(decision.action));
    return {
      id: `bot_decision_${index + 1}`,
      createdAtMs,
      deliberationMs: decision.totalMs,
      boardOccupancy: boardOccupancy(decision.board),
      hold: decision.hold,
      action: decision.action,
      actionIndex,
      pieces,
      logits: Array.from(decision.logits),
      probabilities: Array.from(decision.probabilities),
      inferenceMs: decision.inferenceMs,
      samplingMs: decision.samplingMs,
      totalDecisionMs: decision.totalMs,
      reward: null,
    };
  });
};

export const trainBotPolicyOneShot = async (
  config: BotTrainOneShotConfig,
): Promise<BotTrainOneShotResult> => {
  const episodes = Math.max(1, Math.trunc(config.episodes ?? 24));
  const maxPiecesPerEpisode = Math.max(
    8,
    Math.trunc(config.maxPiecesPerEpisode ?? 120),
  );
  const gamma = clamp(config.gamma ?? 0.995, 0.8, 0.9999);
  const learningRate = clamp(config.learningRate ?? 0.0015, 1e-5, 0.05);
  const entropyBeta = clamp(config.entropyBeta ?? 0.01, 0, 1);
  const valueWeight = clamp(config.valueWeight ?? 0.5, 0, 10);
  const epochs = Math.max(1, Math.trunc(config.epochs ?? 6));
  const seed = Math.max(1, Math.trunc(config.seed ?? Date.now()));
  const pieceSourceProfile = normalizePieceSourceProfile(
    config.pieceSourceProfile,
  );

  const inputDim = encodeObservation(
    buildHeadlessGame({
      modeId: config.modeId,
      settings: config.settings,
      model: config.model,
      modelRunner: config.modelRunner,
      modelAxes: config.modelAxes,
      pieceSourceProfile,
      seed,
      onDecision: () => {},
    }).game.state,
  ).length;
  let params = randomizeParams(
    inputDim,
    64,
    actionSpace.length,
    new XorShift32(seed),
  );
  const transitions: Transition[] = [];
  const episodeReturns: number[] = [];

  for (let episode = 0; episode < episodes; episode += 1) {
    const rollout = runRollout({
      modeId: config.modeId,
      settings: config.settings,
      model: config.model,
      modelRunner: config.modelRunner,
      modelAxes: config.modelAxes,
      pieceSourceProfile,
      policyParams: params,
      seed: seed + episode * 997,
      maxPieces: maxPiecesPerEpisode,
      greedy: false,
    });
    episodeReturns.push(rollout.episodeReturn);
    const rewards = rollout.transitions.map((entry) => entry.reward ?? 0);
    const discounted = computeDiscountedReturns(rewards, gamma);
    for (let i = 0; i < rollout.transitions.length; i += 1) {
      transitions.push({
        observation: rollout.transitions[i].observation,
        actionIndex: rollout.transitions[i].actionIndex,
        reward: discounted[i],
      });
    }
  }

  if (transitions.length === 0) {
    return {
      ok: false,
      message: 'Bot training failed: no transitions collected.',
      episodes,
      meanReturn: 0,
      finalLoss: null,
      policyArtifact: null,
    };
  }

  const observations = new Float32Array(transitions.length * inputDim);
  const actions = new Int32Array(transitions.length);
  const returns = new Float32Array(transitions.length);
  for (let i = 0; i < transitions.length; i += 1) {
    observations.set(transitions[i].observation, i * inputDim);
    actions[i] = transitions[i].actionIndex;
    returns[i] = transitions[i].reward ?? 0;
  }

  const trained = await trainWithTfjs({
    params,
    observations,
    actions,
    returns,
    learningRate,
    entropyBeta,
    valueWeight,
    epochs,
  });
  params = trained.params;
  const policyArtifact = toArtifact(config.modeId, params);
  policyArtifact.archId = config.modelAxes.arch;
  policyArtifact.queuePolicyId = config.modelAxes.queuePolicyId;
  policyArtifact.pipelineId = 'bot_reinforce_v1';
  policyArtifact.pieceSourceProfile = pieceSourceProfile;

  return {
    ok: true,
    message:
      `Bot policy training complete (episodes=${episodes}, transitions=${transitions.length}, ` +
      `gamma=${gamma.toFixed(4)}).`,
    episodes,
    meanReturn: mean(episodeReturns),
    finalLoss: trained.finalLoss,
    policyArtifact,
  };
};

export const generateBotTrajectoryBatch = async (
  config: BotGenerateBatchConfig,
): Promise<BotGenerateBatchResult> => {
  const sessions = Math.max(1, Math.trunc(config.sessions));
  const maxPiecesPerEpisode = Math.max(
    8,
    Math.trunc(config.maxPiecesPerEpisode ?? 140),
  );
  const seed = Math.max(1, Math.trunc(config.seed ?? Date.now()));
  const pieceSourceProfile = normalizePieceSourceProfile(
    config.pieceSourceProfile ?? 'active_generator',
  );
  const params = fromArtifact(config.policy);
  const drafts: BotTrajectoryDraft[] = [];
  let samplesGenerated = 0;

  for (let index = 0; index < sessions; index += 1) {
    const rollout = runRollout({
      modeId: config.modeId,
      settings: config.settings,
      model: config.model,
      modelRunner: config.modelRunner,
      modelAxes: config.modelAxes,
      pieceSourceProfile,
      policyParams: params,
      seed: seed + index * 811,
      maxPieces: maxPiecesPerEpisode,
      greedy: false,
    });
    const samples = toDraftSamples(rollout.decisions, rollout.startedAtMs);
    if (samples.length === 0) {
      continue;
    }
    samplesGenerated += samples.length;
    drafts.push({
      modeId: config.modeId,
      startedAtMs: rollout.startedAtMs,
      endedAtMs: rollout.endedAtMs,
      durationMs: Math.max(0, rollout.endedAtMs - rollout.startedAtMs),
      outcome: rollout.outcome,
      terminal: rollout.terminal,
      samples,
    });
  }

  return {
    sessionsGenerated: drafts.length,
    samplesGenerated,
    drafts,
    trainingIntent: config.trainingIntent ?? null,
  };
};

export const runCapabilityBenchmark = async (
  config: CapabilityBenchmarkConfig,
): Promise<CapabilityBenchmarkResult> => {
  const episodes = Math.max(1, Math.trunc(config.episodes ?? 3));
  const maxPiecesPerEpisode = Math.max(
    8,
    Math.trunc(config.maxPiecesPerEpisode ?? 160),
  );
  const seed = Math.max(1, Math.trunc(config.seed ?? 42_030));
  const pieceSourceProfile = normalizePieceSourceProfile(
    config.pieceSourceProfile,
  );
  const params = fromArtifact(config.policy);
  const decisionDurations: number[] = [];
  const tickDurations: number[] = [];
  let totalSteps = 0;
  let totalDurationMs = 0;

  for (let episode = 0; episode < episodes; episode += 1) {
    const rollout = runRollout({
      modeId: config.modeId,
      settings: config.settings,
      model: config.model,
      modelRunner: config.modelRunner,
      modelAxes: config.modelAxes,
      pieceSourceProfile,
      policyParams: params,
      seed: seed + episode * 101,
      maxPieces: maxPiecesPerEpisode,
      greedy: true,
    });
    for (const decision of rollout.decisions) {
      if (Number.isFinite(decision.totalMs)) {
        decisionDurations.push(Math.max(0, decision.totalMs));
      }
    }
    tickDurations.push(...rollout.tickDurationsMs);
    totalSteps += rollout.steps;
    totalDurationMs += Math.max(0, rollout.endedAtMs - rollout.startedAtMs);
  }

  const meanDecisionMs = mean(decisionDurations);
  const p95DecisionMs = percentile(decisionDurations, 0.95);
  const meanTickMs = mean(tickDurations);
  const p95TickMs = percentile(tickDurations, 0.95);
  const simFps =
    totalDurationMs > 0 ? (totalSteps * 1000) / totalDurationMs : 0;
  const targetSimFps = 120;
  const simThroughput = targetSimFps > 0 ? simFps / targetSimFps : 0;

  const frameBudgetMs = 1000 / targetSimFps;
  let verdict: CapabilityBenchmarkResult['verdict'] = 'constrained';
  if (
    p95TickMs <= frameBudgetMs * 0.9 &&
    p95DecisionMs <= frameBudgetMs * 0.7
  ) {
    verdict = 'excellent';
  } else if (
    p95TickMs <= frameBudgetMs * 1.25 &&
    p95DecisionMs <= frameBudgetMs * 1.0
  ) {
    verdict = 'good';
  }
  const recommendedArch: CapabilityBenchmarkResult['recommendedArch'] =
    verdict === 'constrained' ? 'lean' : 'full';

  return {
    verdict,
    recommendedArch,
    metrics: {
      meanDecisionMs,
      p95DecisionMs,
      meanTickMs,
      p95TickMs,
      simThroughput,
      targetSimFps,
    },
  };
};

export const runHeadlessBotValidation = async (
  config: BotHeadlessValidateConfig,
): Promise<BotHeadlessValidateResult> => {
  const maxPieces = Math.max(1, Math.trunc(config.maxPieces ?? 10_000));
  const seed = Math.max(1, Math.trunc(config.seed ?? 42_030));
  const pieceSourceProfile = normalizePieceSourceProfile(
    config.pieceSourceProfile,
  );
  const params = fromArtifact(config.policy);
  const rollout = runRollout({
    modeId: config.modeId,
    settings: config.settings,
    model: config.model,
    modelRunner: config.modelRunner,
    modelAxes: config.modelAxes,
    pieceSourceProfile,
    policyParams: params,
    seed,
    maxPieces,
    greedy: true,
  });
  const decisionDurations = rollout.decisions
    .map((decision) =>
      Number.isFinite(decision.totalMs) ? Math.max(0, decision.totalMs) : null,
    )
    .filter((value): value is number => value != null);
  const totalReward = rollout.transitions.reduce(
    (sum, transition) => sum + (transition.reward ?? 0),
    0,
  );
  return {
    piecesSurvived: rollout.piecesPlaced,
    outcome: rollout.outcome,
    meanDecisionMs: mean(decisionDurations),
    p95DecisionMs: percentile(decisionDurations, 0.95),
    meanTickMs: mean(rollout.tickDurationsMs),
    p95TickMs: percentile(rollout.tickDurationsMs, 0.95),
    meanBoardScoreDelta: mean(rollout.boardScoreDeltas),
    totalReward,
    startedAtMs: rollout.startedAtMs,
    endedAtMs: rollout.endedAtMs,
    durationMs: Math.max(0, rollout.endedAtMs - rollout.startedAtMs),
  };
};

export type BotGuiInputSourceConfig = {
  policy: BotPolicyArtifact;
  apmInput: number;
  seed?: number;
  greedy?: boolean;
};

export const createGuiInspectBotInputSource = (
  config: BotGuiInputSourceConfig,
): InputSource => {
  const params = fromArtifact(config.policy);
  const seed = Math.max(1, Math.trunc(config.seed ?? Date.now()));
  const greedy = config.greedy !== false;
  const clampedApm = clamp(config.apmInput, 20, 1200);
  const actionIntervalMs = 60_000 / clampedApm;
  let rng = new XorShift32(seed ^ 0x517cc1b7);
  let activeRef: GameState['active'] | null = null;
  let queue: InputFrame[] = [];
  let cooldownMs = 0;

  const nextFrame = (frame: InputFrame): InputFrame => ({
    ...EMPTY_INPUT,
    moveX: frame.moveX,
    rotate: frame.rotate,
    rotate180: frame.rotate180,
    softDrop: frame.softDrop,
    hardDrop: frame.hardDrop,
    hold: frame.hold,
  });

  const queueFromMacro = (action: BotMacroAction): InputFrame[] => {
    const frames: InputFrame[] = [];
    if (action.rotation === 'cw') {
      frames.push(
        nextFrame({
          ...EMPTY_INPUT,
          moveX: 0,
          rotate: 1,
          rotate180: false,
          softDrop: false,
          hardDrop: false,
          hold: false,
          restart: false,
        }),
      );
    } else if (action.rotation === 'ccw') {
      frames.push(
        nextFrame({
          ...EMPTY_INPUT,
          moveX: 0,
          rotate: -1,
          rotate180: false,
          softDrop: false,
          hardDrop: false,
          hold: false,
          restart: false,
        }),
      );
    } else if (action.rotation === '180') {
      frames.push(
        nextFrame({
          ...EMPTY_INPUT,
          moveX: 0,
          rotate: 0,
          rotate180: true,
          softDrop: false,
          hardDrop: false,
          hold: false,
          restart: false,
        }),
      );
    }

    const steps = Math.max(0, Math.min(10, Math.abs(Math.trunc(action.moveX))));
    const dir = action.moveX < 0 ? -1 : 1;
    for (let i = 0; i < steps; i += 1) {
      frames.push(
        nextFrame({
          ...EMPTY_INPUT,
          moveX: dir,
          rotate: 0,
          rotate180: false,
          softDrop: false,
          hardDrop: false,
          hold: false,
          restart: false,
        }),
      );
    }

    frames.push(
      nextFrame({
        ...EMPTY_INPUT,
        moveX: 0,
        rotate: 0,
        rotate180: false,
        softDrop: false,
        hardDrop: true,
        hold: false,
        restart: false,
      }),
    );
    return frames;
  };

  return {
    sample: (state, dtMs) => {
      cooldownMs = Math.max(0, cooldownMs - Math.max(0, dtMs));
      if (state.active !== activeRef) {
        activeRef = state.active;
        const observation = encodeObservation(state);
        const forward = forwardPolicy(params, observation);
        let actionIndex = 0;
        if (greedy) {
          let best = forward.logits[0] ?? Number.NEGATIVE_INFINITY;
          for (let i = 1; i < forward.logits.length; i += 1) {
            if (forward.logits[i] > best) {
              best = forward.logits[i];
              actionIndex = i;
            }
          }
        } else {
          actionIndex = sampleIndex(forward.probabilities, rng);
        }
        const action = actionSpace[actionIndex] ?? actionSpace[0];
        queue = queueFromMacro(action);
      }
      if (queue.length === 0) return EMPTY_INPUT;
      if (cooldownMs > 0) return EMPTY_INPUT;
      cooldownMs = actionIntervalMs;
      return queue.shift() ?? EMPTY_INPUT;
    },
    reset: (nextSeed) => {
      const seeded = Math.max(1, Math.trunc(nextSeed));
      rng = new XorShift32(seeded ^ 0x517cc1b7);
      activeRef = null;
      queue = [];
      cooldownMs = 0;
    },
  };
};
