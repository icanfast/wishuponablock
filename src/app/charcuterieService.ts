import type { Game } from '../core/game';
import type { GameMode, ModeOptions } from '../core/modes';
import { GameRunner, type InputSource } from '../core/runner';
import type { Settings } from '../core/settings';
import { CharcuterieBot, runBotForPieces } from '../bot/charcuterieBot';
import type { Board } from '../core/types';
import {
  createGuiInspectBotInputSource,
  type BotPolicyArtifact,
} from './headlessBotService';

export type CharcuterieScoreWeights = {
  height: number;
  holes: number;
  blocks: number;
  clears: number;
};

export type CharcuterieHoleWeights = {
  bottom: number;
  mid: number;
};

export type CharcuterieServiceOptions = {
  rows: number;
  defaultSimCount: number;
  defaultTargetFilledCells: number;
  defaultTemperature: number;
  scoreWeights: CharcuterieScoreWeights;
  holeWeights: CharcuterieHoleWeights;
  buildGame: (cfg: Settings, mode: GameMode, seed: number) => Game;
  createFinalGame: (cfg: Settings, mode: GameMode, seed: number) => Game;
  resolveBotPolicy?: () => BotPolicyArtifact | null;
  onDebug?: (message: string) => void;
};

const defaultOnDebug = (message: string) => {
  console.info(message);
};

const BOT_CHARCUTERIE_HEIGHT_LIMIT = 15;
const BOT_CHARCUTERIE_MAX_HEIGHT_RESTARTS = 2;

export function createCharcuterieGame(
  cfg: Settings,
  mode: GameMode,
  options: ModeOptions,
  serviceOptions: CharcuterieServiceOptions,
): Game {
  const {
    rows,
    defaultSimCount,
    defaultTargetFilledCells,
    defaultTemperature,
    scoreWeights,
    holeWeights,
    buildGame,
    createFinalGame,
    resolveBotPolicy,
    onDebug = defaultOnDebug,
  } = serviceOptions;
  const pieces = Math.max(0, Math.trunc(Number(options.pieces ?? 0)));
  const sims = Math.max(
    1,
    Math.trunc(Number(options.simCount ?? defaultSimCount)),
  );
  const targetFilledCells = Math.max(
    1,
    Math.min(
      rows * 10,
      Math.trunc(Number(options.targetFilledCells ?? defaultTargetFilledCells)),
    ),
  );
  const legacyPieces =
    pieces > 0 ? pieces : Math.max(1, Math.round(targetFilledCells / 4));
  const temperature = Math.max(
    0.05,
    Number(options.temperature ?? defaultTemperature),
  );
  const seedOverride = options.seed;
  const baseSeed =
    seedOverride !== undefined ? Math.trunc(seedOverride) : Date.now();
  const simStart = performance.now();
  const policy = resolveBotPolicy?.() ?? null;

  if (policy) {
    try {
      let acceptedResult: ReturnType<typeof runBotUntilFilledCells> | null =
        null;
      let acceptedSeed = baseSeed;
      let heightRestarts = 0;
      let attempts = 0;
      for (
        let attempt = 0;
        attempt <= BOT_CHARCUTERIE_MAX_HEIGHT_RESTARTS;
        attempt += 1
      ) {
        attempts = attempt + 1;
        const attemptSeed = baseSeed + attempt * 977;
        const game = buildGame(cfg, mode, attemptSeed);
        const bot = createGuiInspectBotInputSource({
          model: null,
          policy,
          apmInput: 1200,
          executionMode: 'instant',
          seed: attemptSeed ^ 0x9e3779b9,
          greedy: false,
          samplingTemperature: temperature,
          debugTrace: false,
        });
        const result = runBotUntilFilledCells(
          game,
          bot,
          targetFilledCells,
          1000 / 120,
          BOT_CHARCUTERIE_HEIGHT_LIMIT,
        );
        acceptedResult = result;
        acceptedSeed = attemptSeed;
        if (
          result.heightExceeded &&
          attempt < BOT_CHARCUTERIE_MAX_HEIGHT_RESTARTS
        ) {
          heightRestarts += 1;
          continue;
        }
        break;
      }
      const result = acceptedResult;
      if (!result) {
        throw new Error('Bot policy generation returned no rollout result.');
      }
      const finalBoard = result.board;
      const filledCells = result.filledCells;
      const simElapsedMs = performance.now() - simStart;
      onDebug(
        `[Charcuterie] bot policy=${policy.id} targetFilled=${targetFilledCells} ` +
          `filled=${filledCells} pieces=${result.placed} temp=${temperature.toFixed(
            2,
          )} lines=${result.linesCleared} outcome=${result.outcome} ` +
          `maxHeight=${result.maxHeightReached} heightLimit=${BOT_CHARCUTERIE_HEIGHT_LIMIT} ` +
          `heightRestarts=${heightRestarts}/${BOT_CHARCUTERIE_MAX_HEIGHT_RESTARTS} ` +
          `heightExceeded=${result.heightExceeded ? 'y' : 'n'} attempts=${attempts} ` +
          `seed=${acceptedSeed} elapsedMs=${simElapsedMs.toFixed(1)}`,
      );
      const finalGame = createFinalGame(cfg, mode, acceptedSeed);
      finalGame.applyInitialBoard(finalBoard);
      finalGame.markInitialBlocks();
      return finalGame;
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      onDebug(
        `[Charcuterie] bot policy generation failed (${detail}); falling back to legacy random simulation.`,
      );
    }
  }

  if (legacyPieces === 0 || sims === 1) {
    const game = buildGame(cfg, mode, baseSeed);
    if (legacyPieces > 0) {
      const bot = new CharcuterieBot(baseSeed ^ 0x9e3779b9);
      runBotForPieces(game, bot, legacyPieces);
    }
    const finalGame = createFinalGame(cfg, mode, baseSeed);
    if (legacyPieces > 0) {
      finalGame.applyInitialBoard(game.state.board);
    }
    finalGame.markInitialBlocks();
    return finalGame;
  }

  onDebug(
    '[Charcuterie] no bot policy available; falling back to legacy random simulation.',
  );
  let bestGame: Game | null = null;
  let bestScore = Number.POSITIVE_INFINITY;
  let bestHeight = Number.POSITIVE_INFINITY;
  let bestHoles = Number.POSITIVE_INFINITY;
  let bestBlocks = Number.POSITIVE_INFINITY;
  let bestClears = 0;

  let totalClears = 0;
  let simsWithClears = 0;
  let maxClears = 0;

  for (let i = 0; i < sims; i++) {
    const seed = baseSeed + i * 977;
    const game = buildGame(cfg, mode, seed);
    const bot = new CharcuterieBot(seed ^ 0x9e3779b9);
    runBotForPieces(game, bot, legacyPieces);

    const clears = game.totalLinesCleared;
    const scored = scoreCharcuterieBoard(
      game.state.board,
      game.state.gameOver,
      clears,
      rows,
      scoreWeights,
      holeWeights,
    );
    totalClears += clears;
    if (clears > 0) simsWithClears += 1;
    if (clears > maxClears) maxClears = clears;

    if (
      scored.score < bestScore ||
      (scored.score === bestScore && scored.height < bestHeight) ||
      (scored.score === bestScore &&
        scored.height === bestHeight &&
        scored.holes < bestHoles)
    ) {
      bestGame = game;
      bestScore = scored.score;
      bestHeight = scored.height;
      bestHoles = scored.holes;
      bestBlocks = scored.blocks;
      bestClears = clears;
    }
  }

  const simElapsedMs = performance.now() - simStart;
  onDebug(
    `[Charcuterie] sims=${sims} pieces=${pieces} ` +
      `(legacyPieces=${legacyPieces}) ` +
      `bestScore=${bestScore.toFixed(2)} ` +
      `height=${bestHeight} holes=${bestHoles} blocks=${bestBlocks.toFixed(
        0,
      )} clears=${bestClears} ` +
      `clearsTotal=${totalClears} simsWithClears=${simsWithClears} maxClears=${maxClears} ` +
      `seed=${baseSeed} ` +
      `elapsedMs=${simElapsedMs.toFixed(1)}`,
  );

  const finalGame = createFinalGame(cfg, mode, baseSeed);
  if (bestGame) {
    finalGame.applyInitialBoard(bestGame.state.board);
  }
  finalGame.markInitialBlocks();
  return finalGame;
}

function runBotUntilFilledCells(
  game: Game,
  bot: InputSource,
  targetFilledCells: number,
  fixedStepMs = 1000 / 120,
  heightLimit = BOT_CHARCUTERIE_HEIGHT_LIMIT,
): {
  placed: number;
  board: Board;
  filledCells: number;
  linesCleared: number;
  maxHeightReached: number;
  heightExceeded: boolean;
  outcome: 'target' | 'game_over' | 'piece_cap' | 'step_cap' | 'height_cap';
} {
  const runner = new GameRunner(game, { fixedStepMs });
  let placed = 0;
  let lastActive = game.state.active;
  let previousBoard = cloneBoard(game.state.board);
  let previousLines = game.state.totalLinesCleared;
  const maxPieces = Math.max(32, targetFilledCells * 2);
  const maxSteps = maxPieces * 32;
  let bestBoard: Board = cloneBoard(game.state.board);
  let bestFilledCells = countBlocks(game.state.board);
  let bestDistance = Number.POSITIVE_INFINITY;
  let maxHeightReached = getStackHeight(
    game.state.board,
    game.state.board.length,
  );
  let heightExceeded = false;
  let outcome:
    | 'target'
    | 'game_over'
    | 'piece_cap'
    | 'step_cap'
    | 'height_cap' = 'step_cap';

  for (let step = 0; step < maxSteps; step += 1) {
    if (game.state.gameOver) {
      outcome = 'game_over';
      break;
    }
    runner.step(bot);
    if (game.state.active !== lastActive) {
      const boardChanged = !boardsEqual(previousBoard, game.state.board);
      const linesChanged = game.state.totalLinesCleared !== previousLines;
      lastActive = game.state.active;
      if (boardChanged || linesChanged) {
        placed += 1;
        previousBoard = cloneBoard(game.state.board);
        previousLines = game.state.totalLinesCleared;
        const filledCells = countBlocks(game.state.board);
        const distance = Math.abs(filledCells - targetFilledCells);
        if (
          distance < bestDistance ||
          (distance === bestDistance && filledCells >= targetFilledCells)
        ) {
          bestBoard = cloneBoard(game.state.board);
          bestFilledCells = filledCells;
          bestDistance = distance;
        }
        const stackHeight = getStackHeight(
          game.state.board,
          game.state.board.length,
        );
        maxHeightReached = Math.max(maxHeightReached, stackHeight);
        if (stackHeight > heightLimit) {
          heightExceeded = true;
          outcome = 'height_cap';
          break;
        }
        if (filledCells >= targetFilledCells) {
          outcome = 'target';
          break;
        }
        if (placed >= maxPieces) {
          outcome = 'piece_cap';
          break;
        }
      }
    }
  }

  return {
    placed,
    board: bestBoard,
    filledCells: bestFilledCells,
    linesCleared: Math.max(0, Math.trunc(game.totalLinesCleared)),
    maxHeightReached,
    heightExceeded,
    outcome,
  };
}

function scoreCharcuterieBoard(
  board: Board,
  gameOver: boolean,
  clears: number,
  rows: number,
  scoreWeights: CharcuterieScoreWeights,
  holeWeights: CharcuterieHoleWeights,
): {
  score: number;
  height: number;
  holes: number;
  blocks: number;
  clears: number;
} {
  const height = getStackHeight(board, rows) + (gameOver ? rows : 0);
  const holes = getHolePenalty(board, rows, holeWeights);
  const blocks = countBlocks(board);
  const score =
    height * scoreWeights.height +
    holes * scoreWeights.holes +
    blocks * scoreWeights.blocks -
    clears * scoreWeights.clears;
  return { score, height, holes, blocks, clears };
}

function getStackHeight(board: Board, rows: number): number {
  for (let y = 0; y < rows; y++) {
    if (board[y].some((cell) => cell != null)) {
      return rows - y;
    }
  }
  return 0;
}

function countBlocks(board: Board): number {
  return board.reduce((sum, row) => {
    const rowCount = row.reduce((rowSum, cell) => rowSum + (cell ? 1 : 0), 0);
    return sum + rowCount;
  }, 0);
}

function cloneBoard(board: Board): Board {
  return board.map((row) => row.slice());
}

function boardsEqual(left: Board, right: Board): boolean {
  if (left.length !== right.length) return false;
  for (let y = 0; y < left.length; y += 1) {
    const leftRow = left[y];
    const rightRow = right[y];
    if (leftRow.length !== rightRow.length) return false;
    for (let x = 0; x < leftRow.length; x += 1) {
      if (leftRow[x] !== rightRow[x]) {
        return false;
      }
    }
  }
  return true;
}

function getHolePenalty(
  board: Board,
  rows: number,
  holeWeights: CharcuterieHoleWeights,
): number {
  let penalty = 0;
  for (let y = rows - 1; y >= 0; y--) {
    const row = board[y];
    let empty = 0;
    let anyFilled = false;
    for (const cell of row) {
      if (cell == null) {
        empty++;
      } else {
        anyFilled = true;
      }
    }
    if (!anyFilled) continue;
    const extraHoles = Math.max(0, empty - 1);
    if (extraHoles === 0) continue;

    const depth = rows - 1 - y;
    if (depth < 4) {
      penalty += extraHoles * holeWeights.bottom;
    } else if (depth < 8) {
      penalty += extraHoles * holeWeights.mid;
    }
  }
  return penalty;
}
