import type { Game } from '../core/game';
import type { GameMode, ModeOptions } from '../core/modes';
import type { Settings } from '../core/settings';
import { CharcuterieBot, runBotForPieces } from '../bot/charcuterieBot';
import type { Board } from '../core/types';

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
  scoreWeights: CharcuterieScoreWeights;
  holeWeights: CharcuterieHoleWeights;
  buildGame: (cfg: Settings, mode: GameMode, seed: number) => Game;
  createFinalGame: (cfg: Settings, mode: GameMode, seed: number) => Game;
  onDebug?: (message: string) => void;
};

const defaultOnDebug = (message: string) => {
  console.info(message);
};

export function createCharcuterieGame(
  cfg: Settings,
  mode: GameMode,
  options: ModeOptions,
  serviceOptions: CharcuterieServiceOptions,
): Game {
  const {
    rows,
    defaultSimCount,
    scoreWeights,
    holeWeights,
    buildGame,
    createFinalGame,
    onDebug = defaultOnDebug,
  } = serviceOptions;
  const pieces = Math.max(0, Math.trunc(Number(options.pieces ?? 0)));
  const sims = Math.max(
    1,
    Math.trunc(Number(options.simCount ?? defaultSimCount)),
  );
  const seedOverride = options.seed;
  const baseSeed =
    seedOverride !== undefined ? Math.trunc(seedOverride) : Date.now();
  const simStart = performance.now();

  if (pieces === 0 || sims === 1) {
    const game = buildGame(cfg, mode, baseSeed);
    if (pieces > 0) {
      const bot = new CharcuterieBot(baseSeed ^ 0x9e3779b9);
      runBotForPieces(game, bot, pieces);
    }
    const finalGame = createFinalGame(cfg, mode, baseSeed);
    if (pieces > 0) {
      finalGame.applyInitialBoard(game.state.board);
    }
    finalGame.markInitialBlocks();
    return finalGame;
  }

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
    runBotForPieces(game, bot, pieces);

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
