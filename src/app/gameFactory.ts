import { Game } from '../core/game';
import type { GameSession } from '../core/gameSession';
import { createGameSession } from '../core/gameSession';
import type { Settings } from '../core/settings';
import type { GameMode, ModeOptions } from '../core/modes';
import type { Board, PieceKind } from '../core/types';
import { createGeneratorFactory } from '../core/generators';
import { GameRunner } from '../core/runner';
import { applyModeSettings, runModeStart } from './modeService';
import {
  createCharcuterieGame,
  type CharcuterieHoleWeights,
  type CharcuterieScoreWeights,
} from './charcuterieService';
import type { ModelService } from './modelService';

export type GameFactoryOptions = {
  settings: Settings;
  initialMode: GameMode;
  initialModeOptions: ModeOptions;
  modelService: ModelService;
  onPieceLock: (board: Board, hold: PieceKind | null) => void;
  onHold: (board: Board, hold: PieceKind | null) => void;
  onBeforeRestart?: () => void;
  setLockEffectsSuppressed: (value: boolean) => void;
  charcuterie: {
    rows: number;
    defaultSimCount: number;
    scoreWeights: CharcuterieScoreWeights;
    holeWeights: CharcuterieHoleWeights;
    onDebug?: (message: string) => void;
  };
  runnerOptions?: {
    fixedStepMs: number;
    maxElapsedMs?: number;
    maxStepsPerTick?: number;
  };
};

export function createGameSessionFactory(
  options: GameFactoryOptions,
): GameSession {
  const {
    settings,
    initialMode,
    initialModeOptions,
    modelService,
    onPieceLock,
    onHold,
    onBeforeRestart,
    setLockEffectsSuppressed,
    charcuterie,
    runnerOptions = {
      fixedStepMs: 1000 / 120,
      maxElapsedMs: 250,
      maxStepsPerTick: 10,
    },
  } = options;

  const buildGame = (cfg: Settings, mode: GameMode, seed: number): Game => {
    const merged = applyModeSettings(cfg, mode);
    return new Game({
      seed,
      ...merged.game,
      lockNudgeRate: cfg.butterfinger.enabled
        ? cfg.butterfinger.lockNudgeRate
        : 0,
      gravityDropRate: cfg.butterfinger.enabled
        ? cfg.butterfinger.gravityDropRate
        : 0,
      lockRotateRate: cfg.butterfinger.enabled
        ? cfg.butterfinger.lockRotateRate
        : 0,
      generatorFactory: createGeneratorFactory(merged.generator, {
        mlModel: modelService.getModel(),
        mlModelPromise: modelService.getModelPromise() ?? undefined,
      }),
      onPieceLock,
      onHold,
    });
  };

  const createGame = (
    cfg: Settings,
    mode: GameMode,
    options: ModeOptions,
  ): Game => {
    if (mode.id === 'charcuterie') {
      setLockEffectsSuppressed(true);
      try {
        return createCharcuterieGame(cfg, mode, options, {
          rows: charcuterie.rows,
          defaultSimCount: charcuterie.defaultSimCount,
          scoreWeights: charcuterie.scoreWeights,
          holeWeights: charcuterie.holeWeights,
          buildGame: (nextCfg, nextMode, seed) =>
            buildGame(
              {
                ...nextCfg,
                generator: {
                  ...nextCfg.generator,
                  type: 'bag7',
                },
              },
              nextMode,
              seed,
            ),
          createFinalGame: (nextCfg, nextMode, seed) =>
            buildGame(nextCfg, nextMode, seed),
          onDebug: charcuterie.onDebug,
        });
      } finally {
        setLockEffectsSuppressed(false);
      }
    }

    const game = buildGame(cfg, mode, Date.now());
    runModeStart(game, mode, options);
    return game;
  };

  const session: GameSession = createGameSession(settings, {
    initialMode,
    initialModeOptions,
    buildGame: createGame,
    createRunner: (g, mode, options, onRestart) =>
      new GameRunner(g, {
        fixedStepMs: runnerOptions.fixedStepMs,
        maxElapsedMs: runnerOptions.maxElapsedMs,
        maxStepsPerTick: runnerOptions.maxStepsPerTick,
        onRestart,
      }),
    onRestart: ({ settings, mode, options, resetActive, rebuild }) => {
      onBeforeRestart?.();
      if (mode.id === 'charcuterie') {
        rebuild(settings);
        return;
      }
      resetActive();
      runModeStart(session.getGame(), mode, options);
    },
  });

  return session;
}
