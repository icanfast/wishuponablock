import type { GameMode, ModeOptions } from './modes';
import type { Game } from './game';
import type { GameRunner } from './runner';
import type { Settings } from './settings';

export type GameSession = {
  getGame: () => Game;
  getRunner: () => GameRunner;
  getMode: () => GameMode;
  getModeOptions: () => ModeOptions;
  setMode: (mode: GameMode, options: ModeOptions) => void;
  rebuild: (settings: Settings) => void;
  resetActive: (seed?: number) => void;
  setConfig: (settings: Settings) => void;
};

type RestartContext = {
  settings: Settings;
  mode: GameMode;
  options: ModeOptions;
  game: Game;
  resetActive: (seed?: number) => void;
  rebuild: (settings: Settings) => void;
};

type GameSessionOptions = {
  initialMode: GameMode;
  initialModeOptions: ModeOptions;
  buildGame: (settings: Settings, mode: GameMode, options: ModeOptions) => Game;
  createRunner: (
    game: Game,
    mode: GameMode,
    options: ModeOptions,
    onRestart: () => void,
  ) => GameRunner;
  onRestart: (context: RestartContext) => void;
};

export function createGameSession(
  settings: Settings,
  options: GameSessionOptions,
): GameSession {
  const {
    initialMode,
    initialModeOptions,
    buildGame,
    createRunner,
    onRestart,
  } = options;

  let currentSettings = settings;
  let mode: GameMode = initialMode;
  let modeOptions: ModeOptions = initialModeOptions;

  let game: Game = buildGame(currentSettings, mode, modeOptions);
  let runner: GameRunner;

  const rebuild = (nextSettings: Settings) => {
    currentSettings = nextSettings;
    game = buildGame(currentSettings, mode, modeOptions);
    runner = createRunner(game, mode, modeOptions, handleRestart);
  };

  const resetActive = (seed = Date.now()) => {
    game.reset(seed);
  };

  const handleRestart = () => {
    onRestart({
      settings: currentSettings,
      mode,
      options: modeOptions,
      game,
      resetActive,
      rebuild,
    });
  };

  runner = createRunner(game, mode, modeOptions, handleRestart);

  const setMode = (nextMode: GameMode, nextOptions: ModeOptions) => {
    mode = nextMode;
    modeOptions = nextOptions;
  };

  const setConfig = (nextSettings: Settings) => {
    currentSettings = nextSettings;
    const butterfinger = nextSettings.butterfinger;
    game.setConfig({
      ...nextSettings.game,
      lockNudgeRate: butterfinger.enabled ? butterfinger.lockNudgeRate : 0,
      gravityDropRate: butterfinger.enabled ? butterfinger.gravityDropRate : 0,
      lockRotateRate: butterfinger.enabled ? butterfinger.lockRotateRate : 0,
    });
  };

  return {
    getGame: () => game,
    getRunner: () => runner,
    getMode: () => mode,
    getModeOptions: () => modeOptions,
    setMode,
    rebuild,
    resetActive,
    setConfig,
  };
}
