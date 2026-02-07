import type { Settings } from '../core/settings';
import type { GameMode, ModeOptions } from '../core/modes';
import type { GameSession } from '../core/gameSession';

export type SessionController = {
  getSettings: () => Settings;
  setSettings: (settings: Settings) => void;
  rebuildSession: () => void;
  setMode: (mode: GameMode, options: ModeOptions) => void;
  handleGeneratorChange: (settings: Settings) => void;
  applyConfig: (settings: Settings) => void;
};

type SessionControllerOptions = {
  settings: Settings;
  session: GameSession;
  onRebuild?: (settings: Settings) => void;
};

export function createSessionController(
  options: SessionControllerOptions,
): SessionController {
  let currentSettings = options.settings;
  const { session } = options;

  return {
    getSettings: () => currentSettings,
    setSettings: (settings) => {
      currentSettings = settings;
    },
    rebuildSession: () => {
      session.rebuild(currentSettings);
      options.onRebuild?.(currentSettings);
    },
    setMode: (mode, options) => {
      session.setMode(mode, options);
    },
    handleGeneratorChange: (settings) => {
      currentSettings = settings;
      session.rebuild(settings);
      options.onRebuild?.(settings);
    },
    applyConfig: (settings) => {
      currentSettings = settings;
      session.setConfig(settings);
    },
  };
}
